import os
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import yaml
import argparse
import pickle
import logging

from schnetpack.data import ASEAtomsData

from mlff_qd.utils.logging_utils import timer, setup_logging
from mlff_qd.utils.data_processing import ( load_data, preprocess_data, setup_logging_and_dataset,
        prepare_transformations, setup_data_module, show_dataset_info )
from mlff_qd.utils.model import setup_model
from mlff_qd.utils.helpers import load_config, parse_args

def convert_units(value, from_unit, to_unit):
    """Convert energy or force values between different unit systems."""
    conversion_factors = {
        # Energy conversions
        ("Hartree", "eV"): 27.2114,
        ("eV", "Hartree"): 1 / 27.2114,
        ("kcal/mol", "eV"): 0.0433641,
        ("eV", "kcal/mol"): 1 / 0.0433641,
        ("kJ/mol", "eV"): 0.010364,
        ("eV", "kJ/mol"): 1 / 0.010364,

        # Force conversions
        ("Hartree/Bohr", "eV/Ang"): 51.4221,
        ("eV/Ang", "Hartree/Bohr"): 1 / 51.4221,
        ("kcal/mol/Ang", "eV/Ang"): 0.0433641,
        ("eV/Ang", "kcal/mol/Ang"): 1 / 0.0433641,
        ("kJ/mol/Ang", "eV/Ang"): 0.010364,
        ("eV/Ang", "kJ/mol/Ang"): 1 / 0.010364
    }

    if from_unit == to_unit:
        return value  # No conversion needed

    key = (from_unit, to_unit)
    if key in conversion_factors:
        return value * conversion_factors[key]
    else:
        raise ValueError(f"Unsupported conversion: {from_unit} to {to_unit}")
        
def save_energies(all_actual_energy,all_predicted_energy, dataset_type):

    try:
        data = {
            'Actual Energy': np.concatenate(all_actual_energy).flatten(),
            'Predicted Energy': np.concatenate(all_predicted_energy).flatten(),
        }
        df = pd.DataFrame(data)
        csv_file_path = os.path.join(os.getcwd(), f'{dataset_type}_predictions.csv')
        df.to_csv(csv_file_path, index=False)
        print(f"Results saved to {csv_file_path}")
    except Exception as e:
        print(f"Error saving energy results: {e}")

def save_forces(all_actual_forces_flat,all_predicted_forces_flat, dataset_type):
    
    try:
        forces_data = {
            'Actual Forces': all_actual_forces_flat,
            'Predicted Forces': all_predicted_forces_flat,
        }
        forces_pkl_file_path = os.path.join(os.getcwd(), f'{dataset_type}_forces.pkl')
        with open(forces_pkl_file_path, 'wb') as f:
            pickle.dump(forces_data, f)
        print(f"Forces data saved to {forces_pkl_file_path}")
    except Exception as e:
        print(f"Error saving force results: {e}")
        
def run_inference(loader, dataset_type, best_model, device, property_units, new_dataset):
    all_actual_energy = []
    all_predicted_energy = []
    all_actual_forces = []
    all_predicted_forces = []
    
    for batch in tqdm(loader, desc=f"Running inference on {dataset_type} data"):
        batch = {key: value.to(device) for key, value in batch.items()}
        batch['positions'] = batch['_positions']
        batch['positions'].requires_grad_()
        exclude_keys = ['energy', 'forces']

        input_batch = {k: batch[k] for k in batch if k not in exclude_keys}

        result = best_model(input_batch)

        # Collect energies
        actual_energy = batch['energy'].detach().cpu().numpy()
        predicted_energy = result['energy'].detach().cpu().numpy()
        
        all_actual_energy.append(actual_energy)
        all_predicted_energy.append(predicted_energy)

        # Collect forces
        actual_forces = batch['forces'].detach().cpu().numpy()
        predicted_forces = result['forces'].detach().cpu().numpy()
        all_actual_forces.append(actual_forces)
        all_predicted_forces.append(predicted_forces)
     
    # Save results for this dataset
    save_energies(all_actual_energy,all_predicted_energy, dataset_type)

    # Reshape force arrays to maintain three-dimensional vector form
    all_actual_forces_flat = np.concatenate(all_actual_forces).reshape(-1, 3)
    all_predicted_forces_flat = np.concatenate(all_predicted_forces).reshape(-1, 3)

    # Compute MAEs for all properties
    energy_mae = mean_absolute_error(np.concatenate(all_actual_energy), np.concatenate(all_predicted_energy))
    forces_mae = mean_absolute_error(all_actual_forces_flat, all_predicted_forces_flat)

    total_atoms = new_dataset[0]['_n_atoms'].item()  # Get total atoms from the dataset
    energy_mae_per_atom = (energy_mae / total_atoms)

    print(f"Energy MAE on {dataset_type} data: {energy_mae} {property_units['energy']}") 
    print(f"Energy MAE per Atom on {dataset_type} data: {energy_mae_per_atom} {property_units['energy']}") 
    print(f"Forces MAE on {dataset_type} data: {forces_mae} {property_units['forces']}") 

    # Save forces in a pickle file
    save_forces(all_actual_forces_flat,all_predicted_forces_flat, dataset_type)


    # Read energy and force units from MLFF setup
    energy_unit = property_units['energy']
    force_unit = property_units['forces']

    # Define reference convergence values in eV and eV/Ang
    energy_convergence_eV = 0.01  # 10 meV = 0.01 eV
    force_convergence_strict_eV_A = 0.05  # 0.05 eV/Ang
    force_convergence_loose_eV_A = 0.1  # 0.1 eV/Ang

    # Convert MLFF values to eV and eV/Ang
    energy_mae_per_atom_eV = convert_units(energy_mae_per_atom, energy_unit, "eV")
    forces_mae_eV_A = convert_units(forces_mae, force_unit, "eV/Ang")

    # Check convergence
    if energy_mae_per_atom_eV < energy_convergence_eV and forces_mae_eV_A < force_convergence_strict_eV_A:
        print("MLFF converged!")
    elif energy_mae_per_atom_eV < 2 * energy_convergence_eV and forces_mae_eV_A < force_convergence_loose_eV_A:
        print("MLFF near convergence...")
    else:
        print("MLFF not yet converged. Continuing training...")

@timer
def main():
    args = parse_args()
    config = load_config(args.config)

    trained_model_path = config['settings']['testing']['trained_model_path']
    print(f"Trained model path: {trained_model_path}")
    db_path = os.path.join(trained_model_path, config['settings']['general']['database_name'])
    property_units = {
        'energy': config['settings']['model']['property_unit_dict']['energy'],
        'forces': config['settings']['model']['property_unit_dict']['forces']
    }
    # Prepare transformations and data module   
    transformations = prepare_transformations(config,"infer")
           
    custom_data = setup_data_module(
        config,
        db_path,
        transformations,
        property_units
    )
    
    new_dataset = ASEAtomsData(db_path)
    # Show dataset information
    show_dataset_info(new_dataset)

    train_loader = custom_data.train_dataloader()
    validation_loader = custom_data.val_dataloader()
    
    # Setup model
    nnpot, outputs = setup_model(config)

    # Load the best model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_name = config['settings']['logging']['checkpoint_dir']
    best_model_path = os.path.join(trained_model_path, model_name)
    best_model = torch.load(best_model_path, map_location=device)

    # Filter out None postprocessors
    filtered_postprocessors = [pp for pp in best_model.postprocessors if pp is not None]
    best_model.postprocessors = torch.nn.ModuleList(filtered_postprocessors)
    best_model.to(device)
    best_model.eval()

    # Run inference on both datasets
    run_inference(train_loader, "train", best_model, device, property_units, new_dataset)
    run_inference(validation_loader, "validation", best_model, device, property_units, new_dataset)

if __name__ == '__main__':
    setup_logging()  # Initialize logging before main
    logging.info(f"{'*' * 30} Started... {'*' * 30}")
    main()
