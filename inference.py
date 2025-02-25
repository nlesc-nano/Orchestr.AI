import os
import pandas as pd

from schnetpack.data import ASEAtomsData
import schnetpack as spk

from schnetpack.data import AtomsDataModule
import schnetpack.transform as trn

import torch
import numpy as np
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import yaml
import argparse
import pickle
import functools
import logging
import time

from utils.logging_utils import  timer,setup_logging

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


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with SchNet using a configuration file.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration YAML file")
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


@timer
def main():
    args = parse_args()
    config = load_config(args.config)

    output_type = config['settings']['data']['output_type']
    include_homo_lumo = output_type == 2
    include_bandgap = output_type == 3
    include_eigenvalues_vector = output_type == 4  # New flag for output_type 4

    # Read eigenvalue_labels if output_type is 4
    if include_eigenvalues_vector:
        eigenvalue_labels = config['settings']['data']['eigenvalue_labels']
    else:
        eigenvalue_labels = []

    trained_model_path = config['settings']['testing']['trained_model_path']
    print(f"Trained model path: {trained_model_path}")

    db_name =  config['settings']['general']['database_name']
    db_path = os.path.join(trained_model_path, db_name) 

    batch_size = config['settings']['training']['batch_size']
    num_train = config['settings']['training']['num_train']
    num_val = config['settings']['training']['num_val']
    num_workers = config['settings']['training']['num_workers']
    pin_memory = config['settings']['training']['pin_memory']
    cutoff = config['settings']['model']['cutoff']
    distance_unit = config['settings']['model']['distance_unit']

    property_units = {
        'energy': config['settings']['model']['property_unit_dict']['energy'],
        'forces': config['settings']['model']['property_unit_dict']['forces']
    }
    if include_homo_lumo:
        property_units.update({
            'homo': config['settings']['model']['property_unit_dict']['homo'],
            'lumo': config['settings']['model']['property_unit_dict']['lumo'],
        })
    if include_bandgap:
        property_units.update({
            'bandgap': config['settings']['model']['property_unit_dict']['bandgap'],
        })
    if include_eigenvalues_vector:
        property_units.update({
            'eigenvalues_vector': config['settings']['model']['property_unit_dict']['eigenvalues_vector'],
        })

    dataset = ASEAtomsData(db_path)

    print('Available properties in the dataset:')
    print(dataset.available_properties)
    
    total_atoms = dataset[0]['_n_atoms'].item()  
    print(f'Total atoms: {total_atoms}')

    data_module = spk.data.AtomsDataModule(
        db_path,
        batch_size=batch_size,
        num_train=num_train,
        num_val=num_val,
        transforms= [
        trn.ASENeighborList(cutoff=cutoff),
        trn.CastTo32(),
        ],
        distance_unit=distance_unit,
        property_units=property_units,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    data_module.prepare_data()
    data_module.setup()

    train_loader = data_module.train_dataloader()
    validation_loader = data_module.val_dataloader()

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

    @timer
    def run_inference(loader, dataset_type, include_eigenvalues_vector, eigenvalue_labels):
        all_actual_energy = []
        all_predicted_energy = []
        all_actual_forces = []
        all_predicted_forces = []

        if include_homo_lumo:
            all_actual_homo = []
            all_predicted_homo = []
            all_actual_lumo = []
            all_predicted_lumo = []

        if include_bandgap:
            all_actual_bandgap = []
            all_predicted_bandgap = []

        if include_eigenvalues_vector:
            all_actual_eigenvalues_vector = []
            all_predicted_eigenvalues_vector = []

        for batch in tqdm(loader, desc=f"Running inference on {dataset_type} data"):
            batch = {key: value.to(device) for key, value in batch.items()}
            batch['positions'] = batch['_positions']
            batch['positions'].requires_grad_()
            exclude_keys = ['energy', 'forces']
            if include_homo_lumo:
                exclude_keys += ['homo', 'lumo']
            if include_bandgap:
                exclude_keys.append('bandgap')
            if include_eigenvalues_vector:
                exclude_keys.append('eigenvalues_vector')  # Exclude eigenvalues_vector from input

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

            # Collect HOMO and LUMO if applicable
            if include_homo_lumo:
                actual_homo = batch['homo'].detach().cpu().numpy()
                predicted_homo = result['homo'].detach().cpu().numpy()
                all_actual_homo.append(actual_homo)
                all_predicted_homo.append(predicted_homo)

                actual_lumo = batch['lumo'].detach().cpu().numpy()
                predicted_lumo = result['lumo'].detach().cpu().numpy()
                all_actual_lumo.append(actual_lumo)
                all_predicted_lumo.append(predicted_lumo)

            # Collect bandgap if applicable
            if include_bandgap:
                actual_bandgap = batch['bandgap'].detach().cpu().numpy()
                predicted_bandgap = result['bandgap'].detach().cpu().numpy()
                all_actual_bandgap.append(actual_bandgap)
                all_predicted_bandgap.append(predicted_bandgap)

            # Collect eigenvalues_vector if applicable
            if include_eigenvalues_vector:
                actual_eigenvalues = batch['eigenvalues_vector'].detach().cpu().numpy()
                predicted_eigenvalues = result['eigenvalues_vector'].detach().cpu().numpy()
                all_actual_eigenvalues_vector.append(actual_eigenvalues)
                all_predicted_eigenvalues_vector.append(predicted_eigenvalues)

        # Save results for this dataset
        data = {
            'Actual Energy': np.concatenate(all_actual_energy).flatten(),
            'Predicted Energy': np.concatenate(all_predicted_energy).flatten(),
        }

        if include_homo_lumo:
            data['Actual HOMO'] = np.concatenate(all_actual_homo).flatten()
            data['Predicted HOMO'] = np.concatenate(all_predicted_homo).flatten()
            data['Actual LUMO'] = np.concatenate(all_actual_lumo).flatten()
            data['Predicted LUMO'] = np.concatenate(all_predicted_lumo).flatten()

        if include_bandgap:
            data['Actual Bandgap'] = np.concatenate(all_actual_bandgap).flatten()
            data['Predicted Bandgap'] = np.concatenate(all_predicted_bandgap).flatten()

        if include_eigenvalues_vector:
            # Concatenate all eigenvalues vectors
            all_actual_eigenvalues_vector = np.concatenate(all_actual_eigenvalues_vector, axis=0)
            all_predicted_eigenvalues_vector = np.concatenate(all_predicted_eigenvalues_vector, axis=0)

            for i, label in enumerate(eigenvalue_labels):
                data[f'Actual {label}'] = all_actual_eigenvalues_vector[:, i]
                data[f'Predicted {label}'] = all_predicted_eigenvalues_vector[:, i]

        df = pd.DataFrame(data)
        csv_file_path = os.path.join(os.getcwd(), f'{dataset_type}_predictions.csv')
        df.to_csv(csv_file_path, index=False)
        print(f"Results saved to {csv_file_path}")

        # Reshape force arrays to maintain three-dimensional vector form
        all_actual_forces_flat = np.concatenate(all_actual_forces).reshape(-1, 3)
        all_predicted_forces_flat = np.concatenate(all_predicted_forces).reshape(-1, 3)

        # Compute MAEs for all properties
        energy_mae = mean_absolute_error(np.concatenate(all_actual_energy), np.concatenate(all_predicted_energy))
        forces_mae = mean_absolute_error(all_actual_forces_flat, all_predicted_forces_flat)


        energy_mae_per_atom = (energy_mae/total_atoms) 

        print(f"Energy MAE on {dataset_type} data: {energy_mae} {property_units['energy']}") 
        print(f"Energy MAE per Atom on {dataset_type} data: {energy_mae_per_atom} {property_units['energy']}") 
        print(f"Forces MAE on {dataset_type} data: {forces_mae} {property_units['forces']}") 


        if include_homo_lumo:
            homo_mae = mean_absolute_error(np.concatenate(all_actual_homo), np.concatenate(all_predicted_homo))
            lumo_mae = mean_absolute_error(np.concatenate(all_actual_lumo), np.concatenate(all_predicted_lumo))


            print(f"HOMO MAE on {dataset_type} data: {homo_mae} {property_units['homo']}") 
            print(f"LUMO MAE on {dataset_type} data: {lumo_mae} {property_units['lumo ']}") 

        if include_bandgap:
            bandgap_mae = mean_absolute_error(np.concatenate(all_actual_bandgap), np.concatenate(all_predicted_bandgap))


            print(f"Bandgap MAE on {dataset_type} data: {bandgap_mae} {property_units['bandgap']}")  

        if include_eigenvalues_vector:
            # Compute MAE for each eigenvalue label
            for i, label in enumerate(eigenvalue_labels):
                actual = all_actual_eigenvalues_vector[:, i]
                predicted = all_predicted_eigenvalues_vector[:, i]
                mae = mean_absolute_error(actual, predicted)
                
                print(f"{label} MAE on {dataset_type} data: {mae} {property_units[label]}")  

        # Save forces in a pickle file
        forces_data = {
            'Actual Forces': all_actual_forces_flat,
            'Predicted Forces': all_predicted_forces_flat,
        }
        forces_pkl_file_path = os.path.join(os.getcwd(), f'{dataset_type}_forces.pkl')
        with open(forces_pkl_file_path, 'wb') as f:
            pickle.dump(forces_data, f)
        print(f"Forces data saved to {forces_pkl_file_path}")

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
            print("MLFF near convergence")
        else:
            print("MLFF not yet converged. Continuing training...")

    # Run inference on both datasets
    run_inference(train_loader, "train", include_eigenvalues_vector, eigenvalue_labels)
    run_inference(validation_loader, "validation", include_eigenvalues_vector, eigenvalue_labels)


if __name__ == '__main__':
    setup_logging()  # Initialize logging before main
    logging.info(f"{'*' * 30} Started {'*' * 30}")
    main()