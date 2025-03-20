#!/usr/bin/env python3

import os
import pickle
import random
import yaml
import pprint
import logging
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from dscribe.descriptors import SOAP
from ase import Atoms
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from periodictable import elements
from scm.plams import Molecule
from CAT.recipes import replace_surface

from mlff_qd.utils.analysis import ( compute_global_distance_fluctuation_cdist,
        compute_rmsd_matrix )
from mlff_qd.utils.cluster import cluster_trajectory, compute_soap_descriptors
from mlff_qd.utils.config import load_config
from mlff_qd.utils.constants import ( hartree_bohr_to_eV_angstrom, hartree_to_eV,
        bohr_to_angstrom, amu_to_kg, c )
from mlff_qd.utils.io import ( save_xyz, reorder_xyz_trajectory, parse_positions_xyz,
        parse_forces_xyz, get_num_atoms )
from mlff_qd.utils.pca import ( generate_surface_core_pca_samples,
        generate_pca_samples_in_pca_space, generate_structures_from_pca )
from mlff_qd.utils.preprocessing import ( create_mass_dict, center_positions,
        align_to_reference, iterative_alignment_fixed, rotate_forces, find_medoid_structure,
        generate_randomized_samples )

# --- Set up logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("mlff_dataset.log")])
logger = logging.getLogger(__name__)

# --- Utility Functions ---
def plot_generated_samples_extended(combined_samples, atom_types, soap):
    """
    Plot PCA visualizations using:
      - Positions (flattened)
      - SOAP descriptors (averaged over atoms)
      - Global distance fluctuation (rotationally invariant metric)
    
    Saves the figure to a file.
    
    Parameters:
        combined_samples (dict): Dictionary with dataset names as keys and
          position arrays as values (shape: (num_samples, num_atoms, 3)).
        atom_types (list): List of atom type labels.
        soap: DScribe SOAP descriptor object.
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np
    
    logger.info("Preparing extended PCA plots for positions, SOAP, and global distance fluctuation...")
    
    # --- Panel 1: Positions PCA ---
    flattened_pos = {}
    for name, samples in combined_samples.items():
        samples = np.array(samples)
        if samples.ndim == 3:
            flattened_pos[name] = samples.reshape(samples.shape[0], -1)
        elif samples.ndim == 2:
            flattened_pos[name] = samples
        else:
            raise ValueError(f"Unexpected shape for {name}: {samples.shape}")
    all_pos = np.concatenate(list(flattened_pos.values()), axis=0)
    pos_labels = sum([[name]*len(data) for name, data in flattened_pos.items()], [])
    pca_positions = PCA(n_components=2).fit_transform(all_pos)
    
    # --- Panel 2: SOAP PCA ---
    soap_desc = {}
    for name, samples in combined_samples.items():
        samples = np.array(samples)
        if samples.ndim == 3:
            desc_list = []
            for pos in samples:
                atoms = Atoms(symbols=atom_types, positions=pos)
                d = soap.create(atoms)
                desc_list.append(np.mean(d, axis=0))
            soap_desc[name] = np.array(desc_list)
        else:
            soap_desc[name] = samples
    all_soap = np.concatenate(list(soap_desc.values()), axis=0)
    soap_labels = sum([[name]*len(data) for name, data in soap_desc.items()], [])
    pca_soap = PCA(n_components=2).fit_transform(all_soap)
    
    # --- Panel 3: Global Distance Fluctuation ---
    fluct_dict = {}
    for name, samples in combined_samples.items():
        samples = np.array(samples)
        fluct_dict[name] = compute_global_distance_fluctuation_cdist(samples)
    
    # --- Plotting ---
    unique_labels = sorted(set(pos_labels))
    cmap = {label: plt.cm.tab10(i/len(unique_labels)) for i, label in enumerate(unique_labels)}
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Positions PCA
    for label in unique_labels:
        idxs = [i for i, l in enumerate(pos_labels) if l == label]
        axes[0].scatter(pca_positions[idxs, 0], pca_positions[idxs, 1],
                        label=label, color=cmap[label], alpha=0.7)
    axes[0].set_title("PCA: Positions")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend()
    axes[0].grid(True)
    
    # SOAP PCA
    for label in unique_labels:
        idxs = [i for i, l in enumerate(soap_labels) if l == label]
        axes[1].scatter(pca_soap[idxs, 0], pca_soap[idxs, 1],
                        label=label, color=cmap[label], alpha=0.7)
    axes[1].set_title("PCA: SOAP Descriptor")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend()
    axes[1].grid(True)
    
    # Global Distance Fluctuation: boxplot per dataset.
    data_to_plot = [fluct_dict[label] for label in unique_labels]
    axes[2].boxplot(data_to_plot, labels=unique_labels)
    axes[2].set_title("Global Distance Fluctuation")
    axes[2].set_ylabel("Std. deviation of interatomic distances")
    
    plt.tight_layout()
    plt.savefig("extended_pca_plots.png")
    logger.info("Saved extended PCA plots to 'extended_pca_plots.png'")
    plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLFF Data Generation")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to the YAML config file.")
    args = parser.parse_args()
    
    # Load yaml config
    config = load_config(config_file=args.config)

    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    # Extract config values
    pos_file_name = config.get("pos_file", "trajectory_pos.xyz")
    frc_file_name = config.get("frc_file", "trajectory_frc.xyz")
    pos_file_path = PROJECT_ROOT / pos_file_name
    frc_file_path = PROJECT_ROOT / frc_file_name

    max_random_displacement = config.get("max_random_displacement", 0.25)
    scaling_factor = config.get("scaling_factor", 0.40)
    scaling_surf = config.get("scaling_surf", 0.60)
    scaling_core = config.get("scaling_core", 0.40) 
    surface_atom_types = config.get("surface_atom_types", ["Cs", "Br"])
    clustering_method = config.get("clustering_method", "KMeans")
    num_clusters = config.get("num_clusters", 100)
    num_samples_pca = config.get("num_samples_pca", 600)
    num_samples_pca_surface = config.get("num_samples_pca_surface", 300)
    num_samples_randomization = config.get("num_samples_randomization", 100)
    soap_params = config["SOAP"]
    logger.info("Configuration:")
    pprint.pprint(config, indent=4, width=80)

    num_atoms = get_num_atoms(pos_file_path)
    positions, atom_types, energies_hartree = parse_positions_xyz(pos_file_path, num_atoms)
    mass_dict = create_mass_dict(atom_types)
    masses = np.array([mass_dict[atom] for atom in atom_types])
    
    reordered_positions_path = PROJECT_ROOT / "data" / "processed" / "reordered_positions.xyz"
    reordered_forces_path    = PROJECT_ROOT / "data" / "processed" / "reordered_forces.xyz"
    reorder_xyz_trajectory(pos_file_path, reordered_positions_path, num_atoms)
    reorder_xyz_trajectory(frc_file_path, reordered_forces_path, num_atoms)

    positions, atom_types, energies_hartree = parse_positions_xyz(reordered_positions_path, num_atoms)
    forces = parse_forces_xyz(reordered_forces_path, num_atoms)

    # Preprocessing
    centered_positions = center_positions(positions, masses)
    aligned_positions, rotations, _ = iterative_alignment_fixed(centered_positions)
    aligned_forces = rotate_forces(forces, rotations)
    medoid_structure, rep_idx = find_medoid_structure(aligned_positions)
    
    save_xyz("medoid_structure.xyz", medoid_structure[np.newaxis,:,:], atom_types)
    save_xyz("aligned_positions.xyz", aligned_positions, atom_types, energies_hartree, comment="Aligned positions")
    save_xyz("aligned_forces.xyz", aligned_forces, atom_types, comment="Aligned forces")

    energies_ev = [e * hartree_to_eV if e is not None else None for e in energies_hartree]
    aligned_forces_ev = aligned_forces * hartree_bohr_to_eV_angstrom
    
    save_xyz("aligned_positions_ev.xyz", aligned_positions, atom_types, energies_ev, comment="Aligned positions")
    save_xyz("aligned_forces_eV.xyz", aligned_forces_ev, atom_types, comment="Aligned forces")

    aligned_positions_path = PROJECT_ROOT / "data" / "processed" / "aligned_positions.xyz"
    aligned_forces_path = PROJECT_ROOT / "data" / "processed" / "aligned_forces.xyz"
    md_positions, atom_types, _ = parse_positions_xyz(aligned_positions_path, num_atoms)
    md_forces = parse_forces_xyz(aligned_forces_path, num_atoms)
    
    species = sorted(list(set(atom_types)))
    
    soap = SOAP(
        species=species,
        r_cut=soap_params["r_cut"],
        n_max=soap_params["n_max"],
        l_max=soap_params["l_max"],
        sigma=soap_params["sigma"],
        periodic=soap_params["periodic"],
        sparse=soap_params["sparse"]
    )
    logger.info("Computing SOAP descriptors for MD frames...")
    precomputed_soap = compute_soap_descriptors(md_positions, atom_types, soap)
    logger.info("Performing clustering based on SOAP descriptors...")
    representative_md = cluster_trajectory(precomputed_soap, clustering_method, num_clusters, md_positions, atom_types)
    pca_samples = generate_structures_from_pca(md_positions, md_forces, representative_md, atom_types, 
                                               num_samples_pca, scaling_factor, pca_variance_threshold=0.90)
    
    medoid_structure_path = PROJECT_ROOT / "data" / "processed" / "medoid_structure.xyz"
    surface_replaced_path = PROJECT_ROOT / "data" / "processed" / "surface_replaced.xyz"
    pca_surface_samples = generate_surface_core_pca_samples(md_positions, md_forces, atom_types, surface_atom_types,
            representative_md, num_samples_pca_surface, scaling_surf, scaling_core,
            medoid_structure_path, surface_replaced_path) 
    
    randomized_samples = generate_randomized_samples(representative_md, atom_types, num_samples_randomization,
                                                     max_random_displacement)

    combined_samples = {
        "MD": md_positions,
        "PCA": pca_samples,
        "PCA_Surface": pca_surface_samples,
        "Randomized": randomized_samples,
    }
    for name, samples in combined_samples.items():
        logger.info(f"{name}: {np.array(samples).shape}")

    plot_generated_samples_extended(combined_samples, atom_types, soap)

    # Combine non-MD samples for training dataset.
    combined_list = []
    frame_titles = []
    for name, samples in combined_samples.items():
        if name != "MD":
            samples_arr = np.array(samples)
            combined_list.append(samples_arr)
            for i in range(len(samples)):
                frame_titles.append(f"Frame {i+1} from {name}")
    
    training_dataset = np.vstack(combined_list).reshape(-1, num_atoms, 3)
    
    training_dataset_path = PROJECT_ROOT / "data" / "processed" / "training_dataset.xyz"
    with open(training_dataset_path, "w") as f:
        for i, (struct, title) in enumerate(zip(training_dataset, frame_titles)):
            f.write(f"{len(struct)}\n{title}\n")
            for atom, coords in zip(atom_types, struct):
                f.write(f"{atom} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}\n")
    
    logger.info(f"Saved training dataset with {training_dataset.shape[0]} structures to 'training_dataset.xyz'")

