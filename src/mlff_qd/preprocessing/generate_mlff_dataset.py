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

from mlff_qd.utils.analysis import compute_global_distance_fluctuation_cdist
from mlff_qd.utils.config import load_config
from mlff_qd.utils.constants import ( hartree_bohr_to_eV_angstrom, hartree_to_eV,
        bohr_to_angstrom, amu_to_kg, c )
from mlff_qd.utils.io import ( save_xyz, reorder_xyz_trajectory, parse_positions_xyz,
        parse_forces_xyz, get_num_atoms )
from mlff_qd.utils.pca import ( generate_surface_core_pca_samples,
        generate_pca_samples_in_pca_space )
from mlff_qd.utils.preprocessing import ( create_mass_dict, center_positions,
        align_to_reference, iterative_alignment_fixed, rotate_forces, find_medoid_structure )

# --- Set up logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("mlff_dataset.log")])
logger = logging.getLogger(__name__)

# --- Utility Functions ---
def compute_rmsd_matrix(structures1, structures2=None):
    """Compute RMSD matrix between two sets of structures."""
    structures1 = np.array(structures1)
    if structures2 is None:
        structures2 = structures1
        symmetric = True
    else:
        structures2 = np.array(structures2)
        symmetric = False
    N1, A, _ = structures1.shape
    N2 = structures2.shape[0]
    if symmetric:
        rmsd_mat = np.zeros((N1, N2))
        for i in range(N1):
            diff = structures1[i:] - structures1[i]
            rmsd_vals = np.sqrt(np.sum(diff**2, axis=(1,2)) / A)
            rmsd_mat[i, i:] = rmsd_vals
            rmsd_mat[i:, i] = rmsd_vals
    else:
        diff = structures1[:, None, :, :] - structures2[None, :, :, :]
        rmsd_mat = np.sqrt(np.sum(diff**2, axis=(2,3)) / A)
    return rmsd_mat

def plot_rmsd_histogram(rmsd_matrix, bins=50, title="RMSD Histogram", xlabel="RMSD (Å)", ylabel="Frequency", savefile="rmsd_histogram.png"):
    """Plot and save RMSD histogram."""
    rmsd_vals = rmsd_matrix[np.triu_indices_from(rmsd_matrix, k=1)]
    plt.figure(figsize=(8, 6))
    plt.hist(rmsd_vals, bins=bins, edgecolor="black", alpha=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(savefile)
    logger.info(f"Saved RMSD histogram plot to {savefile}")
    plt.close()

def compute_soap_descriptors(md_positions, atom_types, soap):
    """Compute a global (averaged) SOAP descriptor for each MD frame."""
    descriptors = []
    for pos in md_positions:
        atoms = Atoms(symbols=atom_types, positions=pos)
        soap_desc = soap.create(atoms)
        descriptors.append(np.mean(soap_desc, axis=0))
    return np.array(descriptors)

def cluster_trajectory(descriptor_matrix, method, num_clusters, md_positions, atom_types):
    """Cluster the MD trajectory using the provided descriptor matrix and return representative structures."""
    logger.info("Clustering MD trajectory based on SOAP descriptors...")
    if method == "KMeans":
        model = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = model.fit_predict(descriptor_matrix)
    elif method == "DBSCAN":
        model = DBSCAN(eps=0.2, min_samples=10)
        cluster_labels = model.fit_predict(descriptor_matrix)
    elif method == "GMM":
        model = GaussianMixture(n_components=num_clusters, random_state=42)
        cluster_labels = model.fit_predict(descriptor_matrix)
    else:
        raise ValueError(f"Invalid clustering method: {method}")
    rep_indices = []
    unique = np.unique(cluster_labels)
    for cid in unique:
        if cid == -1:
            continue
        members = np.where(cluster_labels == cid)[0]
        centroid = np.mean(descriptor_matrix[members], axis=0)
        distances = np.linalg.norm(descriptor_matrix[members] - centroid, axis=1)
        rep_idx = members[np.argmin(distances)]
        rep_indices.append(rep_idx)
    logger.info(f"Selected {len(rep_indices)} representatives from {len(unique)} clusters.")
    rep_structs = [md_positions[i] for i in rep_indices]
    save_xyz("clustered_md_sample.xyz", rep_structs, atom_types)
    return rep_structs

def generate_structures_from_pca(
    md_positions,
    md_forces,
    representative_md,
    atom_types,
    num_samples,
    scaling_factor=0.4,
    pca_variance_threshold=0.90
):
    """
    Generate PCA-based structures using combined descriptors of positions and forces ONLY.
    Skips SOAP entirely.

    Parameters:
        md_positions (np.ndarray): MD frames, shape (num_frames, num_atoms, 3)
        md_forces (np.ndarray): MD forces, shape (num_frames, num_atoms, 3)
        representative_md (list of np.ndarray): Subset of frames used as "reference" seeds
        atom_types (list): Atom types
        num_samples (int): How many new structures to generate
        pca_variance_threshold (float): Fraction of variance to keep (e.g. 0.90)

    Returns:
        np.ndarray: Array of shape (num_samples, num_atoms, 3) with the new generated structures.
    """
    logger.info("Generating PCA-based structures from positions+forces only (no SOAP).")

    # 1) Build descriptors (positions and forces flattened)
    #    shape => (num_frames, 6*num_atoms)
    descriptors = []
    for pos, frc in zip(md_positions, md_forces):
        pf = np.concatenate([pos.flatten(), frc.flatten()])  # shape => (6*num_atoms,)
        descriptors.append(pf)
    descriptors = np.array(descriptors)
    logger.info(f"Built descriptor matrix of shape: {descriptors.shape}")

    # 2) Normalize (standardize) each feature
    scaler = StandardScaler()
    descriptors_norm = scaler.fit_transform(descriptors)

    # 3) PCA on normalized descriptors
    pca = PCA()
    pca.fit(descriptors_norm)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    opt_comp = np.argmax(cumvar >= pca_variance_threshold) + 1
    logger.info(f"Optimal PCA components: {opt_comp} (captures {cumvar[opt_comp-1]*100:.2f}% variance)")
    pca = PCA(n_components=opt_comp)
    pca.fit(descriptors_norm)

    # 4) We'll just set scaling_factor = max_displacement directly
    #    If you prefer a distance-based approach, do that here.
    distance_flucts = compute_global_distance_fluctuation_cdist(md_positions)
    mean_dist_fluct = np.mean(distance_flucts)
    max_dist_fluct  = np.max(distance_flucts)
    print(f"Mean distance fluctuation = {mean_dist_fluct:.3f}")
    print(f"Max distance fluctuation  = {max_dist_fluct:.3f}")
    logger.info(f"Sampling amplitude (scaling_factor) set to {scaling_factor:.2f}")

    # 5) Loop over representative frames, generate new structures
    new_structures = []
    for i in range(num_samples):
        idx = random.choice(range(len(representative_md)))
        ref_struct = representative_md[idx]
        ref_index = np.where(np.all(md_positions == ref_struct, axis=(1, 2)))[0][0]

        # Build the reference descriptor (pos + frc), skipping SOAP
        ref_pos = md_positions[ref_index].flatten()
        ref_frc = md_forces[ref_index].flatten()
        ref_desc = np.concatenate([ref_pos, ref_frc])  # shape => (6*num_atoms,)

        # Transform to normalized, PCA space
        ref_desc_norm = scaler.transform(ref_desc.reshape(1, -1))[0]          # shape => (6*N,)
        ref_desc_pca = pca.transform(ref_desc_norm.reshape(1, -1))[0]         # shape => (opt_comp,)

        # Perturb in PCA space
        perturbation = np.random.normal(scale=scaling_factor, size=ref_desc_pca.shape)
        new_desc_pca = ref_desc_pca + perturbation

        # Inverse transform: PCA -> normalized descriptor space
        new_desc_norm = pca.inverse_transform(new_desc_pca.reshape(1, -1))[0] # shape => (6*N,)

        # Un-normalize => original scale
        new_desc = scaler.inverse_transform(new_desc_norm.reshape(1, -1))[0]  # shape => (6*N,)

        # Extract new positions from the first 3*N
        n_atoms = ref_struct.shape[0]
        new_pos = new_desc[:n_atoms * 3].reshape(n_atoms, 3)
        new_structures.append(new_pos)

        # Log progress
        if (i + 1) % 100 == 0 or i == 0:
            logger.info(f"Generated {i+1}/{num_samples} PCA samples...")

    # 6) Convert to NumPy array and save
    new_structures = np.array(new_structures)
    save_xyz("pca_samples_no_soap.xyz", new_structures, atom_types)
    logger.info("Saved PCA-based samples (no SOAP) to 'pca_samples_no_soap.xyz'")

    return new_structures

def generate_randomized_samples(md_positions, atom_types, num_samples=100, base_scale=0.1):
    """Generate random structures by Gaussian perturbation."""
    randomized = []
    for i in range(num_samples):
        ref = random.choice(md_positions)
        disp = np.random.normal(0, base_scale, size=ref.shape)
        disp -= np.mean(disp, axis=0)
        randomized.append(ref + disp)
        if (i+1) % 100 == 0 or i==0:
            logger.info(f"Generated {i+1}/{num_samples} randomized samples...")
    save_xyz("randomized_samples.xyz", randomized, atom_types)
    logger.info("Saved randomized samples to 'randomized_samples.xyz'")
    return np.array(randomized)

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
    from ase import Atoms
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

