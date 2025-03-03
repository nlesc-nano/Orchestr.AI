import numpy as np
import pickle 
import random
import matplotlib.pyplot as plt
import yaml 
import pprint
import argparse
from pathlib import Path
from periodictable import elements
from scipy.spatial.transform import Rotation as R
from scm.plams import Molecule
from CAT.recipes import replace_surface
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from mlff_qd.utils.analysis import ( compute_rmsd_matrix, plot_rmsd_histogram,
        plot_generated_samples )
from mlff_qd.utils.cluster import cluster_trajectory
from mlff_qd.utils.config import load_config
from mlff_qd.utils.io import ( save_xyz, save_frequencies, save_binary,
        load_binary, reorder_xyz_trajectory, parse_positions_xyz, parse_forces_xyz,
        get_num_atoms )
from mlff_qd.utils.pca import ( generate_pca_samples, perform_pca_and_plot, 
        generate_structures_from_pca, generate_surface_core_pca_samples )
from mlff_qd.utils.preprocessing import ( center_positions, align_to_reference, rotate_forces, 
        create_mass_dict, generate_randomized_samples )
from mlff_qd.utils.surface import compute_surface_indices_with_replace_surface_dynamic
from mlff_qd.utils.constants import ( hartree_bohr_to_eV_angstrom, hartree_to_eV,
        bohr_to_angstrom, amu_to_kg, c )

np.set_printoptions(threshold=np.inf)

def main():
    # Load configuration
    parser = argparse.ArgumentParser(description="MLFF Data Generation")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to the YAML config file.")
    args = parser.parse_args()

    config = load_config(config_file=args.config)

    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    # Default values used with .get()
    pos_file_name = config.get("pos_file", "trajectory_pos.xyz")
    frc_file_name = config.get("frc_file", "trajectory_frc.xyz")
    pos_file_path = PROJECT_ROOT / pos_file_name
    frc_file_path = PROJECT_ROOT / frc_file_name
    pos_file_str = str(pos_file_path)
    frc_file_str = str(frc_file_path)
    temperature = config.get("temperature", 300.0)
    temperature_target = config.get("temperature_target", 300.0)
    temperature_target_surface = config.get("temperature_target_surface", 450.0)
    max_displacement = config.get("max_displacement", 2.0)
    max_random_displacement = config.get("max_random_displacement", 0.1)
    surface_atom_types = config.get("surface_atom_types", ["Cs", "Br"])
    clustering_method = config.get("clustering_method", "KMeans")  # Options: "KMeans", "DBSCAN", "GMM"
    num_clusters = config.get("num_clusters", 100)
    num_samples_pca = config.get("num_samples_pca", 600)
    num_samples_pca_surface = config.get("num_samples_pca_surface", 300)
    num_samples_randomization = config.get("num_samples_randomization", 100)

    # Print the configuration for confirmation
    print(f"Using configuration:")
    pprint.pprint(config, indent=4, width=80)

    # Read total number of atoms
    num_atoms = get_num_atoms(pos_file_str) 

    # Get atom_types 
    _, atom_types, _ = parse_positions_xyz(pos_file_str, num_atoms)

    # Dynamically create the mass dictionary
    print("Creating mass dictionary...")
    mass_dict = create_mass_dict(atom_types)
    masses = np.array([mass_dict[atom] for atom in atom_types])

    # Reorder raw XYZ -> processed "reordered_positions.xyz" / "reordered_forces.xyz"
    reordered_positions_path = PROJECT_ROOT / "data" / "processed" / "reordered_positions.xyz"
    reordered_forces_path    = PROJECT_ROOT / "data" / "processed" / "reordered_forces.xyz"

    # Write them out
    reorder_xyz_trajectory(pos_file_str, str(reordered_positions_path), num_atoms)
    reorder_xyz_trajectory(frc_file_str, str(reordered_forces_path), num_atoms)

    # Now parse those reordered files:
    positions, atom_types, energies_hartree = parse_positions_xyz(str(reordered_positions_path), num_atoms)
    forces = parse_forces_xyz(str(reordered_forces_path), num_atoms)

    # Center, align, & save results
    centered_positions = center_positions(positions, masses)
    aligned_positions = align_to_reference(centered_positions, centered_positions[0])

    rotation_matrices = [R.align_vectors(frame, centered_positions[0])[0].as_matrix() for frame in centered_positions]
    aligned_forces = rotate_forces(forces, rotation_matrices)

    mean_positions = np.mean(aligned_positions, axis=0)

    # Save them to data/processed:
    save_xyz("mean_structure.xyz", mean_positions[np.newaxis, :, :], atom_types)
    save_xyz("aligned_positions.xyz", aligned_positions, atom_types, energies_hartree, comment="Aligned positions")
    save_xyz("aligned_forces.xyz", aligned_forces, atom_types, comment="Aligned forces")

    # Convert energies/forces
    energies_ev = [e * hartree_to_eV if e is not None else None for e in energies_hartree]
    aligned_forces_ev = aligned_forces * hartree_bohr_to_eV_angstrom

    save_xyz("aligned_positions_ev.xyz", aligned_positions, atom_types, energies_ev, comment="Aligned positions")
    save_xyz("aligned_forces_eV.xyz", aligned_forces_ev, atom_types, comment="Aligned forces")

    # Parse aligned positions from processed
    aligned_positions_path = PROJECT_ROOT / "data" / "processed" / "aligned_positions.xyz"
    md_positions, atom_types, _ = parse_positions_xyz(str(aligned_positions_path), num_atoms)

    print("Computing RMSD matrix for MD trajectory...")
    rmsd_md_internal = compute_rmsd_matrix(md_positions)
    num_frames = md_positions.shape[0]
    print(f"Number of frames is : {num_frames}")

    # Clustering + PCA + Random
    mean_structure_path = PROJECT_ROOT / "data" / "processed" / "mean_structure.xyz"
    surface_replaced_path = PROJECT_ROOT / "data" / "processed" / "surface_replaced.xyz"

    print("Performing clustering on MD trajectory...")
    representative_md = cluster_trajectory(
        rmsd_md_internal, clustering_method, num_clusters, md_positions, atom_types
    )

    pca_samples = generate_structures_from_pca(
        rmsd_md_internal,
        md_positions,
        representative_md,
        atom_types,
        num_samples_pca,
        max_displacement,
        temperature,
        temperature_target
    )

    pca_surface_samples = generate_surface_core_pca_samples(
        md_positions,
        atom_types,
        surface_atom_types,
        rmsd_md_internal,
        representative_md,
        num_samples_pca_surface,
        temperature,
        temperature_target,
        temperature_target_surface,
        max_displacement,
        mean_structure_file=str(mean_structure_path),
        surface_replaced_file=str(surface_replaced_path)
    )

    randomized_samples = generate_randomized_samples(
        representative_md,
        atom_types,
        num_samples_randomization,
        max_random_displacement
    )

    # Combine data sets + final training dataset
    combined_samples = {
        "MD": md_positions,
        "PCA": pca_samples,
        "PCA_Surface": pca_surface_samples,
        "Randomized": randomized_samples,
    }

    print("Dataset shapes before flattening:")
    for name, samples in combined_samples.items():
        print(f"{name}: {np.array(samples).shape}")

    plot_generated_samples(combined_samples)

    combined_sample_list = []
    frame_titles = []

    for name, samples in combined_samples.items():
        if name != "MD":  # Exclude MD samples
            samples_array = np.array(samples)
            combined_sample_list.append(samples_array)
            for i in range(len(samples)):
                frame_titles.append(f"Frame {i + 1} from {name}")

    combined_sample_reshaped = np.vstack(combined_sample_list).reshape(-1, num_atoms, 3)

    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "training_dataset.xyz"

    with open(output_path, "w") as xyz_file:
        for i, (structure, title) in enumerate(zip(combined_sample_reshaped, frame_titles)):
            xyz_file.write(f"{len(structure)}\n")
            xyz_file.write(f"{title}\n")
            for atom, coords in zip(atom_types, structure):
                xyz_file.write(f"{atom} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}\n")

    print(f"Saved non-MD training dataset with {combined_sample_reshaped.shape[0]} structures.")

if __name__ == "__main__":
    main()
