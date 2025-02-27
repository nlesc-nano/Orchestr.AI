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

from mlff_qd.utils.analysis import compute_rmsd_matrix, plot_rmsd_histogram
from mlff_qd.utils.cluster import cluster_trajectory
from mlff_qd.utils.io import ( save_xyz, save_frequencies, save_binary,
        load_binary, reorder_xyz_trajectory, parse_positions_xyz, parse_forces_xyz,
        get_num_atoms )
from mlff_qd.utils.pca import ( generate_pca_samples, perform_pca_and_plot, 
        generate_structures_from_pca, generate_surface_core_pca_samples )
from mlff_qd.utils.preprocessing import ( center_positions, align_to_reference, rotate_forces, 
        create_mass_dict )
from mlff_qd.utils.surface import compute_surface_indices_with_replace_surface_dynamic
from mlff_qd.utils.constants import ( hartree_bohr_to_eV_angstrom, hartree_to_eV,
        bohr_to_angstrom, amu_to_kg, c )

np.set_printoptions(threshold=np.inf)

def generate_randomized_samples(
    md_positions,
    atom_types,
    num_samples=100,
    base_scale=0.1
):
    """
    Generate random configurations by applying Gaussian perturbations to atomic positions.
    The displacements are fixed in scale and not adjusted to match any RMSD target.

    Parameters:
        md_positions (list of np.ndarray): List of MD frames (num_frames, num_atoms, 3).
        atom_types (list of str): Atom types for each atom in the structure.
        num_samples (int): Number of randomized configurations to generate.
        base_scale (float): Standard deviation of the Gaussian noise applied to displace each atom.

    Returns:
        np.ndarray: Array of randomized configurations (num_samples, num_atoms, 3).
    """
    randomized_structures = []

    for i in range(num_samples):
        # Select a random reference frame
        start_idx = np.random.choice(len(md_positions))
        reference_positions = md_positions[start_idx]

        # Generate random displacements
        displacement = np.random.normal(loc=0.0, scale=base_scale, size=reference_positions.shape)

        # Remove net translation
        mean_disp = np.mean(displacement, axis=0)  # Average over all atoms
        displacement -= mean_disp  # Now no net translation

        # Apply displacements to the reference positions
        randomized_structure = reference_positions + displacement
        randomized_structures.append(randomized_structure)

        # Print progress every 100 samples or for the first sample
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Generating sample {i + 1}/{num_samples}...")

    print(f"Generated {len(randomized_structures)} randomized samples.")

    # Save the randomized samples
    save_xyz("randomized_samples.xyz", randomized_structures, atom_types)
    print(f"Saved {len(randomized_structures)} randomized samples to 'randomized_samples.xyz'.")

    return np.array(randomized_structures)

def plot_generated_samples(combined_samples):
    """
    Plot PCA visualizations of the combined samples dataset for both "positions" and "RMSD" representations,
    without outlier detection.

    Parameters:
        combined_samples (dict): Dictionary with dataset names as keys and samples as values. 
                                 Each dataset should have shape (num_samples, num_atoms, 3) or (num_samples, flattened_dim).
    """
    # Flatten datasets for PCA
    print("Flattening datasets for PCA...")
    flattened_samples = {}
    for name, samples in combined_samples.items():
        samples = np.array(samples)  # Ensure data is a NumPy array
        if samples.ndim == 3:  # Shape: (num_samples, num_atoms, 3)
            flattened_samples[name] = samples.reshape(samples.shape[0], -1)  # Flatten to (num_samples, 3*num_atoms)
        elif samples.ndim == 2:  # Already flattened
            flattened_samples[name] = samples
        else:
            raise ValueError(f"Unexpected shape for dataset '{name}': {samples.shape}")

    # Concatenate all datasets
    print("Concatenating datasets for PCA...")
    all_samples_positions = np.concatenate(list(flattened_samples.values()), axis=0)  # Combine datasets row-wise
    sample_labels = sum([[name] * len(data) for name, data in flattened_samples.items()], [])

    # Prepare RMSD matrix for RMSD-based PCA
    print("Computing RMSD matrix for all samples...")
    all_samples_reshaped = all_samples_positions.reshape(all_samples_positions.shape[0], -1, 3)  # Reshape to 3D
    rmsd_matrix = compute_rmsd_matrix(all_samples_reshaped)

    # Perform PCA for "positions"
    print("Performing PCA for 'positions'...")
    pca_positions = PCA(n_components=2)
    pca_transformed_positions = pca_positions.fit_transform(all_samples_positions)
    explained_variance_positions = np.sum(pca_positions.explained_variance_ratio_) * 100
    print(f"Explained variance by first two components (positions): {explained_variance_positions:.2f}%")

    # Perform PCA for "RMSD"
    print("Performing PCA for 'RMSD'...")
    pca_rmsd = PCA(n_components=2)
    pca_transformed_rmsd = pca_rmsd.fit_transform(rmsd_matrix)
    explained_variance_rmsd = np.sum(pca_rmsd.explained_variance_ratio_) * 100
    print(f"Explained variance by first two components (RMSD): {explained_variance_rmsd:.2f}%")

    # Create color map for datasets
    unique_labels = sorted(set(sample_labels))
    color_map = {label: plt.cm.tab10(idx / len(unique_labels)) for idx, label in enumerate(unique_labels)}

    # Plotting
    print("Plotting PCA results...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot for "positions"
    for label in unique_labels:
        label_indices = [i for i, lbl in enumerate(sample_labels) if lbl == label]
        axes[0].scatter(
            pca_transformed_positions[label_indices, 0],
            pca_transformed_positions[label_indices, 1],
            label=label,
            color=color_map[label],
            alpha=0.7
        )
    axes[0].set_title("PCA: Positions Representation")
    axes[0].set_xlabel("Principal Component 1")
    axes[0].set_ylabel("Principal Component 2")
    axes[0].legend()
    axes[0].grid(True)

    # Plot for "RMSD"
    for label in unique_labels:
        label_indices = [i for i, lbl in enumerate(sample_labels) if lbl == label]
        axes[1].scatter(
            pca_transformed_rmsd[label_indices, 0],
            pca_transformed_rmsd[label_indices, 1],
            label=label,
            color=color_map[label],
            alpha=0.7
        )
    axes[1].set_title("PCA: RMSD Representation")
    axes[1].set_xlabel("Principal Component 1")
    axes[1].set_ylabel("Principal Component 2")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def load_config(config_file=None):
    """
    Load configuration from a YAML file. Use `.get()` for defaults in the main code.
    If config_file is not provided, defaults to 'config/preprocess_config.yaml'
    in the project root (relative to this script).
    """

    # If no config file is specified, use a default path relative to this script.
    # Example: go up 3 or 4 levels to reach the project root, then into config/preprocess_config.yaml.
    if config_file is None:
        # Adjust parents[...] as necessary based on your directory structure
        default_path = Path(__file__).resolve().parents[3] / "config" / "preprocess_config.yaml"
        config_file = str(default_path)  # convert Path object to string if needed

    try:
        # Load user-defined configuration
        with open(config_file, "r") as file:
            user_config = yaml.safe_load(file)
            if user_config is None:
                user_config = {}
    except FileNotFoundError:
        print(f"Configuration file '{config_file}' not found. Using only default settings.")
        user_config = {}

    return user_config

if __name__ == "__main__":
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
    representation = config.get("representation", "rmsd")  # Change to "positions" for position-based PCA
    num_samples_pca = config.get("num_samples_pca", 600)
    num_samples_pca_surface = config.get("num_samples_pca_surface", 300)
    num_samples_randomization = config.get("num_samples_randomization", 100)
    num_samples_dataset = config.get("num_samples_dataset", 2000)
    sampling_fractions = {"PCA": 0.6, "PCA_Surface": 0.3, "Randomized": 0.1}

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
