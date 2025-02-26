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
from mlff_qd.utils.io import ( save_xyz, save_frequencies, save_binary,
        load_binary, reorder_xyz_trajectory, parse_positions_xyz, parse_forces_xyz,
        get_num_atoms )
from mlff_qd.utils.pca import generate_pca_samples, perform_pca_and_plot
from mlff_qd.utils.preprocessing import ( center_positions, align_to_reference, rotate_forces, 
        create_mass_dict )
from mlff_qd.utils.surface import compute_surface_indices_with_replace_surface_dynamic
from mlff_qd.utils.constants import ( hartree_bohr_to_eV_angstrom, hartree_to_eV,
        bohr_to_angstrom, amu_to_kg, c )

np.set_printoptions(threshold=np.inf)

def cluster_trajectory(rmsd_md_internal, clustering_method, num_clusters, md_positions, atom_types):
    """
    Perform clustering on MD trajectory based on RMSD and return a subset of representative structures.
    The representative structure for each cluster is chosen as the centroid in RMSD space.

    Parameters:
        rmsd_md_internal (np.ndarray): RMSD matrix of shape (N, N), where N is the number of frames.
        clustering_method (str): Clustering method to use ("KMeans", "DBSCAN", or "GMM").
        num_clusters (int): Number of clusters to form (if applicable).
        md_positions (list of np.ndarray): List of MD frames, each frame is an array of shape (num_atoms, 3).
        atom_types (list of str): Atom type labels corresponding to each atom in the frames.

    Returns:
        list of np.ndarray: The clustered_md, a list of representative structures selected from each cluster.
    """
    print("Performing clustering on MD trajectory...")

    # Select clustering method
    if clustering_method == "KMeans":
        print("Using KMeans clustering...")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(rmsd_md_internal)
    elif clustering_method == "DBSCAN":
        print("Using DBSCAN clustering...")
        eps = 0.2  # Distance threshold for forming clusters
        min_samples = 10  # Minimum number of points to form a dense region
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        cluster_labels = dbscan.fit_predict(rmsd_md_internal)
    elif clustering_method == "GMM":
        print("Using Gaussian Mixture Model clustering...")
        gmm = GaussianMixture(n_components=num_clusters, random_state=42)
        cluster_labels = gmm.fit_predict(rmsd_md_internal)
    else:
        raise ValueError(f"Invalid clustering method: {clustering_method}")

    # Select one representative structure (centroid) from each cluster
    clustered_indices = []
    unique_clusters = np.unique(cluster_labels)
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            # Skip noise points (only applicable to DBSCAN)
            continue
        cluster_members = np.where(cluster_labels == cluster_id)[0]

        # Compute the centroid of the cluster in RMSD space
        # The centroid is defined as the structure that has the minimum average RMSD to all others in the cluster.
        cluster_rmsd_matrix = rmsd_md_internal[np.ix_(cluster_members, cluster_members)]
        centroid_idx_within_cluster = np.argmin(cluster_rmsd_matrix.mean(axis=1))
        representative_idx = cluster_members[centroid_idx_within_cluster]
        clustered_indices.append(representative_idx)

    clustered_md = [md_positions[i] for i in clustered_indices]
    print(f"Clustered MD trajectory: {len(clustered_md)} structures from {len(unique_clusters)} clusters.")

    # Save the clustered MD sample
    save_xyz("clustered_md_sample.xyz", clustered_md, atom_types)

    return clustered_md

def generate_structures_from_pca(
    rmsd_md_internal,
    md_positions,
    representative_md,
    atom_types,
    num_samples,
    max_displacement,
    temperature,
    temperature_target
):
    """
    Generate PCA-based structures from a given MD trajectory and save them to "pca_generated_samples.xyz".

    Parameters:
        rmsd_md_internal (np.ndarray): RMSD matrix of the MD trajectory (N x N).
        md_positions (list of np.ndarray): Aligned MD frames, each frame is (num_atoms, 3).
        representative_md (list of np.ndarray): A subset of MD frames used as reference frames.
        atom_types (list of str): Atom types for each atom in the structure.
        num_samples (int): Number of PCA-based samples to generate.
        generate_pca_samples (function): Function to generate samples given a reference structure and PCA model.
        max_displacement (float): Maximum displacement cap to apply to the scaling factors.
        temperature (float): Temperature at which the MD was performed.
        temperature_target (float): Target temperature for scaling.

    Returns:
        np.ndarray: Array of PCA-generated structures with shape (num_samples, num_atoms, 3).
    """
    print("Computing RMSD-based scaling factor...")
    rmsd_values = np.mean(rmsd_md_internal, axis=1)  # Mean RMSD of each frame to all others
    rmsd_scaling_factor = np.mean(rmsd_values)
    print(f"RMSD-based scaling factor: {rmsd_scaling_factor:.2f}")

    # Perform PCA on the full aligned MD trajectory
    flattened_md_positions = np.array([frame.flatten() for frame in md_positions])  # (N_frames, num_atoms*3)
    pca = PCA()
    pca.fit(flattened_md_positions)

    # Determine the number of components to capture at least 90% variance
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    optimal_components = np.argmax(explained_variance >= 0.90) + 1
    print(
        f"Optimal number of PCA components: {optimal_components} "
        f"(captures {explained_variance[optimal_components - 1] * 100:.2f}% variance)"
    )

    # Refit PCA with the optimal number of components
    pca = PCA(n_components=optimal_components)
    pca.fit(flattened_md_positions)

    # Compute adjusted scaling factor based on target temperature
    scaling_factors_temp_adjusted = rmsd_scaling_factor * np.sqrt(temperature_target / temperature)
    print(f"Scaling factors for PCA components (adjusted for T={temperature_target}K): {scaling_factors_temp_adjusted}")

    # Apply a cap to the scaling factors for maximum displacement
    scaling_factors_temp_adjusted = min(scaling_factors_temp_adjusted, max_displacement)
    print(f"Scaled and capped scaling factors: {scaling_factors_temp_adjusted}")

    # Generate PCA-based samples by applying displacements to representative frames
    print(f"Generating {num_samples} PCA-based samples...")
    pca_samples = []
    for i in range(num_samples):
        start_idx = np.random.choice(len(representative_md))  # Choose random frame from representative_md
        reference_structure = representative_md[start_idx]

        # Generate a new structure by applying PCA perturbations
        pca_sample = generate_pca_samples(
            reference_structure,
            pca,
            1,
            scaling_factor=scaling_factors_temp_adjusted
        )[0]
        pca_samples.append(pca_sample)

        # Print progress every 100 samples or for the first sample
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Generating sample {i + 1}/{num_samples}...")

    pca_samples = np.array(pca_samples)
    print(f"Generated {len(pca_samples)} PCA-based samples with shape {pca_samples.shape}.")

    pca_samples_rmsd = compute_rmsd_matrix(pca_samples)
    plot_rmsd_histogram(pca_samples_rmsd, bins=50, title="RMSD Histogram PCA", xlabel="RMSD (Å)", ylabel="Frequency")

    # Write the PCA-based samples directly to "pca_generated_samples.xyz"
    save_xyz("pca_samples.xyz", pca_samples, atom_types)
    print(f"Saved {len(pca_samples)} PCA-based samples to 'pca_samples.xyz'.")

    return pca_samples

def generate_surface_core_pca_samples(
    md_positions,
    atom_types,
    surface_atom_types,
    rmsd_md_internal,
    representative_md,
    num_samples,
    temperature,
    temperature_target,
    temperature_target_surface,
    max_displacement
):
    # Compute RMSD-based scaling factor
    rmsd_values = np.mean(rmsd_md_internal, axis=1)  # Mean RMSD of each frame to all others
    print(f"rmsd mean of each from to all others : {rmsd_values}")
    rmsd_scaling_factor = np.mean(rmsd_values)
    print(f"RMSD-based scaling factor: {rmsd_scaling_factor:.2f}")

    # Identify surface and core atoms
    mean_positions = np.mean(md_positions, axis=0)
    mean_structure_path = PROJECT_ROOT / "data" / "processed" / "mean_structure.xyz"
    save_xyz(str(mean_structure_path), mean_positions[np.newaxis, :, :], atom_types)
    surface_replaced_file_path = PROJECT_ROOT / "data" / "processed" / "surface_replaced.xyz"
    surface_indices, replaced_atom_types = compute_surface_indices_with_replace_surface_dynamic(
            str(mean_structure_path),
            surface_atom_types,
            str(surface_replaced_file_path),
            f=1.0,
    )
    num_atoms = mean_positions.shape[0]
    core_indices = np.setdiff1d(np.arange(num_atoms), surface_indices)

    # Convert indices to arrays for compatibility
    surface_indices = np.array(surface_indices)
    core_indices = np.array(core_indices)

    # Flatten MD trajectory frames
    flattened_md_positions = np.array([frame.flatten() for frame in md_positions])

    # Convert atom indices (surface_indices) to coordinate indices
    surface_coordinate_indices = []
    for atom_idx in surface_indices:
        surface_coordinate_indices.extend([3 * atom_idx, 3 * atom_idx + 1, 3 * atom_idx + 2])
    surface_coordinate_indices = np.array(surface_coordinate_indices)

    # Convert atom indices (core_indices) to coordinate indices
    core_coordinate_indices = []
    for atom_idx in core_indices:
        core_coordinate_indices.extend([3 * atom_idx, 3 * atom_idx + 1, 3 * atom_idx + 2])
    core_coordinate_indices = np.array(core_coordinate_indices)

    # Separate atomic positions into surface and core
    surface_positions = flattened_md_positions[:, surface_coordinate_indices]
    print(f"surface positions shape {surface_positions.shape}")
    core_positions = flattened_md_positions[:, core_coordinate_indices]

    # Perform PCA for surface atoms
    pca_surface = PCA()
    pca_surface.fit(surface_positions)
    explained_variance_surface = np.cumsum(pca_surface.explained_variance_ratio_)
    optimal_components_surface = np.argmax(explained_variance_surface >= 0.90) + 1
    pca_surface = PCA(n_components=optimal_components_surface)
    pca_surface.fit(surface_positions)

    # Perform PCA for core atoms
    pca_core = PCA()
    pca_core.fit(core_positions)
    explained_variance_core = np.cumsum(pca_core.explained_variance_ratio_)
    optimal_components_core = np.argmax(explained_variance_core >= 0.90) + 1
    pca_core = PCA(n_components=optimal_components_core)
    pca_core.fit(core_positions)

    # Compute RMSD-based scaling factors for surface and core
    scaling_factors_surface = rmsd_scaling_factor * np.sqrt(temperature_target_surface / temperature)
    scaling_factors_core = rmsd_scaling_factor * np.sqrt(temperature_target / temperature)

    scaling_factors_surface = min(scaling_factors_surface, max_displacement)
    print(f"Scaling factors surface for PCA components (adjusted for T={temperature_target_surface}K): {scaling_factors_surface}")

    scaling_factors_core = min(scaling_factors_core, max_displacement)
    print(f"Scaling factors core for PCA components (adjusted for T={temperature_target}K): {scaling_factors_core}")

    # Generate PCA-based samples for surface and core
    print(f"Generating {num_samples} PCA-based samples for surface and core atoms...")
    combined_samples = []
    for i in range(num_samples):
        start_idx = np.random.choice(len(representative_md))  # Choose random frame from representative_md
        reference_structure = representative_md[start_idx]

        # Flatten the surface positions and generate perturbations
        reference_surface_flat = reference_structure[surface_indices].flatten()
        perturbed_surface_flat = generate_pca_samples(
            reference_surface_flat,
            pca_surface,
            1,
            scaling_factor=scaling_factors_surface
        )[0]
        perturbed_surface = perturbed_surface_flat.reshape(len(surface_indices), 3)

        # Flatten the core positions and generate perturbations
        reference_core_flat = reference_structure[core_indices].flatten()
        perturbed_core_flat = generate_pca_samples(
            reference_core_flat,
            pca_core,
            1,
            scaling_factor=scaling_factors_core
        )[0]
        perturbed_core = perturbed_core_flat.reshape(len(core_indices), 3)

        # Combine perturbed surface and core atoms into one structure
        combined_structure = np.zeros_like(reference_structure)
        combined_structure[surface_indices] = perturbed_surface
        combined_structure[core_indices] = perturbed_core

        combined_samples.append(combined_structure)

        # Print progress every 100 samples or for the first sample
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Generating sample {i + 1}/{num_samples}...")

    # Save combined structures
    combined_samples = np.array(combined_samples)
    save_xyz("pca_surface_core_combined_samples.xyz", combined_samples, atom_types)
    print(f"Saved {len(combined_samples)} PCA-based samples to 'pca_surface_core_combined_samples.xyz'.")

    return combined_samples

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
        max_displacement
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
