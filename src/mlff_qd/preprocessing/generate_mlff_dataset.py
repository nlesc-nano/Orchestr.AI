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

np.set_printoptions(threshold=np.inf)

# --- Constants ---
hartree_bohr_to_eV_angstrom = 51.42208619083232
hartree_to_eV = 27.211386245988  # 1 Hartree = 27.211386245988 eV
bohr_to_angstrom = 0.529177210903  # 1 Bohr = 0.529177210903 Å
amu_to_kg = 1.66053906660e-27  # 1 amu = 1.66053906660e-27 kg
c = 2.99792458e10  # Speed of light in cm/s

# --- Utility Functions ---
def generate_pca_samples(reference_structure, pca_model, num_samples, scaling_factor=1.0):
    """
    Generate new structures by perturbing along PCA components.

    Parameters:
        reference_structure (np.ndarray): Atomic positions of the reference structure (num_atoms, 3).
        pca_model (PCA): Precomputed PCA model.
        num_samples (int): Number of new structures to generate.
        scaling_factor (float): Scaling factor for perturbations.

    Returns:
        np.ndarray: Array of generated structures (num_samples, num_atoms, 3).
    """
    flattened_ref = reference_structure.flatten()  # Flatten the reference structure
    num_features = pca_model.components_.shape[1]  # Number of features used in PCA

    # Validate that the reference structure matches PCA components
    if flattened_ref.size != num_features:
        raise ValueError(
            f"Mismatch in size: flattened_ref ({flattened_ref.size}) does not match PCA components ({num_features})"
        )

    generated_structures = []

    for _ in range(num_samples):
        perturbation = np.zeros_like(flattened_ref)
        for pc_idx in range(pca_model.n_components_):
            perturbation += (
                scaling_factor
                * np.random.uniform(-1, 1)
                * pca_model.components_[pc_idx]
            )
        new_structure = flattened_ref + perturbation
        generated_structures.append(new_structure.reshape(-1, 3))

    return np.array(generated_structures)


def compute_surface_indices_with_replace_surface_dynamic(
    input_file, surface_atom_types, f=1.0, surface_replaced_file="surface_replaced.xyz"
):
    """
    Dynamically replace surface atoms with random elements from the periodic table that are not in the molecule.

    Parameters:
        input_file (str): Path to the input .xyz file.
        surface_atom_types (list): List of atom types considered as surface atoms to be replaced.
        f (float): Fraction of surface atoms to replace (default: 1.0).
        surface_replaced_file (str): Path to the output .xyz file with surface atoms replaced.

    Returns:
        surface_indices (list): Indices of the replaced surface atoms in the structure.
        replaced_atom_types (list): Updated atom types after replacement.
    """

    processed_dir = Path(__file__).resolve().parents[3] / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    surface_replaced_file = processed_dir / surface_replaced_file

    print(f"Reading molecule from {input_file}...")
    mol_original = Molecule(input_file)  # Load the original molecule
    mol_updated = mol_original.copy()  # Create a copy to track cumulative changes

    # Identify atom types in the molecule
    molecule_atom_types = {atom.symbol for atom in mol_original}

    # Generate a list of replacement elements from the periodic table
    available_elements = [el.symbol for el in elements if el.symbol not in molecule_atom_types]

    # Prepare replacements dynamically
    replacements = [
        (atom_type, random.choice(available_elements)) for atom_type in surface_atom_types
    ]
    print(f"Dynamic replacements: {replacements}")

    surface_indices = []  # Collect indices of replaced surface atoms
    for i, (original_symbol, replacement_symbol) in enumerate(replacements):
        print(f"Replacing surface atoms: {original_symbol} -> {replacement_symbol} (f={f})...")

        # Create a new molecule for this replacement
        mol_new = replace_surface(mol_updated, symbol=original_symbol, symbol_new=replacement_symbol, f=f)

        # Update `mol_updated` to incorporate the changes
        mol_updated = mol_new.copy()

        # Identify the replaced atoms in the molecule
        for idx, atom in enumerate(mol_new):
            if atom.symbol == replacement_symbol:
                surface_indices.append(idx)

        print(f"Replacement {i+1}: {len(surface_indices)} surface atoms replaced so far.")

    # Save the final updated molecule to the output file
    print(f"Writing modified molecule with replacements to {surface_replaced_file}...")
    mol_updated.write(str(surface_replaced_file))

    # Extract updated atom types
    replaced_atom_types = [atom.symbol for atom in mol_updated]

    print(f"Surface replacements completed. {len(surface_indices)} surface atoms identified and replaced.")
    return surface_indices, replaced_atom_types

def save_positions_xyz(filename, positions, atom_types, energies=None):
    """
    Save aligned positions to an XYZ file, optionally including energies.

    Parameters:
        filename (str): Path to the output XYZ file.
        positions (np.ndarray): Atomic positions (num_frames, num_atoms, 3).
        atom_types (list): Atomic types (num_atoms).
        energies (list or np.ndarray): Total energies for each frame (optional).
    """
    processed_dir = Path(__file__).resolve().parents[3] / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_path = processed_dir / filename
    print(f"Saving aligned positions to {output_path}...")

    num_frames, num_atoms, _ = positions.shape

    with open(output_path, "w") as f:
        for frame_idx in range(num_frames):
            f.write(f"{num_atoms}\n")

            # Include energy in the title if provided
            if energies is not None and energies[frame_idx] is not None:
                f.write(f"Frame {frame_idx + 1}: Aligned positions, Energy = {energies[frame_idx]:.6f} eV\n")
            else:
                f.write(f"Frame {frame_idx + 1}: Aligned positions\n")

            for atom, (x, y, z) in zip(atom_types, positions[frame_idx]):
                f.write(f"{atom} {x:.6f} {y:.6f} {z:.6f}\n")

    print(f"Aligned positions saved to {output_path}.")


def save_forces_xyz(filename, forces, atom_types):
    """
    Save aligned forces to an XYZ-like file.

    Parameters:
        filename (str): Path to the output file.
        forces (np.ndarray): Atomic forces (num_frames, num_atoms, 3).
        atom_types (list): Atomic types (num_atoms).
    """
    processed_dir = Path(__file__).resolve().parents[3] / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_path = processed_dir / filename
    print(f"Saving aligned forces to {output_path}...")

    num_frames, num_atoms, _ = forces.shape
    with open(output_path, "w") as f:
        for frame_idx in range(num_frames):
            f.write(f"{num_atoms}\n")
            f.write(f"Frame {frame_idx + 1}: Aligned forces\n")
            for atom, (fx, fy, fz) in zip(atom_types, forces[frame_idx]):
                f.write(f"{atom} {fx:.6f} {fy:.6f} {fz:.6f}\n")
    
    print(f"Aligned forces saved to {output_path}.")

def save_frequencies(filename, frequencies):
    """
    Save vibrational frequencies to a file.

    Parameters:
        filename (str): Path to the output file.
        frequencies (np.ndarray): Vibrational frequencies (cm^-1).
    """
    print(f"Saving vibrational frequencies to {filename}...")
    with open(filename, "w") as f:
        f.write("Vibrational Frequencies (cm^-1):\n")
        for i, freq in enumerate(frequencies, start=1):
            f.write(f"Mode {i}: {freq:.6f} cm^-1\n")
    print(f"Frequencies saved to {filename}.")

def center_positions(positions, masses):
    """
    Center atomic positions by translating the center of mass (COM) to the origin.

    Parameters:
        positions (np.ndarray): Atomic positions (num_frames, num_atoms, 3).
        masses (np.ndarray): Atomic masses (num_atoms).

    Returns:
        np.ndarray: Centered atomic positions (num_frames, num_atoms, 3).
    """
    num_frames, num_atoms, _ = positions.shape
    com_frames = np.zeros((num_frames, 3))

    # Calculate COM for each frame
    for frame_idx in range(num_frames):
        com_frames[frame_idx] = np.sum(
            positions[frame_idx] * masses[:, np.newaxis], axis=0
        ) / masses.sum()

    # Subtract COM from all positions
    centered_positions = positions - com_frames[:, np.newaxis, :]
    return centered_positions

def align_to_reference(positions, reference_positions):
    """
    Align atomic positions of each frame to a reference structure using least-squares fitting.

    Parameters:
        positions (np.ndarray): Atomic positions (num_frames, num_atoms, 3).
        reference_positions (np.ndarray): Reference positions (num_atoms, 3).

    Returns:
        np.ndarray: Aligned atomic positions (num_frames, num_atoms, 3).
    """
    aligned_positions = np.zeros_like(positions)
    for frame_idx, frame in enumerate(positions):
        # Calculate optimal rotation matrix
        H = frame.T @ reference_positions
        U, _, Vt = np.linalg.svd(H)
        rotation_matrix = U @ Vt

        # Apply rotation to align the frame
        aligned_positions[frame_idx] = frame @ rotation_matrix.T
    return aligned_positions

def rotate_forces(forces, rotation_matrices):
    """
    Rotate forces for each frame to match the aligned orientation.

    Parameters:
        forces (np.ndarray): Atomic forces (num_frames, num_atoms, 3).
        rotation_matrices (np.ndarray): Rotation matrices (num_frames, 3, 3).

    Returns:
        np.ndarray: Rotated forces (num_frames, num_atoms, 3).
    """
    rotated_forces = np.zeros_like(forces)
    for frame_idx, frame_forces in enumerate(forces):
        rotated_forces[frame_idx] = frame_forces @ rotation_matrices[frame_idx]
    return rotated_forces

def compute_rmsd_matrix(structures1, structures2=None):
    """
    Compute the RMSD matrix for pairs of structures efficiently.

    Parameters:
        structures1 (list or np.ndarray): Array of shape (N1, num_atoms, 3)
                                          representing the first set of structures.
        structures2 (list or np.ndarray): Array of shape (N2, num_atoms, 3)
                                          representing the second set of structures.
                                          If None, computes the RMSD matrix within structures1.

    Returns:
        np.ndarray: RMSD matrix of shape (N1, N2).
    """
    # Ensure we have NumPy arrays
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
        # Compute only upper triangle and mirror it to avoid redundant computation
        rmsd_matrix = np.zeros((N1, N2), dtype=float)
        for i in range(N1):
            # Only compute from j = i to the end
            diff = structures2[i:] - structures1[i]  # shape: (N2 - i, A, 3)
            # sum of squared differences per structure
            sq_diff_sum = np.sum(diff**2, axis=(1, 2)) / A
            rmsd_vals = np.sqrt(sq_diff_sum)
            rmsd_matrix[i, i:] = rmsd_vals
            rmsd_matrix[i:, i] = rmsd_vals  # mirror to maintain symmetry
    else:
        # When structures2 is different, compute fully
        # Vectorized approach
        # Broadcast to create a differences array of shape (N1, N2, A, 3)
        diff = structures1[:, None, :, :] - structures2[None, :, :, :]
        # sum over atoms and coordinates
        sq_diff_sum = np.sum(diff**2, axis=(2, 3)) / A
        rmsd_matrix = np.sqrt(sq_diff_sum)

    return rmsd_matrix

def plot_rmsd_histogram(rmsd_matrix, bins=50, title="RMSD Histogram", xlabel="RMSD (Å)", ylabel="Frequency"):
    """
    Plot a histogram of RMSD values from an RMSD matrix.

    Parameters:
        rmsd_matrix (np.ndarray): A symmetric RMSD matrix (shape: num_samples x num_samples).
        bins (int): Number of bins for the histogram.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    # Extract the upper triangle of the RMSD matrix (excluding the diagonal)
    rmsd_values = rmsd_matrix[np.triu_indices_from(rmsd_matrix, k=1)]
    
    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(rmsd_values, bins=bins, edgecolor="black", alpha=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

def save_binary(filename, frequencies, positions, eigenvectors, atom_types):
    """Save frequencies, positions, eigenvectors, and atom types to a binary file."""
    data = {
        "frequencies": frequencies,
        "positions": positions,
        "eigenvectors": eigenvectors,
        "atom_types": atom_types
    }
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved data to binary file: {filename}")

def load_binary(filename):
    """Load frequencies, positions, eigenvectors, and atom types from a binary file."""
    with open(filename, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded data from binary file: {filename}")
    return data["frequencies"], data["eigenvectors"]

def save_xyz(filename, positions, atom_types, energies=None):
    """
    Save atomic positions to an XYZ file in 'data/processed'.

    Parameters:
        filename (str): The filename (e.g. 'aligned_positions.xyz').
        positions (list or np.ndarray): Atomic positions of shape (num_frames, num_atoms, 3).
        atom_types (list[str]): List of atom types (e.g. ["Cs", "Br", ...]).
        energies (list[float], optional): If provided, each energy is written in the frame comment line.
    """

    # Adjust parents[3] or parents[4] depending on how many levels you need to go up
    processed_dir = Path(__file__).resolve().parents[3] / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    output_path = processed_dir / filename

    with open(output_path, "w") as f:
        # positions: shape (num_frames, num_atoms, 3)
        num_frames = len(positions)
        num_atoms = len(atom_types)

        for i, frame in enumerate(positions):
            f.write(f"{num_atoms}\n")
            if energies and energies[i] is not None:
                f.write(f"Frame {i+1}, Energy = {energies[i]:.6f}\n")
            else:
                f.write(f"Frame {i+1}\n")

            for atom, (x, y, z) in zip(atom_types, frame):
                f.write(f"{atom} {x:.6f} {y:.6f} {z:.6f}\n")

def reorder_xyz_trajectory(input_file, output_file, num_atoms):
    """Reorder atoms in the XYZ trajectory."""
    processed_dir = Path(__file__).resolve().parents[3] / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_path = processed_dir / output_file

    print(f"Reordering atoms in trajectory file: {input_file}")
    
    with open(input_file, "r") as infile, open(output_path, "w") as outfile:
        lines = infile.readlines()
        num_lines_per_frame = num_atoms + 2
        for i in range(0, len(lines), num_lines_per_frame):
            header = lines[i:i + 2]
            atom_lines = lines[i + 2:i + 2 + num_atoms]
            sorted_atoms = sorted(atom_lines, key=lambda x: x.split()[0])
            outfile.writelines(header)
            outfile.writelines(sorted_atoms)
    print(f"Reordered trajectory saved to: {output_path}")

def parse_positions_xyz(filename, num_atoms):
    """
    Parse positions from an XYZ file.

    Parameters:
        filename (str): Path to the XYZ file.
        num_atoms (int): Number of atoms in each frame.

    Returns:
        np.ndarray: Atomic positions (num_frames, num_atoms, 3).
        list: Atomic types.
        list: Total energies for each frame (if available; otherwise, empty list).
    """
    print(f"Parsing positions XYZ file: {filename}")
    positions = []
    atom_types = []
    total_energies = []

    with open(filename, "r") as f:
        lines = f.readlines()
        num_lines_per_frame = num_atoms + 2  # 2 lines for header and comment

        for i in range(0, len(lines), num_lines_per_frame):
            atom_lines = lines[i + 2:i + 2 + num_atoms]
            comment_line = lines[i + 1]

            # Try parsing the energy; otherwise, skip
            try:
                total_energy = float(comment_line.split("=")[-1].strip())
                total_energies.append(total_energy)
            except ValueError:
                total_energies.append(None)  # Placeholder for missing energy

            frame_positions = []
            for line in atom_lines:
                parts = line.split()
                atom_types.append(parts[0])
                frame_positions.append([float(x) for x in parts[1:4]])

            positions.append(frame_positions)

    return np.array(positions), atom_types[:num_atoms], total_energies

def parse_forces_xyz(filename, num_atoms):
    """Parse forces from an XYZ file."""
    print(f"Parsing forces XYZ file: {filename}")
    forces = []
    with open(filename, "r") as f:
        lines = f.readlines()
        num_lines_per_frame = num_atoms + 2
        for i in range(0, len(lines), num_lines_per_frame):
            frame_forces = []
            for j in range(2, 2 + num_atoms):
                parts = lines[i + j].split()
                frame_forces.append(list(map(float, parts[1:4])))
            forces.append(frame_forces)
    return np.array(forces)

def get_num_atoms(filename):
    """
    Retrieve the number of atoms from the first line of an XYZ file.

    Parameters:
        filename (str): Path to the XYZ file.

    Returns:
        int: Number of atoms in the structure.
    """
    with open(filename, "r") as f:
        num_atoms = int(f.readline().strip())
    print(f"Number of atoms: {num_atoms}")
    return num_atoms

def create_mass_dict(atom_types):
    """
    Create a dictionary mapping atom types to their atomic masses.

    Parameters:
        atom_types (list): List of atomic types as strings.

    Returns:
        dict: Dictionary where keys are atom types and values are atomic masses.
    """

    mass_dict = {atom: elements.symbol(atom).mass for atom in set(atom_types)}
    print(f"Generated mass dictionary: {mass_dict}")
    return mass_dict

def perform_pca_and_plot(datasets, num_components=2, labels=None, ax=None):
    """
    Perform PCA and plot the results.

    Parameters:
        datasets (dict): A dictionary of dataset names and their corresponding samples.
        num_components (int): Number of PCA components.
        labels (list): List of sample labels (optional).
        ax (matplotlib.axes.Axes): Pre-created axis for plotting (optional).
    """

    print(f"Performing PCA for {len(datasets)} datasets...")
    combined_data = np.concatenate(list(datasets.values()))
    combined_labels = labels if labels else sum([[name] * len(data) for name, data in datasets.items()], [])

    # Perform PCA
    pca = PCA(n_components=num_components)
    pca_transformed = pca.fit_transform(combined_data)

    # Plot results
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    unique_labels = sorted(set(combined_labels))
    colors = plt.cm.get_cmap("tab10", len(unique_labels))
    for idx, label in enumerate(unique_labels):
        label_indices = [i for i, lbl in enumerate(combined_labels) if lbl == label]
        ax.scatter(
            pca_transformed[label_indices, 0],
            pca_transformed[label_indices, 1],
            label=label,
            alpha=0.7,
            color=colors(idx),
        )

    explained_variance = pca.explained_variance_ratio_ * 100
    ax.set_title(
        f"PCA Analysis of Molecular Configurations\n"
        f"Principal Component 1 ({explained_variance[0]:.2f}%) vs "
        f"Principal Component 2 ({explained_variance[1]:.2f}%)"
    )
    ax.set_xlabel(f"Principal Component 1 ({explained_variance[0]:.2f}%)")
    ax.set_ylabel(f"Principal Component 2 ({explained_variance[1]:.2f}%)")
    ax.legend()

    if ax is None:
        plt.tight_layout()
        plt.show()

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
    save_xyz("mean_structure.xyz", mean_positions[np.newaxis, :, :], atom_types)
    surface_replaced_file = "surface_replaced.xyz"
    surface_indices, replaced_atom_types = compute_surface_indices_with_replace_surface_dynamic(
        "data/processed/mean_structure.xyz",
        surface_atom_types=surface_atom_types,
        f=1.0,
        surface_replaced_file=surface_replaced_file,
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

    # Default values used with .get()
    pos_file = config.get("pos_file", "trajectory_pos.xyz")
    frc_file = config.get("frc_file", "trajectory_frc.xyz")
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
    num_atoms = get_num_atoms(pos_file) 

    # Get atom_types 
    _, atom_types, _ = parse_positions_xyz(pos_file, num_atoms)

    # Dynamically create the mass dictionary
    print("Creating mass dictionary...")
    mass_dict = create_mass_dict(atom_types)
    masses = np.array([mass_dict[atom] for atom in atom_types])

    # Reorder positions and forces by atom type
    reorder_xyz_trajectory(pos_file, "reordered_positions.xyz", num_atoms)
    reorder_xyz_trajectory(frc_file, "reordered_forces.xyz", num_atoms)

    # Parse reordered positions and forces
    positions, atom_types, energies_hartree = parse_positions_xyz("data/processed/reordered_positions.xyz", num_atoms)
    forces = parse_forces_xyz("data/processed/reordered_forces.xyz", num_atoms)

    # Center and align positions
    centered_positions = center_positions(positions, masses)
    aligned_positions = align_to_reference(centered_positions, centered_positions[0])

    # Align forces
    rotation_matrices = [R.align_vectors(frame, centered_positions[0])[0].as_matrix() for frame in centered_positions]
    aligned_forces = rotate_forces(forces, rotation_matrices)

    mean_positions = np.mean(aligned_positions, axis=0)

    # Save aligned positions and forces
    save_xyz("mean_structure.xyz", mean_positions[np.newaxis, :, :], atom_types)
    save_positions_xyz("aligned_positions.xyz", aligned_positions, atom_types, energies_hartree)
    save_forces_xyz("aligned_forces.xyz", aligned_forces, atom_types)

    # Convert energies to eV
    energies_ev = [e * hartree_to_eV if e is not None else None for e in energies_hartree]
    # Convert forces to eV/Å
    aligned_forces_ev = aligned_forces * hartree_bohr_to_eV_angstrom 
    save_positions_xyz("aligned_positions_ev.xyz", aligned_positions, atom_types, energies_ev)
    save_forces_xyz("aligned_forces_eV.xyz", aligned_forces_ev, atom_types)

    # Load aligned MD structures
    print("Parsing aligned MD structures...")
    md_positions, atom_types, _ = parse_positions_xyz("data/processed/aligned_positions.xyz", num_atoms)

    print("Computing RMSD matrix for MD trajectory...")
    rmsd_md_internal = compute_rmsd_matrix(md_positions)

    num_frames = md_positions.shape[0]  # Total number of frames
    print(f"Number of frames is : {num_frames}")

    # Clustering-Based Sampling
    print("Performing clustering on MD trajectory...")
    representative_md = cluster_trajectory(rmsd_md_internal, clustering_method, num_clusters, md_positions, atom_types) 

    # Generate sample dataset based on PCA  
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

    # Generate sample dataset based on PCA but separating core from surface, allowing for surface atoms to be displace more
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

    # Generate Randomized Samples
    randomized_samples = generate_randomized_samples(
        representative_md,
        atom_types,
        num_samples_randomization, 
        max_random_displacement
    )

    # Load or generate your datasets (MD, Wigner, Randomized, Normal Modes, Surface)
    combined_samples = { 
        "MD": md_positions, 
        "PCA": pca_samples, 
        "PCA_Surface": pca_surface_samples,
        "Randomized": randomized_samples,
    }

    # Print the shape of each dataset
    print("Dataset shapes before flattening:")
    for name, samples in combined_samples.items():
        print(f"{name}: {np.array(samples).shape}")

    # Plot the PCA of the generated sampls on the MD with either rmsd or positons  
    plot_generated_samples(combined_samples)

    # Combine all non-MD datasets into a single NumPy array
    combined_sample_list = []  # List to hold individual datasets
    frame_titles = []  # List to hold titles for each frame in the XYZ file

    for name, samples in combined_samples.items():
        if name != "MD":  # Exclude MD samples
            samples_array = np.array(samples)  # Ensure samples are a NumPy array
            combined_sample_list.append(samples_array)

            # Generate titles for each frame in this dataset
            for i in range(len(samples)):
                frame_titles.append(f"Frame {i + 1} from {name}")

    # Concatenate the non-MD datasets along the first axis
    combined_sample_reshaped = np.vstack(combined_sample_list).reshape(-1, num_atoms, 3)

    # Save the non-MD training dataset to XYZ file with titles
    processed_dir = Path(__file__).resolve().parents[3] / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "training_dataset.xyz"

    with open(output_path, "w") as xyz_file:
        for i, (structure, title) in enumerate(zip(combined_sample_reshaped, frame_titles)):
            xyz_file.write(f"{len(structure)}\n")  # Write the number of atoms
            xyz_file.write(f"{title}\n")  # Write the title of the frame
            for atom, coords in zip(atom_types, structure):
                xyz_file.write(f"{atom} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}\n")

    print(f"Saved non-MD training dataset with {combined_sample_reshaped.shape[0]} structures.")

