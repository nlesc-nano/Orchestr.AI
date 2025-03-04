import numpy as np
import yaml 
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from ase import Atoms
from dscribe.descriptors import SOAP

def load_config(config_file):
    """Load input parameters from a YAML configuration file."""
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

def save_to_npz(filename, atomic_numbers, positions, energies, forces):
    """
    Save dataset to an .npz file in the required dictionary format.

    Args:
        filename (str): Output file name.
        atomic_numbers (list or np.array): Atomic numbers of selected atoms.
        positions (np.array): Atomic positions (N, 3).
        energies (np.array): Energies corresponding to the configurations.
        forces (np.array): Forces acting on the atoms (N, 3).
    """
    # Extract the filename without suffix for 'name' field
    name = os.path.splitext(os.path.basename(filename))[0]
    
    # Construct dictionary in the required format
    base_vars = {
        'type': 'dataset',
        'name': name,
        'R': positions,   # Atomic positions
        'z': np.array(atomic_numbers, dtype=np.int32),  # Atomic numbers
        'F': forces       # Forces
    }
    
    # Save as .npz
    np.savez_compressed(filename, **base_vars)
    print(f"Saved dataset to {filename}")

def plot_energy_and_forces(energies, forces, filename='analysis.png'):
    """
    Plot total energy, energy per atom (with average ± 2σ, 3σ lines, and 0.05 eV/atom line),
    max force, and average force (with ±2σ, ±3σ lines).

    Parameters:
        energies (np.ndarray): Total energies (eV) of shape [num_frames].
        forces (np.ndarray): Forces of shape [num_frames, num_atoms, 3] in eV/Å.
        filename (str): Filename for the saved .png plot.
    """
    num_frames = len(energies)
    frames = np.arange(num_frames)

    # Number of atoms per frame (assumed constant)
    num_atoms = forces.shape[1]

    # --- ENERGY PER ATOM ---
    energy_per_atom = energies / num_atoms

    # Statistics for energy per atom
    mean_epa = np.mean(energy_per_atom)
    std_epa = np.std(energy_per_atom)
    epa_2sigma_plus = mean_epa + 2 * std_epa
    epa_3sigma_plus = mean_epa + 3 * std_epa
    # If you also want negative thresholds, e.g., mean - 2σ:
    epa_2sigma_minus = mean_epa - 2 * std_epa
    epa_3sigma_minus = mean_epa - 3 * std_epa

    # 0.05 eV/atom line (often cited as "chemical accuracy")
    chem_accuracy_plus = mean_epa + 0.05
    chem_accuracy_minus = mean_epa - 0.05

    # --- FORCES ---
    force_magnitudes = np.linalg.norm(forces, axis=2)  # shape: [num_frames, num_atoms]
    max_force_per_frame = np.max(force_magnitudes, axis=1)
    avg_force_per_frame = np.mean(force_magnitudes, axis=1)

    # Statistics for average force
    mean_avgF = np.mean(avg_force_per_frame)
    std_avgF = np.std(avg_force_per_frame)
    avgF_2sigma = mean_avgF + 2 * std_avgF
    avgF_3sigma = mean_avgF + 3 * std_avgF
    # If you want negative lines (uncommon for force magnitudes), you could do mean_avgF - 2σ, etc.

    # --- PLOTTING ---
    fig, axes = plt.subplots(4, 1, figsize=(10, 20))

    # 1) TOTAL ENERGY
    axes[0].plot(frames, energies, marker='o', linestyle='-', color='blue', label='Total Energy')
    axes[0].set_xlabel('Frame Index')
    axes[0].set_ylabel('Total Energy (eV)')
    axes[0].set_title('Total Energy per Frame')
    axes[0].legend()

    # 2) ENERGY PER ATOM
    axes[1].plot(frames, energy_per_atom, marker='o', linestyle='-', color='purple', label='Energy/Atom')
    # Add lines for mean ± 2σ and ±3σ
    axes[1].axhline(mean_epa, color='gray', linestyle='--', label='Mean')
    axes[1].axhline(epa_2sigma_plus, color='orange', linestyle='--', label='Mean + 2σ')
    axes[1].axhline(epa_3sigma_plus, color='red', linestyle='--', label='Mean + 3σ')
    axes[1].axhline(epa_2sigma_minus, color='orange', linestyle='--')
    axes[1].axhline(epa_3sigma_minus, color='red', linestyle='--')
    # Chemical accuracy line
    axes[1].axhline(chem_accuracy_plus, color='green', linestyle=':', label='0.05 eV/atom')
    axes[1].axhline(chem_accuracy_minus, color='green', linestyle=':', label='0.05 eV/atom')
    axes[1].set_xlabel('Frame Index')
    axes[1].set_ylabel('Energy per Atom (eV/atom)')
    axes[1].set_title('Energy per Atom per Frame')
    axes[1].legend()

    # 3) MAX FORCE
    axes[2].plot(frames, max_force_per_frame, marker='o', linestyle='-', color='red', label='Max Force')
    axes[2].set_xlabel('Frame Index')
    axes[2].set_ylabel('Force Magnitude (eV/Å)')
    axes[2].set_title('Max Force per Frame')
    axes[2].legend()

    # 4) AVERAGE FORCE
    axes[3].plot(frames, avg_force_per_frame, marker='o', linestyle='-', color='green', label='Avg Force')
    axes[3].axhline(mean_avgF, color='gray', linestyle='--', label='Mean')
    axes[3].axhline(avgF_2sigma, color='orange', linestyle='--', label='Mean + 2σ')
    axes[3].axhline(avgF_3sigma, color='red', linestyle='--', label='Mean + 3σ')
    axes[3].set_xlabel('Frame Index')
    axes[3].set_ylabel('Force Magnitude (eV/Å)')
    axes[3].set_title('Average Force per Frame (with thresholds)')
    axes[3].legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    print(f"Plots saved to {filename}")


def analyze_fluctuations(energies, forces):
    """
    Analyze the fluctuations of energies and forces to suggest loss weights
    with and without normalization.

    Parameters:
        energies (np.ndarray): Array of total energies (shape: [num_samples]).
        forces (np.ndarray): Array of forces (shape: [num_samples, num_atoms, 3]).

    Returns:
        dict: Suggested loss weights for energies and forces in both normalized and unnormalized cases.
    """
    # Compute standard deviation of energies
    energy_fluctuation = np.std(energies)

    # Compute standard deviation of force components
    force_fluctuation = np.std(forces)  # Automatically flattens the array

    # Compute raw weights
    weight_energy_raw = force_fluctuation / energy_fluctuation
    weight_forces_raw = 1.0  # Forces are baseline

    # Compute normalized weights
    total_weight = weight_energy_raw + weight_forces_raw
    weight_energy_norm = weight_energy_raw / total_weight
    weight_forces_norm = weight_forces_raw / total_weight

    # Print results
    print(f"Energy fluctuation (std): {energy_fluctuation:.6f}")
    print(f"Force fluctuation (std): {force_fluctuation:.6f}")
    print("\nSuggested loss weights:")
    print(f"  Without normalization: Energy = {weight_energy_raw:.6f}, Forces = {weight_forces_raw:.6f}")
    print(f"  With normalization:    Energy = {weight_energy_norm:.6f}, Forces = {weight_forces_norm:.6f}")

    return {
        "raw": {"energy": weight_energy_raw, "forces": weight_forces_raw},
        "normalized": {"energy": weight_energy_norm, "forces": weight_forces_norm},
    }


def parse_stacked_xyz(filename):
    energies = []
    positions = []
    forces = []
    atom_types = []
    
    with open(filename, "r") as f:
        lines = f.readlines()
        idx = 0
        while idx < len(lines):
            num_atoms = int(lines[idx].strip())
            idx += 1
            energy_line = lines[idx].strip().split()
            energy = float(energy_line[0])
            energies.append(energy)
            idx += 1
            
            frame_positions = []
            frame_forces = []
            
            if not atom_types:
                frame_atom_types = []
                for _ in range(num_atoms):
                    parts = lines[idx].split()
                    frame_atom_types.append(parts[0])
                    frame_positions.append([float(x) for x in parts[1:4]])
                    frame_forces.append([float(x) for x in parts[4:7]])
                    idx += 1
                atom_types = frame_atom_types
            else:
                for _ in range(num_atoms):
                    parts = lines[idx].split()
                    frame_positions.append([float(x) for x in parts[1:4]])
                    frame_forces.append([float(x) for x in parts[4:7]])
                    idx += 1
            
            positions.append(frame_positions)
            forces.append(frame_forces)
    
    return np.array(energies), np.array(positions), np.array(forces), atom_types

def create_labels_from_counts(counts):
    total = sum(counts)
    labels = np.empty(total, dtype=int)
    start = 0
    for i, c in enumerate(counts):
        end = start + c
        labels[start:end] = i
        start = end
    return labels

def plot_pca(features, labels, title="Feature PCA", filename="pca_plot.png"):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)
    
    plt.figure(figsize=(8,6))
    color_map = ["blue", "green", "red", "orange", "purple", "brown", "pink", "gray"]
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        mask = (labels == lbl)
        plt.scatter(reduced[mask, 0], reduced[mask, 1], 
                    label=f"Group {lbl}", alpha=0.7, color=color_map[lbl % len(color_map)])
    plt.legend()
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_outliers(features, labels, outliers, title="Outlier Detection", filename="outliers_plot.png"):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)
    
    plt.figure(figsize=(8,6))
    color_map = ["blue", "green", "red", "orange"]
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        lbl_mask = (labels == lbl)
        inliers_mask = (outliers == 1) & lbl_mask
        outliers_mask = (outliers == -1) & lbl_mask
        plt.scatter(reduced[inliers_mask, 0], reduced[inliers_mask, 1], 
                    alpha=0.7, label=f"Label {lbl} Inliers", 
                    color=color_map[lbl % len(color_map)])
        plt.scatter(reduced[outliers_mask, 0], reduced[outliers_mask, 1], 
                    alpha=0.9, label=f"Label {lbl} Outliers", 
                    color=color_map[lbl % len(color_map)], marker='x', s=60)
    
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_final_selection(features, labels, selected_indices, title="Final Selection", filename="final_selection.png"):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)
    
    plt.figure(figsize=(8,6))
    color_map = ["blue", "green", "red", "orange"]
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        lbl_mask = (labels == lbl)
        plt.scatter(reduced[lbl_mask, 0], reduced[lbl_mask, 1], 
                    alpha=0.5, label=f"Label {lbl} All", 
                    color=color_map[lbl % len(color_map)])
    plt.scatter(reduced[selected_indices, 0], reduced[selected_indices, 1], 
                facecolors='none', edgecolors='black', s=100, label="Selected Representatives")
    
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.savefig(filename, dpi=300)
    plt.close()

def save_stacked_xyz(filename, energies, positions, forces, atom_types):
    """
    Save data to a SchNetPack-compatible XYZ file.

    Parameters:
        filename (str): Output file path.
        energies (np.ndarray): Energies for each frame (shape: num_frames).
        positions (np.ndarray): Positions of atoms (shape: num_frames x num_atoms x 3).
        forces (np.ndarray): Forces on atoms (shape: num_frames x num_atoms x 3).
        atom_types (list): List of atom types (length: num_atoms).
    """
    num_frames, num_atoms, _ = positions.shape
    print(f"Saving SchNetPack-compatible XYZ file to {filename}...")
    with open(filename, "w") as f:
        for frame_idx in range(num_frames):
            # Write the number of atoms
            f.write(f"{num_atoms}\n")
            
            # Write the energy line with the required `E=` prefix
            f.write(f"{energies[frame_idx]:.6f}\n")
            
            # Write atomic positions and forces
            for atom, (x, y, z), (fx, fy, fz) in zip(atom_types, positions[frame_idx], forces[frame_idx]):
                f.write(f"{atom:<2} {x:>12.6f} {y:>12.6f} {z:>12.6f} {fx:>12.6f} {fy:>12.6f} {fz:>12.6f}\n")
            


def compute_local_descriptors(positions, atom_types, soap):
    num_frames = positions.shape[0]
    descriptors = []
    print("Computing SOAP descriptors for each frame...")
    for i in range(num_frames):
        if i % 100 == 0:
            print(f"Processing frame {i+1}/{num_frames}...")
        atoms_frame = Atoms(symbols=atom_types, positions=positions[i], pbc=False)
        
        # Compute SOAP for all atoms
        frame_soap = soap.create(atoms_frame)
        # Average over atoms to get one descriptor per frame
        frame_desc = frame_soap.mean(axis=0)
        descriptors.append(frame_desc)
    
    descriptors = np.array(descriptors)
    print(f"Finished computing SOAP descriptors.")
    print(f"Local descriptor shape: {descriptors.shape}")
    if descriptors.size > 0:
        print("Sample local descriptor vector:", descriptors[0])
    return descriptors

def save_to_npz(filename, atomic_numbers, positions, energies, forces, cells=None, pbc=None):
    """
    Save dataset in SchNetPack-compatible .npz format.

    Parameters:
        filename (str): Name of the output .npz file.
        atomic_numbers (list of list): List of atomic numbers for each frame.
        positions (np.ndarray): Atomic positions (shape: num_frames x num_atoms x 3).
        energies (np.ndarray): Energies for each frame (shape: num_frames).
        forces (np.ndarray): Atomic forces (shape: num_frames x num_atoms x 3).
        cells (np.ndarray, optional): Unit cell matrices (shape: num_frames x 3 x 3).
        pbc (np.ndarray, optional): Periodic boundary conditions (shape: num_frames x 3).
    """
    print(f"Saving dataset to {filename}...")
    data = {
        "atomic_numbers": np.array(atomic_numbers, dtype=np.int32),
        "positions": positions,
        "energies": energies,
        "forces": forces,
    }
    
    if cells is not None:
        data["cells"] = cells
    if pbc is not None:
        data["pbc"] = pbc
    
    np.savez_compressed(filename, **data)
    print(f"Dataset saved to {filename}.")

def analyze_reference_forces(forces, atom_types):
    """
    Analyze forces from the reference dataset.

    Args:
        forces (numpy array): Array of shape (n_frames, n_atoms, 3) containing forces.
        atom_types (list): List of atom types corresponding to each atom.

    Returns:
        dict: Summary statistics including per-atom and per-frame analysis.
    """
    # Compute force magnitudes
    force_magnitudes = np.linalg.norm(forces, axis=2)  # Shape: (n_frames, n_atoms)

    # Per-atom analysis
    per_atom_means = np.mean(force_magnitudes, axis=0)  # Shape: (n_atoms,)
    per_atom_stds = np.std(force_magnitudes, axis=0)    # Shape: (n_atoms,)
    per_atom_ranges = np.ptp(force_magnitudes, axis=0)  # Shape: (n_atoms,)

    # Per-frame analysis
    per_frame_means = np.mean(force_magnitudes, axis=1)  # Shape: (n_frames,)
    per_frame_stds = np.std(force_magnitudes, axis=1)    # Shape: (n_frames,)
    per_frame_ranges = np.ptp(force_magnitudes, axis=1)  # Shape: (n_frames,)

    # Overall statistics
    overall_mean = np.mean(force_magnitudes)
    overall_std = np.std(force_magnitudes)
    overall_range = np.ptp(force_magnitudes)

    # Per-atom-type analysis
    atom_type_means = {}
    atom_type_stds = {}
    atom_type_ranges = {}

    unique_atom_types = np.unique(atom_types)
    for atom_type in unique_atom_types:
        indices = [i for i, at in enumerate(atom_types) if at == atom_type]
        atom_type_forces = force_magnitudes[:, indices]  # Shape: (n_frames, n_atoms_of_type)

        atom_type_means[atom_type] = np.mean(atom_type_forces)
        atom_type_stds[atom_type] = np.std(atom_type_forces)
        atom_type_ranges[atom_type] = np.ptp(atom_type_forces)

    summary = {
        "per_atom_means": per_atom_means,
        "per_atom_stds": per_atom_stds,
        "per_atom_ranges": per_atom_ranges,
        "per_frame_means": per_frame_means,
        "per_frame_stds": per_frame_stds,
        "per_frame_ranges": per_frame_ranges,
        "overall_mean": overall_mean,
        "overall_std": overall_std,
        "overall_range": overall_range,
        "atom_type_means": atom_type_means,
        "atom_type_stds": atom_type_stds,
        "atom_type_ranges": atom_type_ranges,
    }

    return summary

def suggest_thresholds(force_stats, std_fraction=0.1, range_fraction=0.1):
    """
    Suggest typical thresholds for standard deviation and range to detect problematic frames.

    Args:
        force_stats (dict): Force statistics from the analyze_reference_forces function.
        std_fraction (float): Fraction of overall standard deviation to suggest variance threshold.
        range_fraction (float): Fraction of overall range to suggest range threshold.

    Returns:
        dict: Suggested thresholds for standard deviation and range, including per-atom-type thresholds.
    """
    overall_std = force_stats["overall_std"]
    overall_range = force_stats["overall_range"]

    # Compute overall thresholds
    std_threshold = std_fraction * overall_std
    range_threshold = range_fraction * overall_range

    print(f"Suggested Std Threshold (based on {std_fraction*100:.1f}% of Overall Std): {std_threshold:.6f} eV/Å")
    print(f"Suggested Range Threshold (based on {range_fraction*100:.1f}% of Overall Range): {range_threshold:.6f} eV/Å")

    # Compute per-atom-type thresholds
    atom_type_thresholds = {}
    for atom_type, atom_std in force_stats["atom_type_stds"].items():
        atom_range = force_stats["atom_type_ranges"][atom_type]
        atom_std_threshold = std_fraction * atom_std
        atom_range_threshold = range_fraction * atom_range

        atom_type_thresholds[atom_type] = {
            "std_threshold": atom_std_threshold,
            "range_threshold": atom_range_threshold,
        }

        print(f"Atom Type {atom_type}: Suggested Std Threshold = {atom_std_threshold:.6f} eV/Å, "
              f"Range Threshold = {atom_range_threshold:.6f} eV/Å")

    return {
        "overall": {
            "std_threshold": std_threshold,
            "range_threshold": range_threshold,
        },
        "per_atom_type": atom_type_thresholds,
    }

def generate_random_subsets(input_file, output_prefix="random_dataset", sizes=[500, 1000, 2000, 4000]):
    """Randomly selects structures from input_file and saves them as new datasets."""
    
    print(f"Loading dataset for random selection: {input_file}")
    energies, positions, forces, atom_types = parse_stacked_xyz(input_file)

    total_structures = len(energies)
    
    for size in sizes:
        if size > total_structures:
            print(f"Warning: Requested {size} samples, but dataset only contains {total_structures}. Using all available structures.")
            size = total_structures
        
        # Select random indices
        selected_indices = np.random.choice(total_structures, size, replace=False)
        
        # Save randomly selected dataset
        xyz_filename = f"{output_prefix}_{size}.xyz"
        save_stacked_xyz(
            xyz_filename,
            energies[selected_indices],
            positions[selected_indices],
            forces[selected_indices],
            atom_types
        )

        # Save as npz 
        npz_filename = f"{output_prefix}_{size}.npz"
        atomic_numbers = [[atom_types.index(atom) + 1 for atom in atom_types]]
        save_to_npz(npz_filename, atomic_numbers, positions[selected_indices], energies[selected_indices], forces[selected_indices])

        plot_energy_and_forces(energies[selected_indices], forces[selected_indices], f'plot_energy_and_forces_{output_prefix}_{size}.png')

        print(f"Random dataset of {size} samples saved as {xyz_filename}")

def generate_md_random_subsets(input_file, md_subset_size=2001, output_prefix="md_random_dataset", sizes=[500, 1000, 2000]):
    """Randomly selects structures only from the MD subset (first 2001 frames) and saves them."""
    
    print(f"Loading dataset for MD-only random selection: {input_file}")
    energies, positions, forces, atom_types = parse_stacked_xyz(input_file)

    if md_subset_size > len(energies):
        print(f"Warning: MD subset size ({md_subset_size}) is larger than available dataset ({len(energies)}). Adjusting to available size.")
        md_subset_size = len(energies)

    # Keep only the first 2001 structures (MD subset)
    energies, positions, forces = energies[:md_subset_size], positions[:md_subset_size], forces[:md_subset_size]

    for size in sizes:
        if size > md_subset_size:
            print(f"Warning: Requested {size} samples, but MD subset only contains {md_subset_size}. Using all available MD structures.")
            size = md_subset_size
        
        # Select random indices from the MD subset
        selected_indices = np.random.choice(md_subset_size, size, replace=False)
        
        # Save MD-randomly selected dataset
        xyz_filename = f"{output_prefix}_{size}.xyz"
        save_stacked_xyz(
            xyz_filename,
            energies[selected_indices],
            positions[selected_indices],
            forces[selected_indices],
            atom_types
        )
   
        # Save as .npz
        npz_filename = f"{output_prefix}_{size}.npz"
        atomic_numbers = [[atom_types.index(atom) + 1 for atom in atom_types]]
        save_to_npz(npz_filename, atomic_numbers, positions[selected_indices], energies[selected_indices], forces[selected_indices])

        plot_energy_and_forces(energies[selected_indices], forces[selected_indices], f'plot_energy_and_forces_{output_prefix}_{size}.png')

        print(f"MD-only random dataset of {size} samples saved as {xyz_filename}")


def consolidate_dataset(config):
    """Main dataset consolidation pipeline based on YAML configuration."""
    
    # Load dataset parameters from config
    input_file = config["dataset"]["input_file"]
    output_prefix = config["dataset"]["output_prefix"]
    sizes = config["dataset"]["sizes"]
    subset_counts_dict = config["dataset"]["subset_counts"]
    contamination = config["dataset"]["contamination"]
    
    # Extract subset counts from dictionary
    subset_counts = list(subset_counts_dict.values())  # Extract values in order
    total_structures = sum(subset_counts)  # Compute total number of structures

    print(f"Parsing input file: {input_file}")
    energies, positions, forces, atom_types = parse_stacked_xyz(input_file)
    
    print(f"Parsed positions shape: {positions.shape}")
    print(f"Parsed forces shape: {forces.shape}")
    print(f"Parsed energies shape: {energies.shape}")
    
    if total_structures != len(energies):
        print(f"Warning: The sum of subset_counts {total_structures} does not match the total frames {len(energies)}.")
   
    plot_energy_and_forces(energies, forces, 'starting_dataset_energy_forces.png')
 
    labels = create_labels_from_counts(subset_counts)  # Adjusted for dictionary format
    
    # Compute global features
    avg_force_magnitudes = np.linalg.norm(forces, axis=2).mean(axis=1)
    force_variance = forces.var(axis=(1, 2))

    # Analysis detailed on global features
    force_stats = analyze_reference_forces(forces, atom_types)
    
    print(f"Overall Mean Force Magnitude: {force_stats['overall_mean']:.6f} eV/Å")
    print(f"Overall Force Standard Deviation: {force_stats['overall_std']:.6f} eV/Å")
    print(f"Overall Force Range: {force_stats['overall_range']:.6f} eV/Å")

    for atom_type, mean in force_stats["atom_type_means"].items():
        std = force_stats["atom_type_stds"][atom_type]
        rng = force_stats["atom_type_ranges"][atom_type]
        print(f"Atom Type {atom_type}: Mean = {mean:.6f} eV/Å, Std = {std:.6f} eV/Å, Range = {rng:.6f} eV/Å")

    thresholds = suggest_thresholds(force_stats, std_fraction=0.1, range_fraction=0.1)
    print("Suggested thresholds:", thresholds)

    species = sorted(list(set(atom_types)))
    atomic_numbers = [[atom_types.index(atom) + 1 for atom in atom_types]]

    # Load SOAP parameters from YAML
    soap_params = config["SOAP"]
    soap = SOAP(
        species=species,
        r_cut=soap_params["r_cut"],
        n_max=soap_params["n_max"],
        l_max=soap_params["l_max"],
        sigma=soap_params["sigma"],
        periodic=soap_params["periodic"],
        sparse=soap_params["sparse"]
    )
    
    print("Computing local descriptors (SOAP)...")
    local_features = compute_local_descriptors(positions, atom_types, soap)
    
    global_features = np.vstack((energies, avg_force_magnitudes, force_variance)).T
    raw_features = np.hstack((global_features, local_features))
    
    print("Scaling features...")
    scaler = StandardScaler()
    features = scaler.fit_transform(raw_features)
    
    print("Detecting gross outliers with Isolation Forest...")
    outlier_detector = IsolationForest(contamination=contamination, random_state=42)
    outliers_if = outlier_detector.fit_predict(features)
    
    plot_outliers(features, labels, outliers_if, title="Outlier Detection (IsolationForest)", filename=f"{output_prefix}_outliers_if.png")
    
    inlier_indices = np.where(outliers_if == 1)[0]
    energies = energies[inlier_indices]
    positions = positions[inlier_indices]
    forces = forces[inlier_indices]
    features = features[inlier_indices]
    labels = labels[inlier_indices]

    plot_energy_and_forces(energies, forces, 'plot_energy_and_forces_after_filtering.png') 
    print(f"Dataset size after removing IF outliers: {len(energies)}")
    
    if len(energies) == 0:
        print("No inlier samples remain after outlier removal. Please adjust your parameters or check data quality.")
        return
    
    print("Plotting final inlier distribution PCA...")
    plot_pca(features, labels, title="Final Inlier Features PCA", filename=f"{output_prefix}_final_inliers_pca.png")

    inlier_filename = f"inliers_full_dataset.xyz"
    save_stacked_xyz(inlier_filename, energies, positions, forces, atom_types)

    # Generate random subsets (all structures)
    generate_random_subsets(inlier_filename, "random_dataset", sizes)

    for target_size in sizes:
        print(f"Consolidating to {target_size} samples...")
        
        if len(features) < target_size:
            print(f"Warning: Target size {target_size} exceeds available samples ({len(features)}). Using all samples instead.")
            target_size = len(features)
        
        kmeans = KMeans(n_clusters=target_size, random_state=0)
        clusters = kmeans.fit_predict(features)
        cluster_centers = kmeans.cluster_centers_
        
        selected_indices = []
        for cluster_label in range(target_size):
            cluster_indices = np.where(clusters == cluster_label)[0]
            distances = np.linalg.norm(features[cluster_indices] - cluster_centers[cluster_label], axis=1)
            representative_index = cluster_indices[np.argmin(distances)]
            selected_indices.append(representative_index)
        
        selected_indices = np.array(selected_indices)
        print(f"Final dataset for size {target_size}: {len(selected_indices)} samples")

        xyz_filename = f"{output_prefix}_{target_size}.xyz"
        save_stacked_xyz(
            xyz_filename,
            energies[selected_indices],
            positions[selected_indices],
            forces[selected_indices],
            atom_types
        )

        plot_energy_and_forces(energies[selected_indices], forces[selected_indices], f'plot_energy_and_forces_{output_prefix}_{target_size}.png')

        # Save as .npz
        npz_filename = f"{output_prefix}_{target_size}.npz"
        save_to_npz(npz_filename, atomic_numbers, positions, energies, forces)

        loss_weights = analyze_fluctuations(energies[selected_indices], forces[selected_indices])

if __name__ == "__main__":
    config_file = "input.yaml"
    config = load_config(config_file)

    # Generate random subsets (only from MD structures)
    generate_md_random_subsets(config["dataset"]["input_file"], config["dataset"]["subset_counts"]["MD"], "md_random_dataset", config["dataset"]["sizes"])

    # Run the main dataset consolidation pipeline
    consolidate_dataset(config)
