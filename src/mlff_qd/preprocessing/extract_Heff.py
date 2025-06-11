import os
import argparse
import re
import h5py
import numpy as np
import torch
import logging

from mlff_qd.utils.constants import hartree_to_eV

# Set up logging: log to both console and file.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# File handler
file_handler = logging.FileHandler("process_log.txt")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

########################
# HDF5 Reading Section #
########################

def read_hdf5_mo_data(hdf5_file):
    """Reads MO eigenvectors and eigenvalues from the given HDF5 file."""
    mo_data = {}
    with h5py.File(hdf5_file, "r") as f:
        coeff_grp = f["coefficients"]
        eigval_grp = f["eigenvalues"]
        for point in coeff_grp.keys():
            U = np.array(coeff_grp[point])
            energies = np.array(eigval_grp[point])
            mo_data[point] = (U, energies)
    logger.info(f"Loaded MO data from {hdf5_file} for {len(mo_data)} frames.")
    return mo_data

##############################
# KS Matrix Reading Section  #
##############################

def read_ks_matrix_csr(file_path, size):
    """Reads the KS matrix from KS.txt in CSR format."""
    row_indices, col_indices, values = [], [], []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) == 3:
                row, col, value = int(parts[0]) - 1, int(parts[1]) - 1, float(parts[2])
                row_indices.append(row)
                col_indices.append(col)
                values.append(value)
                if row != col:  # Ensure symmetry
                    row_indices.append(col)
                    col_indices.append(row)
                    values.append(value)
    indices = torch.tensor([row_indices, col_indices], dtype=torch.long)
    values = torch.tensor(values, dtype=torch.float64)
    sparse_matrix = torch.sparse_coo_tensor(indices, values, (size, size))
    logger.info(f"Read KS matrix from {file_path} with shape ({size}, {size}).")
    return sparse_matrix

def evaluate_sparsity(sparse_matrix):
    """Evaluates sparsity of the KS matrix."""
    total_elements = sparse_matrix.shape[0] * sparse_matrix.shape[1]
    non_zero_elements = sparse_matrix._nnz()
    sparsity = 1 - (non_zero_elements / total_elements)
    logger.info(f"Matrix Sparsity: {sparsity:.6f}")
    return sparsity

#########################################
# Effective KS Matrix and Diagonalization
#########################################

def compute_effective_ks_matrix(H, U):
    """Computes the effective KS matrix Heff = U^T H U."""
    device = H.device
    U_torch = torch.tensor(U, dtype=torch.float64, device=device)
    H_dense = H.to_dense() if H.is_sparse else H
    H_eff = U_torch.T @ (H_dense @ U_torch)
    logger.info("Computed effective KS matrix (Heff).")
    return H_eff

def diagonalize_heff(H_eff):
    """Diagonalizes Heff and returns eigenvalues and eigenvectors."""
    H_eff_dense = H_eff.to_dense() if H_eff.is_sparse else H_eff
    eigvals, eigvecs = torch.linalg.eigh(H_eff_dense)
    logger.info("Diagonalized effective KS matrix (Heff).")
    return eigvals, eigvecs

#############################################
# Folder Processing and Frame Matching     #
#############################################

def match_frame_to_folder(cp2k_folder):
    """
    Checks if KS.txt exists in the cp2k_folder and extracts the point label from a .wfn file.
    Returns the point label (e.g., 'point_123') if found, otherwise None.
    """
    ks_path = os.path.join(cp2k_folder, "KS.txt")
    if not os.path.exists(ks_path):
        return None
    for fname in os.listdir(cp2k_folder):
        if fname.endswith(".wfn"):
            m = re.match(r"(point_\d+)-", fname)
            if m:
                return m.group(1)
    return None

#############################################
# Main Processing: Process Each Scratch Chunk
#############################################

def process_all_frames(matrix_size, device=torch.device("cpu")):
    """
    Processes frames by iterating over each scratch_chunk_* folder.
    In each scratch_chunk folder, it reads the HDF5 file (containing the MO data)
    and then processes the cp2k_job folders within that chunk.
    """
    results = {}

    # Get all scratch_chunk directories in the current directory.
    scratch_chunks = [
        d for d in os.listdir(".")
        if d.startswith("scratch_chunk_") and os.path.isdir(d)
    ]
    def extract_chunk_number(chunk):
        match = re.search(r"scratch_chunk_(\d+)", chunk)
        return int(match.group(1)) if match else -1
    scratch_chunks = sorted(scratch_chunks, key=extract_chunk_number)
    logger.info(f"Found {len(scratch_chunks)} scratch_chunk directories.")

    for chunk in scratch_chunks:
        chunk_path = os.path.join(".", chunk)
        # Find the HDF5 file in the scratch_chunk folder.
        hdf5_files = [f for f in os.listdir(chunk_path) if f.endswith(".hdf5")]
        if not hdf5_files:
            logger.warning(f"No HDF5 file found in {chunk_path}. Skipping this chunk.")
            continue
        hdf5_file_path = os.path.join(chunk_path, hdf5_files[0])
        mo_data = read_hdf5_mo_data(hdf5_file_path)

        # Get all cp2k_job folders in this scratch_chunk folder.
        cp2k_folders = [
            os.path.join(chunk_path, d)
            for d in os.listdir(chunk_path)
            if d.startswith("cp2k_job") and os.path.isdir(os.path.join(chunk_path, d))
        ]
        def extract_job_number(folder_path):
            folder_name = os.path.basename(folder_path)
            match = re.search(r"cp2k_job\.(\d+)", folder_name)
            return int(match.group(1)) if match else -1
        cp2k_folders = sorted(cp2k_folders, key=extract_job_number)
        logger.info(f"In {chunk_path}, found {len(cp2k_folders)} cp2k_job folders.")

        for folder in cp2k_folders:
            point_label = match_frame_to_folder(folder)
            if point_label is None:
                logger.warning(f"Folder {folder} does not have KS.txt or a .wfn file. Skipping...")
                continue
            if point_label not in mo_data:
                logger.warning(f"Frame {point_label} not found in HDF5 file {hdf5_file_path}. Skipping folder {folder}...")
                continue

            ks_file = os.path.join(folder, "KS.txt")
            logger.info(f"Processing folder {folder} for frame {point_label}...")
            H = read_ks_matrix_csr(ks_file, matrix_size).to(device)
            evaluate_sparsity(H)

            U, mo_energies = mo_data[point_label]
            H_eff = compute_effective_ks_matrix(H, U)
            eigvals, eigvecs = diagonalize_heff(H_eff)
            eigvals_eV = eigvals.cpu().numpy() * hartree_to_eV
            mo_energies_eV = mo_energies * hartree_to_eV

            # In case of duplicate point labels, the last one will override.
            results[point_label] = {
                "H_eff": H_eff.to_dense().cpu().numpy(),
                "eigvals": eigvals_eV,
                "eigvecs": eigvecs.cpu().numpy(),
                "mo_energies": mo_energies_eV
            }
            logger.info(f"Frame {point_label} processed. Heff shape: {H_eff.shape}, Eigenvalues count: {len(eigvals_eV)}")

    # Save everything into a single .npz file.
    np.savez("Heff_data.npz",
             H_eff=np.array([results[k]["H_eff"] for k in results]),
             eigvals=np.array([results[k]["eigvals"] for k in results]),
             eigvecs=np.array([results[k]["eigvecs"] for k in results]),
             mo_energies=np.array([results[k]["mo_energies"] for k in results]),
             frame_labels=np.array(list(results.keys())))
    logger.info("All data saved to Heff_data.npz")

    return results

#######################
# Main Script Section #
#######################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process KS matrices from scratch_chunk folders and store results in a single file.")
    parser.add_argument("--matrix_size", type=int, default=647, help="Size of the KS matrix.")
    args = parser.parse_args()

    device = torch.device("cpu")  # Change to "cuda" if GPU is available
    logger.info("Starting processing...")
    results = process_all_frames(args.matrix_size, device=device)
    logger.info("Processing completed.")


