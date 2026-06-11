import os
import logging
import numpy as np
from scipy.spatial import distance_matrix
from orchestr_ai.utils.centering import estimate_padding, process_xyz
from orchestr_ai.utils.io import parse_stacked_xyz, parse_dual_spin_xyz

logger = logging.getLogger(__name__)

def load_frames(f_path, spin_state):
    """Loads XYZ file using parse_dual_spin_xyz or parse_stacked_xyz from io.py."""
    if not os.path.exists(f_path):
        logger.warning(f"File not found: {f_path}")
        return []
        
    logger.info(f"Reading: {f_path}")
    if spin_state == "dual":
        E_s, E_t, dE, P, F_s, F_t, atoms = parse_dual_spin_xyz(f_path)
        frames = [{
            'num_atoms': len(atoms),
            'atoms': atoms,
            'pos': P[i],
            'f_singlet': F_s[i],
            'f_triplet': F_t[i],
            'E_singlet': E_s[i],
            'E_triplet': E_t[i],
            'Delta_E': dE[i]
        } for i in range(len(E_s))]
    else:
        energies, positions, forces, atom_types = parse_stacked_xyz(f_path)
        frames = [{
            'num_atoms': len(atom_types),
            'atoms': atom_types,
            'pos': positions[i],
            'forces': forces[i],
            'energy': energies[i]
        } for i in range(len(energies))]
    logger.info(f"Loaded {len(frames)} frames from {f_path}")
    return frames

def resolve_file_list(raw_files):
    """Converts a comma-separated string or a YAML list of file paths to a clean list."""
    if isinstance(raw_files, str):
        return [f.strip() for f in raw_files.split(",") if f.strip()]
    if isinstance(raw_files, list):
        return [f.strip() for f in raw_files if f and f.strip()]
    return []

def pool_and_sample_frames(pool_files, num_picked, spin_state):
    """Pools frames from files and samples equally-spaced configurations if requested."""
    pooled = []
    for f in pool_files:
        pooled.extend(load_frames(f, spin_state))
        
    total_pooled = len(pooled)
    if pool_files:
        logger.info(f"Total source frames pooled for subsetting: {total_pooled}")
        
    if num_picked and 0 < num_picked < total_pooled:
        indices = np.round(np.linspace(0, total_pooled - 1, num_picked)).astype(int)
        sampled = [pooled[idx] for idx in indices]
        logger.info(f"Selected {len(sampled)} equally spaced frames.")
        return sampled
    return pooled

def compute_global_box(frames):
    """Finds the largest system and computes its global cubic box length with neighbor padding."""
    max_atoms = max(f['num_atoms'] for f in frames)
    largest_frames = [f for f in frames if f['num_atoms'] == max_atoms]
    
    L_max = 0.0
    for f in largest_frames:
        pos = np.asarray(f['pos'])
        lengths = np.max(pos, axis=0) - np.min(pos, axis=0)
        
        # Calculate padding: 2 * average nearest neighbor distance (constrained between 3.0 and 10.0 A)
        dmat = distance_matrix(pos, pos)
        dmat[dmat == 0] = np.inf
        padding = np.clip(2.0 * np.mean(np.min(dmat, axis=1)), 3.0, 10.0)
        
        max_len = np.max(lengths + padding)
        if max_len > L_max:
            L_max = max_len
    return float(L_max)

def run_merge_and_center(merge_and_center_cfg, spin_state="single"):
    """Unified entry point to merge and center datasets."""
    output_file = merge_and_center_cfg.get("output_file")
    if not output_file:
        raise ValueError("merge_and_center configuration must specify output_file.")

    pool_files = resolve_file_list(merge_and_center_cfg.get("pool_files", []))
    merge_files = resolve_file_list(merge_and_center_cfg.get("merge_files", []))
    num_picked = merge_and_center_cfg.get("num_picked", 0)

    # 1. Load, pool, and sample picked files
    picked_frames = pool_and_sample_frames(pool_files, num_picked, spin_state)
    
    # 2. Load merge files in full
    full_frames = []
    for f in merge_files:
        full_frames.extend(load_frames(f, spin_state))
    if merge_files:
        logger.info(f"Loaded {len(full_frames)} frames to merge in full.")

    # 3. Merge
    merged_frames = picked_frames + full_frames
    if not merged_frames:
        raise ValueError("No frames found to process.")

    # 4. Compute global box size from the largest configurations
    max_atoms = max(f['num_atoms'] for f in merged_frames)
    largest_frames = [f for f in merged_frames if f['num_atoms'] == max_atoms]
    logger.info(f"Detected largest system with {max_atoms} atoms (found in {len(largest_frames)} frames)")
    logger.info("Computing global maximum box size based on the largest system...")
    L_max = compute_global_box(merged_frames)
    logger.info(f"Global maximum cubic box size (L_max_global): {L_max:.6f} A")
    logger.info(f"Merged total frames: {len(merged_frames)} (Picked: {len(picked_frames)}, Full: {len(full_frames)})")
    
    # 5. Save raw uncentered merged trajectory to a temp file
    logger.info(f"Centering all frames inside cubic box of size {L_max:.6f} A and writing to {output_file}...")
    temp_xyz = output_file + ".temp_raw"
    with open(temp_xyz, "w") as f:
        for frame in merged_frames:
            num_atoms = frame['num_atoms']
            atoms = frame['atoms']
            pos = frame['pos']
            f.write(f"{num_atoms}\n")
            
            if spin_state == "dual":
                header = f"E_singlet: {frame['E_singlet']:.12f} E_triplet: {frame['E_triplet']:.12f} Delta_E: {frame['Delta_E']:.12f}\n"
                f.write(header)
                fs = frame['f_singlet']
                ft = frame['f_triplet']
                for j in range(num_atoms):
                    f.write(f"{atoms[j]} {pos[j][0]:.6f} {pos[j][1]:.6f} {pos[j][2]:.6f} "
                            f"{fs[j][0]:.6f} {fs[j][1]:.6f} {fs[j][2]:.6f} "
                            f"{ft[j][0]:.6f} {ft[j][1]:.6f} {ft[j][2]:.6f}\n")
            else:
                header = f"{frame['energy']:.6f}\n"
                f.write(header)
                forces = frame['forces']
                for j in range(num_atoms):
                    f.write(f"{atoms[j]:<2} {pos[j][0]:12.6f} {pos[j][1]:12.6f} {pos[j][2]:12.6f} "
                            f"{forces[j][0]:12.6f} {forces[j][1]:12.6f} {forces[j][2]:12.6f}\n")
                            
    # 6. Delegate centering, PNG plotting, and NPZ generation to platform's process_xyz
    png_file = os.path.splitext(output_file)[0] + ".png"
    process_xyz(temp_xyz, output_file, png_file, spin_state=spin_state, L_max_global=L_max)
    
    # Cleanup temp raw file
    if os.path.exists(temp_xyz):
        os.remove(temp_xyz)
        
    logger.info(f"Successfully wrote {len(merged_frames)} centered, merged frames to {output_file}")
    return output_file
