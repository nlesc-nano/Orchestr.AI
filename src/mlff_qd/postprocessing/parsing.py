"""
parsing.py

This module provides functions to parse custom extended XYZ files and to save
stacked XYZ files in a format compatible with the rest of the pipeline.
"""

import numpy as np

def parse_extxyz(file_path, label="dataset"):
    """
    Extracts energies, forces, and positions from a custom plain XYZ file.
    Format expected:
    Line 1: n_atoms
    Line 2: energy (float)
    Line 3+: Symbol  x  y  z  fx  fy  fz
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return [], [], []

    energies, forces, positions = [], [], []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
            
        n_atoms = int(line)
        energy_line = lines[i+1].strip()
        
        # 1. Parse Energy (robustly handle plain float or 'energy=...' formats)
        try:
            energy = float(energy_line)
        except ValueError:
            energy_str = [x for x in energy_line.split() if '=' not in x or 'energy' in x.lower()][-1].split('=')[-1]
            energy = float(energy_str) if energy_str else np.nan
        
        energies.append(energy)
        
        # 2. Parse Positions and Forces using fast list comprehensions
        block = [l.split() for l in lines[i+2 : i+2+n_atoms]]
        
        pos = np.array([[float(p[1]), float(p[2]), float(p[3])] for p in block], dtype=np.float64)
        
        if len(block[0]) >= 7:
            frc = np.array([[float(p[4]), float(p[5]), float(p[6])] for p in block], dtype=np.float64)
        else:
            frc = np.full((n_atoms, 3), np.nan, dtype=np.float64)
            
        positions.append(pos)
        forces.append(frc)
        
        i += n_atoms + 2

    print(f"Fast-parsed {len(energies)} frames from {file_path} as {label}")
    return energies, forces, positions


def save_stacked_xyz_schnetpack(filename, energies, positions, forces, atom_types):
    """
    Saves data back out to the same custom XYZ format.
    """
    num_frames, num_atoms, _ = positions.shape
    print(f"Saving custom XYZ file to {filename} ({num_frames} frames, {num_atoms} atoms)...")
    
    try:
        with open(filename, "w") as f:
            for idx in range(num_frames):
                f.write(f"{num_atoms}\n")
                f.write(f" {energies[idx]:.8f}\n")
                for a_idx in range(num_atoms):
                    atom = atom_types[a_idx]
                    x, y, z = positions[idx, a_idx]
                    fx, fy, fz = forces[idx, a_idx]
                    f.write(f"{atom:<3s} {x:15.8f} {y:15.8f} {z:15.8f} {fx:15.8f} {fy:15.8f} {fz:15.8f}\n")
        print(f"File saved: {filename}")
    except Exception as e:
        print(f"Error saving to {filename}: {e}")

