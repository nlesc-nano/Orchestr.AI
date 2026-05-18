"""
parsing.py

This module provides functions to parse custom extended XYZ files and to save
stacked XYZ files in a format compatible with the rest of the pipeline.
"""

import numpy as np
import re

def parse_extxyz(file_path, label="dataset"):
    """
    Extracts energies, forces, and positions from a custom plain or ASE extended XYZ file.
    Format expected:
    Line 1: n_atoms
    Line 2: energy (float) OR ASE properties string containing energy=...
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
    
    # Pre-compile the regex for maximum speed during the loop
    energy_pattern = re.compile(r'energy=([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)', re.IGNORECASE)
    
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
            
        n_atoms = int(line)
        energy_line = lines[i+1].strip()
        
        # 1. Parse Energy (robustly handle plain float or 'energy=...' ASE formats)
        try:
            # First try the fastest route: is the whole line just a float?
            energy = float(energy_line)
        except ValueError:
            # If it fails, use the fast regex to hunt down "energy=X" in the ASE string
            match = energy_pattern.search(energy_line)
            if match:
                energy = float(match.group(1))
            else:
                energy = np.nan
        
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
    frame_positions = [np.asarray(p, dtype=float) for p in positions]
    frame_forces = [np.asarray(f, dtype=float) for f in forces]
    energies = np.asarray(energies, dtype=float)

    if len(frame_positions) != len(frame_forces) or len(frame_positions) != len(energies):
        raise ValueError("positions, forces, and energies must contain one entry per frame")

    if len(atom_types) > 0 and isinstance(atom_types[0], str):
        frame_atom_types = [atom_types for _ in frame_positions]
    else:
        frame_atom_types = atom_types

    num_frames = len(frame_positions)
    print(f"Saving custom XYZ file to {filename} ({num_frames} frames)...")
    
    try:
        with open(filename, "w") as f:
            for idx in range(num_frames):
                num_atoms = frame_positions[idx].shape[0]
                if frame_forces[idx].shape != frame_positions[idx].shape:
                    raise ValueError(
                        f"Frame {idx} force shape {frame_forces[idx].shape} "
                        f"does not match positions {frame_positions[idx].shape}"
                    )
                if len(frame_atom_types[idx]) != num_atoms:
                    raise ValueError(
                        f"Frame {idx} has {num_atoms} atoms but "
                        f"{len(frame_atom_types[idx])} atom labels"
                    )
                f.write(f"{num_atoms}\n")
                f.write(f" {energies[idx]:.8f}\n")
                for a_idx in range(num_atoms):
                    atom = frame_atom_types[idx][a_idx]
                    x, y, z = frame_positions[idx][a_idx]
                    fx, fy, fz = frame_forces[idx][a_idx]
                    f.write(f"{atom:<3s} {x:15.8f} {y:15.8f} {z:15.8f} {fx:15.8f} {fy:15.8f} {fz:15.8f}\n")
        print(f"File saved: {filename}")
    except Exception as e:
        print(f"Error saving to {filename}: {e}")
