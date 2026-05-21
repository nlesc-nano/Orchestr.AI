"""
parsing.py

This module provides functions to parse custom extended XYZ files and to save
stacked XYZ files in a format compatible with the rest of the pipeline.
"""

import numpy as np
import re

def parse_properties_string(prop_str):
    """
    Parses ASE Properties string, e.g. species:S:1:pos:R:3:f_singlet:R:3:f_triplet:R:3
    Returns a dict mapping property names to their corresponding 0-indexed column list.
    """
    parts = prop_str.split(':')
    mapping = {}
    current_col = 0
    i = 0
    while i < len(parts) - 2:
        name = parts[i]
        try:
            count = int(parts[i+2])
        except ValueError:
            break
        mapping[name] = list(range(current_col, current_col + count))
        current_col += count
        i += 3
    return mapping


def parse_extxyz(file_path, label="dataset", energy_key=None, forces_key=None):
    """
    Extracts energies, forces, and positions from a custom plain or ASE extended XYZ file.
    Format expected:
    Line 1: n_atoms
    Line 2: energy (float) OR ASE properties string containing energy=... (or custom keys)
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
        
        # 1. Parse Energy
        energy = np.nan
        try:
            # First try the fastest route: is the whole line just a float?
            energy = float(energy_line)
        except ValueError:
            # Look for specific energy key if requested (matching key=value or key: value)
            if energy_key:
                key_pattern = re.compile(rf'{energy_key}\s*[=:]\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)', re.IGNORECASE)
                match = key_pattern.search(energy_line)
                if match:
                    energy = float(match.group(1))
            
            # Fallback to standard energy= if still nan
            if np.isnan(energy):
                match = energy_pattern.search(energy_line)
                if match:
                    energy = float(match.group(1))
        
        energies.append(energy)
        
        # 2. Parse Properties metadata if present
        prop_match = re.search(r'Properties=([^\s]+)', energy_line)
        prop_map = None
        if prop_match:
            prop_map = parse_properties_string(prop_match.group(1))
        
        # 3. Parse Positions and Forces using fast list comprehensions
        block = [l.split() for l in lines[i+2 : i+2+n_atoms]]
        
        pos = np.array([[float(p[1]), float(p[2]), float(p[3])] for p in block], dtype=np.float64)
        
        # Determine force columns dynamically
        f_cols = [4, 5, 6]  # default fallback
        if len(block[0]) >= 10:
            # If line is long and contains triplet/singlet columns, detect which one
            is_triplet = False
            if forces_key and "triplet" in forces_key.lower():
                is_triplet = True
            elif energy_key and "triplet" in energy_key.lower():
                is_triplet = True
                
            if is_triplet:
                f_cols = [7, 8, 9]
                
        if prop_map and forces_key and forces_key in prop_map:
            f_cols = prop_map[forces_key]
            
        if len(block[0]) >= max(f_cols) + 1:
            frc = np.array([[float(p[f_cols[0]]), float(p[f_cols[1]]), float(p[f_cols[2]])] for p in block], dtype=np.float64)
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

