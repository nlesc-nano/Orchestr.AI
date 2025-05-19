"""
parsing.py

This module provides functions to parse extended XYZ (extXYZ) files and to save
stacked XYZ files in a format compatible with SchNetPack. It includes functionality
to extract energies, forces, and positions from extXYZ files, and to write out data
in a well-structured extended XYZ format.
"""

import os
import numpy as np


def parse_extxyz(file_path, label="dataset"):
    """
    Parses energies, forces, and positions from an extended XYZ file.

    The file is expected to have the following structure for each frame:
      - The number of atoms on the first line.
      - A comment line on the second line, containing the energy information.
      - Atom lines (one per atom) containing the atom type followed by at least three
        numbers for position. Optionally, force data may be present (three additional values).

    Robustness checks are implemented to handle truncated files, malformed lines,
    and missing energy information.

    Parameters:
        file_path (str): Path to the extended XYZ file.
        label (str): A label used when printing summary information (default: "dataset").

    Returns:
        tuple:
          - energies (list of float): Parsed energies for each frame.
          - forces (list of np.ndarray): Array (per frame) of forces with shape (n_atoms, 3).
          - positions (list of np.ndarray): Array (per frame) of positions with shape (n_atoms, 3).
    """
    energies = []
    forces = []
    positions = []
    current_frame = 0

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            try:
                # Read number of atoms from the header
                n_atoms = int(lines[i].strip())
                if n_atoms <= 0:
                    print(f"Warning: Frame {current_frame} in {file_path} reports {n_atoms} atoms. Skipping frame.")
                    i += 1
                    while i < len(lines) and not lines[i].strip().isdigit():
                        i += 1  # Skip to next potential header
                    continue

                # Check that there are enough lines to cover header + atoms
                if i + 1 + n_atoms > len(lines):
                    print(f"Warning: Frame {current_frame} in {file_path} seems truncated (line {i}). Stopping parse.")
                    break

                comment_line = lines[i + 1].strip()
                energy = np.nan  # Default energy value
                
                # Attempt to extract energy from the comment line
                try:
                    lower_comment = comment_line.lower()
                    if "energy=" in lower_comment:
                        for part in comment_line.split():
                            if part.lower().startswith("energy="):
                                energy = float(part.split('=')[1])
                                break
                    else:
                        # Fallback: Use the last word in the comment if it can be converted to float
                        energy_str = comment_line.split()[-1]
                        try:
                            energy = float(energy_str)
                        except ValueError:
                            pass  # Keep energy as NaN
                except Exception:
                    pass  # On any error, keep energy as NaN

                if np.isnan(energy):
                    print(f"Warning: Could not parse energy for frame {current_frame} from '{comment_line}'.")
                energies.append(energy)

                # Parse atom lines for positions and forces
                frame_positions = []
                frame_forces = []
                malformed_lines = 0
                for j in range(i + 2, i + 2 + n_atoms):
                    parts = lines[j].split()
                    if len(parts) >= 4:
                        # Expecting at least the atom symbol and three positional values
                        frame_positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                        if len(parts) >= 7:
                            frame_forces.append([float(parts[4]), float(parts[5]), float(parts[6])])
                        else:
                            frame_forces.append([np.nan, np.nan, np.nan])
                    else:
                        malformed_lines += 1
                        frame_positions.append([np.nan, np.nan, np.nan])
                        frame_forces.append([np.nan, np.nan, np.nan])
                
                if malformed_lines > 0:
                    print(f"Warning: Frame {current_frame} had {malformed_lines} malformed atom lines.")
                
                forces.append(np.array(frame_forces, dtype=np.float64))
                positions.append(np.array(frame_positions, dtype=np.float64))
                
                i += n_atoms + 2  # Move to the next frame (header + comment + atoms)
                current_frame += 1
                
            except ValueError:
                print(f"Warning: ValueError parsing frame header/atom line near line {i}. Attempting to find next frame.")
                i += 1
                while i < len(lines) and not lines[i].strip().isdigit():
                    i += 1  # Move to next potential header
                continue
            
            except IndexError:
                print(f"Warning: IndexError parsing frame near line {i} (likely truncated). Stopping parse.")
                break

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return [], [], []
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return [], [], []

    print(f"Manually parsed {len(energies)} frames from {file_path} as {label}")
    return energies, forces, positions


def save_stacked_xyz(filename, energies, positions, forces, atom_types):
    """
    Saves data to a SchNetPack-compatible extended XYZ file.

    The function checks that the number of frames and atom counts are consistent,
    then writes out the data in the following format for each frame:
      - Number of atoms.
      - A comment line containing the energy (in key=value format).
      - One line per atom: atom label, x, y, z coordinates followed by fx, fy, fz forces.

    Parameters:
        filename (str): Output file path.
        energies (list of float): Energies for each frame.
        positions (np.ndarray): Array of positions with shape (n_frames, n_atoms, 3).
        forces (np.ndarray): Array of forces with shape (n_frames, n_atoms, 3).
        atom_types (list of str): List of atom symbols (length must equal number of atoms per frame).

    Raises:
        ValueError: If there is a mismatch in frame count or atom count.
    """
    if not (len(energies) == positions.shape[0] == forces.shape[0]):
        raise ValueError("Frame count mismatch.")
    if not (positions.shape[1] == forces.shape[1] == len(atom_types)):
        raise ValueError("Atom count mismatch.")
    if positions.ndim != 3 or forces.ndim != 3 or positions.shape[2] != 3 or forces.shape[2] != 3:
        raise ValueError("Invalid shapes for positions/forces.")

    num_frames, num_atoms, _ = positions.shape
    print(f"Saving SchNetPack-compatible XYZ file to {filename} ({num_frames} frames, {num_atoms} atoms)...")
    try:
        with open(filename, "w") as f:
            for frame_idx in range(num_frames):
                f.write(f"{num_atoms}\n")
                f.write(f" {energies[frame_idx]:.8f}\n")
                for i in range(num_atoms):
                    atom = atom_types[i]
                    x, y, z = positions[frame_idx, i]
                    fx, fy, fz = forces[frame_idx, i]
                    f.write(f"{atom:<3s} {x:15.8f} {y:15.8f} {z:15.8f} {fx:15.8f} {fy:15.8f} {fz:15.8f}\n")
        print(f"File saved: {filename}")
    except IOError as e:
        print(f"Error writing file {filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during saving: {e}")



