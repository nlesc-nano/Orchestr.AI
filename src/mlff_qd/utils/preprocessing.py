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
