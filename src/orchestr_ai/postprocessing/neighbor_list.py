"""
neighbor_list.py

SchNetPack-specific neighbor-list utilities for Orchestr.AI postprocessing.

Important:
This module imports SchNetPack classes at top level, so it must only be imported
for SchNetPack-based engines such as schnet, painn, so3net, field_schnet, fusion.
"""

from __future__ import annotations

import os

import numpy as np
import torch

try:
    from schnetpack.transform import ASENeighborList
except ImportError as e:
    raise ImportError(
        "SchNetPack is required for SchNetPack-based postprocessing. "
        "Use the Orchestr.AI core environment for engines such as "
        "schnet, painn, so3net, field_schnet, and fusion."
    ) from e


class SmartNeighborList(ASENeighborList):
    """
    ASE-based neighbor list with displacement-based caching (Verlet list).
    
    This is used to accelerate Molecular Dynamics (MD) by only recomputing
    neighbor indices when atoms have moved beyond the 'skin' distance.
    """

    def __init__(self, cutoff: float, skin: float = 2.0):
        super().__init__(cutoff=cutoff)
        self.skin = skin
        self.last_positions = None
        self.last_cell = None

    def update(self, atoms):
        current_positions = atoms.get_positions()
        current_cell = atoms.get_cell()

        # Always update if cell changes or it's the first call
        if self.last_positions is None or not np.array_equal(current_cell, self.last_cell):
            self.last_positions = current_positions.copy()
            self.last_cell = current_cell.copy()
            return super().update(atoms)

        # Update only if an atom moved more than half the skin distance
        # (Standard Verlet list logic: skin/2 per atom)
        displacements = current_positions - self.last_positions
        max_displacement = np.max(np.linalg.norm(displacements, axis=1))

        if max_displacement > (self.skin / 2.0):
            self.last_positions = current_positions.copy()
            self.last_cell = current_cell.copy()
            return super().update(atoms)

        return False


class NeighborListProvider:
    """
    Provides SchNetPack-compatible neighbor lists.
    """

    def __init__(self, config: dict, existing_nl=None):
        self.run_type = config.get("run_type", "MD").upper()
        self.backend = config.get("nl_backend", "ase").lower()
        self.cutoff = config.get("cutoff", 12.0)
        self.skin = config.get("skin", 0.0)  # Default 0 means always recompute

        self.ase_nl = existing_nl

        if self.ase_nl is None:
            if self.backend == "ase" or self.backend == "legacy":
                # For EVAL, we strongly recommend skin=0 (always recompute) for trust.
                # For MD, a skin > 0 is used for performance.
                if self.skin > 0:
                    self.ase_nl = SmartNeighborList(cutoff=self.cutoff, skin=self.skin)
                    print(f"NeighborList: Using SmartNeighborList with skin={self.skin}Å (Performance mode).")
                else:
                    self.ase_nl = ASENeighborList(cutoff=self.cutoff)
                    print(f"NeighborList: Using standard ASENeighborList (Trust/EVAL mode).")

        if self.backend == "alchemy":
            print("NeighborList: 'alchemy' backend selected. Note: This requires matscipy.")

    def get_ase_nl(self):
        """
        Return the ASE-compatible neighbor list for SchNetPack.
        """
        if self.ase_nl is None:
            # Fallback if somehow not initialized
            return ASENeighborList(cutoff=self.cutoff)
        return self.ase_nl

    def compute_alchemy_edges(self, atoms):
        """
        Optional matscipy-based neighbor-list computation.

        This is not normally needed for the SchNetPack ASE calculator,
        but is kept for backward compatibility with existing code.
        """
        try:
            import matscipy.neighbors
        except ImportError as e:
            raise ImportError(
                "matscipy is required for the 'alchemy' backend. "
                "Install it with: pip install matscipy"
            ) from e

        i, j = matscipy.neighbors.neighbor_list("ij", atoms, self.cutoff)
        return torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long)


def setup_neighbor_list(config):
    """
    Legacy helper function used by postprocessing and evaluation.
    """
    provider = NeighborListProvider(config)
    return provider.get_ase_nl()