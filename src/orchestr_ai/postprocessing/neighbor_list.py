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
    from schnetpack.transform import ASENeighborList, CachedNeighborList
except ImportError as e:
    raise ImportError(
        "SchNetPack is required for SchNetPack-based postprocessing "
        "because the neighbor-list implementation depends on "
        "schnetpack.transform.ASENeighborList and CachedNeighborList. "
        "Use the Orchestr.AI core environment for engines such as "
        "schnet, painn, so3net, field_schnet, and fusion."
    ) from e


class SmartNeighborList(ASENeighborList):
    """
    Legacy ASE-based neighbor list with skin-based caching.

    This class extends SchNetPack's ASENeighborList, so it belongs in this
    SchNetPack-only module.
    """

    def __init__(self, cutoff, update_threshold, skin):
        super().__init__(cutoff=cutoff)
        self.update_threshold = update_threshold
        self.skin = skin
        self.last_positions = None
        self.last_cell = None

    def update(self, atoms):
        current_positions = atoms.get_positions()
        current_cell = atoms.get_cell()
        needs_update = False

        if self.last_positions is None or not np.array_equal(current_cell, self.last_cell):
            needs_update = True
        else:
            displacements = current_positions - self.last_positions
            max_displacement = np.max(np.linalg.norm(displacements, axis=1))
            total_displacement = np.sum(np.linalg.norm(displacements, axis=1))

            if max_displacement > self.skin or total_displacement > self.update_threshold:
                needs_update = True

        if needs_update:
            self.last_positions = current_positions.copy()
            self.last_cell = current_cell.copy()
            return super().update(atoms)

        return False


class NeighborListProvider:
    """
    Provides SchNetPack-compatible neighbor lists.

    This provider is intended only for SchNetPack-based engines.
    MACE, NequIP, and Allegro should not use this class.
    """

    def __init__(self, config, existing_nl=None):
        self.backend = config.get("nl_backend", "legacy").lower()
        self.cutoff = config.get("cutoff", 12.0)
        self.skin = config.get("skin", 2.0)
        self.update_threshold = config.get("update_threshold", 2.0)
        self.cache_path = config.get("cache_path", "neighbor_cache")

        self.ase_nl = existing_nl

        if self.backend == "legacy" and self.ase_nl is None:
            cache_dir = os.path.dirname(self.cache_path)
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)

            smart_nl = SmartNeighborList(
                cutoff=self.cutoff,
                update_threshold=self.update_threshold,
                skin=self.skin,
            )

            self.ase_nl = CachedNeighborList(
                neighbor_list=smart_nl,
                cache_path=self.cache_path,
            )

            print("NeighborList initialized with 'legacy' ASE backend.")

        elif self.backend == "alchemy":
            print("NeighborList initialized with 'alchemy' backend (matscipy).")

    def get_ase_nl(self):
        """
        Return the ASE-compatible neighbor list for SchNetPack.
        """
        if self.backend != "legacy":
            print(
                "Warning: Model requested legacy ASE neighbor list, "
                "but backend is 'alchemy'. Falling back to ASE."
            )

            smart_nl = SmartNeighborList(
                cutoff=self.cutoff,
                update_threshold=self.update_threshold,
                skin=self.skin,
            )

            return CachedNeighborList(
                neighbor_list=smart_nl,
                cache_path=self.cache_path,
            )

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