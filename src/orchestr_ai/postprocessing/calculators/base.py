# src/orchestr_ai/postprocessing/calculators/base.py

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseCalculator(ABC):
    """
    Abstract interface for all ML framework calculators.

    Every engine-specific calculator must implement:
      1. prepare_batch()
      2. forward()
    """

    @abstractmethod
    def prepare_batch(self, frames):
        """
        Convert a list of ASE Atoms into the model-specific input format.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs, n_atoms_list):
        """
        Run inference and return:
          energies,
          forces_list,
          latent_frame_list,
          latent_atom_list
        """
        raise NotImplementedError