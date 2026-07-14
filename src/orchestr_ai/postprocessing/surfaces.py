from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

import numpy as np

from orchestr_ai.postprocessing.simulation import get_ase_calculator


@dataclass
class TwoStateSurface:
    """Container for two adiabatic ML surfaces at one geometry."""

    energy_s0: float
    forces_s0: np.ndarray
    energy_s1: float
    forces_s1: np.ndarray

    @property
    def gap(self) -> float:
        return float(self.energy_s1 - self.energy_s0)


class MultiStateSurfaceEvaluator:
    """
    Evaluate S0 and S1 surfaces for NAMD.

    The initial implementation targets the MACE multihead setup used in this
    project: S0 from the singlet head and S1 either from a triplet head or from
    singlet + delta reconstruction.
    """

    def __init__(self, model_obj, device, config, neighbor_list=None):
        self.model_obj = model_obj
        self.device = device
        self.config = config
        self.neighbor_list = neighbor_list
        self.framework = config.get("model_framework", "schnetpack").lower()

        namd = config.get("namd", {})
        states = namd.get("states", {})
        self.s0_head = states.get("s0", "singlet")
        self.s1_head = states.get("s1", "triplet_reconstructed")

        if self.framework != "mace":
            raise ValueError(
                "NAMD two-state surface evaluation currently supports MACE "
                "multihead models. Add a framework-specific evaluator for "
                f"'{self.framework}' before using NAMD with this engine."
            )

        self.calc_s0 = self._make_calc(self.s0_head)
        self.calc_s1 = self._make_calc(self.s1_head)

    def _make_calc(self, head):
        cfg = deepcopy(self.config)
        cfg["mace_head"] = head
        return get_ase_calculator(
            self.model_obj,
            cfg,
            self.device,
            self.neighbor_list,
        )

    @staticmethod
    def _energy_from_results(calc, fallback):
        results = getattr(calc, "results", {}) or {}
        per_atom = results.get("energies")
        if per_atom is not None:
            arr = np.asarray(per_atom, dtype=np.float64)
            if arr.size and np.all(np.isfinite(arr)):
                return float(arr.sum(dtype=np.float64))
        return float(fallback)

    def evaluate(self, atoms) -> TwoStateSurface:
        atoms_s0 = atoms.copy()
        atoms_s0.calc = self.calc_s0
        e0_raw = atoms_s0.get_potential_energy()
        f0 = np.asarray(atoms_s0.get_forces(), dtype=np.float64)
        e0 = self._energy_from_results(self.calc_s0, e0_raw)

        atoms_s1 = atoms.copy()
        atoms_s1.calc = self.calc_s1
        e1_raw = atoms_s1.get_potential_energy()
        f1 = np.asarray(atoms_s1.get_forces(), dtype=np.float64)
        e1 = self._energy_from_results(self.calc_s1, e1_raw)

        return TwoStateSurface(
            energy_s0=e0,
            forces_s0=f0,
            energy_s1=e1,
            forces_s1=f1,
        )
