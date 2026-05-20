# src/orchestr_ai/postprocessing/calculators/mace_calculator.py

from __future__ import annotations

import numpy as np
import torch

from orchestr_ai.postprocessing.calculators.base import BaseCalculator


class MaceCalculator(BaseCalculator):
    """
    Wrapper for MACE models.
    """

    def __init__(self, model, device, cutoff=12.0):
        self.model = model
        self.device = device
        self.cutoff = cutoff
        self.z_table = None

        self.model.to(self.device)
        self.model.eval()

        self._prepare_z_table()
        self._detect_cuequivariance()

    def _prepare_z_table(self):
        try:
            from mace.tools import utils
        except ImportError as e:
            raise ImportError(
                "MACE is required for MACE postprocessing, but it is not installed."
            ) from e

        raw_z = None

        if hasattr(self.model, "z_table"):
            raw_z = self.model.z_table
        elif hasattr(self.model, "atomic_numbers"):
            raw_z = self.model.atomic_numbers

        if raw_z is None:
            return

        if isinstance(raw_z, torch.Tensor):
            z_list = raw_z.detach().cpu().numpy().astype(int).tolist()
            self.z_table = utils.get_atomic_number_table_from_zs(z_list)
            self.model.z_table = self.z_table
            print(f"MACE: Converted model tensor to AtomicNumberTable: {z_list}")
        else:
            self.z_table = raw_z

    def _detect_cuequivariance(self):
        try:
            import cuequivariance_torch  # noqa: F401

            self.use_cueq = True
            print("MACE: cuEquivariance detected and enabled.")

            if hasattr(self.model, "enable_cueq"):
                self.model.enable_cueq = True

        except ImportError:
            self.use_cueq = False
            print("MACE: cuEquivariance not found. Using standard PyTorch ops.")

    def prepare_batch(self, frames):
        try:
            from mace.data.utils import config_from_atoms
            from mace.data.atomic_data import AtomicData
            from mace.tools.torch_geometric.dataloader import Collater
            from mace.tools import utils
        except ImportError as e:
            raise ImportError(
                "MACE is required for MACE postprocessing, but it is not installed."
            ) from e

        if self.z_table is None:
            print("Warning: z_table not found in model. Inferring from batch frames.")
            z_all = []
            for atoms in frames:
                z_all.extend(atoms.get_atomic_numbers())
            self.z_table = utils.get_atomic_number_table_from_zs(z_all)

        data_list = []

        for atoms in frames:
            atoms_config = config_from_atoms(atoms)

            data = AtomicData.from_config(
                atoms_config,
                z_table=self.z_table,
                cutoff=self.cutoff,
            )

            data_list.append(data)

        collater = Collater(follow_batch=[], exclude_keys=[])
        batch = collater(data_list).to(self.device)

        return batch

    def forward(self, inputs, n_atoms_list):
        with torch.set_grad_enabled(True):
            results = self.model(inputs.to_dict())

        energies_np = (
            results["energy"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
            .flatten()
        )

        forces_cpu = results["forces"].detach().cpu()
        forces_list = [
            f.numpy().astype(np.float64)
            for f in torch.split(forces_cpu, n_atoms_list, dim=0)
        ]

        latent_frame_list = [np.array([], dtype=np.float64)] * len(n_atoms_list)
        latent_atom_list = [None] * len(n_atoms_list)

        if "node_feats" in results and results["node_feats"] is not None:
            latents_cpu = results["node_feats"].detach().cpu()

            latent_atom_list = [
                l.numpy()
                for l in torch.split(latents_cpu, n_atoms_list, dim=0)
            ]

            latent_frame_list = [
                np.sum(l, axis=0).astype(np.float64)
                for l in latent_atom_list
            ]

        return energies_np, forces_list, latent_frame_list, latent_atom_list