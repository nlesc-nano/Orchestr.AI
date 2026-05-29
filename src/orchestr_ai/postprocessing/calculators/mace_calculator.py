# src/orchestr_ai/postprocessing/calculators/mace_calculator.py

from __future__ import annotations

import numpy as np
import torch

from orchestr_ai.postprocessing.calculators.base import BaseCalculator


class MaceCalculator(BaseCalculator):
    """
    Wrapper for MACE models.
    """

    def __init__(self, model, device, cutoff=12.0, head=None):
        self.model = model
        self.device = device
        self.cutoff = cutoff
        self.z_table = None

        self.model.to(self.device)
        self.model.eval()

        self._prepare_z_table()
        self._detect_cuequivariance()

        # Handle MACE heads dynamically for multi-head models
        try:
            self.available_heads = self.model.heads
        except AttributeError:
            self.available_heads = ["Default"]

        if head is not None:
            self.head = head
        elif len(self.available_heads) == 1:
            self.head = self.available_heads[0]
        else:
            default_heads = [h for h in self.available_heads if h.lower() == "default"]
            if default_heads:
                self.head = default_heads[0]
            else:
                self.head = self.available_heads[0]

        print(f"MACE: Using head '{self.head}' out of available heads {self.available_heads}")


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
            atoms_config = config_from_atoms(
                atoms,
                head_name=self.head,
            )

            data = AtomicData.from_config(
                atoms_config,
                z_table=self.z_table,
                cutoff=self.cutoff,
                heads=self.available_heads,
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


class AutoScaledReconstructedMaceCalculator(MaceCalculator):
    """
    Wrapper for MACE models running under Auto-Scaled Reconstruction Mode.
    Reconstructs physical triplet state from singlet (base) and delta heads.
    """
    def __init__(self, model, device, cutoff=12.0, scale_metadata_path="mace_scale_metadata.json"):
        import json
        import os

        # Initialize base MaceCalculator
        super().__init__(model, device, cutoff, head=None)

        self.scale_metadata_path = scale_metadata_path

        # Load metadata values
        if not os.path.exists(scale_metadata_path):
            raise FileNotFoundError(
                f"Scaling metadata file not found at: {scale_metadata_path}. "
                "Ensure that the preprocessing scaling script has run and generated this file."
            )

        with open(scale_metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.k_E = float(meta["k_E"])
        self.k_F = float(meta["k_F"])
        self.base_head = meta.get("base_head", "singlet")
        self.delta_head = meta.get("delta_head", "delta")

        # Verify values to avoid division by zero or negative values
        if self.k_E <= 0.0 or self.k_F <= 0.0:
            raise ValueError(f"Scaling factors must be strictly positive. Got k_E={self.k_E}, k_F={self.k_F}")

        print(
            f"MACE Reconstruction Calculator Initialized: "
            f"k_E = {self.k_E:.12f}, k_F = {self.k_F:.12f}, "
            f"base_head = '{self.base_head}', delta_head = '{self.delta_head}'"
        )

    def prepare_batch(self, frames):
        # We need to build the inputs using MACE's native layout.
        # But we need one batch prepared for base_head, and another for delta_head.
        # We temporarily set self.head and call the superclass's prepare_batch.
        orig_head = self.head

        try:
            self.head = self.base_head
            batch_base = super().prepare_batch(frames)

            self.head = self.delta_head
            batch_delta = super().prepare_batch(frames)
        finally:
            self.head = orig_head

        return (batch_base, batch_delta)

    def forward(self, inputs, n_atoms_list):
        batch_base, batch_delta = inputs

        # Pass 1: Base head (singlet)
        with torch.set_grad_enabled(True):
            results_base = self.model(batch_base.to_dict())

        E_singlet_pred = (
            results_base["energy"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
            .flatten()
        )

        forces_singlet_cpu = results_base["forces"].detach().cpu()
        F_singlet_pred = [
            f.numpy().astype(np.float64)
            for f in torch.split(forces_singlet_cpu, n_atoms_list, dim=0)
        ]

        # Extract node features / latent representations from the base head (singlet)
        latent_frame_list = [np.array([], dtype=np.float64)] * len(n_atoms_list)
        latent_atom_list = [None] * len(n_atoms_list)

        if "node_feats" in results_base and results_base["node_feats"] is not None:
            latents_cpu = results_base["node_feats"].detach().cpu()
            latent_atom_list = [
                l.numpy()
                for l in torch.split(latents_cpu, n_atoms_list, dim=0)
            ]
            latent_frame_list = [
                np.sum(l, axis=0).astype(np.float64)
                for l in latent_atom_list
            ]

        # Pass 2: Delta head (delta)
        with torch.set_grad_enabled(True):
            results_delta = self.model(batch_delta.to_dict())

        Delta_E_delta_pred_scaled = (
            results_delta["energy"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
            .flatten()
        )

        forces_delta_cpu = results_delta["forces"].detach().cpu()
        F_delta_pred_scaled = [
            f.numpy().astype(np.float64)
            for f in torch.split(forces_delta_cpu, n_atoms_list, dim=0)
        ]

        # Recovery Math:
        # E_triplet_reconstructed = E_singlet_pred - (Delta_E_delta_pred_scaled / k_E)
        # F_triplet_reconstructed = F_singlet_pred - (F_delta_pred_scaled / k_F)
        energies_np = E_singlet_pred - (Delta_E_delta_pred_scaled / self.k_E)

        forces_list = []
        for f_s, f_d_scaled in zip(F_singlet_pred, F_delta_pred_scaled):
            f_triplet_rec = f_s - (f_d_scaled / self.k_F)
            forces_list.append(f_triplet_rec)

        # Store for InferenceRunner retrieval
        self.last_E_singlet = E_singlet_pred
        self.last_F_singlet = F_singlet_pred
        self.last_Delta_E_scaled = Delta_E_delta_pred_scaled
        self.last_F_delta_scaled = F_delta_pred_scaled

        return energies_np, forces_list, latent_frame_list, latent_atom_list


try:
    from mace.calculators import MACECalculator
    from ase.calculators.calculator import all_changes
except ImportError:
    MACECalculator = object
    all_changes = None


class ReconstructedMACECalculator(MACECalculator):
    """
    ASE-compatible calculator for reconstructed triplet state.
    Used for MD, GEO_OPT, and VIB simulation run types.
    """
    def __init__(self, model, device, cutoff=12.0, scale_metadata_path="mace_scale_metadata.json", **kwargs):
        import json
        import os

        if MACECalculator is object:
            raise ImportError("MACE is not installed in the current environment.")

        if not os.path.exists(scale_metadata_path):
            raise FileNotFoundError(
                f"Scaling metadata file not found at: {scale_metadata_path}. "
                "Ensure that the preprocessing scaling script has run and generated this file."
            )

        with open(scale_metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.k_E = float(meta["k_E"])
        self.k_F = float(meta["k_F"])
        self.base_head = meta.get("base_head", "singlet")
        self.delta_head = meta.get("delta_head", "delta")

        # Pass base_head as the default head to MACE's constructor
        kwargs["head"] = self.base_head

        # Initialize base MACECalculator using 'models' parameter
        super().__init__(models=[model], device=str(device), default_dtype="float32", **kwargs)

        # Set default head to base_head initially
        self.head = self.base_head

        print(
            f"ASE Reconstructed MACECalculator Initialized: "
            f"k_E = {self.k_E:.12f}, k_F = {self.k_F:.12f}, "
            f"base_head = '{self.base_head}', delta_head = '{self.delta_head}'"
        )

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties

        # 1. Run base head (singlet) calculation
        self.head = self.base_head
        super().calculate(atoms, properties, system_changes)
        E_singlet = self.results["energy"]
        F_singlet = self.results["forces"]

        # 2. Run delta head calculation
        # Note: system_changes is reset to ensure MACE recomputes for the new head
        self.head = self.delta_head
        super().calculate(atoms, properties, all_changes)
        Delta_E_scaled = self.results["energy"]
        F_delta_scaled = self.results["forces"]

        # 3. Perform reconstruction
        E_triplet = E_singlet - (Delta_E_scaled / self.k_E)
        F_triplet = F_singlet - (F_delta_scaled / self.k_F)

        # 4. Save results back
        self.results["energy"] = E_triplet
        self.results["free_energy"] = E_triplet
        self.results["forces"] = F_triplet  