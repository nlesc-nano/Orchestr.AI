# src/orchestr_ai/postprocessing/calculators/schnetpack_calculator.py

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from orchestr_ai.postprocessing.calculators.base import BaseCalculator


try:
    from schnetpack.interfaces import AtomsConverter
    from schnetpack import properties as Properties
except ImportError as e:
    raise ImportError(
        "SchNetPack is required for SchNetPack-based postprocessing. "
        "Use the Orchestr.AI core environment for engines such as "
        "schnet, painn, so3net, field_schnet, and fusion."
    ) from e


class SchnetpackCalculator(BaseCalculator):
    """
    Wrapper for SchNetPack-based models:
      - SchNet
      - PaiNN
      - SO3net
      - FieldSchNet
      - Fusion, if it still uses SchNetPack-style inference
    """

    def __init__(self, model, device, neighbor_list_provider):
        self.model = model
        self.device = device
        self.neighbor_list_provider = neighbor_list_provider

        self.mean_offset = 0.0
        self.atomref = None
        self.latent = None

        print("--- SchnetpackCalculator: Attempting to surgically extract offsets ---")
        self._extract_and_disable_postprocessors()

        self.model.to(self.device, dtype=torch.float32)
        self.model.eval()

        rep = getattr(self.model, "representation", None)
        if rep is not None and not hasattr(rep, "electronic_embeddings"):
            rep.electronic_embeddings = nn.ModuleList([])

        self.atoms_converter = AtomsConverter(
            neighbor_list=self.neighbor_list_provider.get_ase_nl(),
            device=self.device,
            dtype=torch.float32,
        )

        self._register_latent_hook()

    def _extract_and_disable_postprocessors(self):
        try:
            if not hasattr(self.model, "postprocessors"):
                return

            for pp in self.model.postprocessors:
                add_mean_enabled = bool(getattr(pp, "add_mean", False))
                add_atomrefs_enabled = bool(getattr(pp, "add_atomrefs", True))
                extracted_mean = 0.0

                if hasattr(pp, "state_dict") and "mean" in pp.state_dict():
                    extracted_mean = pp.state_dict()["mean"].item()
                elif hasattr(pp, "mean") and isinstance(getattr(pp, "mean"), torch.Tensor):
                    extracted_mean = getattr(pp, "mean").item()

                if add_mean_enabled and abs(extracted_mean) > 1e-8:
                    self.mean_offset = extracted_mean
                    print(
                        f"\n⚠️  FLAG: Non-zero dataset mean offset detected: "
                        f"{self.mean_offset:.6f} eV/atom"
                    )
                elif abs(extracted_mean) > 1e-8:
                    self.mean_offset = 0.0
                    print(
                        "\nFLAG: Stored dataset mean detected, but "
                        "AddOffsets.add_mean is false."
                    )
                else:
                    self.mean_offset = 0.0
                    print("\n✅ FLAG: Mean offset is 0.0.")

                if add_atomrefs_enabled:
                    for ref_name in ["atomref", "z_offsets"]:
                        if hasattr(pp, ref_name) and getattr(pp, ref_name) is not None:
                            ref_val = getattr(pp, ref_name)

                            if isinstance(ref_val, torch.Tensor):
                                self.atomref = (
                                    ref_val.detach()
                                    .cpu()
                                    .numpy()
                                    .astype(np.float64)
                                    .flatten()
                                )
                            elif hasattr(ref_val, "weight"):
                                self.atomref = (
                                    ref_val.weight.detach()
                                    .cpu()
                                    .numpy()
                                    .astype(np.float64)
                                    .flatten()
                                )

                            print(
                                f"Successfully extracted '{ref_name}' "
                                "(isolated atomic energies)."
                            )

            self.model.postprocessors = torch.nn.ModuleList([])
            print("Successfully disabled model's internal postprocessors.")

        except Exception as e:
            print(f"WARNING: Surgical offset extraction failed: {e}")

    def _register_latent_hook(self):
        rep = getattr(self.model, "representation", None)
        if rep is None:
            return

        def hook(module, input, output):
            if isinstance(output, dict) and "scalar_representation" in output:
                self.latent = (
                    output["scalar_representation"]
                    .detach()
                    .to("cpu", dtype=torch.float32)
                )
            else:
                self.latent = None

        rep.register_forward_hook(hook)

    def prepare_batch(self, frames):
        self.latent = None

        inputs = self.atoms_converter(frames)

        pos_key = Properties.R if Properties.R in inputs else None
        if not pos_key:
            pos_key = next(
                (k for k in ["_positions", "positions"] if k in inputs),
                None,
            )

        if pos_key:
            inputs[pos_key].requires_grad_(True)

        return inputs

    def forward(self, inputs, n_atoms_list):
        with torch.set_grad_enabled(True):
            results = self.model(inputs)

        energies_np = (
            results["energy"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
            .flatten()
        )

        if self.atomref is not None:
            z_key = None

            if hasattr(Properties, "Z") and Properties.Z in inputs:
                z_key = Properties.Z
            elif "_atomic_numbers" in inputs:
                z_key = "_atomic_numbers"
            elif "atomic_numbers" in inputs:
                z_key = "atomic_numbers"

            if z_key:
                z_values = inputs[z_key].detach().cpu().numpy().astype(np.int64)
                valid_z = np.clip(z_values, 0, len(self.atomref) - 1)
                e0_values = self.atomref[valid_z]

                splits = np.cumsum(n_atoms_list)[:-1]
                e0_per_frame = [
                    np.sum(e0) for e0 in np.split(e0_values, splits)
                ]

                energies_np += np.array(e0_per_frame)
            else:
                print("WARNING: Could not find atomic numbers to apply atomref.")

        if self.mean_offset != 0.0:
            offsets = np.array(n_atoms_list, dtype=np.float64) * self.mean_offset
            energies_np += offsets

        forces_cpu = results["forces"].detach().cpu()
        forces_list = [
            f.numpy().astype(np.float64)
            for f in torch.split(forces_cpu, n_atoms_list, dim=0)
        ]

        latent_atom_list = [None] * len(n_atoms_list)
        latent_frame_list = [np.array([], dtype=np.float64)] * len(n_atoms_list)

        if self.latent is not None:
            latents_cpu = self.latent.detach().cpu()
            latent_atom_list = [
                l.numpy()
                for l in torch.split(latents_cpu, n_atoms_list, dim=0)
            ]
            latent_frame_list = [
                np.mean(l, axis=0).astype(np.float64)
                for l in latent_atom_list
            ]

        return energies_np, forces_list, latent_frame_list, latent_atom_list
