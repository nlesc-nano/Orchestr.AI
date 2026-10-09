# src/orchestr_ai/postprocessing/calculators/nequip_calculator.py

from __future__ import annotations

import numpy as np
import torch

from orchestr_ai.postprocessing.calculators.base import BaseCalculator


class NequipCalculator(BaseCalculator):
    """
    Wrapper for NequIP / Allegro models using the official ASE calculator.

    model_obj should be the compiled model path.
    """

    def __init__(self, model_obj, device):
        self.device = device
        self.model_path = model_obj
        self.calc = None

    def prepare_batch(self, frames):
        if self.calc is None:
            try:
                from nequip.ase import NequIPCalculator
            except ImportError as e:
                raise ImportError(
                    "NequIP is required for NequIP/Allegro postprocessing, "
                    "but it is not installed."
                ) from e
            try:  # TorchScript ops of models compiled with cuEquivariance (need a GPU)
                import cuequivariance_torch  # noqa: F401
                import cuequivariance_ops_torch  # noqa: F401
            except ImportError:
                pass

            species = sorted(list(set(frames[0].get_chemical_symbols())))
            chemical_map = {s: s for s in species}

            self.calc = NequIPCalculator.from_compiled_model(
                compile_path=self.model_path,
                chemical_species_to_atom_type_map=chemical_map,
                device=str(self.device),
            )

        return frames

    def forward(self, frames, n_atoms_list):
        """Energies, forces and invariant latents straight from the model output: the final node features
        (NequIP) or the final edge features summed on their central atom (Allegro)."""
        e_list = []
        f_list = []
        latent_frame_list = []
        latent_atom_list = []

        for atoms in frames:
            out = self.calc.call_model(self.calc.atoms_to_data(atoms))
            e_list.append(float(out["total_energy"].detach().cpu()) * self.calc.energy_units_to_eV)
            f_list.append(
                out["forces"].detach().cpu().numpy().astype(np.float64)
                * self.calc.energy_units_to_eV / self.calc.length_units_to_A
            )

            if "node_features" in out:
                latent = out["node_features"].detach()
            else:
                edge = out["edge_features"].detach()
                latent = torch.zeros(len(atoms), edge.shape[1], dtype=edge.dtype, device=edge.device)
                latent.index_add_(0, out["edge_index"][0], edge)
            atom_latents = latent.cpu().numpy().astype(np.float64)

            latent_frame_list.append(atom_latents.mean(axis=0))
            latent_atom_list.append(atom_latents)

        return np.array(e_list), f_list, latent_frame_list, latent_atom_list


class NequipCalculatorExtended(BaseCalculator):
    """
    Extended NequIP / Allegro wrapper for direct model-object inference.
    Keep this only if you still need direct non-ASE NequIP inference.
    """

    def __init__(self, model, device, cutoff=12.0):
        self.model = model
        self.device = device
        self.cutoff = cutoff
        self.type_mapper = None
        self.r_max = float(cutoff)

        self.model.to(self.device)
        self.model.eval()

        try:
            import cuequivariance_torch  # noqa: F401

            print(
                "cuEquivariance detected! "
                "NequIP/Allegro ops will be hardware accelerated."
            )
        except ImportError:
            pass

    def prepare_batch(self, frames):
        try:
            from nequip.data import from_ase, AtomicDataDict
            import ase.neighborlist
        except ImportError as e:
            raise ImportError(
                "NequIP is required for extended NequIP/Allegro inference."
            ) from e

        if self.type_mapper is None:
            type_names = getattr(self.model, "type_names", None)
            if type_names is None:
                type_names = sorted(list(set(frames[0].get_chemical_symbols())))

            try:
                from nequip.data import TypeMapper

                self.type_mapper = TypeMapper(type_names=type_names)
            except ImportError:
                self.type_mapper = type_names

        data_list = []

        for atoms in frames:
            if isinstance(self.type_mapper, list):
                data_dict = from_ase(atoms)
                symbols = atoms.get_chemical_symbols()
                atom_types = [self.type_mapper.index(s) for s in symbols]

                data_dict[AtomicDataDict.ATOM_TYPE_KEY] = torch.tensor(
                    atom_types,
                    dtype=torch.long,
                )
            else:
                try:
                    data_dict = from_ase(atoms, type_mapper=self.type_mapper)
                except TypeError:
                    data_dict = from_ase(atoms)
                    symbols = atoms.get_chemical_symbols()
                    atom_types = [
                        self.type_mapper.type_names.index(s)
                        for s in symbols
                    ]

                    data_dict[AtomicDataDict.ATOM_TYPE_KEY] = torch.tensor(
                        atom_types,
                        dtype=torch.long,
                    )

            i, j, s = ase.neighborlist.neighbor_list("ijS", atoms, self.r_max)

            data_dict[AtomicDataDict.EDGE_INDEX_KEY] = torch.stack(
                [
                    torch.tensor(i, dtype=torch.long),
                    torch.tensor(j, dtype=torch.long),
                ],
                dim=0,
            )

            data_dict[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = torch.tensor(
                s,
                dtype=torch.float32,
            )

            r_max_key = getattr(AtomicDataDict, "R_MAX_KEY", "r_max")
            data_dict[r_max_key] = torch.tensor([self.r_max], dtype=torch.float32)

            for key, value in data_dict.items():
                if isinstance(value, torch.Tensor):
                    data_dict[key] = value.to(self.device)

            data_list.append(data_dict)

        return data_list

    def forward(self, inputs, n_atoms_list):
        from nequip.data import AtomicDataDict

        e_list = []
        f_list = []
        latent_frame_list = []
        latent_atom_list = []

        with torch.set_grad_enabled(True):
            for data_dict in inputs:
                results = self.model(data_dict)

                energy = results.get(
                    AtomicDataDict.TOTAL_ENERGY_KEY,
                    results.get("energy"),
                )

                if energy is not None:
                    e_list.append(
                        energy.detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                        .flatten()[0]
                    )
                else:
                    e_list.append(0.0)

                forces = results.get(
                    AtomicDataDict.FORCE_KEY,
                    results.get("forces"),
                )

                if forces is not None:
                    f_list.append(
                        forces.detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                else:
                    num_atoms = data_dict["pos"].shape[0]
                    f_list.append(np.zeros((num_atoms, 3), dtype=np.float64))

                latent_key = getattr(
                    AtomicDataDict,
                    "NODE_FEATURES_KEY",
                    "node_features",
                )

                latent_atom = results.get(latent_key, results.get("features"))

                if latent_atom is not None:
                    latent_np = (
                        latent_atom.detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )

                    latent_atom_list.append(latent_np)
                    latent_frame_list.append(latent_np.mean(axis=0))
                else:
                    num_atoms = f_list[-1].shape[0]
                    dummy = np.zeros((num_atoms, 1), dtype=np.float64)
                    latent_atom_list.append(dummy)
                    latent_frame_list.append(dummy.mean(axis=0))

        return np.array(e_list), f_list, latent_frame_list, latent_atom_list
