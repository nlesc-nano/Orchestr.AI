"""
calculator.py

This module provides a clean, framework-agnostic interface for ML force-field inference.
It defines a NeighborList provider, a BaseCalculator interface with implementations for 
different frameworks (SchNetPack, MACE, NequIP), and an InferenceRunner to handle 
batching, precise timings, and logging.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
import traceback

# Legacy SchNetPack imports
from schnetpack.interfaces import AtomsConverter
from schnetpack import properties as Properties
from schnetpack.transform import ASENeighborList, CachedNeighborList

# ==========================================
# 1. NEIGHBOR LIST MANAGEMENT
# ==========================================

class SmartNeighborList(ASENeighborList):
    """Legacy ASE-based neighbor list with skin-based caching."""
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
            if (np.max(np.linalg.norm(displacements, axis=1)) > self.skin or
                np.sum(np.linalg.norm(displacements, axis=1)) > self.update_threshold):
                needs_update = True

        if needs_update:
            self.last_positions = current_positions.copy()
            self.last_cell = current_cell.copy()
            return super().update(atoms)
        return False


class NeighborListProvider:
    """Provides either the legacy ASE neighbor list or the blazing fast 'alchemy' list."""
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
            smart_nl = SmartNeighborList(self.cutoff, self.update_threshold, self.skin)
            self.ase_nl = CachedNeighborList(neighbor_list=smart_nl, cache_path=self.cache_path)
            print("NeighborList initialized with 'legacy' ASE backend.")
            
        elif self.backend == "alchemy":
            print("NeighborList initialized with 'alchemy' backend (matscipy).")

    def get_ase_nl(self):
        """Returns the ASE neighbor list for legacy frameworks (SchNetPack)."""
        if self.backend != "legacy":
            print("Warning: Model requested legacy ASE NL, but backend is 'alchemy'. Falling back to ASE.")
            # Fallback initialized safely
            smart_nl = SmartNeighborList(self.cutoff, self.update_threshold, self.skin)
            return CachedNeighborList(neighbor_list=smart_nl, cache_path=self.cache_path)
        return self.ase_nl

    def compute_alchemy_edges(self, atoms):
        """
        Blazingly fast neighbor list computation using matscipy (C++ optimized).
        Used natively by modern MACE/NequIP implementations.
        """
        try:
            import matscipy.neighbors
        except ImportError:
            raise ImportError("matscipy is required for the 'alchemy' backend. Run: pip install matscipy")
            
        # Returns sender (i) and receiver (j) indices instantly
        i, j = matscipy.neighbors.neighbor_list('ij', atoms, self.cutoff)
        return torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long)


def setup_neighbor_list(config):
    """Legacy helper function to maintain compatibility with evaluate.py initialization."""
    provider = NeighborListProvider(config)
    return provider.get_ase_nl()


# ==========================================
# 2. FRAMEWORK-SPECIFIC CALCULATORS
# ==========================================

class BaseCalculator:
    """Abstract interface for all ML framework calculators."""
    def prepare_batch(self, frames):
        """Converts a list of ASE Atoms into the model's specific tensor dictionary."""
        raise NotImplementedError

    def forward(self, inputs, n_atoms_list):
        """Runs inference and returns standard arrays: (energy, forces, latent_frame, latent_atom)"""
        raise NotImplementedError


class SchnetpackCalculator(BaseCalculator):
    """Wrapper for SchNet and PaiNN models."""
    def __init__(self, model, device, nl_provider):
        self.model = model
        self.device = device
        self.model.to(self.device, dtype=torch.float32)
        self.model.eval()

        # Safely handle PaiNN electronic embeddings bug
        rep = getattr(self.model, "representation", None)
        if rep is not None and not hasattr(rep, "electronic_embeddings"):
            rep.electronic_embeddings = nn.ModuleList([])

        self.atoms_converter = AtomsConverter(
            neighbor_list=nl_provider.get_ase_nl(),
            device=self.device,
            dtype=torch.float32
        )

        # Hook to extract latent representations
        self.latent = None
        if hasattr(self.model, 'representation') and self.model.representation is not None:
            def hook(module, input, output):
                if isinstance(output, dict) and 'scalar_representation' in output:
                    self.latent = output['scalar_representation'].detach().to('cpu', dtype=torch.float32)
                else:
                    self.latent = None
            self.model.representation.register_forward_hook(hook)

    def prepare_batch(self, frames):
        self.latent = None
        inputs = self.atoms_converter(frames)
        
        # Identify the positions tensor to track force gradients
        pos_key = Properties.R if Properties.R in inputs else None
        if not pos_key:
            pos_key = next((k for k in ["_positions", "positions"] if k in inputs), None)
        if pos_key:
            inputs[pos_key].requires_grad_(True)
            
        return inputs

    def forward(self, inputs, n_atoms_list):
        with torch.set_grad_enabled(True):
            results = self.model(inputs)

        # Extract predictions
        energies_np = results["energy"].detach().cpu().numpy().astype(np.float64).flatten()
        forces_cpu = results["forces"].detach().cpu()
        forces_list = [f.numpy().astype(np.float64) for f in torch.split(forces_cpu, n_atoms_list, dim=0)]

        # Extract latents
        latent_atom_list = [None] * len(n_atoms_list)
        latent_frame_list = [np.array([], dtype=np.float64)] * len(n_atoms_list)
        
        if self.latent is not None:
            latents_cpu = self.latent.detach().cpu()
            latent_atom_list = [l.numpy() for l in torch.split(latents_cpu, n_atoms_list, dim=0)]
            latent_frame_list = [np.sum(l, axis=0).astype(np.float64) for l in latent_atom_list]

        return energies_np, forces_list, latent_frame_list, latent_atom_list

class MaceCalculator(BaseCalculator):
    """Wrapper for MACE models."""
    def __init__(self, model, device, nl_provider):
        self.model = model
        self.device = device
        self.nl_provider = nl_provider
        self.model.to(self.device)
        self.model.eval()
        
        # --- cuEquivariance Detection ---
        try:
            import cuequivariance_torch
            self.use_cueq = True
            print("cuEquivariance detected! MACE tensor ops will run blazingly fast.")
            if hasattr(self.model, "enable_cueq"):
                self.model.enable_cueq = True 
        except ImportError:
            self.use_cueq = False
            print("cuEquivariance not found. MACE will use standard PyTorch ops.")

    def prepare_batch(self, frames):
        try:
            from mace.data.utils import config_from_atoms
            from mace.data.atomic_data import AtomicData
            from mace.tools.torch_geometric.dataloader import Collater
        except ImportError:
            raise ImportError("MACE is not installed. Please run: pip install mace-torch")

        data_list = []
        for atoms in frames:
            # 1. Convert ASE to MACE configuration
            config = config_from_atoms(atoms)
            
            # 2. Build AtomicData using the model's exact atomic numbers (z_table)
            data = AtomicData.from_config(
                config, 
                z_table=self.model.z_table, 
                cutoff=self.nl_provider.cutoff
            )
            data_list.append(data)

        # 3. Collate into a single batch
        collater = Collater(follow_batch=[])
        batch = collater(data_list).to(self.device)
        
        return batch

    def forward(self, inputs, n_atoms_list):
        with torch.set_grad_enabled(True):
            # MACE usually expects a dictionary representation of the batch
            results = self.model(inputs.to_dict())

        # Extract MACE specific keys
        energies_np = results["energy"].detach().cpu().numpy().astype(np.float64).flatten()
        forces_cpu = results["forces"].detach().cpu()
        forces_list = [f.numpy().astype(np.float64) for f in torch.split(forces_cpu, n_atoms_list, dim=0)]

        # Extract Latents (MACE calls this 'node_feats')
        latent_frame_list = [np.array([], dtype=np.float64)] * len(n_atoms_list)
        latent_atom_list = [None] * len(n_atoms_list)
        
        if "node_feats" in results and results["node_feats"] is not None:
            latents_cpu = results["node_feats"].detach().cpu()
            latent_atom_list = [l.numpy() for l in torch.split(latents_cpu, n_atoms_list, dim=0)]
            latent_frame_list = [np.sum(l, axis=0).astype(np.float64) for l in latent_atom_list]

        return energies_np, forces_list, latent_frame_list, latent_atom_list

class NequipCalculator(BaseCalculator):
    """Wrapper for NequIP / Allegro models using the official ASE calculator."""
    def __init__(self, model_obj, device, nl_provider):
        self.device = device
        # model_obj is now the direct string path passed from evaluate.py!
        self.model_path = model_obj 
        self.calc = None

    def prepare_batch(self, frames):
        # Initialize the ASE calculator on the first batch
        if self.calc is None:
            from nequip.ase import NequIPCalculator
            
            # IDENTITY MAPPING: Extract species from the first frame and map them
            # Exact logic matching run_nequip_sim.py (e.g., {'Cd': 'Cd', 'Se': 'Se'})
            species = sorted(list(set(frames[0].get_chemical_symbols())))
            chemical_map = {s: s for s in species}
            
            # Load directly from the file path! This perfectly preserves 
            # the hidden r_max and TypeMapper metadata inside the TorchScript zip.
            self.calc = NequIPCalculator.from_compiled_model(
                compile_path=self.model_path,
                chemical_species_to_atom_type_map=chemical_map,
                device=str(self.device)
            )
                    
        return frames

    def forward(self, frames, n_atoms_list):
        e_list, f_list, l_frame_list, l_atom_list = [], [], [], []
        
        for atoms in frames:
            # 1. Secure Forward Pass via official ASE Calculator
            atoms.calc = self.calc
            
            e = atoms.get_potential_energy()
            f = atoms.get_forces()
            
            e_list.append(e)
            f_list.append(f)
            
            # 2. Synthesize Latent Features for Active Learning
            f_mags = np.linalg.norm(f, axis=1)
            
            # Create a histogram of force magnitudes (captures the dynamic state of the QD)
            hist, _ = np.histogram(f_mags, bins=20, range=(0.0, 10.0))
            
            # Add basic statistical moments (mean force, max force, per-atom energy)
            moments = np.array([f_mags.mean(), f_mags.std(), f_mags.max(), e / len(atoms)])
            
            # Combine into a single descriptive frame latent vector
            frame_latent = np.concatenate([hist, moments]).astype(np.float64)
            
            # Atom-level latents (Force vector + Magnitude)
            atom_latents = np.hstack([f, f_mags.reshape(-1, 1)]).astype(np.float64)
            
            l_frame_list.append(frame_latent)
            l_atom_list.append(atom_latents)
            
        return np.array(e_list), f_list, l_frame_list, l_atom_list


class NequipCalculator_extended(BaseCalculator):
    """Wrapper for NequIP / Allegro models (supports v0.16.0+)."""
    def __init__(self, model, device, nl_provider):
        self.model = model
        self.device = device
        self.nl_provider = nl_provider
        self.model.to(self.device)
        self.model.eval()
        self.type_mapper = None
        self.r_max = float(self.nl_provider.cutoff)

        try:
            import cuequivariance_torch
            print("cuEquivariance detected! NequIP/Allegro ops will be hardware accelerated.")
        except ImportError:
            pass

    def prepare_batch(self, frames):
        from nequip.data import from_ase, AtomicDataDict
        import ase.neighborlist
        
        # 1. Initialize the TypeMapper on the first run
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
            # 2. Extract ASE data
            if isinstance(self.type_mapper, list):
                data_dict = from_ase(atoms)
                syms = atoms.get_chemical_symbols()
                atom_types = [self.type_mapper.index(s) for s in syms]
                data_dict[AtomicDataDict.ATOM_TYPE_KEY] = torch.tensor(atom_types, dtype=torch.long)
            else:
                try:
                    data_dict = from_ase(atoms, type_mapper=self.type_mapper)
                except TypeError:
                    # Fallback if from_ase API changes
                    data_dict = from_ase(atoms)
                    syms = atoms.get_chemical_symbols()
                    atom_types = [self.type_mapper.type_names.index(s) for s in syms]
                    data_dict[AtomicDataDict.ATOM_TYPE_KEY] = torch.tensor(atom_types, dtype=torch.long)

            # 3. MANUALLY COMPUTE THE NEIGHBOR LIST GRAPH
            # 'i' = sender atom, 'j' = receiver atom, 'S' = unit cell shift vector
            i, j, S = ase.neighborlist.neighbor_list('ijS', atoms, self.r_max)
            
            # Inject the graph directly into the NequIP dictionary
            data_dict[AtomicDataDict.EDGE_INDEX_KEY] = torch.stack([
                torch.tensor(i, dtype=torch.long),
                torch.tensor(j, dtype=torch.long)
            ], dim=0)
            
            # The shift vector must be float32 to match the cell matrix multiplication
            data_dict[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = torch.tensor(S, dtype=torch.float32)
            
            # Inject r_max just in case the model asserts it
            r_max_key = getattr(AtomicDataDict, 'R_MAX_KEY', 'r_max')
            data_dict[r_max_key] = torch.tensor([self.r_max], dtype=torch.float32)

            # 4. Move tensors to GPU/CPU safely
            for k, v in data_dict.items():
                if isinstance(v, torch.Tensor):
                    data_dict[k] = v.to(self.device)
            data_list.append(data_dict)
            
        return data_list

    def forward(self, inputs, n_atoms_list):
        from nequip.data import AtomicDataDict

        with torch.set_grad_enabled(True):
            e_list, f_list, l_frame_list, l_atom_list = [], [], [], []
            
            for data_dict in inputs:
                results = self.model(data_dict)
                
                # Extract Energy
                e = results.get(AtomicDataDict.TOTAL_ENERGY_KEY, results.get('energy'))
                if e is not None:
                    e_list.append(e.detach().cpu().numpy().astype(np.float64).flatten()[0])
                else:
                    e_list.append(0.0)
                
                # Extract Forces
                f = results.get(AtomicDataDict.FORCE_KEY, results.get('forces'))
                if f is not None:
                    f_list.append(f.detach().cpu().numpy().astype(np.float64))
                else:
                    num_atoms = data_dict['pos'].shape[0]
                    f_list.append(np.zeros((num_atoms, 3), dtype=np.float64))
                
                # Extract Latents (Node Features)
                l_key = getattr(AtomicDataDict, 'NODE_FEATURES_KEY', 'node_features')
                l_atom = results.get(l_key, results.get('features'))
                if l_atom is not None:
                    l_np = l_atom.detach().cpu().numpy().astype(np.float64)
                    l_atom_list.append(l_np)
                    l_frame_list.append(l_np.mean(axis=0))
                else:
                    num_atoms = f_list[-1].shape[0]
                    dummy = np.zeros((num_atoms, 1), dtype=np.float64)
                    l_atom_list.append(dummy)
                    l_frame_list.append(dummy.mean(axis=0))
                    
            return np.array(e_list), f_list, l_frame_list, l_atom_list

# ==========================================
# 3. THE INFERENCE ORCHESTRATOR
# ==========================================

class InferenceRunner:
    """Orchestrates batching, precise timings, and logging for ML inference."""
    def __init__(self, calculator: BaseCalculator, batch_size: int, log_file: str = None):
        self.calculator = calculator
        self.batch_size = batch_size
        self.log_file = log_file

    def run(self, frames, true_energies=None, true_forces=None):
        n_frames = len(frames)
        all_energy_pred, all_forces_pred = [], []
        all_latent_frame, all_latent_atom = [], []

        cum_eval_time = 0.0
        batches_processed = 0
        log_lines_buffer = []

        print(f"Starting generic inference for {n_frames} frames (Batch Size: {self.batch_size})...")

        for batch_start in range(0, n_frames, self.batch_size):
            batch_frames = frames[batch_start : batch_start + self.batch_size]
            actual_size = len(batch_frames)
            n_atoms_list = [len(f) for f in batch_frames]

            batch_start_time = time.time()

            try:
                # 1. Data Preparation (Includes Neighbor List time)
                t0_prep = time.time()
                inputs = self.calculator.prepare_batch(batch_frames)
                prep_time = time.time() - t0_prep

                # 2. Forward Pass (Includes cuEquivariance tensor ops time)
                t0_forward = time.time()
                energies, forces_list, lat_frame, lat_atom = self.calculator.forward(inputs, n_atoms_list)
                forward_time = time.time() - t0_forward

                # 3. Store Results
                all_energy_pred.extend(energies)
                all_forces_pred.extend(forces_list)
                all_latent_frame.extend(lat_frame)
                all_latent_atom.extend(lat_atom)

                # 4. Logging
                if self.log_file:
                    for i in range(actual_size):
                        global_idx = batch_start + i
                        true_e = true_energies[global_idx] if true_energies and global_idx < len(true_energies) else np.nan
                        pred_e = energies[i]
                        diff_e = pred_e - true_e

                        if not log_lines_buffer:
                            header = f"{'Frame':>6s} | {'True_E(eV)':>15s} | {'Pred_E(eV)':>15s} | {'Diff(eV)':>12s} | {'PrepTime(s)':>12s} | {'FwdTime(s)':>12s}\n"
                            log_lines_buffer.append(header)
                        
                        log_lines_buffer.append(
                            f"{global_idx:6d} | {true_e:15.6f} | {pred_e:15.6f} | {diff_e:12.6f} | "
                            f"{prep_time/actual_size:12.6f} | {forward_time/actual_size:12.6f}\n"
                        )

            except Exception as e:
                print(f"Error processing batch {batches_processed}: {e}")
                traceback.print_exc()
                # Append NaNs to maintain array shapes on failure
                all_energy_pred.extend([np.nan] * actual_size)
                all_forces_pred.extend([np.full((n, 3), np.nan) for n in n_atoms_list])
                all_latent_frame.extend([np.nan] * actual_size)
                all_latent_atom.extend([np.nan] * actual_size)

            cum_eval_time += (time.time() - batch_start_time)
            batches_processed += 1
            if batches_processed % 10 == 0 or batches_processed == 1:
                print(f"  Processed batch {batches_processed} (Avg total frame time: {cum_eval_time / len(all_energy_pred):.5f}s)")

        # Flush logs to disk
        if self.log_file and log_lines_buffer:
            try:
                with open(self.log_file, "w") as elog:
                    elog.writelines(log_lines_buffer)
            except IOError as log_e:
                print(f"Warning: Failed to write log file: {log_e}")

        print("\n--- Inference Summary ---")
        print(f"Total Time: {cum_eval_time:.3f}s | Avg Time/Frame: {cum_eval_time/max(1, n_frames):.5f}s")
        print("-------------------------\n")

        return all_energy_pred, all_forces_pred, all_latent_frame, all_latent_atom

# ==========================================
# 4. ENTRY POINT FOR EVALUATE.PY
# ==========================================

def evaluate_model(frames, true_energies, true_forces, model_obj, device, batch_size,
                   eval_log_file, config, neighbor_list):
    """
    Standard entry point used by evaluate.py. Determines the framework,
    instantiates the correct BaseCalculator, and runs the InferenceRunner.
    """
    try:
        nl_provider = NeighborListProvider(config, existing_nl=neighbor_list)
        framework = config.get("model_framework", "schnetpack").lower()

        if framework == "schnetpack":
            calc = SchnetpackCalculator(model_obj, device, nl_provider)
        elif framework == "mace":
            calc = MaceCalculator(model_obj, device, nl_provider)
        elif framework == "nequip":
            calc = NequipCalculator(model_obj, device, nl_provider)
        else:
            raise ValueError(f"Unknown framework '{framework}'. Cannot instantiate calculator.")

        runner = InferenceRunner(calc, batch_size, eval_log_file)
        return runner.run(frames, true_energies, true_forces)

    except Exception as e:
        print(f"Critical error initializing evaluate_model: {e}")
        traceback.print_exc()
        return None, None, None, None

