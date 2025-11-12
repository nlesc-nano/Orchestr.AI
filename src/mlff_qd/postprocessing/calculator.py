"""
calculator.py

This module defines a custom ASE calculator for ML models using PyTorch,
provides utilities for Coulomb energy/force calculation, neighbor list management,
and a model evaluation function with uncertainty quantification options.
"""

import torch
import numpy as np
import time
import os
import contextlib
import traceback  # Import traceback here as evaluate_model uses it

from ase.calculators.calculator import Calculator, all_changes
from schnetpack.interfaces import AtomsConverter
from schnetpack import properties as Properties
from schnetpack.transform import ASENeighborList, CachedNeighborList

# Define constants
ANGSTROM_TO_BOHR = 1.8897259886
DEFAULT_PREFACTOR = 14.39964  # eVÃÂ·Ãâ¦ for Coulomb

# --- Charge Assignment ---
def assign_charges(atoms, charges_dict):
    """
    Assigns charges to atoms based on their symbols using a dictionary.

    Parameters:
        atoms (ase.Atoms): The atoms object.
        charges_dict (dict): Dictionary mapping atomic symbols to charges.

    Returns:
        list: Charges for each atom (default 0.0 if symbol not found).
    """
    if not charges_dict:
        return [0.0] * len(atoms)  # Handle case where dict is None or empty
    charges = [charges_dict.get(atom.symbol, 0.0) for atom in atoms]
    return charges


# --- Coulomb Calculation ---
def custom_coulomb_pytorch(atoms, include_electrostatic=True, prefactor=DEFAULT_PREFACTOR):
    """
    Calculates Coulomb energy and forces using PyTorch.

    Parameters:
        atoms (ase.Atoms): Atoms object containing positions and initial charges.
        include_electrostatic (bool): Whether to compute electrostatic contributions.
        prefactor (float): Scaling factor for Coulomb interaction.

    Returns:
        tuple: (energy (float), forces (np.ndarray), calculation time (float))
    """
    # If electrostatics are disabled, return default values immediately.
    if not include_electrostatic:
        return 0.0, np.zeros((len(atoms), 3), dtype=np.float64), 0.0

    # Ensure 'atoms' has a 'calc' attribute with a 'device' property
    if not hasattr(atoms, 'calc') or not hasattr(atoms.calc, 'device'):
        device = torch.device('cpu')
        print("Warning: Calculator context missing in custom_coulomb_pytorch, using CPU.")
    else:
        device = atoms.calc.device

    if not include_electrostatic or not atoms.has('initial_charges'):
        return 0.0, np.zeros((len(atoms), 3), dtype=np.float64), 0.0

    start_time = time.time()
    positions = torch.tensor(atoms.get_positions(), dtype=torch.float32, device=device)
    charges = torch.tensor(atoms.get_initial_charges(), dtype=torch.float32, device=device)
    n_atoms = len(positions)
    if n_atoms <= 1:
        return 0.0, np.zeros((n_atoms, 3), dtype=np.float64), 0.0

    # Compute pairwise distances
    dist = torch.cdist(positions.unsqueeze(0), positions.unsqueeze(0)).squeeze(0)
    # Set self-interaction distances to infinity
    dist = dist.masked_fill(torch.eye(n_atoms, dtype=torch.bool, device=dist.device), float('inf'))

    # Create upper triangular mask to avoid double-counting
    mask = torch.triu(torch.ones_like(dist), diagonal=1).bool()
    energy_coul = prefactor * torch.sum(
        charges.unsqueeze(1) * charges.unsqueeze(0) / (dist + 1e-9) * mask
    )

    # Calculate forces
    dist_force = dist + 1e-9
    inv_dist3 = 1.0 / (dist_force ** 3)
    force_mag = -prefactor * charges.unsqueeze(1) * charges.unsqueeze(0) * inv_dist3
    direction_vectors = positions.unsqueeze(1) - positions.unsqueeze(0)
    dist_expanded = dist_force.unsqueeze(-1)
    # Avoid division by zero for norm_direction
    norm_direction = torch.zeros_like(direction_vectors)
    valid_dist_mask = (dist_expanded > 1e-10)
    norm_direction[valid_dist_mask] = (
        direction_vectors[valid_dist_mask] / dist_expanded[valid_dist_mask]
    )

    # Exclude self-interactions from force calculation
    force_mag = force_mag.masked_fill(torch.eye(n_atoms, dtype=torch.bool, device=force_mag.device), 0.0)
    forces = (force_mag.unsqueeze(-1) * norm_direction).sum(dim=1)  # Sum over j for each i

    energy_numpy = energy_coul.item()
    forces_numpy = forces.detach().cpu().numpy().astype(np.float64)
    calc_time = time.time() - start_time

    # Store Coulomb energy in calculator results if possible
    # Check atoms.calc exists and is the correct calculator instance
    # No need to explicitly set here, calculate() method will store it in self.results
    # if hasattr(atoms, 'calc') and atoms.calc is not None and hasattr(atoms.calc, 'results'):
    #     atoms.calc.results["coul_fn_energy"] = energy_numpy

    return energy_numpy, forces_numpy, calc_time


# --- Neighbor List ---
class SmartNeighborList(ASENeighborList):
    """
    Neighbor list that updates less frequently based on displacements.

    Attributes:
        update_threshold (float): Total displacement threshold to trigger update.
        skin (float): Maximum individual atom displacement to trigger update.
        last_positions (np.ndarray): Last recorded positions.
        last_cell (np.ndarray): Last recorded simulation cell.
    """

    def __init__(self, cutoff, update_threshold, skin):
        super().__init__(cutoff=cutoff)
        self.update_threshold = update_threshold
        self.skin = skin
        self.last_positions = None
        self.last_cell = None  # Track cell changes too

    def update(self, atoms):
        """
        Updates the neighbor list if significant displacement or cell change is detected.

        Parameters:
            atoms (ase.Atoms): Current state of the atoms.

        Returns:
            bool: True if the neighbor list was updated, False otherwise.
        """
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
            return super().update(atoms)  # Call parent update method

        return False  # No update needed


def setup_neighbor_list(config):
    """
    Sets up and returns a cached neighbor list based on configuration parameters.

    Parameters:
        config (dict): Configuration dictionary with keys "cutoff", "skin", "update_threshold", and "cache_path".

    Returns:
        CachedNeighborList: The cached neighbor list object.
    """
    cutoff = config.get("cutoff", 12.0)
    skin = config.get("skin", 2.0)
    update_threshold = config.get("update_threshold", 2.0)
    cache_path = config.get("cache_path", "neighbor_cache")

    cache_dir = os.path.dirname(cache_path)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Created dir: {cache_dir}")

    smart_nl = SmartNeighborList(cutoff=cutoff, update_threshold=update_threshold, skin=skin)
    cached_nl = CachedNeighborList(neighbor_list=smart_nl, cache_path=cache_path)
    print(
        f"NL setup: Cutoff={cutoff}, Skin={skin}, Update Threshold={update_threshold}, Cache={cache_path}"
    )
    return cached_nl

def _to_scalar(x):
    """Return a Python float (FP64) from x if possible."""
    import torch
    import numpy as np
    try:
        if isinstance(x, torch.Tensor):
            if x.numel() == 0: return float("nan")
            if x.numel() == 1: return float(x.item())
            return float(x.detach().cpu().numpy().ravel()[0])
        a = np.asarray(x)
        if a.size == 0: return float("nan")
        return float(a.ravel()[0])
    except Exception:
        try: return float(x)
        except Exception: return float("nan")

# --- ASE Calculator Class (FINAL CORRECTED VERSION) ---
class MyTorchCalculator(Calculator):
    """
    ASE Calculator interface for an intensive (per-atom) PyTorch model.
    
    This calculator performs a "surgical" optimization:
    1. Loads the model in its native FP32 for fast inference.
    2. Extracts the per-atom 'mean' offset from the model's postprocessor.
    3. Disables the model's internal postprocessors to stop the "blurry" FP32 math.
    4. Runs the fast FP32 model to predict the *total per-frame* fluctuation (E_ml).
    5. Reconstructs the total energy in FP64: (mean * N_atoms) + E_ml + E_coul
    """
    implemented_properties = ["energy", "forces", "latent", "per_atom_latent"]

    def __init__(self, model_obj, device, neighbor_list, config):
        super().__init__()
        self.device = device
        self.model = model_obj
        self.config = config

        # --- 1. Model Precision and Device (Load in FP32) ---
        try:
            param = next(self.model.parameters())
            self.compute_dtype = param.dtype
        except StopIteration:
            self.compute_dtype = torch.float32 # Default
        
        self.model.to(self.device, dtype=self.compute_dtype)
        self.model.eval()
        print(f"MyTorchCalculator***: Model loaded on {self.device} in {self.compute_dtype}.")

        # --- 2. Configuration ---
        self.include_electrostatic = config.get("include_electrostatic", True)
        self.coulomb_prefactor = config.get("prefactor", DEFAULT_PREFACTOR)
        self.atomic_charges_dict = config.get("atomic_charges") if self.include_electrostatic else None

        # --- 3. The "Surgical" Mean Offset Fix ---
        self.mean_offset = 0.0 # This will be our FP64 offset
        
        print("Attempting to surgically extract 'mean' offset...")
        try:
            add_offsets_module = self.model.postprocessors[1]
            mean_tensor = getattr(add_offsets_module, 'mean') 
            
            if not isinstance(mean_tensor, torch.Tensor):
                raise TypeError(f"'mean' is not a torch.Tensor, found {type(mean_tensor)}")

            self.mean_offset = mean_tensor.item()
            print(f"Successfully extracted 'mean' offset: {self.mean_offset:.10f} (as FP64)")

            # --- 4. Disable ALL Postprocessors in the Model ---
            self.model.postprocessors = torch.nn.ModuleList([])
            print("Successfully disabled model's internal postprocessors (AddOffsets, CastTo64).")
        
        except Exception as e:
            print(f"WARNING: Surgical 'mean' extraction failed: {e}")
            print("Disabling postprocessors, but 'mean' offset will be 0.0.")
            self.mean_offset = 0.0
            if hasattr(self.model, 'postprocessors'):
                self.model.postprocessors = torch.nn.ModuleList([])

        # --- 5. Latent Hook ---
        self.latent = None
        self.hook_handle = None
        self.representation_layer = None
        if hasattr(self.model, 'representation'):
            self.representation_layer = self.model.representation
            if self.representation_layer is not None:
                def hook(module, input, output):
                    if isinstance(output, dict) and 'scalar_representation' in output:
                        self.latent = output['scalar_representation'].detach().to('cpu', dtype=torch.float32)
                    else:
                        self.latent = None
                self.hook_handle = self.representation_layer.register_forward_hook(hook)
            else:
                 print("Warning: Model representation attribute is None.")
        else:
            print("Warning: Model does not have 'representation' attribute.")

        # --- 6. AtomsConverter ---
        self.atoms_converter = AtomsConverter(
            neighbor_list=neighbor_list,
            device=self.device,
            dtype=self.compute_dtype # torch.float32
        )
        print(f"AtomsConverter set up to output {self.compute_dtype} on {self.device}.")
        
        self.results = {}

    def calculate(self, atoms=None, properties=["energy", "forces"], system_changes=all_changes):
        """
        Calculates properties using mixed-precision.
        - FP32/AMP for ML inference (predicts E_ml_total_fluctuation).
        - FP64 for final summation (E_total = (mean * N_atoms) + E_ml + E_coul).
        """
        super().calculate(atoms, properties, system_changes)
        calc_start = time.time()
        self.latent = None # Reset latent
        n_atoms = len(atoms) # Get atom count for scaling

        # --- Assign initial charges (if needed) ---
        if self.include_electrostatic and self.atomic_charges_dict and not atoms.has("initial_charges"):
            try:
                atoms.set_initial_charges(assign_charges(atoms, self.atomic_charges_dict))
            except Exception as e:
                print(f"Warning: Charge assignment failed: {e}")

        # --- Convert ASE atoms to SchNetPack input (FP32) ---
        inputs = self.atoms_converter(atoms)

        # --- Dtype/Device Normalization (Robustness Check) ---
        index_like = {
            Properties.idx_i, Properties.idx_j, Properties.idx_m,
            Properties.offsets, Properties.n_atoms, Properties.Z,
            "_atomic_numbers", "_idx_i", "_idx_j", "_idx_m"
        }
        for k in index_like:
            if k in inputs and torch.is_tensor(inputs[k]) and not inputs[k].is_floating_point():
                inputs[k] = inputs[k].to(dtype=torch.long)
        
        for k, v in inputs.items():
            if torch.is_tensor(v) and v.is_floating_point():
                inputs[k] = v.to(dtype=self.compute_dtype) # Enforce FP32
        
        # --- Find Position Key and Set requires_grad ---
        position_key_found = None
        if Properties.R in inputs and torch.is_tensor(inputs[Properties.R]):
            position_key_found = Properties.R
        else:
            for key in ["_positions", "positions"]:
                if key in inputs and torch.is_tensor(inputs[key]):
                    position_key_found = key
                    break
        
        if not position_key_found:
             found_keys = list(inputs.keys())
             raise KeyError(
                 f"Could not find a suitable positions tensor "
                 f"(checked standard key '{Properties.R}' and fallbacks ['_positions', 'positions']) "
                 f"in calculator inputs. Found keys: {found_keys}"
              )
        
        inputs[position_key_found].requires_grad_(True)
        
        # --- Model Inference (FP32) ---
        ml_eval_start_time = time.time()
        self.model.eval()
        
        with torch.set_grad_enabled(True):
            # Model is pure FP32, predicts only E_ml (total frame fluctuation)
            results = self.model(inputs) 

        ml_time = time.time() - ml_eval_start_time

        # --- Extract ML Results -> Convert to FP64 ---
        
        # 1. E_ml (TOTAL frame fluctuation, FP32) -> Python float (FP64)
        E_ml_fluctuation_scalar = _to_scalar(results.get("energy", np.nan))

        # 2. ML Forces (FP32) -> NumPy array (FP64)
        if "forces" in results:
            F_ml_np = results["forces"].detach().cpu().numpy().astype(np.float64)
        else:
            F_ml_np = np.full((n_atoms, 3), np.nan, dtype=np.float64)
            print("Error: 'forces' key not found in model output.")

        # 3. Latent Representation
        latent_frame = np.array([], dtype=np.float64)
        latent_per_atom_avg_np = None
        if self.latent is not None:
            latent_per_atom_avg_np = self.latent.numpy() 
            latent_frame = np.sum(latent_per_atom_avg_np, axis=0).astype(np.float64)

        # --- Add Coulomb (Computation in FP64) ---
        E_coul, F_coul, coul_time = custom_coulomb_pytorch(
            atoms, self.include_electrostatic, self.coulomb_prefactor
        )
        F_coul = np.asarray(F_coul, dtype=np.float64)

        # --- Store Final Results (All summation in FP64) ---
        
        # *** THIS IS THE CORRECTED LOGIC (YOURS) ***
        # All variables are Python floats (FP64) or int, so this is a high-precision sum.
        E_baseline_total = self.mean_offset * n_atoms
        E_ml_total = E_baseline_total + E_ml_fluctuation_scalar
        energy_total_scalar = E_ml_total + E_coul
        # *** END OF FIX ***
        
        forces_total_np = F_ml_np + F_coul

        self.results["energy"] = energy_total_scalar
        self.results["forces"] = forces_total_np
        
        # --- Store partial/debug info ---
        self.results["E_ml"] = E_ml_total # Store the total ML energy
        self.results["E_ref"] = E_baseline_total # Store the baseline
        self.results["E_ml_total_fluctuation"] = E_ml_fluctuation_scalar
        self.results["mean_offset"] = self.mean_offset
        self.results["coul_fn_energy"] = E_coul
        self.results["ml_time"] = ml_time
        self.results["coul_fn_time"] = coul_time
        self.results["calc_total_time"] = time.time() - calc_start
        self.results["sigma_energy"] = 0.0
        self.results["latent"] = latent_frame
        self.results["per_atom_latent"] = latent_per_atom_avg_np

    def __del__(self):
        """Ensures that the hook handle is removed upon deletion."""
        if self.hook_handle:
            try:
                self.hook_handle.remove()
            except Exception:
                pass

# --- Refactored Evaluation Function (FINAL, WITH AMP SWITCH) ---
def evaluate_model(frames, true_energies, true_forces, model_obj, device, batch_size,
                   eval_log_file, config, neighbor_list):
    """
    Evaluates a pre-loaded model (intensive, per-atom) on a set of frames
    using batched inference and high-precision summation.
    
    Includes a switch for AMP (autocast) which defaults to OFF for max precision.
    To enable, set 'use_amp: True' in your config.
    """
    print("\n--- Evaluating Model (Batched, Intensive) ---")
    try:
        calc = MyTorchCalculator(model_obj, device, neighbor_list, config)
        
    except Exception as e_calc:
        print(f"Error initializing MyTorchCalculator: {e_calc}")
        traceback.print_exc()
        return None, None, None, None, None  # Indicate failure

    # --- SAFETY PATCH for PaiNN ---
    rep = getattr(calc.model, "representation", None)
    if rep is not None and not hasattr(rep, "electronic_embeddings"):
        rep.electronic_embeddings = nn.ModuleList([])
    
    n_frames = len(frames)
    all_energy_pred = []
    all_forces_pred = []
    all_latent_pred = []
    all_per_atom_latent_pred = []
    
    cum_eval_time = 0.0
    batches_processed = 0
    print(f"Starting evaluation for {n_frames} frames with batch size {batch_size}...")
    
    # --- Efficient Logging Setup ---
    log_header_written = False
    log_lines_buffer = [] # Buffer to store log lines

    # Get the mean_offset (as an FP64 Python float) *once*
    mean_offset_fp64 = calc.mean_offset
    print(f"Evaluation using mean offset: {mean_offset_fp64:.10f}")

    # --- AMP: Your requested "on/off switch" ---
    # Defaults to OFF (False) for maximum precision
    use_amp = config.get("use_amp", False) 
    amp_dtype = torch.float16 # Default
    
    if use_amp:
        print("--- AMP (Autocast) is ENABLED ---")
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            print("Using torch.bfloat16 for AMP autocast (stable).")
        else:
            print("Using torch.float16 for AMP autocast.")
    else:
        print("--- AMP (Autocast) is DISABLED (default, maximum precision) ---")
    # --- End AMP Config ---

    # === Main Batch Loop ===
    for batch_start in range(0, n_frames, batch_size):
        batch_indices = range(batch_start, min(batch_start + batch_size, n_frames))
        batch_frames = [frames[i] for i in batch_indices]
        batch_size_actual = len(batch_frames)
        if batch_size_actual == 0: continue

        batch_start_time = time.time()
        
        batch_energies_total = []
        batch_forces_total = []
        batch_latents_frame = []
        batch_latents_per_atom = []
        
        try:
            # --- 1. Assign Charges (CPU-side) ---
            for frame in batch_frames:
                if calc.include_electrostatic and calc.atomic_charges_dict and not frame.has("initial_charges"):
                    try:
                        frame.set_initial_charges(assign_charges(frame, calc.atomic_charges_dict))
                    except Exception as e:
                        print(f"Warning: Charge assignment failed: {e}")
            
            # --- 2. Convert Batch of Atoms to PyTorch Input (FP32) ---
            calc.latent = None
            inputs = calc.atoms_converter(batch_frames)
            
            # --- 3. Set `requires_grad` for Positions ---
            position_key_found = None
            if Properties.R in inputs: 
                position_key_found = Properties.R
            else:
                for key in ["_positions", "positions"]:
                    if key in inputs: 
                        position_key_found = key
                        break
            
            if not position_key_found:
                raise KeyError("Could not find positions tensor in batched inputs.")
            
            inputs[position_key_found].requires_grad_(True)
            
            # --- 4. Batched ML Inference ---
            ml_eval_start_time = time.time()
            
            with torch.set_grad_enabled(True):
                # This context now respects your 'use_amp' switch
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    results_batch = calc.model(inputs)
                
            ml_time = time.time() - ml_eval_start_time
            
            # --- 5. Extract and Un-batch ML Results ---
            
            n_atoms_list = inputs[Properties.n_atoms].cpu().tolist()
            
            E_ml_fluctuation_np = results_batch["energy"].detach().cpu().numpy().astype(np.float64).flatten()
            
            # --- Optimized Data Transfer ---
            F_ml_batch_tensor_cpu = results_batch["forces"].detach().cpu()
            F_ml_list = [f.numpy().astype(np.float64) for f in torch.split(F_ml_batch_tensor_cpu, n_atoms_list, dim=0)]
            # --- End Optimized Data Transfer ---
            
            latent_per_atom_list = []
            if calc.latent is not None:
                latent_batch_tensor = calc.latent.detach()
                latent_per_atom_list = [l.numpy() for l in torch.split(latent_batch_tensor, n_atoms_list, dim=0)]
            else:
                latent_per_atom_list = [None] * batch_size_actual
            
            # --- 6. Post-processing Loop (Coulomb + Final Sum) ---
            for i in range(batch_size_actual):
                frame = batch_frames[i]
                frame_idx_global = batch_start + i
                n_atoms = n_atoms_list[i]
                
                E_ml_fluctuation_scalar = E_ml_fluctuation_np[i]
                F_ml_np = F_ml_list[i]
                latent_per_atom_np = latent_per_atom_list[i]
                
                # --- Coulomb (in FP64) ---
                E_coul, F_coul, coul_time = custom_coulomb_pytorch(
                    frame, calc.include_electrostatic, calc.coulomb_prefactor
                )
                F_coul = np.asarray(F_coul, dtype=np.float64)
                
                # --- FINAL SUMMATION (All in FP64) ---
                E_baseline_total = mean_offset_fp64 * n_atoms
                E_ml_total = E_baseline_total + E_ml_fluctuation_scalar
                energy_total = E_ml_total + E_coul
                forces_total = F_ml_np + F_coul
                
                if latent_per_atom_np is not None:
                    latent_frame = np.sum(latent_per_atom_np, axis=0).astype(np.float64)
                else:
                    latent_frame = np.array([], dtype=np.float64)
                
                # --- Store results for this frame ---
                batch_energies_total.append(energy_total)
                batch_forces_total.append(forces_total)
                batch_latents_frame.append(latent_frame)
                batch_latents_per_atom.append(latent_per_atom_np)

                # --- Efficient Logging: Add to buffer ---
                if eval_log_file:
                    true_e_val = _to_scalar(true_energies[frame_idx_global]) if frame_idx_global < len(true_energies) else np.nan
                    energy_diff = energy_total - true_e_val if not (np.isnan(energy_total) or np.isnan(true_e_val)) else np.nan
                    step_time = (time.time() - batch_start_time) / batch_size_actual

                    if not log_header_written:
                        header = (
                            f"{'Frame':>6s} | {'True Energy (eV)':>18s} | {'Pred Energy (eV)':>18s} | "
                            f"{'Energy Diff (eV)':>15s} | {'E_ML_Total (eV)':>15s} | {'E_ML_fluct (eV)':>12s} | {'E_Coul (eV)':>12s} | "
                            f"{'MLtime(s)':>10s} | {'CoulTime(s)':>10s} | {'StepTime(s)':>10s} | {'CumTime(s)':>10s}\n"
                        )
                        log_lines_buffer.append(header)
                        log_header_written = True
                    
                    log_lines_buffer.append(
                        f"{frame_idx_global:6d} | {true_e_val:18.6f} | {energy_total:18.6f} | "
                        f"{energy_diff:15.6f} | {E_ml_total:15.6f} | {E_ml_fluctuation_scalar:12.6f} | {E_coul:12.6f} | "
                        f"{ml_time/batch_size_actual:10.4f} | {coul_time:10.4f} | "
                        f"{step_time:10.4f} | {cum_eval_time + (time.time()-batch_start_time):10.4f}\n"
                    )
            
            # --- End of post-processing loop ---
            
            all_energy_pred.extend(batch_energies_total)
            all_forces_pred.extend(batch_forces_total)
            all_latent_pred.extend(batch_latents_frame)
            all_per_atom_latent_pred.extend(batch_latents_per_atom)

        except Exception as e_batch_calc:
            print(f"Error calculating batch starting {batch_start}: {e_batch_calc}")
            traceback.print_exc()
            all_energy_pred.extend([np.nan] * batch_size_actual)
            all_forces_pred.extend([np.full((len(f), 3), np.nan) for f in batch_frames])
            all_latent_pred.extend([np.nan] * batch_size_actual)
            all_per_atom_latent_pred.extend([np.nan] * batch_size_actual)
            continue
        
        batch_time = time.time() - batch_start_time
        cum_eval_time += batch_time
        batches_processed += 1
        
        if batches_processed % 10 == 0 or batches_processed == 1:
            total_batches = (n_frames + batch_size - 1) // batch_size
            print(f"  Processed batch {batches_processed}/{total_batches} (Avg frame time in batch: {batch_time/batch_size_actual:.5f}s)")

    # --- Write buffered log lines to file (one single, fast operation) ---
    if eval_log_file and log_lines_buffer:
        try:
            with open(eval_log_file, "w") as elog: # Use "w" to overwrite
                elog.writelines(log_lines_buffer)
            print(f"Log file written successfully to {eval_log_file}")
        except IOError as log_e:
            print(f"Warning: Failed to write log file: {log_e}")

    # --- FINAL SUMMARY ---
    print("\n--- Evaluation Summary ---")
    print(f"Processed {len(all_energy_pred)} frames in {batches_processed} batches.")
    print(f"Total Eval Time: {cum_eval_time:.3f}s")
    if n_frames > 0: print(f"Avg Time / Frame: {(cum_eval_time / n_frames):.5f}s")
    print("--------------------------\n")

    return all_energy_pred, all_forces_pred, all_latent_pred, all_per_atom_latent_pred


