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


# --- ASE Calculator Class ---
class MyTorchCalculator(Calculator):
    """
    ASE Calculator interface for a PyTorch model.

    Computes energy, forces, and latent representations. Handles electrostatics.
    """
    implemented_properties = ["energy", "forces", "latent", "per_atom_latent"]

    # --- MODIFIED __init__ ---
    def __init__(self, model_obj, device, neighbor_list, config):
        super().__init__()
        self.device = device
        self.model = model_obj
        self.config = config # Store the config dictionary

        # Extract settings from config
        self.apply_atom_offsets = config.get("apply_atom_offsets", False)
        self.include_electrostatic = config.get("include_electrostatic", True)
        self.coulomb_prefactor = config.get("prefactor", DEFAULT_PREFACTOR)
        # Store charges dict IF electrostatics are enabled
        self.atomic_charges_dict = config.get("atomic_charges") if self.include_electrostatic else None

        # Ensure model is on the correct device and in eval mode initially
        self.model.to(self.device)
        self.model.eval()

        # Filter out None postprocessors if they exist
        if hasattr(self.model, 'postprocessors'):
            filtered_postprocessors = [pp for pp in self.model.postprocessors if pp is not None]
            self.model.postprocessors = torch.nn.ModuleList(filtered_postprocessors)

        # Hook to capture SchNet representation output
        self.latent = None
        self.hook_handle = None # Store hook handle to potentially remove it later if needed
        self.representation_layer = None

        if hasattr(self.model, 'representation'):
            self.representation_layer = self.model.representation
            if self.representation_layer is not None:
                def hook(module, input, output):
                    # Ensure output is a dict and key exists, store as float32 CPU tensor
                    if isinstance(output, dict) and 'scalar_representation' in output:
                        # Detach and move to CPU within the hook if possible, ensures latent is available after calc
                        self.latent = output['scalar_representation'].detach().to('cpu', dtype=torch.float32)
                    else:
                        self.latent = None # Reset if not found

                # Register the forward hook
                self.hook_handle = self.representation_layer.register_forward_hook(hook)
            else:
                 print("Warning: Model representation attribute is None.")
        else:
            print("Warning: Model does not have 'representation' attribute for latent hook.")

        # Handle atomrefs
        self.atomrefs = None
        if self.apply_atom_offsets:
            # Simplified atomref finding (adjust if needed based on your model structure)
             atomrefs_attr = getattr(self.model, 'atomrefs', None)
             if atomrefs_attr is None and hasattr(self.model, 'output_modules') and self.model.output_modules:
                  # Check the last output module, assuming it might hold atomrefs
                  atomrefs_attr = getattr(self.model.output_modules[-1], 'atomrefs', None)
             self.atomrefs = atomrefs_attr
             if self.atomrefs is not None:
                  # Ensure atomrefs tensor is on the correct device
                  self.atomrefs = self.atomrefs.to(self.device)
                  print("Atom references loaded.")
             else:
                  print("Warning: apply_atom_offsets=True, but no atomrefs found.")


        # Setup AtomsConverter
        self.atoms_converter = AtomsConverter(neighbor_list=neighbor_list, device=device, dtype=torch.float32)

        # Attributes for dropout mode (set externally before calling calculate)
        self.use_dropout = False
        self.n_mc = 1
        self.results = {} # Initialize results dict

    # --- MODIFIED calculate ---
    def calculate(self, atoms=None, properties=["energy", "forces"], system_changes=all_changes):
        """
        Calculates properties. NOTE: Always calculates energy and forces internally
        to bypass issues with energy-only calls in some models.
        """
        # Call parent calculate, but ignore properties list internally
        super().calculate(atoms, ["energy", "forces"], system_changes) # Request both from cache
        calc_start = time.time()
        self.latent = None # Reset latent

        # --- Assign initial charges if needed ---
        if self.include_electrostatic and self.atomic_charges_dict and not atoms.has("initial_charges"):
            try:
                atoms.set_initial_charges(assign_charges(atoms, self.atomic_charges_dict))
            except Exception as e:
                print(f"Warning: Charge assignment failed in calculate(): {e}")
        # --- End charge assignment ---

        # Convert ASE atoms to SchNetPack input format
        inputs = self.atoms_converter(atoms)

        # === DTYPE / DEVICE NORMALIZATION (Key Fix) ===
        # Use the model's first parameter to infer device & dtype
        param = next(self.model.parameters())
        model_device = param.device
        param_dtype  = param.dtype

        # 1) move all tensors to the model's device
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(device=model_device)

        # 2) ensure index-like tensors are long
        index_like = {
            Properties.idx_i, Properties.idx_j, Properties.idx_m,
            Properties.offsets, Properties.n_atoms, Properties.Z,
            "_atomic_numbers", "_idx_i", "_idx_j", "_idx_m"
        }
        for k in index_like:
            if k in inputs and torch.is_tensor(inputs[k]) and not inputs[k].is_floating_point():
                inputs[k] = inputs[k].to(dtype=torch.long)

        # 3) ensure floating tensors match the model parameter dtype
        for k, v in inputs.items():
            if torch.is_tensor(v) and v.is_floating_point():
                inputs[k] = v.to(dtype=param_dtype)
        # === End normalization ===


        # --- Locate position key and ALWAYS set requires_grad=True ---
        position_key_found = None
        if Properties.R in inputs and torch.is_tensor(inputs[Properties.R]):
            position_key_found = Properties.R
        else:
            fallback_keys = ["_positions", "positions"]
            for key in fallback_keys:
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
        # Always enable gradients for the position tensor
        inputs[position_key_found].requires_grad_(True)
        # --- End position key logic ---


        # --- Model Inference (Always calculates forces) ---
        ml_eval_start_time = time.time()
        amp_enabled = bool(self.config.get("md", {}).get("use_amp", False)) and (param.device.type == "cuda")
        E_ml = 0.0
        F_ml = np.zeros((len(atoms), 3))
        sigma_energy = 0.0
        latent_frame = None
        latent_per_atom_avg_np = None

        if self.use_dropout:
            # --- Dropout Branch ---
            self.model.train()
            energies_list = []
            forces_tmp_list = []
            latent_tmp_list = []

            inference_context = torch.set_grad_enabled(True) # Gradients always needed
            for mc_pass in range(self.n_mc):
                 self.latent = None
                 with inference_context, torch.amp.autocast("cuda", enabled=amp_enabled):
                     results_mc = self.model(inputs) # Model calculates E and F

                 # Store results, check if forces key exists
                 energies_list.append(results_mc["energy"])
                 if "forces" in results_mc:
                     forces_tmp_list.append(results_mc["forces"])
                 else:
                     forces_tmp_list.append(None)
                     print("Warning: 'forces' key not found in MC pass results.")

                 if self.latent is not None: latent_tmp_list.append(self.latent.clone())
                 else: latent_tmp_list.append(None)

            # Process results
            if energies_list:
                 energies = torch.stack(energies_list).float()
                 E_ml = energies.mean().item()
                 sigma_energy = energies.std(dim=0, unbiased=True).item() if len(energies_list) > 1 else 0.0
            valid_forces = [f for f in forces_tmp_list if f is not None and torch.is_tensor(f)]
            if valid_forces:
                 forces_stack = torch.stack(valid_forces).float()
                 F_ml = np.mean(forces_stack.detach().cpu().numpy(), axis=0).astype(np.float64)
            else:
                 F_ml = np.full((len(atoms), 3), np.nan) # Indicate failure if no forces collected
                 print("Error: No valid forces collected during dropout.")
            valid_latents = [l for l in latent_tmp_list if l is not None and torch.is_tensor(l)]
            if valid_latents:
                 latent_per_atom_tensor = torch.stack(valid_latents)
                 latent_per_atom_avg = torch.mean(latent_per_atom_tensor, dim=0)
                 latent_per_atom_avg_np = latent_per_atom_avg.cpu().numpy().astype(np.float32)
                 latent_frame = np.sum(latent_per_atom_avg_np, axis=0).astype(np.float64)
            else:
                 latent_frame = np.array([])
                 latent_per_atom_avg_np = None

        else:  # Standard Inference (No Dropout)
            self.model.eval()
        
            # --- Force FP64 inference but KEEP autograd enabled (required for forces) ---
            # Convert model parameters to double precision (in-place). This ensures forward
            # is executed in FP64 and avoids float32 quantization of large energies.
            try:
                self.model = self.model.double()
            except Exception:
                # Fallback: convert parameters individually if .double() is not supported
                for p in self.model.parameters():
                    p.data = p.data.to(torch.float64)
                    if p.grad is not None:
                        p.grad.data = p.grad.data.to(torch.float64)
        
            # Ensure all floating inputs are float64 and on the model device
            param = next(self.model.parameters())
            model_device = param.device
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(device=model_device, dtype=torch.float64)
        
            # Make sure the positions tensor requires gradients (needed for autograd-based forces)
            if position_key_found and torch.is_tensor(inputs[position_key_found]):
                inputs[position_key_found].requires_grad_(True)
        
            # Do forward WITHOUT autocast; gradients must be enabled for force computation.
            # NOTE: do NOT use torch.no_grad() here because the model uses autograd internally.
            results = self.model(inputs)   # model computes energy + autograd-based forces
        
            # --- Extract predictions preserving FP64 precision ---
            # energy: keep a numpy float64 array representation
            energy_tensor = results["energy"].detach().cpu()
            # Convert to 0-d or 1-d numpy array (we keep the array for UQ precision)
            energy_np = np.asarray(energy_tensor.numpy(), dtype=np.float64)
            # Also create a python scalar for legacy logging/formatting compatibility
            # (safe conversion only if energy_np is scalar-like)
            try:
                energy_scalar = float(np.ravel(energy_np)[0])
            except Exception:
                energy_scalar = float(energy_np.item()) if hasattr(energy_np, "item") else float(energy_np.tolist()[0])
        
            # forces: detach to CPU and keep float64
            if "forces" in results:
                F_ml = results["forces"].detach().cpu().numpy().astype(np.float64)
            else:
                F_ml = np.full((len(atoms), 3), np.nan)
                print("Error: 'forces' key not found in model output during standard inference.")
        
            # latent handling unchanged, just keep types consistent
            if self.latent is not None:
                latent_per_atom_avg_np = self.latent.cpu().numpy().astype(np.float32)
                latent_frame = np.sum(latent_per_atom_avg_np, axis=0).astype(np.float64)
            else:
                latent_frame = np.array([])
                latent_per_atom_avg_np = None
        
            # Use both high-precision array and scalar for compatibility:
            E_ml = energy_np       # high-precision numpy array (for UQ)
            E_ml_scalar = energy_scalar  # python float for logs/backwards-compatible uses

            # forces: detach to CPU and keep float64
            if "forces" in results:
                F_ml = results["forces"].detach().cpu().numpy().astype(np.float64)
            else:
                F_ml = np.full((len(atoms), 3), np.nan)
                print("Error: 'forces' key not found in model output during standard inference.")

            sigma_energy = 0.0
            if self.latent is not None:
                latent_per_atom_avg_np = self.latent.cpu().numpy().astype(np.float32)
                latent_frame = np.sum(latent_per_atom_avg_np, axis=0).astype(np.float64)
            else:
                latent_frame = np.array([])
                latent_per_atom_avg_np = None

        # Calculate ML time AFTER inference completes
        ml_time = time.time() - ml_eval_start_time
        # --- End Model Inference ---

        # --- Add Atom Refs ---
        E_ref = 0.0
        if self.apply_atom_offsets and self.atomrefs is not None:
            try:
                 atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), device=self.device, dtype=torch.long)
                 E_ref = self.atomrefs[atomic_numbers].sum().float().item()
            except IndexError as e: print(f"Warning: Atom numbers mismatch with atomrefs: {e}.")
            except Exception as e: print(f"Warning: Error applying atomrefs: {e}.")
        # --- End Atom Refs ---

        # --- Add Coulomb ---
        # Use the Coulomb function defined in this module
        E_coul, F_coul, coul_time = custom_coulomb_pytorch(
            atoms, self.include_electrostatic, self.coulomb_prefactor
        )
        F_coul = np.asarray(F_coul, dtype=np.float64)
        # --- End Coulomb ---

        # --- Store Results ---
        # Store both a high-precision numpy energy array (for UQ) and a scalar for legacy code/logging.
        # E_ml is the numpy array; E_ml_scalar is the python float.
        energy_total_np = np.asarray(E_ml, dtype=np.float64) + np.float64(E_ref) + np.float64(E_coul)
        energy_total_scalar = float(np.ravel(energy_total_np)[0])
        
        self.results["energy_np"] = energy_total_np            # high-precision array (use this for UQ)
        self.results["energy"] = energy_total_scalar           # legacy scalar for logs and formatting
        self.results["forces"] = (np.asarray(F_ml, dtype=np.float64) + np.asarray(F_coul, dtype=np.float64))
        self.results["ml_time"] = ml_time
        self.results["coul_fn_time"] = coul_time
        self.results["calc_total_time"] = time.time() - calc_start
        self.results["sigma_energy"] = sigma_energy
        self.results["latent"] = latent_frame
        self.results["per_atom_latent"] = latent_per_atom_avg_np
        self.results["E_ml_avg"] = energy_total_np             # keep array average for downstream UQ if needed
        self.results["E_ref"] = E_ref
        self.results["coul_fn_energy"] = E_coul


    def __del__(self):
        """Ensures that the hook handle is removed upon deletion."""
        if self.hook_handle:
            try:
                self.hook_handle.remove()
            except Exception:
                pass # Ignore errors during cleanup

# --- Model Evaluation Function (Minimal Fixes) ---
# --- safe scalarizer helper (place inside evaluate_model or module top) ---
def _to_scalar(x):
    """Return a Python float from x if possible.
    Works for Python numbers, numpy scalars/0-d arrays, 1-D arrays (takes first element),
    and torch tensors (takes item() if scalar or ravel()[0] if array-like).
    """
    import torch
    try:
        # torch tensors
        if isinstance(x, torch.Tensor):
            if x.numel() == 0:
                return float("nan")
            if x.numel() == 1:
                return float(x.item())
            return float(x.detach().cpu().numpy().ravel()[0])
        # numpy and array-like
        a = np.asarray(x)
        if a.size == 0:
            return float("nan")
        # if it's already a scalar numpy type this cast will work
        return float(a.ravel()[0])
    except Exception:
        try:
            return float(x)
        except Exception:
            return float("nan")


def evaluate_model(frames, true_energies, true_forces, model_obj, device, batch_size,
                   uq_method_flag, n_mc, eval_log_file, unique_mc_log_file, config, neighbor_list):
    """
    Evaluates a pre-loaded model on a set of frames with support for uncertainty quantification (UQ).
    """
    print(f"\n--- Evaluating Model ({uq_method_flag} mode) ---")
    try:
        # Correct instantiation using config
        calc = MyTorchCalculator(model_obj, device, neighbor_list, config)
    except Exception as e_calc:
        print(f"Error initializing MyTorchCalculator: {e_calc}")
        traceback.print_exc()
        return None, None, None, None, None  # Indicate failure

    # ------------------------------------------------------------------
    # SAFETY PATCH for PaiNN checkpoints trained with older schnetpack
    import torch.nn as nn

    rep = getattr(calc.model, "representation", None)
    if rep is not None and not hasattr(rep, "electronic_embeddings"):
        rep.electronic_embeddings = nn.ModuleList([])
    # ------------------------------------------------------------------

    # Configure calculator for UQ
    if uq_method_flag == "dropout":
        if n_mc is None or n_mc <= 1:
            n_mc = 10
        print(f"Using Dropout mode with n_mc = {n_mc}")
        calc.use_dropout = True
        calc.n_mc = n_mc
    else:
        calc.use_dropout = False
        calc.n_mc = 1

    n_frames = len(frames)
    all_energy_pred = []
    all_forces_pred = []
    all_latent_pred = []          # List for frame-averaged latent vectors
    all_per_atom_latent_pred = [] # List for per-atom latent vectors
    all_sigma_energy = []         # List for energy std dev (if dropout)
    cum_eval_time = 0.0
    batches_processed = 0

    print(f"Starting evaluation for {n_frames} frames with batch size {batch_size}...")
    log_header_written = False

    for batch_start in range(0, n_frames, batch_size):
        batch_indices = range(batch_start, min(batch_start + batch_size, n_frames))
        batch_frames = [frames[i] for i in batch_indices]
        batch_size_actual = len(batch_frames)
        if batch_size_actual == 0: continue

        batch_start_time = time.time()
        batch_energies = []
        batch_forces = []
        batch_latents = []             # Batch frame-averaged latents
        batch_per_atom_latents = []    # Batch per-atom latents
        batch_sigmas = []              # Batch energy sigmas

        calc_successful = True
        try:
            for i, frame in enumerate(batch_frames):
                frame_idx_global = batch_start + i

                # !!! No need to assign charges here, calc.calculate() handles it !!!
                # if calc.include_electrostatic and calc.atomic_charges_dict and not frame.has("initial_charges"):
                #     try: frame.set_initial_charges(assign_charges(frame, calc.atomic_charges_dict))
                #     except Exception as e_ch: print(f"Warn Frame {frame_idx_global}: Charge assign failed: {e_ch}")

                # Call calculate - properties request includes latent
                calc.calculate(atoms=frame, properties=["energy", "forces", "latent", "per_atom_latent"])
                results = calc.results # Get results dictionary from calculator

                # Append results, using .get for safety
                batch_energies.append(_to_scalar(results.get("energy", np.nan)))
                batch_forces.append(results.get("forces", np.full((len(frame), 3), np.nan)))
                batch_latents.append(results.get("latent")) # Frame average latent (could be None or np.array)
                batch_per_atom_latents.append(results.get("per_atom_latent")) # Per-atom latent (could be None or np.array)
                batch_sigmas.append(_to_scalar(results.get("sigma_energy", 0.0)))

                # Logging (using results directly) - safe scalarized logging
                if frame_idx_global == 0 or frame_idx_global == n_frames - 1:
                    # true energy (coerce to float if possible)
                    true_e_val = float(true_energies[frame_idx_global]) if frame_idx_global < len(true_energies) else np.nan

                    # Prefer the scalar 'energy' for legacy logging; if missing, try energy_np
                    pred_raw = results.get("energy", None)
                    if pred_raw is None:
                        pred_raw = results.get("energy_np", np.nan)

                    # Safe conversion of prediction and other numeric fields
                    pred_e_val = _to_scalar(pred_raw)
                    e_ml_raw = results.get("E_ml_avg", np.nan)
                    e_coul_raw = results.get("coul_fn_energy", np.nan)
                    step_time_raw = results.get("calc_total_time", 0.0)
                    ml_time_raw = results.get("ml_time", 0.0)
                    coul_time_raw = results.get("coul_fn_time", 0.0)

                    # Scalarize everything before formatting
                    e_ml = _to_scalar(e_ml_raw)
                    e_coul = _to_scalar(e_coul_raw)
                    step_time = _to_scalar(step_time_raw)
                    ml_time = _to_scalar(ml_time_raw)
                    coul_time = _to_scalar(coul_time_raw)

                    energy_diff = pred_e_val - true_e_val if not (np.isnan(pred_e_val) or np.isnan(true_e_val)) else np.nan

                    try:
                        if not log_header_written and eval_log_file:
                            header = (
                                f"{'Frame':>6s} | {'True Energy (eV)':>18s} | {'Pred Energy (eV)':>18s} | "
                                f"{'Energy Diff (eV)':>15s} | {'E_ML_avg (eV)':>15s} | {'E_Coul (eV)':>12s} | "
                                f"{'MLtime(s)':>10s} | {'CoulTime(s)':>10s} | {'StepTime(s)':>10s} | {'CumTime(s)':>10s}\n"
                            )
                            with open(eval_log_file, "a") as elog_h:
                                elog_h.write(header)
                            log_header_written = True

                        if eval_log_file:
                            with open(eval_log_file, "a") as elog:
                                elog.write(
                                    f"{frame_idx_global:6d} | {true_e_val:18.6f} | {pred_e_val:18.6f} | "
                                    f"{energy_diff:15.6f} | {e_ml:15.6f} | {e_coul:12.6f} | "
                                    f"{ml_time:10.4f} | {coul_time:10.4f} | "
                                    f"{step_time:10.4f} | {cum_eval_time + (time.time()-batch_start_time):10.4f}\n"
                                )
                    except IOError as log_e:
                        print(f"Warn: Log failed frame {frame_idx_global}: {log_e}")


        except Exception as e_batch_calc:
            print(f"Error calculating batch starting {batch_start}: {e_batch_calc}")
            traceback.print_exc()
            calc_successful = False
            continue # Skip rest of batch processing

        if calc_successful:
            all_energy_pred.extend(batch_energies)
            all_forces_pred.extend(batch_forces)
            all_latent_pred.extend(batch_latents)
            all_per_atom_latent_pred.extend(batch_per_atom_latents)
            all_sigma_energy.extend(batch_sigmas)

        batch_time = time.time() - batch_start_time
        cum_eval_time += batch_time
        batches_processed += 1

        if batches_processed % 10 == 0 or batches_processed == 1:
            total_batches = (n_frames + batch_size - 1) // batch_size
            print(f"  Processed batch {batches_processed}/{total_batches}...")

    # --- FINAL SUMMARY ---
    print("\n--- Evaluation Summary ---")
    print(f"Processed {n_frames} frames in {batches_processed} batches.")
    print(f"Total Eval Time: {cum_eval_time:.3f}s")
    if n_frames > 0: print(f"Avg Time / Frame: {(cum_eval_time / n_frames):.5f}s")
    print("--------------------------\n")

    # Return all collected data
    sigma_energy_array = np.array(all_sigma_energy) if uq_method_flag == "dropout" else None
    return all_energy_pred, all_forces_pred, all_latent_pred, all_per_atom_latent_pred, sigma_energy_array
