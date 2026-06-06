"""
schnetpack_ase.py

Contains the ASE calculator for SchNetPack models used in Orchestr.AI.
"""

import numpy as np
import torch
from ase.calculators.calculator import all_changes
from schnetpack.interfaces import SpkCalculator

class LegacyOffsetSpkCalculator(SpkCalculator):
    def __init__(self, model_obj, **kwargs):
        self.mean_offset = 0.0
        self.atomref = None
        print("--- Attempting to surgically extract 'mean' and 'atomref' offsets ---")
        try:
            if hasattr(model_obj, 'postprocessors'):
                for pp in model_obj.postprocessors:
                    add_mean_enabled = bool(getattr(pp, "add_mean", False))
                    add_atomrefs_enabled = bool(getattr(pp, "add_atomrefs", True))
                    
                    # 1. Aggressively extract mean
                    extracted_mean = 0.0
                    if hasattr(pp, 'state_dict') and 'mean' in pp.state_dict():
                        extracted_mean = pp.state_dict()['mean'].item()
                    elif hasattr(pp, 'mean') and isinstance(getattr(pp, 'mean'), torch.Tensor):
                        extracted_mean = getattr(pp, 'mean').item()
                    
                    # 2. Flag and process the mean
                    if add_mean_enabled and abs(extracted_mean) > 1e-8:
                        self.mean_offset = extracted_mean
                        print(f"\n⚠️  FLAG: Non-zero dataset mean offset detected: {self.mean_offset:.6f} eV/atom")
                        print("    -> The model was trained with 'remove_mean: true'.")
                        print("    -> For optimal transferability, consider training future")
                        print("       models with 'remove_mean: false'.\n")
                    elif abs(extracted_mean) > 1e-8:
                        self.mean_offset = 0.0
                        print("\nFLAG: Stored dataset mean detected, but AddOffsets.add_mean is false.")
                        print("    -> Not applying a size-extensive mean offset during postprocessing.\n")
                    else:
                        self.mean_offset = 0.0
                        print("\n✅ FLAG: Mean offset is 0.0 (Trained with 'remove_mean: false').")
                        print("    -> Model relies purely on isolated atomic energies.\n")
                        
                    # 3. Extract atomic references
                    if add_atomrefs_enabled:
                        for ref_name in ['atomref', 'z_offsets']:
                            if hasattr(pp, ref_name) and getattr(pp, ref_name) is not None:
                                ref_val = getattr(pp, ref_name)
                                if isinstance(ref_val, torch.Tensor):
                                    self.atomref = ref_val.detach().cpu().numpy().astype(np.float64).flatten()
                                elif hasattr(ref_val, 'weight'):
                                    self.atomref = ref_val.weight.detach().cpu().numpy().astype(np.float64).flatten()
                                print(f"Successfully extracted '{ref_name}' (isolated atomic energies).")
                            
                # Disable internal postprocessors 
                model_obj.postprocessors = torch.nn.ModuleList([])
                print("Successfully disabled model's internal postprocessors.")
        except Exception as e:
             print(f"WARNING: Surgical extraction failed: {e}")
        
        super().__init__(model=model_obj, **kwargs)

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        
        # Apply the surgical offset fix on-the-fly in float64
        if 'energy' in self.results:
            total_offset = 0.0
            
            if self.atomref is not None:
                Z = self.atoms.numbers
                valid_Z = np.clip(Z, 0, len(self.atomref) - 1)
                total_offset += np.sum(self.atomref[valid_Z])
                
            if self.mean_offset != 0.0:
                total_offset += self.mean_offset * len(self.atoms)
                
            self.results['energy'] += total_offset
            
            if 'E_ml_avg' in self.results:
                self.results['E_ml_avg'] += total_offset

        # Clear model_results to release the PyTorch autograd graph from memory
        self.model_results = None
