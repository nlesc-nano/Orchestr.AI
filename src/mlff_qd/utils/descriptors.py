# mlff_qd/utils/descriptors.py
import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP

def compute_local_descriptors(positions, atom_types, soap):
    print("[SOAP] Computing descriptors...")
    desc=[]
    for i in range(positions.shape[0]):
        if i%100==0: print(f" SOAP frame {i+1}/{positions.shape[0]}")
        A = Atoms(symbols=atom_types, positions=positions[i], pbc=False)
        S = soap.create(A).mean(axis=0)
        desc.append(S)
    desc=np.array(desc)
    print(f"[SOAP] Done, shape={desc.shape}")
    return desc