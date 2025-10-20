# mlff_qd/utils/descriptors.py
import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP

import logging
logger = logging.getLogger(__name__)


def get_species(atom_types):
    """
    Return a deterministic list of unique species from 'atom_types' (sequence of symbols).
    Using sorted() for stable, reproducible order across runs.
    """
    return sorted(set(atom_types))

def init_soap(species,
              r_cut: float = 12.0,
              n_max: int = 7,
              l_max: int = 3,
              sigma: float = 0.1,
              periodic: bool = False,
              sparse: bool = False):
    """
    Build a SOAP descriptor object with sane defaults.
    """
    return SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        periodic=periodic,
        sparse=sparse,
    )
    
def compute_local_descriptors(positions, atom_types, soap_params: dict | None = None):
    """
    positions: (N, A, 3)
    atom_types: sequence of length A (symbols)
    soap_params: dict passed to dscribe.descriptors.SOAP(**soap_params)
    """
    if soap_params is not None:
        species = soap_params.get("species", get_species(atom_types))
        logger.info(f"[SOAP] Initializing with explicit params; species={species}")
        soap = SOAP(**{**{"species": species}, **soap_params})
        pbc = bool(soap_params.get("periodic", False))
    else:
        species = get_species(atom_types)
        logger.info(f"[SOAP] Initializing , Auto species={species}")
        soap = init_soap(species)
        pbc = False  
    
    logger.info("[SOAP] Computing descriptors...")
    desc=[]
    
    for i in range(positions.shape[0]):
        if i%100==0: 
            logger.info(f" SOAP frame {i+1}/{positions.shape[0]}")
        A = Atoms(symbols=atom_types, positions=positions[i], pbc=False)
        S = soap.create(A).mean(axis=0)
        desc.append(S)
        
    desc=np.array(desc)
    logger.info(f"[SOAP] Done, shape={desc.shape}")
    return desc