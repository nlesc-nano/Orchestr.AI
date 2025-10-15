import argparse
import os
import yaml
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import numpy as np
import logging
logger = logging.getLogger(__name__)

def load_config_preproc(config_file=None):
    # If no config file is specified, use a default path relative to this script.
    if config_file is None:
        # Adjust parents[...] as necessary based on your directory structure
        default_path = "preprocess_config.yaml"
        config_file = str(default_path)  # convert Path object to string if needed

    try:
        # Load user-defined configuration
        with open(config_file, "r") as file:
            user_config = yaml.safe_load(file)
            if user_config is None:
                user_config = {}
    except FileNotFoundError:
        logger.warning(f"Configuration file '{config_file}' not found. Using only default settings.")
        user_config = {}

    return user_config

def load_config(config_file="config.yaml"):
    """
    Loads configuration from a YAML file.

    Parameters:
        config_file (str): Path to the YAML configuration file (default: "config.yaml").

    Returns:
        dict or None: Configuration dictionary if loaded successfully, or None if there was an error.
    """
    if not os.path.exists(config_file):
        logger.warning(f"Error: Configuration file '{config_file}' not found.")
        return None

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded successfully from {config_file}.")
        return config
    except Exception as e:
        logger.warning(f"Error loading configuration from '{config_file}': {e}")
        return None

def parse_args():
    parser = argparse.ArgumentParser(description="Run Machine Learning Force Field Training with SchNetPack")
    parser.add_argument("--config", type=str, default="input.yaml", help="Path to the configuration YAML file")
    args = parser.parse_args()
    return args

def get_optimizer_class(name):
    optimizers_map = {
        'Adam':  optim.Adam,
        'AdamW': optim.AdamW,
        'SGD':   optim.SGD
    }
    if name not in optimizers_map:
        raise ValueError(f"Unsupported optimizer '{name}'. Available: {list(optimizers_map.keys())}")
    return optimizers_map[name]

def get_scheduler_class(name):
    schedulers_map = {
        'ReduceLROnPlateau': lr_sched.ReduceLROnPlateau,
        'StepLR':            lr_sched.StepLR
    }
    if name not in schedulers_map:
        raise ValueError(f"Unsupported scheduler '{name}'. Available: {list(schedulers_map.keys())}")
    return schedulers_map[name]

def analyze_fluctuations(energies, forces):
    """
    Analyze fluctuation to suggest loss weights.
    """
    energy_fluct = np.std(energies)
    force_fluct  = np.std(forces)
    wE_raw = force_fluct / energy_fluct
    wF_raw = 1.0
    S = wE_raw + wF_raw
    wE_norm = wE_raw/S
    wF_norm = wF_raw/S

    logger.info(f"[Stats] σ(E)={energy_fluct:.4f}, σ(F)={force_fluct:.4f}")
    logger.info(f"[Weights] raw: E={wE_raw:.4f}, F={wF_raw:.1f}; norm: E={wE_norm:.4f}, F={wF_norm:.4f}")
    return {'raw':{'energy':wE_raw,'forces':wF_raw},
            'normalized':{'energy':wE_norm,'forces':wF_norm}}
            

def analyze_reference_forces(forces, atom_types):
    """
    Return dict of per-atom, per-frame, overall force stats.
    """
    fm = np.linalg.norm(forces, axis=2)

    per_atom_mean  = fm.mean(axis=0)
    per_atom_std   = fm.std(axis=0)
    per_atom_rng   = np.ptp(fm, axis=0)
    per_frame_mean = fm.mean(axis=1)
    per_frame_std  = fm.std(axis=1)
    per_frame_rng  = np.ptp(fm, axis=1)
    summary = {
        'per_atom_means':  per_atom_mean,
        'per_atom_stds':   per_atom_std,
        'per_atom_ranges': per_atom_rng,
        'per_frame_means': per_frame_mean,
        'per_frame_stds':  per_frame_std,
        'per_frame_ranges': per_frame_rng,
        'overall_mean':   fm.mean(),
        'overall_std':    fm.std(),
        'overall_range':  np.ptp(fm)      
    }
    # per-type
    for t in set(atom_types):
        idxs = [i for i, a in enumerate(atom_types) if a == t]
        arr  = fm[:, idxs]
        summary.setdefault('atom_type_means', {})[t]   = arr.mean()
        summary.setdefault('atom_type_stds',  {})[t]   = arr.std()
        summary.setdefault('atom_type_ranges', {})[t]  = np.ptp(arr)  # ← changed
    return summary

def suggest_thresholds(force_stats, std_fraction=0.1, range_fraction=0.1):
    overall_std   = force_stats['overall_std']
    overall_rng   = force_stats['overall_range']
    thr_std   = std_fraction  * overall_std
    thr_range = range_fraction* overall_rng
    logger.info(f"[THR] Std thr={thr_std:.4f}, Range thr={thr_range:.4f}")
    per_type={}
    for t in force_stats['atom_type_stds']:
        ts = force_stats['atom_type_stds'][t]*std_fraction
        tr = force_stats['atom_type_ranges'][t]*range_fraction
        per_type[t]={'std_thr':ts,'range_thr':tr}
        logger.info(f" {t}: std_thr={ts:.4f}, range_thr={tr:.4f}")
    return {'overall':{'std_thr':thr_std,'range_thr':thr_range},
            'per_type':per_type}