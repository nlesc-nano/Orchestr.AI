"""
run_md_opt.py

Entry point for running MLFF-based simulations or evaluation tasks.
This script loads the configuration, initializes logging and output buffering,
reads the initial atomic structure, loads the ML model, and executes the specified task:
MD, geometry optimization, vibrational analysis, or evaluation.
"""

import os
import sys
import logging
import traceback
import torch
from ase.io import read

from mlff_qd.postprocessing.calculator import setup_neighbor_list, assign_charges
from mlff_qd.postprocessing.simulation import run_md, run_geo_opt, run_vibrational_analysis
from mlff_qd.postprocessing.evaluate import run_eval
from mlff_qd.utils.helpers import load_config

# === Setup Logging and Unbuffered Output ===
sys.path.insert(0, os.getcwd())
try:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    # Reduce verbosity for specific libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('pycaret').setLevel(logging.WARNING)
    logging.getLogger('ase').propagate = False

    # Set stdout and stderr to be unbuffered
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)
except Exception as e:
    print(f"Warning: Could not set unbuffered output/logging: {e}")


def main():
    """
    Main function to execute MLFF-based simulations or evaluation tasks.
    
    Steps:
    - Load configuration from "config.yaml"
    - Read the initial structure from file
    - Load and set up the ML model and neighbor list
    - Optionally assign initial charges to atoms
    - Execute the task specified by 'run_type': MD, GEO_OPT, VIB, or EVAL
    """
    logging.info("--- Starting MLFF Simulation/Evaluation ---")

    # --- Load Configuration ---
    global config  # Global usage assumed by evaluate.py if needed
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)

    if config is None:
        logging.error("Exiting due to configuration error.")
        sys.exit(1)

    run_type = config.get("run_type", "MD").upper()
    logging.info(f"Run type selected: {run_type}")

    # === Evaluation Run ===
    if run_type == "EVAL":
        try:
            run_eval(config)
        except Exception as e:
            logging.error("\n--- An error occurred during Evaluation ---")
            logging.error(f"{type(e).__name__}: {e}")
            logging.error(traceback.format_exc())
            logging.error("--- Evaluation Run Failed ---")
            sys.exit(1)
        return


    # === Simulation Setup ===
    # Load the initial structure file
    initial_xyz = config.get("initial_xyz")
    if not initial_xyz or not os.path.exists(initial_xyz):
        logging.error(f"Initial structure file '{initial_xyz}' not found or specified.")
        sys.exit(1)

    # Load the ML model path
    model_path = config.get("model_path")
    if not model_path or not os.path.exists(model_path):
        logging.error(f"ML Model path '{model_path}' not found or specified.")
        sys.exit(1)

    try:
        atoms = read(initial_xyz)
        logging.info(f"Read initial structure: {len(atoms)} atoms from {initial_xyz}")
    except Exception as e:
        logging.error(f"Error reading {initial_xyz}: {e}")
        sys.exit(1)

    # Optionally assign initial charges if electrostatics are enabled
    if config.get("include_electrostatic", True):
        charges_dict = config.get("atomic_charges")
        if charges_dict:
            try:
                charges = assign_charges(atoms, charges_dict)
                atoms.set_initial_charges(charges)
                logging.info("Assigned initial charges.")
            except Exception as e:
                logging.error(f"Error assigning charges: {e}")
                sys.exit(1)

    # Set up the ML model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    try:
        logging.info(f"Loading ML model from {model_path}...")
        model_obj = torch.load(model_path, map_location=device)
        model_obj = model_obj.to(torch.float32)
        model_obj.eval()
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model {model_path}: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)

    # Set up the neighbor list based on configuration
    neighbor_list = setup_neighbor_list(config)

    # === Run Simulation ===
    try:
        if run_type == "MD":
            run_md(atoms, model_obj, device, neighbor_list, config)
        elif run_type == "GEO_OPT":
            run_geo_opt(atoms, model_obj, device, neighbor_list, config)
        elif run_type == "VIB":
            run_vibrational_analysis(atoms, model_obj, device, neighbor_list, config)
        else:
            logging.error(f"Invalid run_type '{run_type}'.")
            sys.exit(1)
    except Exception as e:
        logging.error(f"\n--- An error occurred during {run_type} Simulation ---")
        logging.error(f"{type(e).__name__}: {e}")
        logging.error(traceback.format_exc())
        logging.error(f"--- {run_type} Run Failed ---")
        sys.exit(1)

    logging.info("\n--- Script Finished ---")


# === Entry Point ===
if __name__ == "__main__":
    config = None
    main()



