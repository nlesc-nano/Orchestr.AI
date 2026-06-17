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
import torch.serialization
torch.serialization.add_safe_globals([slice])
import torch._inductor.codecache
from ase.io import read

from orchestr_ai.utils.helpers import load_config
from orchestr_ai.utils.env_dispatch import maybe_dispatch_to_engine_env
from orchestr_ai.utils.engine_profiles import (
    detect_engine_from_config,
    get_default_engine_profiles,
    is_schnetpack_engine,
    model_framework_from_engine,
)

def _setup_logging_and_output() -> None:
    sys.path.insert(0, os.getcwd())

    try:
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                handlers=[logging.StreamHandler(sys.stdout)],
            )

        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("pycaret").setLevel(logging.WARNING)
        logging.getLogger("ase").propagate = False

        sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
        sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

    except Exception as e:
        print(f"Warning: Could not set unbuffered output/logging: {e}")

def main():
    """
    Main function to execute MLFF-based simulations or evaluation tasks.
    Dispatch must happen before importing SchNetPack-dependent modules.
    
    Steps:
    - Load configuration from "config.yaml"
    - Read the initial structure from file
    - Load and set up the ML model and neighbor list
    - Execute the task specified by 'run_type': MD, GEO_OPT, VIB, or EVAL
    """
    _setup_logging_and_output()

    logging.info("--- Starting MLFF Simulation/Evaluation ---")

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)

    if config is None:
        logging.error("Exiting due to configuration error.")
        sys.exit(1)

    # Detect engine/platform early from config.
    try:
        engine = detect_engine_from_config(config)
    except Exception as e:
        logging.error(f"Could not detect engine/platform: {e}")
        sys.exit(1)

    # Normalize model_framework for downstream postprocessing.
    # Old configs can still use model_framework directly.
    config.setdefault("model_framework", model_framework_from_engine(engine))

    framework = config.get("model_framework", "schnetpack").lower()
    if framework == "allegro":
        framework = "nequip"
        config["model_framework"] = "nequip"

    # Dispatch before importing calculator/evaluate/simulation.
    maybe_dispatch_to_engine_env(
        engine=engine,
        module="orchestr_ai.postprocessing",
        engine_to_profile=get_default_engine_profiles(),
    )

    run_type = config.get("run_type", "MD").upper()
    logging.info(f"Run type selected: {run_type}")
    logging.info(f"Detected engine/platform: {engine}")
    logging.info(f"Detected model framework: {config.get('model_framework')}")

    # Lazy import evaluation only when needed.
    if run_type == "EVAL":
        try:
            from orchestr_ai.postprocessing.evaluate import run_eval

            run_eval(config)
        except Exception as e:
            logging.error("\n--- An error occurred during Evaluation ---")
            logging.error(f"{type(e).__name__}: {e}")
            logging.error(traceback.format_exc())
            logging.error("--- Evaluation Run Failed ---")
            sys.exit(1)
        return

    # Lazy import simulation only after dispatch.
    from orchestr_ai.postprocessing.simulation import (
        run_md,
        run_geo_opt,
        run_vibrational_analysis,
    )

    initial_xyz = config.get("initial_xyz")
    if not initial_xyz or not os.path.exists(initial_xyz):
        logging.error(f"Initial structure file '{initial_xyz}' not found or specified.")
        sys.exit(1)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")


    try:
        logging.info(f"Loading ML model from {model_path}...")

        if framework == "nequip":
            # NequIP/Allegro ASE calculator loads from compiled model path.
            model_obj = model_path
            logging.info("NequIP/Allegro model path passed to calculator.")
        else:
            best = torch.load(model_path, map_location=device, weights_only=False)
            # Determine precision dynamically (default to float32)
            default_dtype = config.get("default_dtype") or config.get("default_precision") or "float32"
            torch_dtype = torch.float64 if default_dtype == "float64" else torch.float32
            
            try:
                best = best.to(device=device, dtype=torch_dtype)
                logging.info(f"Successfully cast model parameters to {torch_dtype}")
            except Exception as e:
                logging.warning(f"Could not cast model to {torch_dtype}: {e}. Falling back to default cast.")
                best = best.to(device=device)

            if hasattr(best, "postprocessors"):
                try:
                    from torch import nn

                    filtered = [
                        pp for pp in getattr(best, "postprocessors")
                        if pp is not None
                    ]
                    setattr(best, "postprocessors", nn.ModuleList(filtered))
                except Exception:
                    pass

            if hasattr(best, "do_postprocessing"):
                best.do_postprocessing = True

            best.eval()
            model_obj = best
            logging.info("Model loaded and moved to device successfully.")

    except Exception as e:
        logging.error(f"Error loading model {model_path}: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)

    # SchNetPack-only neighbor list.
    neighbor_list = None
    if is_schnetpack_engine(engine):
        from orchestr_ai.postprocessing.neighbor_list import setup_neighbor_list

        neighbor_list = setup_neighbor_list(config)
        logging.info("SchNetPack neighbor list initialized.")
    else:
        logging.info(
            f"No SchNetPack neighbor list required for engine '{engine}' "
            f"with framework '{framework}'."
        )

    try:
        if run_type == "MD":
            run_md(atoms, model_obj, device, config, neighbor_list=neighbor_list)
        elif run_type == "GEO_OPT":
            run_geo_opt(atoms, model_obj, device, config, neighbor_list=neighbor_list)
        elif run_type == "VIB":
            run_vibrational_analysis(
                atoms,
                model_obj,
                device,
                config,
                neighbor_list=neighbor_list,
            )
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



