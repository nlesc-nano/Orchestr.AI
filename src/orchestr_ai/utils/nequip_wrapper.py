import yaml
import logging
import tempfile
import os
import sys
import numpy as np
from ase.io import read, write

def center_spin_targets_xyz(xyz_path: str) -> str:
    """
    Mean-center E_singlet and E_triplet in an extxyz file in-place.

    Returns:
        str: same xyz_path after centering
    """
    frames = read(xyz_path, index=":")
    if len(frames) == 0:
        raise ValueError(f"No frames found in {xyz_path}")

    singlet_vals = []
    triplet_vals = []

    for atoms in frames:
        if "E_singlet" not in atoms.info or "E_triplet" not in atoms.info:
            raise KeyError(
                f"Expected E_singlet and E_triplet in atoms.info for all frames in {xyz_path}"
            )
        singlet_vals.append(float(atoms.info["E_singlet"]))
        triplet_vals.append(float(atoms.info["E_triplet"]))

    mean_s = float(np.mean(singlet_vals))
    mean_t = float(np.mean(triplet_vals))

    logging.info(
        f"[NequIP Wrapper] Centering energies: "
        f"E_singlet mean={mean_s:.10f}, E_triplet mean={mean_t:.10f}"
    )

    for atoms in frames:
        atoms.info["E_singlet"] = float(atoms.info["E_singlet"]) - mean_s
        atoms.info["E_triplet"] = float(atoms.info["E_triplet"]) - mean_t

    write(xyz_path, frames, format="extxyz")
    logging.info(f"[NequIP Wrapper] Wrote mean-centered targets back to {xyz_path}")

    return xyz_path

def run_nequip_training(config_path):
    """
    Run NequIP or Allegro training with the specified config file using the latest NequIP version with Hydra.
    
    Args:
        config_path (str): Path to the NequIP or Allegro YAML config file.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        
        if "optimizer_betas" in config and isinstance(config["optimizer_betas"], list):
            config["optimizer_betas"] = tuple(config["optimizer_betas"])
        
        # if "data" in config and "split_dataset" in config["data"] and "input_xyz_file" in config:
        #     from orchestr_ai.utils.data_conversion import preprocess_data_for_platform
        #     converted_file = preprocess_data_for_platform(config["input_xyz_file"], "nequip" if config.get("model", {}).get("_target_") != "allegro.model.AllegroModel" else "allegro")
        #     config["data"]["split_dataset"]["file_path"] = converted_file

        # if "data" in config and "split_dataset" in config["data"] or "input_xyz_file" in config:
        if "data" in config and ("split_dataset" in config["data"] or "input_xyz_file" in config):
            from orchestr_ai.utils.data_conversion import preprocess_data_for_platform

            platform_name = (
                "nequip"
                if config.get("model", {}).get("_target_") != "allegro.model.AllegroModel"
                else "allegro"
            )

            source_file = config["data"]["split_dataset"]["file_path"]
            converted_file = preprocess_data_for_platform(source_file, platform_name)


            # If this is the custom spin-combined dataset, mean-center energies
            try:
                center_spin_targets_xyz(converted_file)
            except Exception as e:
                logging.warning(
                    f"[NequIP Wrapper] Skipping target centering for {converted_file}: {e}"
                )

            config["data"]["split_dataset"]["file_path"] = converted_file
        
        # Lazy import: only required when actually running NequIP
        from nequip.scripts.train import main as nequip_train_main

        temp_dir = os.path.dirname(config_path) or "."
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w", encoding="utf-8", dir=temp_dir)
        yaml.dump(config, temp_file, allow_unicode=True)
        temp_file.flush()
        temp_file.close()
        updated_config_path = temp_file.name
        config_name = os.path.basename(updated_config_path)

        argv = ["nequip-train", "-cp", temp_dir, "-cn", config_name]

        old_argv = list(sys.argv)
        sys.argv = argv
        try:
            nequip_train_main()
        finally:
            sys.argv = old_argv
            os.unlink(updated_config_path)
    except Exception as e:
        logging.error(f"NequIP/Allegro training failed with config {config_path}: {str(e)}")
        raise