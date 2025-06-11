import yaml
import tempfile
import os
import logging
from mlff_qd.utils.data_conversion import preprocess_data_for_platform

def resolve_placeholders(config, parent_config=None):
    if parent_config is None:
        parent_config = config
    if isinstance(config, dict):
        return {k: resolve_placeholders(v, parent_config) for k, v in config.items()}
    elif isinstance(config, list):
        return [resolve_placeholders(item, parent_config) for item in config]
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        key = config[2:-1]
        return parent_config.get(key, config)
    return config

def extract_engine_yaml(master_yaml_path, platform):
    with open(master_yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    
    engine_cfg = {}
    is_unified = "common" in config
    has_settings = "settings" in config

    # Extract p_mlff_qd_input_xyz from common (for unified) or root (for individual)
    input_xyz = config.get("common", {}).get("p_mlff_qd_input_xyz") or config.get("p_mlff_qd_input_xyz")
    if input_xyz and not os.path.exists(input_xyz):
        raise ValueError(f"Input XYZ file not found or does not exist: {input_xyz}")
    
    # Preprocess data based on platform if input_xyz is provided
    if input_xyz:
        converted_file = preprocess_data_for_platform(input_xyz, platform, output_dir=os.path.join(os.path.dirname(input_xyz), "converted_data"))
        logging.info(f"Converted data file for {platform}: {converted_file}")

    # For MACE, use the platform-specific section, ignoring 'common' for individual YAMLs
    if platform == "mace":
        if is_unified and platform in config and isinstance(config[platform], dict):
            engine_cfg.update(config[platform])
        elif not is_unified:
            engine_cfg.update(config)
        else:
            raise ValueError(f"No '{platform}' section found in the YAML config")
        if input_xyz and not engine_cfg.get("train_file"):
            engine_cfg["train_file"] = converted_file
    # For SchNet-based platforms (painn, fusion, schnet), merge into a flat dictionary
    elif platform in ["painn", "fusion", "schnet"]:
        # Start with platform-specific config if unified, or the whole config if individual
        if is_unified and platform in config and isinstance(config[platform], dict):
            engine_cfg.update(config[platform])
        elif has_settings and "settings" in config:
            engine_cfg.update(config["settings"])  # Use existing settings for individual YAMLs
        elif not is_unified:
            engine_cfg.update(config)
        # Apply model overrides
        if "model" not in engine_cfg:
            engine_cfg["model"] = {}
        if platform == "painn":
            engine_cfg["model"]["model_type"] = "painn"
        elif platform == "fusion":
            engine_cfg["model"].update({
                "model_type": "nequip_mace_interaction_fusion",
                "lmax": 2,
                "cutoff": 12.0,
                "n_rbf": 40,
                "n_atom_basis": 192,
                "n_interactions_nequip": 1,
                "n_interactions_mace": 1
            })
        # Merge with common settings if unified
        if is_unified and "common" in config:
            engine_cfg.update(config["common"])
        if input_xyz:
            if "data" not in engine_cfg:
                engine_cfg["data"] = {}
            engine_cfg["data"]["dataset_path"] = converted_file
    # For other platforms (nequip, allegro), merge 'common' and platform-specific or use individual YAML
    else:
        if is_unified and "common" in config:
            engine_cfg.update(config["common"])
        if is_unified and platform in config and isinstance(config[platform], dict):
            engine_cfg.update(config[platform])
        elif has_settings and "settings" in config:
            engine_cfg.update(config["settings"])  # Use existing settings for individual YAMLs
        elif not is_unified:
            engine_cfg.update(config)
        if platform in ["nequip", "allegro"] and input_xyz:
            if "data" not in engine_cfg:
                engine_cfg["data"] = {}
            if "split_dataset" not in engine_cfg["data"]:
                engine_cfg["data"]["split_dataset"] = {}
            if not engine_cfg["data"]["split_dataset"].get("file_path"):
                engine_cfg["data"]["split_dataset"]["file_path"] = converted_file
    
    # Remove p_mlff_qd_input_xyz from the final config to avoid conflicts with downstream packages
    if "p_mlff_qd_input_xyz" in engine_cfg:
        del engine_cfg["p_mlff_qd_input_xyz"]
    
    # Write to temp YAML
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{platform}.yaml", mode="w", encoding="utf-8")
    yaml.dump(engine_cfg, temp, allow_unicode=True)
    temp.flush()
    temp.close()
    temp_path = temp.name
    
    return temp_path