import yaml
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

def extract_engine_yaml(master_yaml_path, platform, input_xyz=None):
    with open(master_yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    
    engine_cfg = {}
    is_unified = "common" in config

    # Extract p_mlff_qd_input_xyz from command line, common (for unified), or root (for individual)
    if input_xyz is not None:
        p_mlff_qd_input_xyz = input_xyz
    elif is_unified and "common" in config and "p_mlff_qd_input_xyz" in config["common"]:
        p_mlff_qd_input_xyz = config["common"]["p_mlff_qd_input_xyz"]
        if p_mlff_qd_input_xyz is None or p_mlff_qd_input_xyz == "":
            raise ValueError("p_mlff_qd_input_xyz in unified YAML 'common' section is empty or null. Please provide a valid path via --input argument.")
    elif "p_mlff_qd_input_xyz" in config:
        p_mlff_qd_input_xyz = config["p_mlff_qd_input_xyz"]
    else:
        raise ValueError("p_mlff_qd_input_xyz not found in config or --input argument")

    if p_mlff_qd_input_xyz and not os.path.exists(p_mlff_qd_input_xyz):
        raise ValueError(f"Input XYZ file not found or does not exist: {p_mlff_qd_input_xyz}")
    
    # Preprocess data based on platform if input_xyz is provided
    if p_mlff_qd_input_xyz:
        converted_file = preprocess_data_for_platform(p_mlff_qd_input_xyz, platform, output_dir=os.path.join(os.path.dirname(p_mlff_qd_input_xyz), "converted_data"))
        logging.info(f"Converted data file for {platform}: {converted_file}")

    # For MACE, use the platform-specific section, ignoring 'common' for individual YAMLs
    if platform == "mace":
        if is_unified and platform in config and isinstance(config[platform], dict):
            engine_cfg.update(config[platform] or {})
        elif not is_unified:
            engine_cfg.update(config or {})
        else:
            raise ValueError(f"No '{platform}' section found in the YAML config")
        if "train_file" not in engine_cfg:
            engine_cfg["train_file"] = {}
        if p_mlff_qd_input_xyz and not engine_cfg.get("train_file"):
            engine_cfg["train_file"] = converted_file
    # For SchNet-based platforms (painn, fusion, schnet), use schnet as base
    elif platform in ["painn", "fusion", "schnet"]:
        # Initialize key dictionaries
        if "model" not in engine_cfg:
            engine_cfg["model"] = {}
        if "data" not in engine_cfg:
            engine_cfg["data"] = {}
        # Use schnet section as base for unified YAMLs
        if is_unified and "schnet" in config and isinstance(config["schnet"], dict):
            engine_cfg.update(config["schnet"] or {})
        elif not is_unified:
            engine_cfg.update(config or {})
        # Apply model overrides
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
        # Merge only specific common settings for unified YAMLs, ensuring general is a dictionary
        if is_unified and "common" in config:
            if "seed" in config["common"]:
                engine_cfg["general"] = engine_cfg.get("general", {})
                engine_cfg["general"]["seed"] = config["common"]["seed"]
        if p_mlff_qd_input_xyz:
            engine_cfg["data"]["dataset_path"] = converted_file
    # For other platforms (nequip, allegro), merge 'common' and platform-specific or use individual YAML
    else:
        if is_unified and "common" in config:
            engine_cfg.update(config["common"])
        if is_unified and platform in config and isinstance(config[platform], dict):
            engine_cfg.update(config[platform])
        elif not is_unified:
            engine_cfg.update(config or {})
        if platform in ["nequip", "allegro"] and p_mlff_qd_input_xyz:
            if "data" not in engine_cfg:
                engine_cfg["data"] = {}
            if "split_dataset" not in engine_cfg["data"]:
                engine_cfg["data"]["split_dataset"] = {}
            if not engine_cfg["data"]["split_dataset"].get("file_path"):
                engine_cfg["data"]["split_dataset"]["file_path"] = converted_file
    
    # Remove p_mlff_qd_input_xyz from the final config to avoid conflicts with downstream packages
    if "p_mlff_qd_input_xyz" in engine_cfg:
        del engine_cfg["p_mlff_qd_input_xyz"]
    
    return engine_cfg