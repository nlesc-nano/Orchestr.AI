import yaml
import tempfile

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
    with open(master_yaml_path, "r") as f:
        config = yaml.safe_load(f) or {}
    
    engine_cfg = {}
    is_unified = "common" in config or platform in config
    
    # For MACE, only use the platform-specific section, ignoring 'common'
    if platform == "mace":
        if platform in config and isinstance(config[platform], dict):
            engine_cfg.update(config[platform])
        else:
            raise ValueError(f"No '{platform}' section found in the YAML config")
    else:
        # For PAINN, use the 'schnet' section and override model_type
        if platform == "painn":
            if "schnet" in config and isinstance(config["schnet"], dict):
                engine_cfg.update(config["schnet"])
                if "model" not in engine_cfg:
                    engine_cfg["model"] = {}
                engine_cfg["model"]["model_type"] = "painn"
            else:
                raise ValueError("No 'schnet' section found in the YAML config for platform 'painn'")
        # For FUSION (nequip_mace_interaction_fusion), use the 'schnet' section and merge model parameters
        elif platform == "fusion":
            if "schnet" in config and isinstance(config["schnet"], dict):
                engine_cfg.update(config["schnet"])
                # Define new model parameters
                new_model_params = {
                    "model_type": "nequip_mace_interaction_fusion",
                    "lmax": 2,
                    "n_interactions_nequip": 1,
                    "n_interactions_mace": 1
                }
                # Merge with existing model parameters
                if "model" not in engine_cfg:
                    engine_cfg["model"] = {}
                engine_cfg["model"].update(new_model_params)
            else:
                raise ValueError("No 'schnet' section found in the YAML config for platform 'fusion'")

        # For other platforms, merge 'common' and platform-specific sections
        else:
            if "common" in config:
                engine_cfg.update(config["common"])
            if is_unified and platform in config and isinstance(config[platform], dict):
                engine_cfg.update(config[platform])
            elif not is_unified:
                engine_cfg.update(config)
    
    # For SchNet, PAINN, FUSION, and NEQUIP_PAINN_FUSION, wrap the configuration under 'settings', but avoid double wrapping
    if platform in ["schnet", "painn", "fusion"]:
        if "settings" in engine_cfg:
            engine_cfg = {"settings": resolve_placeholders(engine_cfg["settings"], config)}
        else:
            engine_cfg = {"settings": resolve_placeholders(engine_cfg, config)}
    
    # Write to temp YAML
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w")
    yaml.dump(engine_cfg, temp)
    temp.close()
    return temp.name