import yaml
import tempfile

def resolve_placeholders(config, parent_config=None):
    """
    Recursively resolve placeholders in the config dictionary.
    """
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
        # For other platforms, merge 'common' and platform-specific sections
        if "common" in config:
            engine_cfg.update(config["common"])
        if is_unified and platform in config and isinstance(config[platform], dict):
            engine_cfg.update(config[platform])
        elif not is_unified:
            engine_cfg.update(config)
    
    # For SchNet, wrap the configuration under 'settings', but avoid double wrapping
    if platform == "schnet":
        if "settings" in engine_cfg:
            engine_cfg = {"settings": resolve_placeholders(engine_cfg["settings"], config)}
        else:
            engine_cfg = {"settings": resolve_placeholders(engine_cfg, config)}
    
    # Write to temp YAML
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w")
    yaml.dump(engine_cfg, temp)
    temp.close()
    return temp.name