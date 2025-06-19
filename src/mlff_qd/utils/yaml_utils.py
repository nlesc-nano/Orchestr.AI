import yaml
import os
import logging
from mlff_qd.utils.data_conversion import preprocess_data_for_platform
from copy import deepcopy

# Common defaults for all platforms
COMMON_DEFAULTS = {
    "input_xyz_file": None,
    "seed": 42,
    "train_size": 800,
    "val_size": 200,
    "batch_size": 16,
    "epochs": 3,
    "output_dir": "./results",
    "chemical_symbols": [],
    "learning_rate": 0.001,
    "energy_loss_weight": 0.05,
    "forces_loss_weight": 0.95,
    "n_rbf": 20,
    "l_max": 1,
    "optimizer": "AdamW",
    "num_workers": 24,
    "pin_memory": True,
    "scheduler": {
        "type": "ReduceLROnPlateau",
        "factor": 0.8,
        "patience": 5
    },
    "device": "cpu",
    "log_every_n_steps":10
}

# Key mappings for standardized user inputs to platform-specific keys
KEY_MAPPINGS = {
    "nequip": {
        "input_xyz_file": "data.split_dataset.file_path",
        "seed": ["data.seed", "model.seed"],
        "train_size": "data.split_dataset.train",
        "val_size": "data.split_dataset.val",
        "batch_size": ["data.train_dataloader.batch_size", "data.val_dataloader.batch_size"],
        "epochs": "trainer.max_epochs",
        "output_dir": ["trainer.logger.save_dir", "trainer.callbacks[1].dirpath"],
        "chemical_symbols": ["data.transforms[1].chemical_symbols", "model.type_names", "model.pair_potential.chemical_species"],
        "learning_rate": "training_module.optimizer.lr",
        "energy_loss_weight": "training_module.loss.coeffs.total_energy",
        "forces_loss_weight": "training_module.loss.coeffs.forces",
        "n_rbf": "model.num_bessels",
        "l_max": "model.l_max",
        "optimizer": "training_module.optimizer._target_",
        "num_workers": "data.num_workers",
        "pin_memory": "data.pin_memory",
        "scheduler": "training_module.scheduler",
        "log_every_n_steps":"trainer.log_every_n_steps",
        "device": "trainer.accelerator"
    },
    "allegro": {
        "input_xyz_file": "data.split_dataset.file_path",
        "seed": ["data.seed", "model.seed"],
        "train_size": "data.split_dataset.train",
        "val_size": "data.split_dataset.val",
        "batch_size": ["data.train_dataloader.batch_size", "data.val_dataloader.batch_size"],
        "epochs": "trainer.max_epochs",
        "output_dir": ["trainer.logger.save_dir", "trainer.callbacks[1].dirpath"],
        "chemical_symbols": ["data.transforms[1].chemical_symbols", "model.type_names", "model.pair_potential.chemical_species"],
        "learning_rate": "training_module.optimizer.lr",
        "energy_loss_weight": "training_module.loss.coeffs.total_energy",
        "forces_loss_weight": "training_module.loss.coeffs.forces",
        "n_rbf": "model.radial_chemical_embed.num_bessels",
        "l_max": "model.l_max",
        "optimizer": "training_module.optimizer._target_",
        "num_workers": "data.num_workers",
        "pin_memory": "data.pin_memory",
        "scheduler": "training_module.scheduler",
        "log_every_n_steps":"trainer.log_every_n_steps",
        "device": "trainer.accelerator"
    },
    "mace": {
        "input_xyz_file": "train_file",
        "seed": "seed",
        "batch_size": "batch_size",
        "epochs": "max_num_epochs",
        "learning_rate": "lr",
        "energy_loss_weight": None,
        "forces_loss_weight": None,
        "n_rbf": "num_radial_basis",
        "l_max": "max_ell",
        "optimizer": "optimizer.type",
        "num_workers": None,
        "pin_memory": None,
        "scheduler": None,
        "device": "device",
        "layers": "num_interactions",
        "features": "num_channels"
    },
    "schnet": {
        "input_xyz_file": "data.dataset_path",
        "seed": "general.seed",
        "train_size": "training.num_train",
        "val_size": "training.num_val",
        "batch_size": "training.batch_size",
        "epochs": "training.max_epochs",
        "output_dir": ["logging.folder", "testing.trained_model_path"],
        "chemical_symbols": None,
        "learning_rate": "training.optimizer.lr",
        "energy_loss_weight": "outputs.energy.loss_weight",
        "forces_loss_weight": "outputs.forces.loss_weight",
        "n_rbf": "model.n_rbf",
        "l_max": None,
        "optimizer": "training.optimizer.type",
        "num_workers": "training.num_workers",
        "pin_memory": "training.pin_memory",
        "scheduler": "training.scheduler",
        "log_every_n_steps":"training.log_every_n_steps",
        "device": "training.accelerator"
    },
    "painn": {
        "input_xyz_file": "data.dataset_path",
        "seed": "general.seed",
        "train_size": "training.num_train",
        "val_size": "training.num_val",
        "batch_size": "training.batch_size",
        "epochs": "training.max_epochs",
        "output_dir": ["logging.folder", "testing.trained_model_path"],
        "chemical_symbols": None,
        "learning_rate": "training.optimizer.lr",
        "energy_loss_weight": "outputs.energy.loss_weight",
        "forces_loss_weight": "outputs.forces.loss_weight",
        "n_rbf": "model.n_rbf",
        "l_max": None,
        "optimizer": "training.optimizer.type",
        "num_workers": "training.num_workers",
        "pin_memory": "training.pin_memory",
        "scheduler": "training.scheduler",
        "log_every_n_steps":"training.log_every_n_steps",
        "device": "training.accelerator"
    },
    "fusion": {
        "input_xyz_file": "data.dataset_path",
        "seed": "general.seed",
        "train_size": "training.num_train",
        "val_size": "training.num_val",
        "batch_size": "training.batch_size",
        "epochs": "training.max_epochs",
        "output_dir": ["logging.folder", "testing.trained_model_path"],
        "chemical_symbols": None,
        "learning_rate": "training.optimizer.lr",
        "energy_loss_weight": "outputs.energy.loss_weight",
        "forces_loss_weight": "outputs.forces.loss_weight",
        "n_rbf": "model.n_rbf",
        "l_max": "model.l_max",
        "optimizer": "training.optimizer.type",
        "num_workers": "training.num_workers",
        "pin_memory": "training.pin_memory",
        "scheduler": "training.scheduler",
        "log_every_n_steps":"training.log_every_n_steps",
        "device": "training.accelerator"
    }
}

def update_nested_dict(d, keys, value):
    """Update a nested dictionary using a dot-separated key or list of keys."""
    if isinstance(keys, list):
        for key_path in keys:
            current = d
            key_list = key_path.split(".")
            for key in key_list[:-1]:
                current = current.setdefault(key, {})
            current[key_list[-1]] = value
    else:
        current = d
        key_list = keys.split(".")
        for key in key_list[:-1]:
            current = current.setdefault(key, {})
        current[key_list[-1]] = value

def resolve_placeholders(config, parent_config=None):
    if parent_config is None:
        parent_config = config
    if isinstance(config, dict):
        return {k: resolve_placeholders(v, parent_config) for k, v in config.items()}
    elif isinstance(config, list):
        return [resolve_placeholders(item, parent_config) for item in config]
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        key = config[2:-1]
        keys = key.split(".")
        current = parent_config
        for k in keys:
            current = current.get(k, config)
            if current == config:
                return config
        return current
    return config

def load_template(platform):
    """Load platform-specific template from the mlff_qd/templates directory."""
    template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates", f"{platform}.yaml")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file for {platform} not found at {template_path}")
    with open(template_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

# Add this new function at the top of the file, after imports
def validate_and_fill_defaults(cfg, platform, defaults=COMMON_DEFAULTS):
    """Recursively validate, fill missing keys, and filter out unmapped keys based on KEY_MAPPINGS."""
    # Get all valid keys for this platform from KEY_MAPPINGS
    valid_keys = set(KEY_MAPPINGS[platform].keys())
    
    def filter_dict(d, parent_keys=""):
        filtered = {}
        for key, value in list(d.items()):
            current_path = f"{parent_keys}{key}" if not parent_keys else f"{parent_keys}.{key}"
            
            # Check if this key or its nested path is mapped
            is_mapped = any(mapped_key.startswith(current_path + ".") or mapped_key == current_path 
                           for mapped_key in [k for k in KEY_MAPPINGS[platform].values() if k])
            
            if isinstance(value, dict):
                nested = filter_dict(value, current_path)
                if nested or is_mapped:  # Keep nested dict if it has mapped keys or content
                    filtered[key] = nested
            elif key in valid_keys and key in defaults and value is None:
                filtered[key] = defaults[key]  # Fill missing mapped keys with defaults
            elif is_mapped:
                filtered[key] = value  # Keep mapped keys with user/template values
        return filtered
    
    # Apply filtering and validation
    cfg.clear()  # Reset cfg to avoid residual template keys
    updated_cfg = filter_dict(cfg)
    cfg.update(updated_cfg)

# Update the extract_engine_yaml function, uncommenting validation
def extract_engine_yaml(master_yaml_path, platform, input_xyz=None):
    with open(master_yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    
    engine_cfg = deepcopy(load_template(platform))  # Load template as base
    is_unified = "common" in config

    # Extract input_xyz_file from command line, common (for unified), or root (for individual)
    if input_xyz is not None:
        input_xyz_file = input_xyz
    elif is_unified and "common" in config and "input_xyz_file" in config["common"]:
        input_xyz_file = config["common"]["input_xyz_file"]
        if input_xyz_file is None or input_xyz_file == "":
            raise ValueError("input_xyz_file in unified YAML 'common' section is empty or null. Please provide a valid path via --input argument.")
    elif "input_xyz_file" in config:
        input_xyz_file = config["input_xyz_file"]
    else:
        raise ValueError("input_xyz_file not found in config or --input argument")

    if input_xyz_file and not os.path.exists(input_xyz_file):
        raise ValueError(f"Input XYZ file not found or does not exist: {input_xyz_file}")
    
    # Preprocess data based on platform if input_xyz is provided
    if input_xyz_file:
        converted_file = preprocess_data_for_platform(input_xyz_file, platform, output_dir=os.path.join(os.path.dirname(input_xyz_file), "converted_data"))
        logging.info(f"Converted data file for {platform}: {converted_file}")

    # For MACE, use the platform-specific section, ignoring 'common' for individual YAMLs
    if platform == "mace":
        if is_unified:
            # Use template as base and apply remapped overrides for mace
            if "overrides" in config and platform in config["overrides"] and isinstance(config["overrides"][platform], dict):
                overrides_mapped = {}
                for user_key, value in config["overrides"][platform].items():
                    if user_key in KEY_MAPPINGS[platform]:
                        mapped_keys = KEY_MAPPINGS[platform][user_key]
                        if mapped_keys:
                            if isinstance(mapped_keys, list):
                                for mapped_key in mapped_keys:
                                    update_nested_dict(overrides_mapped, mapped_key, value)
                            else:
                                update_nested_dict(overrides_mapped, mapped_keys, value)
                engine_cfg.update(overrides_mapped)
            # Merge common settings with key mapping
            if "common" in config:
                common = config["common"]
                for user_key, value in common.items():
                    if user_key in KEY_MAPPINGS[platform]:
                        mapped_keys = KEY_MAPPINGS[platform][user_key]
                        if mapped_keys:
                            update_nested_dict(engine_cfg, mapped_keys, value)
                    elif user_key == "input_xyz_file":
                        continue  # Handled separately
                    elif user_key == "seed":
                        engine_cfg["general"] = engine_cfg.get("general", {})
                        engine_cfg["general"]["seed"] = value
        elif not is_unified:
            engine_cfg.update(config or {})
        else:
            raise ValueError(f"No '{platform}' section found in the YAML config")
        # Ensure train_file is updated with the converted file path
        if input_xyz_file and "train_file" in engine_cfg:
            engine_cfg["train_file"] = converted_file

    # For SchNet-based platforms (painn, fusion, schnet), use schnet template with overrides
    elif platform in ["painn", "fusion", "schnet"]:
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
        # Merge common settings with key mapping and overrides
        if is_unified and "common" in config:
            common = config["common"]
            overrides = config.get("overrides", {}).get(platform, {})
            for user_key, value in {**common, **overrides}.items():
                if user_key in KEY_MAPPINGS[platform]:
                    mapped_keys = KEY_MAPPINGS[platform][user_key]
                    if mapped_keys:
                        update_nested_dict(engine_cfg, mapped_keys, value)
                elif user_key == "input_xyz_file":
                    continue  # Handled separately
                elif user_key == "seed":
                    engine_cfg["general"] = engine_cfg.get("general", {})
                    engine_cfg["general"]["seed"] = value
        if input_xyz_file:
            engine_cfg["data"]["dataset_path"] = converted_file
    # For other platforms (nequip, allegro), merge 'common' and platform-specific or use individual YAML
    else:
        if is_unified and "common" in config:
            engine_cfg.update(config["common"])
        if is_unified and platform in config and isinstance(config[platform], dict):
            engine_cfg.update(config[platform])
        elif not is_unified:
            engine_cfg.update(config or {})
        if platform in ["nequip", "allegro"] and input_xyz_file:
            if "data" not in engine_cfg:
                engine_cfg["data"] = {}
            if "split_dataset" not in engine_cfg["data"]:
                engine_cfg["data"]["split_dataset"] = {}
            if not engine_cfg["data"]["split_dataset"].get("file_path"):
                engine_cfg["data"]["split_dataset"]["file_path"] = converted_file
    
    # Remove input_xyz_file from the final config to avoid conflicts with downstream packages
    if "input_xyz_file" in engine_cfg:
        del engine_cfg["input_xyz_file"]
    
    # Validate and fill missing defaults
    # validate_and_fill_defaults(engine_cfg, platform)
    
    return engine_cfg
