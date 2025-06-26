import yaml
import os
import logging
from copy import deepcopy
from mlff_qd.utils.data_conversion import preprocess_data_for_platform

# Define nested key mappings for new YAML format
KEY_MAPPINGS = {
    "schnet": {
        "model.cutoff": ["model.cutoff"],
        "model.layers": ["model.num_interactions"],
        "model.features": ["model.n_atom_basis"],
        "model.n_rbf": ["model.n_rbf"],
        "model.l_max": ["model.l_max"],
        "model.chemical_symbols": ["model.chemical_symbols"],
        "training.seed": ["general.seed"],
        "training.batch_size": ["training.batch_size"],
        "training.epochs": ["training.max_epochs"],
        "training.learning_rate": ["training.optimizer.lr"],
        "training.optimizer": ["training.optimizer.type"],
        "training.scheduler": ["training.scheduler"],
        "training.num_workers": ["training.num_workers"],
        "training.pin_memory": ["training.pin_memory"],
        "training.log_every_n_steps": ["training.log_every_n_steps"],
        "training.device": ["training.accelerator"],
        "training.train_size": ["training.num_train"],
        "training.val_size": ["training.num_val"],
        "output.output_dir": ["logging.folder", "testing.trained_model_path"],
        "output.energy_loss_weight": ["outputs.energy.loss_weight"],
        "output.forces_loss_weight": ["outputs.forces.loss_weight"],
        "data.input_xyz_file": ["data.dataset_path"],
    },
    "painn": {},  # Will inherit from schnet and apply patch in code
    "fusion": {}, # Will inherit from schnet and apply patch in code
    "nequip": {
        "model.cutoff": ["training_module.model.r_max"],
        "model.layers": ["training_module.model.num_layers"],
        "model.features": ["training_module.model.num_features"],  
        "model.n_rbf": ["training_module.model.num_bessels"],
        "model.l_max": ["training_module.model.l_max"],
        "model.chemical_symbols": [
            "chemical_symbols",  # root-level in template
            "training_module.model.type_names",
            "data.transforms[1].chemical_symbols",
            "training_module.model.pair_potential.chemical_species"
        ],
        "model.parity": ["training_module.model.parity"],
        "model.model_dtype": ["training_module.model.model_dtype"],
        "training.seed": ["data.seed", "training_module.model.seed"],
        "training.batch_size": [
            "data.train_dataloader.batch_size",
            "data.val_dataloader.batch_size"
        ],
        "training.epochs": ["trainer.max_epochs"],
        "training.learning_rate": ["training_module.optimizer.lr"],
        "training.optimizer": ["training_module.optimizer._target_"],
        "training.scheduler": ["training_module.lr_scheduler.scheduler._target_"],  # Or ...scheduler.type if you use a string
        "training.num_workers": ["data.train_dataloader.num_workers", "data.val_dataloader.num_workers"],
        "training.pin_memory": [],  # not in template, add if needed
        "training.log_every_n_steps": ["trainer.log_every_n_steps"],
        "training.device": ["device", "trainer.accelerator"],  # your template uses 'device'
        "training.train_size": ["data.split_dataset.train"],
        "training.val_size": ["data.split_dataset.val"],
        "data.input_xyz_file": ["data.split_dataset.file_path"],
        "output.output_dir": ["trainer.callbacks[1].dirpath", "trainer.logger[0].save_dir"],
        "loss.energy_weight": ["training_module.loss.coeffs.total_energy"],
        "loss.forces_weight": ["training_module.loss.coeffs.forces"],
    },
    
    "allegro": {
        "model.cutoff": ["training_module.model.r_max"],
        "model.layers": ["training_module.model.num_layers"],
        "model.features": ["training_module.model.num_scalar_features"],  # Allegro: num_scalar_features
        "model.n_rbf": ["training_module.model.radial_chemical_embed.num_bessels"],
        "model.l_max": ["training_module.model.l_max"],
        "model.chemical_symbols": [
            "chemical_symbols",
            "training_module.model.type_names",
            "data.transforms[1].chemical_symbols",
            "training_module.model.pair_potential.chemical_species"
        ],
        "model.parity": ["training_module.model.parity"],
        "model.model_dtype": ["training_module.model.model_dtype"],
        "training.seed": ["data.seed", "training_module.model.seed"],
        "training.batch_size": [
            "data.train_dataloader.batch_size",
            "data.val_dataloader.batch_size"
        ],
        "training.epochs": ["trainer.max_epochs"],
        "training.learning_rate": ["training_module.optimizer.lr"],
        "training.optimizer": ["training_module.optimizer._target_"],
        "training.scheduler": ["training_module.lr_scheduler.scheduler._target_"],  # same as NequIP
        "training.num_workers": ["data.train_dataloader.num_workers", "data.val_dataloader.num_workers"],
        "training.pin_memory": [],  # not in template
        "training.log_every_n_steps": ["trainer.log_every_n_steps"],
        "training.device": ["device", "trainer.accelerator"],
        "training.train_size": ["data.split_dataset.train"],
        "training.val_size": ["data.split_dataset.val"],
        "data.input_xyz_file": ["data.split_dataset.file_path"],
        "output.output_dir": ["trainer.callbacks[0].dirpath", "trainer.logger[0].save_dir"],
        "loss.energy_weight": ["training_module.loss.coeffs.total_energy"],
        "loss.forces_weight": ["training_module.loss.coeffs.forces"],
    },

    "mace": {
        "model.cutoff": ["r_max"],
        "model.layers": ["num_interactions"],
        "model.features": ["num_channels"],
        "model.n_rbf": ["num_radial_basis"],
        "model.l_max": ["max_L"],
        "model.chemical_symbols": ["chemical_symbols"],
        "training.seed": ["seed"],
        "training.batch_size": ["batch_size"],
        "training.epochs": ["max_num_epochs"],
        "training.learning_rate": ["lr"],
        "training.optimizer": ["optimizer"],  # string field in your template
        "training.scheduler": ["scheduler"],  # string field in your template
        "training.num_workers": ["num_workers"],
        "training.pin_memory": ["pin_memory"],
        "training.log_every_n_steps": ["eval_interval"],
        "training.device": ["device"],
        "training.train_size": ["train_file"],  # This is actually a path to file, not a ratio/size; be careful
        "training.val_size": ["valid_file"],
        "data.input_xyz_file": ["train_file"],  # (overwrites train_file path with converted dataset)
        "output.output_dir": [],  # Not present as key, could be directory for output, add if needed
        "loss.energy_weight": ["energy_weight"],
        "loss.forces_weight": ["forces_weight"],
    },
}


OPTIMIZER_TARGETS = {
    "AdamW": "torch.optim.AdamW",
    "Adam": "torch.optim.Adam",
    "SGD": "torch.optim.SGD",
    # add more if needed
}

def preprocess_optimizer(user_cfg):
    # Recursively process nested user_cfg
    if isinstance(user_cfg, dict):
        for k, v in user_cfg.items():
            if k == "optimizer" and isinstance(v, str) and v in OPTIMIZER_TARGETS:
                user_cfg[k] = {"_target_": OPTIMIZER_TARGETS[v]}
            elif isinstance(v, dict):
                preprocess_optimizer(v)
    return user_cfg
    
# Patch painn/fusion mapping to schnet (they use the same template)
for plat in ["painn", "fusion"]:
    KEY_MAPPINGS[plat] = deepcopy(KEY_MAPPINGS["schnet"])

def load_template(platform):
    """Load platform-specific template from the templates directory."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    template_path = os.path.join(base_dir, "templates", f"{platform if platform != 'painn' and platform != 'fusion' else 'schnet'}.yaml")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file for {platform} not found at {template_path}")
    with open(template_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def set_nestedx(cfg, keys, value):
    """Set value in cfg at nested keys (list)."""
    current = cfg
    for k in keys[:-1]:
        if k not in current or not isinstance(current[k], dict):
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value
    
    
def set_nested(cfg, keys, value):
    """Set value in cfg at nested keys (handles dicts and [list] indices)."""
    current = cfg
    for idx, k in enumerate(keys[:-1]):
        # Handle lists, e.g., "callbacks[1]"
        if "[" in k and k.endswith("]"):
            base, idx_str = k[:-1].split("[")
            idx_int = int(idx_str)
            # Create the base list if it doesn't exist
            if base not in current or not isinstance(current[base], list):
                current[base] = []
            # Extend list to desired length if necessary
            while len(current[base]) <= idx_int:
                current[base].append({})
            current = current[base][idx_int]
        else:
            # Standard dict path
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
    last = keys[-1]
    # Handle list on final key
    if "[" in last and last.endswith("]"):
        base, idx_str = last[:-1].split("[")
        idx_int = int(idx_str)
        if base not in current or not isinstance(current[base], list):
            current[base] = []
        while len(current[base]) <= idx_int:
            current[base].append({})
        current[base][idx_int] = value
    else:
        current[last] = value


def get_nested(cfg, keys):
    """Get value from cfg at nested keys (list), return None if not found."""
    current = cfg
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return None
    return current

def flatten_dict(d, parent_key="", sep="."):
    """Flatten nested dictionary to dot-separated keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def apply_key_mapping(user_cfg, engine_cfg, key_mapping):
    """Map user (dot) keys into engine_cfg using mapping."""
    flat_user = flatten_dict(user_cfg)
    for user_key, value in flat_user.items():
        if user_key in key_mapping:
            for engine_path in key_mapping[user_key]:
                set_nested(engine_cfg, engine_path.split("."), value)
                

def prune_to_template(cfg, template):
    """Remove keys in cfg that do not exist in template (recursive)."""
    if not isinstance(cfg, dict) or not isinstance(template, dict):
        return cfg
    pruned = {}
    for k, v in cfg.items():
        if k in template:
            if isinstance(v, dict):
                pruned[k] = prune_to_template(v, template[k])
            else:
                pruned[k] = v
    return pruned

def adjust_splits_for_engine(train_size, val_size, test_size, platform):
    if platform in ["schnet", "painn", "fusion"]:
        # Only use train/val
        if (train_size + val_size) < 1.0:
            missing = 1.0 - (train_size + val_size)
            val_size += missing
            print(f"[WARNING] {platform}: test_size will be ignored. Adjusted val_size to {val_size:.3f} so train+val=1.0")
        elif (train_size + val_size) > 1.0:
            total = train_size + val_size
            train_size /= total
            val_size /= total
            print(f"[WARNING] {platform}: train+val > 1.0, normalized both so train+val=1.0")
        # test_size ignored
        test_size = 0.0
    elif platform in ["nequip", "allegro"]:
        # Use all three, but ensure sum to 1.0
        total = train_size + val_size + test_size
        if abs(total - 1.0) > 1e-6:
            train_size /= total
            val_size  /= total
            test_size /= total
            print(f"[WARNING] {platform}: train+val+test != 1.0; normalized all so they sum to 1.0")
    return train_size, val_size, test_size

def smart_round(x, ndigits=4):
    return round(float(x), ndigits)

def extract_engine_yaml(master_yaml_path, platform, input_xyz=None):
    """
    Generate an engine-specific YAML dict from the unified YAML config file and platform.
    All overrides logic is REMOVED.
    """
    with open(master_yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Load base template for engine (always schnet for painn/fusion)
    engine_base = load_template(platform)

    # --- Flatten user config (we only look at `common` block) ---
    if "common" not in config:
        raise ValueError("New YAML must have a `common:` section at top level.")
    user_cfg = config["common"]
    
    user_cfg = preprocess_optimizer(user_cfg)

    # --- Extract and normalize splits ---
    training_cfg = user_cfg.get("training", {})
    train_size = training_cfg.get("train_size", 0.8)
    val_size   = training_cfg.get("val_size", 0.2)
    test_size  = training_cfg.get("test_size", 0.0)
    
    
    train_size, val_size, test_size = adjust_splits_for_engine(
        train_size, val_size, test_size, platform
    )
    
    train_size = smart_round(train_size)
    val_size   = smart_round(val_size)
    test_size  = smart_round(test_size)

    # -- Find input_xyz_file --
    input_xyz_file = None
    if "data" in user_cfg and "input_xyz_file" in user_cfg["data"]:
        input_xyz_file = user_cfg["data"]["input_xyz_file"]
    elif "input_xyz_file" in user_cfg:
        input_xyz_file = user_cfg["input_xyz_file"]
    if input_xyz:
        input_xyz_file = input_xyz
    if not input_xyz_file or not os.path.exists(input_xyz_file):
        raise ValueError(f"Input XYZ file not found: {input_xyz_file}")

    # -- Data conversion --
    converted_file = preprocess_data_for_platform(
        input_xyz_file, platform, output_dir=os.path.join(os.path.dirname(input_xyz_file), "converted_data")
    )
    logging.info(f"Converted data file for {platform}: {converted_file}")

    # -- Prepare template and mapping as before --
    engine_base = load_template(platform)
    engine_cfg = deepcopy(engine_base)
    apply_key_mapping(user_cfg, engine_cfg, KEY_MAPPINGS[platform])

    # Always patch in the correct data file after mapping
    for p in KEY_MAPPINGS[platform]["data.input_xyz_file"]:
        set_nested(engine_cfg, p.split("."), converted_file)

    # --- Inject split sizes at the right place in the engine config ---
    if platform in ["nequip", "allegro"]:
        set_nested(engine_cfg, ["data", "split_dataset", "train"], train_size)
        set_nested(engine_cfg, ["data", "split_dataset", "val"], val_size)
        set_nested(engine_cfg, ["data", "split_dataset", "test"], test_size)
    elif platform in ["schnet", "painn", "fusion"]:
        set_nested(engine_cfg, ["training", "num_train"], train_size)
        set_nested(engine_cfg, ["training", "num_val"], val_size)

    # --- schnet/painn/fusion tweaks ---
    if platform == "painn":
        engine_cfg["model"]["model_type"] = "painn"
    if platform == "fusion":
        engine_cfg["model"].update({
            "model_type": "nequip_mace_interaction_fusion",
            "lmax": 2,
            "n_interactions_nequip": 1,
            "n_interactions_mace": 1
        })
    
    engine_cfg = prune_to_template(engine_cfg, engine_base)
    
    # Remove any user-facing keys that might conflict (like 'input_xyz_file' at top)
    if "input_xyz_file" in engine_cfg:
        del engine_cfg["input_xyz_file"]
    
    if platform == "mace":
        engine_cfg["valid_file"] = None
        engine_cfg["test_file"] = None

    return engine_cfg

