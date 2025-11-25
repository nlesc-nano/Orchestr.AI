import yaml
import os
import logging
from copy import deepcopy

KEY_MAPPINGS = {
    "schnet": {
        "model.cutoff": ["model.cutoff"],
        "model.mp_layers": ["model.n_interactions"],
        "model.features": ["model.n_atom_basis"],
        "model.n_rbf": ["model.n_rbf"],
        "training.seed": ["general.seed"],
        "training.batch_size": ["training.batch_size"],
        "training.epochs": ["training.max_epochs"],
        "training.learning_rate": ["training.optimizer.lr"],
        "training.optimizer": ["training.optimizer.type"],
        "training.scheduler.type": ["training.scheduler.type"],
        "training.scheduler.factor": ["training.scheduler.factor"],
        "training.scheduler.patience": ["training.scheduler.patience"],
        "training.num_workers": ["training.num_workers"],
        "training.pin_memory": ["training.pin_memory"],
        "training.log_every_n_steps": ["training.log_every_n_steps"],
        "training.accelerator": ["training.accelerator"],
        "training.devices": ["training.devices"],
        "training.train_size": ["training.num_train"],
        "training.val_size": ["training.num_val"],
        "training.test_size": ["training.num_test"],
        "training.early_stopping": ["training.early_stopping"],  # EarlyStopping 
        "training.early_stopping.patience": ["training.early_stopping.patience"],
        "training.early_stopping.min_delta": ["training.early_stopping.min_delta"],
        "training.early_stopping.monitor": ["training.early_stopping.monitor"],
        "output.output_dir": ["logging.folder", "testing.trained_model_path"],
        "loss.energy_weight": ["outputs.energy.loss_weight"],
        "loss.forces_weight": ["outputs.forces.loss_weight"],
        "data.input_xyz_file": ["data.dataset_path"],
        "fine_tuning.pretrained_model": ["fine_tuning.pretrained_checkpoint"],
        "fine_tuning.learning_rate": ["fine_tuning.lr"],
        "fine_tuning.early_stopping_patience": ["fine_tuning.early_stopping_patience"],
        "fine_tuning.freeze_backbone": ["fine_tuning.freeze_all_representation"],

    },
    "painn": {},  
    "fusion": {},
    "nequip": {
        "model.cutoff": ["training_module.model.r_max"],
        "model.mp_layers": ["training_module.model.num_layers"],
        "model.features": ["training_module.model.num_features"],  
        "model.n_rbf": ["training_module.model.num_bessels"],
        "model.l_max": ["training_module.model.l_max"],
        "model.chemical_symbols": [
            "model_type_names",  # root-level in template
            "chemical_species",
            "training_module.model.type_names",
            "data.transforms[0].model_type_names",
            "training_module.model.pair_potential.chemical_species"
        ],
        "model.parity": ["training_module.model.parity"],
        "model.model_dtype": ["training_module.model.model_dtype"],
        "training.seed": ["data.seed", "training_module.model.seed"],
        "training.batch_size": [
            "data.train_dataloader.batch_size",
            "data.val_dataloader.batch_size",
            "data.stats_manager.dataloader_kwargs.batch_size"
        ],
        "training.epochs": ["trainer.max_epochs"],
        "training.learning_rate": ["training_module.optimizer.lr"],
        "training.optimizer": ["training_module.optimizer._target_"],
        "training.scheduler": ["training_module.lr_scheduler.scheduler._target_"],
        "training.scheduler.factor": ["training_module.lr_scheduler.scheduler.factor"],
        "training.scheduler.patience": ["training_module.lr_scheduler.scheduler.patience"],
        "training.num_workers": ["data.train_dataloader.num_workers", "data.val_dataloader.num_workers"],
        "training.pin_memory": [],  # not in template, add if needed
        "training.log_every_n_steps": ["trainer.log_every_n_steps"],
        "training.accelerator": ["device", "trainer.accelerator"],  # template uses 'device'
        "training.devices": ["trainer.devices"],
        "training.train_size": ["data.split_dataset.train"],
        "training.val_size": ["data.split_dataset.val"],
        "training.test_size": ["data.split_dataset.test"],
        "training.early_stopping.patience": ["trainer.callbacks[0].patience"],
        "training.early_stopping.min_delta": ["trainer.callbacks[0].min_delta"],
        "training.early_stopping.monitor":  ["trainer.callbacks[0].monitor"],
        "data.input_xyz_file": ["data.split_dataset.file_path"],
        "output.output_dir": ["trainer.callbacks[1].dirpath", "trainer.logger[0].save_dir"],
        "loss.energy_weight": ["training_module.loss.coeffs.total_energy"],
        "loss.forces_weight": ["training_module.loss.coeffs.forces"],
    },
    
    "allegro": {
        "model.cutoff": ["cutoff_radius", "training_module.model.r_max"],
        "model.mp_layers": ["training_module.model.num_layers"],
        "model.features": ["num_scalar_features", "training_module.model.num_scalar_features"], 
        "model.n_rbf": ["training_module.model.radial_chemical_embed.num_bessels"],
        "model.l_max": ["training_module.model.l_max"],
        "model.chemical_symbols": [
            "model_type_names",  # root-level in template
            "chemical_species",
            "training_module.model.type_names",
            "data.transforms[1].model_type_names",
            "training_module.model.pair_potential.chemical_species"
        ],
        "model.parity": ["training_module.model.parity"],
        "model.model_dtype": ["training_module.model.model_dtype"],
        "training.seed": ["data.seed", "training_module.model.seed"],
        "training.batch_size": [
            "data.train_dataloader.batch_size",
            "data.val_dataloader.batch_size",
            "data.stats_manager.dataloader_kwargs.batch_size"
        ],
        "training.epochs": ["trainer.max_epochs"],
        "training.learning_rate": ["training_module.optimizer.lr"],
        "training.optimizer": ["training_module.optimizer._target_"],
        "training.scheduler": ["training_module.lr_scheduler.scheduler._target_"],  # same as NequIP
        "training.scheduler.factor": ["training_module.lr_scheduler.scheduler.factor"],
        "training.scheduler.patience": ["training_module.lr_scheduler.scheduler.patience"],
        "training.num_workers": ["data.train_dataloader.num_workers", "data.val_dataloader.num_workers"],
        "training.pin_memory": [],  # not in template
        "training.log_every_n_steps": ["trainer.log_every_n_steps"],
        "training.accelerator": ["device", "trainer.accelerator"],
        "training.devices": ["trainer.devices"],
        "training.train_size": ["data.split_dataset.train"],
        "training.val_size": ["data.split_dataset.val"],
        "training.test_size": ["data.split_dataset.test"],
        "training.early_stopping.patience": ["trainer.callbacks[1].patience"],
        "training.early_stopping.min_delta": ["trainer.callbacks[1].min_delta"],
        "training.early_stopping.monitor":  ["trainer.callbacks[1].monitor"], 
        "data.input_xyz_file": ["data.split_dataset.file_path"],
        "output.output_dir": ["trainer.callbacks[0].dirpath", "trainer.logger[0].save_dir"],
        "loss.energy_weight": ["training_module.loss.coeffs.total_energy"],
        "loss.forces_weight": ["training_module.loss.coeffs.forces"],
    },

    "mace": {
        "model.cutoff": ["r_max"],
        "model.mp_layers": ["num_interactions"],
        "model.features": ["num_channels"],
        "model.n_rbf": ["num_radial_basis"],
        "model.l_max": ["max_L"],
        "model.chemical_symbols": ["chemical_symbols"],
        "training.seed": ["seed"],
        "training.batch_size": ["batch_size"],
        "training.epochs": ["max_num_epochs"],
        "training.learning_rate": ["lr"],
        "training.optimizer": ["optimizer"],  # string field in template
        "training.scheduler": ["scheduler"],  # string field in template
        "training.num_workers": ["num_workers"],
        "training.pin_memory": ["pin_memory"],
        "training.log_every_n_steps": ["eval_interval"],
        "training.accelerator": ["device"],
        "training.train_size": ["train_file"],  # This is actually a path to file, not a ratio/size; be careful 
        "training.val_size": ["valid_fraction"],
        "training.early_stopping.patience": ["patience"],
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
}

NPZ_ENGINES = {"schnet", "painn", "fusion"}
XYZ_ENGINES = {"nequip", "allegro", "mace"}

def expected_extension(platform: str) -> str:
    return ".npz" if platform in NPZ_ENGINES else ".xyz"

def validate_input_file(path: str, platform: str) -> str:
    if not path or not os.path.exists(path):
        raise ValueError(f"Input data file not found: {path}")
    need = expected_extension(platform)
    if not str(path).lower().endswith(need):
        raise ValueError(f"[{platform}] Invalid input extension for {path!r}. Expected '{need}'.")
    return path

def _resolve_path(base_dir, p):
    if not p:
        return None
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(base_dir, p))

def get_dataset_paths_from_yaml(platform, config_yaml_path):
    """
    Return a list of dataset files referenced by the engine YAML, resolved relative to the YAML file.
    Only returns existing files. Deduplicated.
    """
    if not config_yaml_path or not os.path.exists(config_yaml_path):
        return []

    with open(config_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    yaml_dir = os.path.dirname(os.path.abspath(config_yaml_path))
    platform = (platform or "").lower()
    paths = []

    if platform in {"schnet", "painn", "fusion"}:
        dp = cfg.get("data", {}).get("dataset_path")
        if dp: paths.append(_resolve_path(yaml_dir, dp))

        for k in ("val_dataset_path", "test_dataset_path"):
            p = cfg.get("data", {}).get(k)
            if p: paths.append(_resolve_path(yaml_dir, p))

    elif platform in {"nequip", "allegro"}:
        sp = cfg.get("data", {}).get("split_dataset", {}).get("file_path")
        if sp: paths.append(_resolve_path(yaml_dir, sp))
        for k in ("test_file", "infer_file"):
            p = cfg.get("data", {}).get(k)
            if p: paths.append(_resolve_path(yaml_dir, p))

    elif platform == "mace":
        tf = cfg.get("train_file")
        if tf: paths.append(_resolve_path(yaml_dir, tf))
        for k in ("valid_file", "val_file", "test_file"):
            p = cfg.get(k)
            if p: paths.append(_resolve_path(yaml_dir, p))

    # Dedup + existing only
    final, seen = [], set()
    for p in paths:
        if p and os.path.exists(p) and p not in seen:
            seen.add(p)
            final.append(p)
    return final

# Patch painn/fusion mapping to schnet (they use the same template)
for plat in ["painn", "fusion"]:
    KEY_MAPPINGS[plat] = deepcopy(KEY_MAPPINGS["schnet"])

def get_early_stopping_monitor(platform):
    if platform in ["schnet", "painn", "fusion"]:
        return "val_loss"
    elif platform in ["nequip", "allegro"]:
        return "val0_epoch/weighted_sum"
    elif platform == "mace":
        return None
    else:
        return None

def remove_early_stopping_callbacks(engine_cfg):
    # Remove from top-level 'callbacks'
    if "callbacks" in engine_cfg and isinstance(engine_cfg["callbacks"], list):
        engine_cfg["callbacks"] = [cb for cb in engine_cfg["callbacks"] if not (isinstance(cb, dict) and cb.get("_target_") == "lightning.pytorch.callbacks.EarlyStopping")]
        if not engine_cfg["callbacks"]:
            del engine_cfg["callbacks"]
    # Remove from trainer.callbacks if present
    if "trainer" in engine_cfg and "callbacks" in engine_cfg["trainer"]:
        if isinstance(engine_cfg["trainer"]["callbacks"], list):
            engine_cfg["trainer"]["callbacks"] = [cb for cb in engine_cfg["trainer"]["callbacks"] if not (isinstance(cb, dict) and cb.get("_target_") == "lightning.pytorch.callbacks.EarlyStopping")]
            if not engine_cfg["trainer"]["callbacks"]:
                del engine_cfg["trainer"]["callbacks"]
                
def update_early_stopping_callbacks(engine_cfg, es_cfg, key_mappings, platform):
    patience = es_cfg.get("patience", 20 if platform in ["nequip", "allegro"] else 30)
    min_delta = es_cfg.get("min_delta", 1e-3)
    monitor = es_cfg.get("monitor", get_early_stopping_monitor(platform))
    for param, val in [("patience", patience), ("min_delta", min_delta), ("monitor", monitor)]:
        user_val = es_cfg.get(param, val)
        for path in key_mappings.get(f"training.early_stopping.{param}", []):
            set_nested(engine_cfg, path.split("."), user_val)

def apply_early_stopping(user_cfg, engine_cfg, platform, key_mappings):
    es_cfg = user_cfg.get("training", {}).get("early_stopping", {})
    enabled = es_cfg.get("enabled", None)
    if enabled is False or enabled is None:
        if platform == "mace":
            engine_cfg.pop("patience", None)
        else:
            remove_early_stopping_callbacks(engine_cfg)
            if platform in ["schnet", "painn", "fusion"]:
                if "training" in engine_cfg:
                    engine_cfg["training"].pop("early_stopping", None)
    elif enabled is True:
        if platform == "mace":
            patience = es_cfg.get("patience", 30)
            engine_cfg["patience"] = patience
        elif platform in ["schnet", "painn", "fusion"]:
            es_cfg_clean = {k: v for k, v in es_cfg.items() if k != "enabled"}
            if "monitor" not in es_cfg_clean or not es_cfg_clean.get("monitor"):
                es_cfg_clean["monitor"] = get_early_stopping_monitor(platform)
            engine_cfg.setdefault("training", {})["early_stopping"] = es_cfg_clean
        elif platform in ["nequip", "allegro"]:
            update_early_stopping_callbacks(engine_cfg, es_cfg, key_mappings, platform)

def preprocess_optimizer(user_cfg):
    # Recursively process nested user_cfg
    if isinstance(user_cfg, dict):
        for k, v in user_cfg.items():
            if k == "optimizer" and isinstance(v, str) and v in OPTIMIZER_TARGETS:
                user_cfg[k] = {"_target_": OPTIMIZER_TARGETS[v]}
            elif isinstance(v, dict):
                preprocess_optimizer(v)
    return user_cfg
    

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

def smart_round(x, ndigits=4):
    return round(float(x), ndigits)

def adjust_splits_for_engine(train_size, val_size, test_size, platform):
    if test_size is None:
        test_size = 0.0
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{platform}: train+val+test != 1.0 (got {total:.4f})! Please fix your splits.")
    if val_size <= 0:
        raise ValueError(f"{platform}: val_size must be >0 for early stopping.")
    if train_size <= 0:
        raise ValueError(f"{platform}: train_size must be >0.")
    if train_size < 0 or val_size < 0 or test_size < 0:
        raise ValueError(f"{platform}: Split sizes cannot be negative.")
    
    if test_size == 0:
        logging.warning(f"{platform}: test_size=0—skipping test metrics/inference.")
    
    return smart_round(train_size), smart_round(val_size), smart_round(test_size)
    
def path_exists_in_template(template: dict, keys: list) -> bool:
    """Check if a nested key path exists in the template dictionary."""
    cur = template
    for k in keys:
        # Support dot notation and list indexes
        if "[" in k and k.endswith("]"):
            base, idx = k[:-1].split("[")
            idx = int(idx)
            if base not in cur or not isinstance(cur[base], list):
                return False
            if idx >= len(cur[base]):
                return False
            cur = cur[base][idx]
        else:
            if not isinstance(cur, dict) or k not in cur:
                return False
            cur = cur[k]
    return True

def path_is_set_from_common(flat_common, key_mapping, dotkey):
    """
    Returns True if this override dotkey is set by mapping from common.
    """
    # For each common key that has a mapping
    for ckey, mapped_paths in key_mapping.items():
        if ckey in flat_common:
            for mp in mapped_paths:
                # If the mapped path matches the override dotkey, it's set by common!
                if mp == dotkey:
                    return True
    return False


def apply_overrides_with_common_check(
    engine_cfg: dict,
    overrides: dict,
    template: dict,
    flat_common: dict,
    key_mapping: dict,
    parent_path=()
):
    for k, v in overrides.items():
        # Always handle dot notation
        key_parts = k.split(".")
        full_path = parent_path + tuple(key_parts)
        dot_key = ".".join(full_path)
            
        # Handle EarlyStopping: skip override if common section has it!
        if (
            dot_key.startswith("training.early_stopping")
            or dot_key.startswith("callbacks")
            or dot_key.startswith("trainer.callbacks")
        ) and any(
            x in flat_common
            for x in [
                "training.early_stopping.patience",
                "training.early_stopping.min_delta",
                "training.early_stopping.monitor",
                "training.early_stopping.enabled",
                "training.early_stopping"
            ]
        ):
            logging.warning(
                f"[OVERRIDE WARNING] EarlyStopping override ignored for {dot_key}: already set from common section."
            )
            continue
            
        # Check if this override key is set from the common section (directly or via key mapping)
        if path_is_set_from_common(flat_common, key_mapping, dot_key):
            logging.warning(f"[OVERRIDE WARNING] Key {dot_key} already set from common section; ignoring expert override.")
            continue

        # If value is a dict, recurse (only if it's not a dot path, which can't be a dict)
        if isinstance(v, dict) and len(key_parts) == 1:
            engine_cfg.setdefault(k, {})
            tmpl_sub = template.get(k, {}) if isinstance(template, dict) else {}
            apply_overrides_with_common_check(engine_cfg[k], v, tmpl_sub, flat_common, key_mapping, parent_path + (k,))
            continue

        # List-style keys (handled by set_nested already)
        if any("[" in part and part.endswith("]") for part in key_parts):
            pass  # The code below already handles list-style keys

        # Check if this key exists in the template
        tmpl_ptr = template
        is_in_template = True
        for part in full_path:
            if tmpl_ptr is None:
                is_in_template = False
                break
            if "[" in part and part.endswith("]"):
                base, idx_str = part[:-1].split("[")
                idx = int(idx_str)
                if (not isinstance(tmpl_ptr, dict)) or (base not in tmpl_ptr) or (not isinstance(tmpl_ptr[base], list)) or (idx >= len(tmpl_ptr[base])):
                    is_in_template = False
                    break
                tmpl_ptr = tmpl_ptr[base][idx]
            else:
                if (not isinstance(tmpl_ptr, dict)) or (part not in tmpl_ptr):
                    is_in_template = False
                    break
                tmpl_ptr = tmpl_ptr[part]
        
        # If the key is not in the template, skip it with a warning
        if not is_in_template:
            logging.warning(f"[OVERRIDE WARNING] Key {dot_key} not present in template! Skipping.")
            continue

        # Apply the override
        set_nested(engine_cfg, list(full_path), v)
        logging.info(f"[OVERRIDE INFO] Key {dot_key} set by expert override (value: {v!r}).")

def warn_unused_common_keys(user_cfg, platform):
    # Flatten user config
    flat_common = flatten_dict(user_cfg)
    # Get mapped keys for platform
    key_mapping = KEY_MAPPINGS[platform]
    mapped_keys = set(key_mapping.keys())
    # Unused keys: in flat_common but not in mapping
    unused_keys = [k for k in flat_common if k not in mapped_keys]
    if unused_keys:
        logging.info(
            f"[INFO] The following keys from `common` are not used by platform '{platform}': {unused_keys}"
        )
    return unused_keys

def handle_pair_potential(user_cfg, engine_cfg, platform):
    if platform in ["nequip", "allegro"]:
        pair_potential_kind = user_cfg.get("model", {}).get("pair_potential", None)
        model_dict = engine_cfg.get("training_module", {}).get("model", {})
        if pair_potential_kind is None or str(pair_potential_kind).strip().lower() == "null":
            if "pair_potential" in model_dict:
                del model_dict["pair_potential"]
                logging.info("Removed pair_potential from extracted YAML (pair_potential: null).")
        elif isinstance(pair_potential_kind, str) and pair_potential_kind.strip().upper() == "ZBL":
            logging.info("ZBL pair_potential retained in extracted YAML.")
        else:
            raise ValueError(
                f"[ERROR] Unsupported value for common.model.pair_potential: {pair_potential_kind!r}. "
                "Allowed values: 'ZBL' (string) or null."
            )
            
def extract_common_config(master_yaml_path):
    with open(master_yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if "common" not in config:
        raise ValueError("New YAML must have a `common:` section at top level.")
    return config["common"], config

def _int_or_len_devices(x):
    """Normalize Lightning 'devices' into an int count.
    Accepts int, list/tuple (e.g., [0,1]), or numeric string. Returns None for 'auto'/unknown.
    """
    if isinstance(x, int):
        return x
    if isinstance(x, (list, tuple)):
        return len(x)
    try:
        return int(x)
    except Exception:
        return None  # e.g., 'auto'
        

def apply_autoddp(engine_cfg: dict):
    """Ensure DDP keys match effective device count.
    - devices <= 1 → remove trainer.strategy/num_nodes
    - devices >= 2 → set trainer.strategy='ddp' and default trainer.num_nodes=1
    """
    trainer = engine_cfg.setdefault("trainer", {})
    devs_raw = trainer.get("devices", 1)
    devs = _int_or_len_devices(devs_raw)
    if devs is None:
        # Unknown or 'auto' → let Lightning decide; don't force DDP keys
        return

    trainer["devices"] = devs  # reflect normalized value

    if devs <= 1:
        trainer.pop("strategy", None)
        trainer.pop("num_nodes", None)
    else:
        trainer.setdefault("strategy", "ddp")
        trainer.setdefault("num_nodes", 1)

def apply_mace_distributed_from_devices(user_cfg: dict, engine_cfg: dict):
    """
    Boolean-only policy for MACE:
    If unified YAML provides training.devices, derive engine_cfg['distributed']:
      - devices <= 1  -> False
      - devices >= 2  -> True
    If training.devices is absent, do nothing (engine-specific YAML stays as-is).
    """
    devv = (user_cfg or {}).get("training", {}).get("devices", None)
    if devv is None:
        return  # unified didn't specify devices; leave engine_cfg['distributed'] untouched
    try:
        n = _int_or_len_devices(devv)
    except NameError:
        try:
            n = int(devv)
        except Exception:
            n = None
    if n is None:
        # devices like "auto" not supported for this policy; default to False
        engine_cfg["distributed"] = False
        return
    engine_cfg["distributed"] = bool(n >= 2)
    
def _env_world_size():
    import os
    for k in ("WORLD_SIZE", "SLURM_NTASKS", "SLURM_NPROCS"):
        v = os.environ.get(k)
        if v and str(v).isdigit():
            return max(1, int(v))
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return max(1, len([x for x in cvd.split(",") if x.strip()]))
    return 1

def validate_mace_launch_policy(cfg: dict):
    """Fail fast if MACE 'distributed' disagrees with the launch environment."""
    import logging
    ws = _env_world_size()
    dist = bool(cfg.get("distributed", False))

    if dist and ws == 1:
        logging.error("[MACE] distributed=true but environment world_size=1. "
                      "Either set distributed:false or launch multi-task (e.g., --ntasks-per-node>=2 or torchrun).")
        raise SystemExit(2)

    if (not dist) and ws > 1:
        logging.error("[MACE] distributed=false but environment world_size=%d. "
                      "Either set distributed:true or run a single task (--ntasks=1).", ws)
        raise SystemExit(2)

    if (not dist) and ws == 1:
        try:
            import torch
            visible = torch.cuda.device_count()
        except Exception:
            visible = 1
        if visible >= 2:
            logging.error("[MACE] Multiple GPUs (%d) visible with distributed=false and a single task. "
                          "Either enable distributed:true or limit to devices: 1.", visible)
            raise SystemExit(2)

    logging.info("[MACE] Launch context OK: distributed=%s, world_size=%d", dist, ws)

def handle_nequip_finetuning(engine_cfg, user_cfg):
    """
    Modifies the NequIP config structure in-memory to enable fine-tuning.
    """
    ft_cfg = user_cfg.get("fine_tuning", {})
    
    if not ft_cfg.get("enabled", False):
        return engine_cfg

    pretrained_path = ft_cfg.get("pretrained_model")
    if not pretrained_path:
        raise ValueError("[NequIP] Fine-tuning enabled but 'pretrained_model' is missing in input.yaml")

    if "learning_rate" in ft_cfg:
        # NequIP expects LR at training_module.optimizer.lr
        if "training_module" in engine_cfg and "optimizer" in engine_cfg["training_module"]:
            engine_cfg["training_module"]["optimizer"]["lr"] = ft_cfg["learning_rate"]

    # Override Early Stopping (if specified)
    if "early_stopping_patience" in ft_cfg:
        callbacks = engine_cfg.get("trainer", {}).get("callbacks", [])
        for cb in callbacks:
            if "EarlyStopping" in cb.get("_target_", ""):
                cb["patience"] = ft_cfg["early_stopping_patience"]

    # Build the Modifier Block (The NequIP specific logic)
    modifier_block = {
        "modifier": "modify_PerTypeScaleShift",
        "scales": "${training_data_stats:per_type_forces_rms}",
        "shifts_trainable": True, 
        "scales_trainable": True,
    }
    
    # Handle custom shifts if provided (Need Fix)
    if "per_species_shifts" in ft_cfg:
        modifier_block["shifts"] = ft_cfg["per_species_shifts"]

    path_str = str(pretrained_path).lower()
    
    if path_str.endswith(".ckpt"):
        model_loader = {
            "_target_": "nequip.model.ModelFromCheckpoint",
            "checkpoint_path": pretrained_path
        }
    else:
        model_loader = {
            "_target_": "nequip.model.ModelFromPackage",
            "package_path": pretrained_path
        }

    new_model_structure = {
        "_target_": "nequip.model.modify",
        "modifiers": [modifier_block],
        "model": model_loader
    }

    # Inject into the correct location in the dictionary
    if "training_module" in engine_cfg and "model" in engine_cfg["training_module"]:
        engine_cfg["training_module"]["model"] = new_model_structure
    elif "model" in engine_cfg:
        engine_cfg["model"] = new_model_structure
    
    return engine_cfg

def handle_mace_finetuning(engine_cfg, user_cfg):
    """
    Injects MACE fine-tuning parameters (foundation_model) if enabled.
    """
    ft_cfg = user_cfg.get("fine_tuning", {})
    
    if not ft_cfg.get("enabled", False):
        return engine_cfg

    pretrained_path = ft_cfg.get("pretrained_model")
    if not pretrained_path:
        raise ValueError("[MACE] Fine-tuning enabled but 'pretrained_model' is missing.")
    
    engine_cfg["foundation_model"] = pretrained_path

    if "learning_rate" in ft_cfg:
        engine_cfg["lr"] = ft_cfg["learning_rate"]
        
    if "early_stopping_patience" in ft_cfg:
        engine_cfg["patience"] = ft_cfg["early_stopping_patience"]

    # MACE often benefits from explicit E0s="average" during fine-tuning 
    if "E0s" not in engine_cfg:
        engine_cfg["E0s"] = "average"

    return engine_cfg

def extract_engine_yaml(master_yaml_path, platform, input_xyz=None):
    # Load configs
    user_cfg, config = extract_common_config(master_yaml_path)
    user_cfg = preprocess_optimizer(user_cfg)

    # Data splits
    training_cfg = user_cfg.get("training", {})
    train_size = training_cfg.get("train_size", 0.8)
    val_size   = training_cfg.get("val_size", 0.2)
    test_size  = training_cfg.get("test_size", 0.0)
    train_size, val_size, test_size = adjust_splits_for_engine(train_size, val_size, test_size, platform)

    # Find input_xyz_file 
    input_xyz_file = None
    if "data" in user_cfg and "input_xyz_file" in user_cfg["data"]:
        input_xyz_file = user_cfg["data"]["input_xyz_file"]
    elif "input_xyz_file" in user_cfg:
        input_xyz_file = user_cfg["input_xyz_file"]
    if input_xyz:
        input_xyz_file = input_xyz
    if not input_xyz_file or not os.path.exists(input_xyz_file):
        raise ValueError(f"Input XYZ file not found: {input_xyz_file}")

    # extension validation only, no conversion
    input_xyz_file = validate_input_file(input_xyz_file, platform)

    # Prepare template and mapping as before
    engine_base = load_template(platform)
    engine_cfg = deepcopy(engine_base)
    apply_key_mapping(user_cfg, engine_cfg, KEY_MAPPINGS[platform])
    warn_unused_common_keys(user_cfg, platform)

    # Always patch in the correct data file after mapping
    for p in KEY_MAPPINGS[platform]["data.input_xyz_file"]:
        set_nested(engine_cfg, p.split("."), input_xyz_file)

    # Patch split sizes
    if platform in ["nequip", "allegro"]:
        set_nested(engine_cfg, ["data", "split_dataset", "train"], smart_round(train_size))
        set_nested(engine_cfg, ["data", "split_dataset", "val"], smart_round(val_size))
        set_nested(engine_cfg, ["data", "split_dataset", "test"], smart_round(test_size))
    elif platform in ["schnet", "painn", "fusion"]:
        set_nested(engine_cfg, ["training", "num_train"], smart_round(train_size))
        set_nested(engine_cfg, ["training", "num_val"], smart_round(val_size))
        set_nested(engine_cfg, ["training", "num_test"], smart_round(test_size))

    # Special patches
    handle_pair_potential(user_cfg, engine_cfg, platform)
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

    # Handle overrides section
    flat_common = flatten_dict(user_cfg)
    if "overrides" in config and platform in config["overrides"]:
        overrides = config["overrides"][platform]
        engine_base_template = load_template(platform)
        apply_overrides_with_common_check(engine_cfg, overrides, engine_base_template, flat_common, KEY_MAPPINGS[platform])

    if "input_xyz_file" in engine_cfg:
        del engine_cfg["input_xyz_file"]
    if platform == "mace":
        engine_cfg["valid_file"] = None
        engine_cfg["test_file"] = None
        
    # EarlyStopping logic
    apply_early_stopping(user_cfg, engine_cfg, platform, KEY_MAPPINGS[platform])

    # Patch for test_size=0 (after splits adjusted)
    if test_size == 0.0:
        if platform in ['nequip', 'allegro']:
            engine_cfg['run'] = ['train', 'val']  # Skip test
        elif platform == 'mace':
            engine_cfg['test_file'] = None  # Skip test
        print(f"Debug: Patched run for {platform}: {engine_cfg.get('run')}")  # Debug

    # Engine-specific post-processing
    if platform in ['nequip', 'allegro']:
        apply_autoddp(engine_cfg)
    elif platform == 'mace':
        apply_mace_distributed_from_devices(user_cfg, engine_cfg)    
        # MACE Fine-Tuning 
        engine_cfg = handle_mace_finetuning(engine_cfg, user_cfg)
   # nequip and allegro Fine-Tuning 
    if platform in ["nequip", "allegro"]:
        engine_cfg = handle_nequip_finetuning(engine_cfg, user_cfg)
    
    return engine_cfg