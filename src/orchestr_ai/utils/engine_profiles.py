# src/orchestr_ai/utils/engine_profiles.py
from __future__ import annotations

import os
from typing import Dict

from orchestr_ai.utils.env_dispatch import EnvProfile


SCHNETPACK_ENGINES = {
    "schnet",
    "painn",
    "so3net",
    "field_schnet",
    "fusion",
}

NEQUIP_ENGINES = {
    "nequip",
    "allegro",
}

MACE_ENGINES = {
    "mace",
}

SUPPORTED_ENGINES = SCHNETPACK_ENGINES | NEQUIP_ENGINES | MACE_ENGINES


def normalize_engine(engine: str | None) -> str:
    return (engine or "").strip().lower()


def is_schnetpack_engine(engine: str | None) -> bool:
    return normalize_engine(engine) in SCHNETPACK_ENGINES


def is_nequip_engine(engine: str | None) -> bool:
    return normalize_engine(engine) in NEQUIP_ENGINES


def is_mace_engine(engine: str | None) -> bool:
    return normalize_engine(engine) in MACE_ENGINES


def model_framework_from_engine(engine: str | None) -> str:
    """
    Convert Orchestr.AI engine/platform name to postprocessing model framework.
    """
    engine = normalize_engine(engine)

    if engine in SCHNETPACK_ENGINES:
        return "schnetpack"
    if engine in NEQUIP_ENGINES:
        return "nequip"
    if engine in MACE_ENGINES:
        return "mace"

    raise ValueError(
        f"Unknown engine/platform '{engine}'. "
        f"Supported engines: {sorted(SUPPORTED_ENGINES)}"
    )


def get_default_engine_profiles() -> Dict[str, EnvProfile]:
    """
    Central source of truth for engine -> Conda environment mapping.

    Environment variables allow HPC/site-specific overrides:
      ORCHESTRAI_CORE_CONDA_ENV
      ORCHESTRAI_NEQUIP_CONDA_ENV
      ORCHESTRAI_MACE_CONDA_ENV
    """
    core_env = os.getenv("ORCHESTRAI_CORE_CONDA_ENV", "orchestr_ai-core")
    nequip_env = os.getenv("ORCHESTRAI_NEQUIP_CONDA_ENV", "orchestr_ai-nequip")
    mace_env = os.getenv("ORCHESTRAI_MACE_CONDA_ENV", "orchestr_ai-mace")

    return {
        "schnet": EnvProfile(conda_env=core_env),
        "painn": EnvProfile(conda_env=core_env),
        "so3net": EnvProfile(conda_env=core_env),
        "field_schnet": EnvProfile(conda_env=core_env),
        "fusion": EnvProfile(conda_env=core_env),
        "nequip": EnvProfile(conda_env=nequip_env),
        "allegro": EnvProfile(conda_env=nequip_env),
        "mace": EnvProfile(conda_env=mace_env),
    }


def detect_engine_from_config(config: dict, cli_engine: str | None = None) -> str:
    """
    Detect engine/platform from either CLI override or config.

    Supports both training-style configs:
      platform: mace

    and postprocessing-style configs:
      platform: mace
      engine: mace
      model_framework: mace
    """
    candidate = (
        cli_engine
        or config.get("platform")
        or config.get("engine")
        or config.get("model_framework")
        or config.get("framework")
        or "schnetpack"
    )

    engine = normalize_engine(candidate)

    # In old postprocessing configs, model_framework may be "schnetpack".
    # Map that to the default SchNetPack engine name for dispatch purposes.
    if engine == "schnetpack":
        engine = "schnet"

    if engine not in SUPPORTED_ENGINES:
        raise ValueError(
            "Could not detect a supported engine/platform from config. "
            "Please set one of: platform, engine, or model_framework. "
            f"Supported engines: {sorted(SUPPORTED_ENGINES)}. "
            f"Got: {candidate!r}"
        )

    return engine