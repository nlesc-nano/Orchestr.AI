# src/orchestr_ai/postprocessing/calculators/factory.py

from __future__ import annotations


def normalize_framework(framework: str | None) -> str:
    framework = (framework or "schnetpack").strip().lower()

    if framework in {"schnet", "painn", "so3net", "field_schnet", "fusion"}:
        return "schnetpack"

    if framework == "allegro":
        return "nequip"

    return framework


_DTYPES = {"float32": "float32", "fp32": "float32", "single": "float32",
           "float64": "float64", "fp64": "float64", "double": "float64"}


def resolve_dtype(config: dict) -> str:
    """Precision of every postprocessing run_type: config "default_dtype" (or "default_precision"); if unset, float64
    for GEO_OPT and float32 for the other run_types."""
    value = config.get("default_dtype") or config.get("default_precision")
    if not value:
        return "float64" if str(config.get("run_type", "")).strip().upper() == "GEO_OPT" else "float32"
    try:
        return _DTYPES[str(value).strip().lower()]
    except KeyError:
        raise ValueError(f"default_dtype must be float32 or float64, got {value!r}") from None


def load_model(model_path: str, framework: str, device, config: dict):
    """
    Load a model for postprocessing at the configured precision (resolve_dtype).

    Also sets torch's default dtype, because the calculators build their input tensors in it; every process that
    runs a model (including each spawned EVAL worker) loads it through here. NequIP/Allegro: returns the path,
    their calculator loads the compiled model in its own precision. The SchNetPack calculator runs in float32.
    """
    import torch

    dtype = resolve_dtype(config)
    framework = normalize_framework(framework)
    if framework == "nequip":
        return model_path
    if dtype == "float64" and framework != "mace":
        print(f"[load_model] WARNING: default_dtype float64 is only supported for MACE; the {framework} calculator runs in float32")
    torch.set_default_dtype(getattr(torch, dtype))
    model = torch.load(model_path, map_location=device, weights_only=False)
    try:
        model = model.to(device=device, dtype=getattr(torch, dtype))
    except Exception as exc:  # e.g. scripted models that cannot be cast
        print(f"[load_model] WARNING: could not cast {model_path} to {dtype} ({exc}); using it as stored")
        model = model.to(device=device)
    model.eval()
    return model


def create_calculator(
    framework: str,
    model_obj,
    device,
    config: dict,
    neighbor_list=None,
):
    """
    Create the correct postprocessing calculator.

    Imports are intentionally lazy so MACE/NequIP environments do not import
    SchNetPack, and SchNetPack environments do not need MACE/NequIP installed.
    """
    framework = normalize_framework(framework)

    if framework == "schnetpack":
        from orchestr_ai.postprocessing.neighbor_list import (
            NeighborListProvider,
            setup_neighbor_list,
        )
        from orchestr_ai.postprocessing.calculators.schnetpack_calculator import (
            SchnetpackCalculator,
        )

        if neighbor_list is None:
            neighbor_list = setup_neighbor_list(config)

        neighbor_list_provider = NeighborListProvider(
            config,
            existing_nl=neighbor_list,
        )

        return SchnetpackCalculator(
            model=model_obj,
            device=device,
            neighbor_list_provider=neighbor_list_provider,
        )

    if framework == "mace":
        import json
        import os

        cutoff = config.get("cutoff", 12.0)
        mace_head = config.get("mace_head", None)
        scale_metadata_path = config.get("scale_metadata_path") or config.get("mace_scale_metadata")
        if not scale_metadata_path:
            scale_metadata_path = "mace_scale_metadata.json"

        if mace_head == "triplet_reconstructed":
            from orchestr_ai.postprocessing.calculators.mace_calculator import (
                AutoScaledReconstructedMaceCalculator,
            )

            return AutoScaledReconstructedMaceCalculator(
                model=model_obj,
                device=device,
                cutoff=cutoff,
                scale_metadata_path=scale_metadata_path,
            )
        else:
            from orchestr_ai.postprocessing.calculators.mace_calculator import (
                MaceCalculator,
            )
            calc = MaceCalculator(
                model=model_obj,
                device=device,
                cutoff=cutoff,
                head=mace_head,
            )
            if "delta" in getattr(calc, "available_heads", []) and os.path.exists(scale_metadata_path):
                try:
                    with open(scale_metadata_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    calc.reconstruction_k_E = float(meta["k_E"])
                    calc.reconstruction_k_F = float(meta.get("k_F", meta["k_E"]))
                    calc.reconstruction_base_head = meta.get("base_head", "singlet")
                    calc.reconstruction_delta_head = meta.get("delta_head", "delta")
                    print(
                        "MACE: Loaded singlet+delta reconstruction metadata "
                        f"for diagnostics from {scale_metadata_path}."
                    )
                except Exception as exc:
                    print(f"MACE: Warning: failed to load reconstruction metadata: {exc}")
            return calc


    if framework == "nequip":
        from orchestr_ai.postprocessing.calculators.nequip_calculator import (
            NequipCalculator,
        )

        return NequipCalculator(
            model_obj=model_obj,
            device=device,
        )

    raise ValueError(
        f"Unknown model framework '{framework}'. "
        "Supported frameworks are: schnetpack, mace, nequip, allegro."
    )
