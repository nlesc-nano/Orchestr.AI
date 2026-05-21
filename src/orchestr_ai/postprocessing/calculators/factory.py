# src/orchestr_ai/postprocessing/calculators/factory.py

from __future__ import annotations


def normalize_framework(framework: str | None) -> str:
    framework = (framework or "schnetpack").strip().lower()

    if framework in {"schnet", "painn", "so3net", "field_schnet", "fusion"}:
        return "schnetpack"

    if framework == "allegro":
        return "nequip"

    return framework


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
        from orchestr_ai.postprocessing.calculators.mace_calculator import (
            MaceCalculator,
        )

        cutoff = config.get("cutoff", 12.0)
        mace_head = config.get("mace_head", None)

        return MaceCalculator(
            model=model_obj,
            device=device,
            cutoff=cutoff,
            head=mace_head,
        )


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