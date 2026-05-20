# src/orchestr_ai/postprocessing/calculator.py

"""
Backward-compatible calculator API.

New code should import from:
  orchestr_ai.postprocessing.calculators.factory
  orchestr_ai.postprocessing.calculators.base
  orchestr_ai.postprocessing.inference_runner
"""

from __future__ import annotations

import traceback

from orchestr_ai.postprocessing.calculators.base import BaseCalculator
from orchestr_ai.postprocessing.calculators.factory import create_calculator
from orchestr_ai.postprocessing.inference_runner import InferenceRunner

def require_schnetpack_interfaces():
    """
    Lazy import SchNetPack interface utilities.

    This prevents MACE/NequIP/Allegro environments from failing when
    calculator.py is imported.
    """
    try:
        from schnetpack.interfaces import AtomsConverter
        from schnetpack import properties as Properties
    except ImportError as e:
        raise ImportError(
            "SchNetPack is required for SchNetPack-based inference, "
            "but it is not installed in the current environment. "
            "Use the Orchestr.AI core environment for engines such as "
            "schnet, painn, so3net, field_schnet, and fusion."
        ) from e

    return AtomsConverter, Properties

def setup_neighbor_list(config):
    from orchestr_ai.postprocessing.neighbor_list import setup_neighbor_list as _setup_neighbor_list
    return _setup_neighbor_list(config)

def evaluate_model(
    frames,
    true_energies,
    true_forces,
    model_obj,
    device,
    batch_size,
    eval_log_file,
    config,
    neighbor_list=None,
):
    """
    Backward-compatible entry point used by evaluate.py.
    """
    try:
        framework = config.get("model_framework", "schnetpack").lower()

        calc = create_calculator(
            framework=framework,
            model_obj=model_obj,
            device=device,
            config=config,
            neighbor_list=neighbor_list,
        )

        runner = InferenceRunner(calc, batch_size, eval_log_file)

        return runner.run(
            frames=frames,
            true_energies=true_energies,
            true_forces=true_forces,
        )

    except Exception as e:
        print(f"Critical error initializing evaluate_model: {e}")
        traceback.print_exc()
        return None, None, None, None