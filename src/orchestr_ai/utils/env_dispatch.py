# src/orchestr_ai/utils/env_dispatch.py
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, NoReturn

class EnvDispatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class EnvProfile:
    """
    Defines where an engine should run.

    - If python_path is set, we use it directly.
    - Else, we resolve python from conda env name.
    """
    conda_env: Optional[str] = None
    python_path: Optional[str] = None


def _which_conda() -> Optional[str]:
    # Prefer the conda executable if present
    return shutil.which("conda") or shutil.which("micromamba")


def _conda_env_prefix_by_name(conda_exe: str) -> Dict[str, str]:
    """
    Return mapping: env_name -> prefix path using `conda env list --json`.
    Works for normal users and HPC modules alike.
    """
    try:
        proc = subprocess.run(
            [conda_exe, "env", "list", "--json"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as e:
        raise EnvDispatchError(f"Failed to run '{conda_exe} env list --json': {e}") from e

    try:
        data = json.loads(proc.stdout)
        env_paths = data.get("envs", [])
    except Exception as e:
        raise EnvDispatchError("Failed to parse conda env list JSON output.") from e

    mapping: Dict[str, str] = {}
    for p in env_paths:
        # env name = last path component in common conda layouts
        # This is robust enough for conda env list output.
        name = Path(p).name
        mapping[name] = p
    return mapping


def _python_from_conda_env(conda_env_name: str) -> str:
    """
    Resolve the python executable for a conda env name.
    """
    conda_exe = _which_conda()
    if not conda_exe:
        raise EnvDispatchError(
            "Cannot find 'conda' in PATH. Activate conda or load the conda module first."
        )

    env_map = _conda_env_prefix_by_name(conda_exe)
    prefix = env_map.get(conda_env_name)
    if not prefix:
        # Give a helpful error listing close candidates
        candidates = ", ".join(sorted(env_map.keys())[:20])
        raise EnvDispatchError(
            f"Conda env '{conda_env_name}' not found. "
            f"Known envs (first 20): {candidates}"
        )

    # Linux conda layout
    py = str(Path(prefix) / "bin" / "python")
    if not Path(py).exists():
        raise EnvDispatchError(f"Python not found at expected location: {py}")
    return py


def resolve_python(profile: EnvProfile) -> str:
    """
    Resolve python executable path for a given profile.
    Priority:
    1) explicit python_path
    2) conda env name
    """
    if profile.python_path:
        py = profile.python_path
        if not Path(py).exists():
            raise EnvDispatchError(f"Configured python_path does not exist: {py}")
        return py

    if profile.conda_env:
        return _python_from_conda_env(profile.conda_env)

    raise EnvDispatchError("EnvProfile must specify either python_path or conda_env.")


def _is_dispatched() -> bool:
    # Recursion guard: once we hop envs, do not hop again.
    return os.environ.get("ORCHESTRAI_DISPATCHED", "").strip() == "1"


def should_dispatch(engine: str, engine_to_profile: Dict[str, EnvProfile]) -> bool:
    """
    Decide if current run should dispatch to another env.
    """
    if _is_dispatched():
        return False

    profile = engine_to_profile.get(engine)
    if profile is None:
        return False  # unknown engine: let caller handle

    target_python = resolve_python(profile)
    # If target python is the same as current interpreter, no need to dispatch
    try:
        same = Path(target_python).resolve() == Path(sys.executable).resolve()
    except Exception:
        same = target_python == sys.executable
    return not same


def dispatch_to_engine_env(
    engine: str,
    engine_to_profile: Dict[str, EnvProfile],
    module: str = "orchestr_ai.training",
    extra_args: Optional[list[str]] = None,
) -> NoReturn:
    """
    Re-exec `python -m <module> ...` under the correct engine environment.

    Examples:
      dispatch_to_engine_env("mace", profiles, module="orchestr_ai.training")
      dispatch_to_engine_env("mace", profiles, module="orchestr_ai.postprocessing")
    """
    profile = engine_to_profile.get(engine)
    if profile is None:
        raise EnvDispatchError(f"No env profile configured for engine '{engine}'.")

    target_python = resolve_python(profile)

    argv = [target_python, "-m", module]
    argv += sys.argv[1:]

    if extra_args:
        argv += extra_args

    env = os.environ.copy()
    env["ORCHESTRAI_DISPATCHED"] = "1"
    env["ORCHESTRAI_ENGINE"] = engine
    env["ORCHESTRAI_DISPATCH_MODULE"] = module

    try:
        os.execve(target_python, argv, env)
    except Exception as e:
        raise EnvDispatchError(
            f"Failed to exec into engine env python '{target_python}'. "
            f"Command would have been: {' '.join(argv)}. Error: {e}"
        ) from e


def _env_label() -> str:
    return os.path.dirname(os.path.dirname(sys.executable))


def maybe_dispatch_to_engine_env(
    engine: str,
    module: str,
    engine_to_profile: Dict[str, EnvProfile],
) -> None:
    """
    Shared high-level dispatch helper.

    Returns normally if:
      - ORCHESTRAI_SINGLE_ENV=1
      - already dispatched
      - current Python is already the target Python

    Otherwise replaces the current process via os.execve().
    """
    single_env_mode = os.getenv("ORCHESTRAI_SINGLE_ENV", "0") == "1"

    print(f"Running in env prefix: {_env_label()}")

    if single_env_mode:
        print(f"[Orchestr.AI] Single-env mode enabled; no dispatch for engine '{engine}'")
        print(f"[Orchestr.AI] Engine: {engine} | Env prefix: {_env_label()}")
        print(f"[Orchestr.AI] Python: {sys.executable}")
        return

    if should_dispatch(engine, engine_to_profile):
        target = engine_to_profile[engine].conda_env or engine_to_profile[engine].python_path
        print(f"[Orchestr.AI] Dispatch: engine '{engine}' → env '{target}'")
        dispatch_to_engine_env(
            engine=engine,
            engine_to_profile=engine_to_profile,
            module=module,
        )

    print(f"[Orchestr.AI] Engine: {engine} | Env prefix: {_env_label()}")
    print(f"[Orchestr.AI] Python: {sys.executable}")