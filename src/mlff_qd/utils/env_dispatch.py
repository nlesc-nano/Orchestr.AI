# src/mlff_qd/utils/env_dispatch.py
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


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
    return shutil.which("conda")


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
    return os.environ.get("MLFFQD_DISPATCHED", "").strip() == "1"


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
    extra_args: Optional[list[str]] = None,
) -> "NoReturn":
    """
    Re-exec `python -m mlff_qd.training ...` under the correct python.

    You call this early, after parsing args enough to know the engine.
    """
    profile = engine_to_profile.get(engine)
    if profile is None:
        raise EnvDispatchError(f"No env profile configured for engine '{engine}'.")

    target_python = resolve_python(profile)

    # Re-run the same module under the other env, passing through args.
    # We preserve the user's args, and optionally allow extra args.
    argv = [target_python, "-m", "mlff_qd.training"]
    argv += sys.argv[1:]
    if extra_args:
        argv += extra_args

    env = os.environ.copy()
    env["MLFFQD_DISPATCHED"] = "1"
    env["MLFFQD_ENGINE"] = engine  # useful for debugging/logging

    # Optional: carry over current working dir and PYTHONPATH
    # - If MLFF_QD is installed in both envs (recommended), no PYTHONPATH is needed.
    # - If in editable mode, still fine.

    try:
        # Replace current process; no double-logs, no nested lightning processes.
        os.execve(target_python, argv, env)
    except Exception as e:
        raise EnvDispatchError(
            f"Failed to exec into engine env python '{target_python}'. "
            f"Command would have been: {' '.join(argv)}. Error: {e}"
        ) from e
