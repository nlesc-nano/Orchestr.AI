"""
`run_type: PROPS` for `python -m orchestr_ai.postprocessing config.yaml`.

Runs the QD properties pipeline over library records (`<id>/record.json` +
`start.xyz`), writing `<id>/props/`. Configuration:

    run_type: PROPS
    platform: mace                  # environment dispatch (orchestr_ai-mace)
    model_path: /path/to/macemh1model
    mace_head: omat_pbe
    props:
      records: [path/to/record_dir, ...]   # and/or
      tree: path/to/library                # every record.json below it
      max_atoms: 300                       # skip larger records (optional)
      steps: [relax, structure, hessian, vibspec, electronic, stability,
              detachment, solvation, sites, report, wigner, md]   # default: all
                                           # (md only with settings.md_enabled: the MD subset)
      force: false                         # ignore cached step results
      cif: null                            # bulk CIF (default: from record.origin.cif)
      settings:                            # any field of orchestr_ai.qd.run.Settings
        device: auto
        dtype: float64
        fmax: 0.01
"""
from __future__ import annotations

import json
import logging
from dataclasses import fields
from pathlib import Path

from . import STEPS
from .run import Settings, run_record


def _records(props: dict) -> list[Path]:
    dirs = [Path(p) for p in props.get("records") or []]
    if props.get("tree"):
        dirs += sorted(p.parent for p in Path(props["tree"]).rglob("record.json"))
    seen, out = set(), []
    for d in dirs:
        key = d.resolve()
        if key not in seen and (d / "record.json").is_file():
            seen.add(key)
            out.append(d)
    return out


def settings_from_config(config: dict) -> Settings:
    props = config.get("props") or {}
    known = {f.name for f in fields(Settings)}
    given = dict(props.get("settings") or {})
    unknown = sorted(set(given) - known)
    if unknown:
        raise ValueError(f"props.settings: unknown keys {', '.join(unknown)}")
    if config.get("mace_head"):
        given.setdefault("head", config["mace_head"])
    if config.get("model_path"):
        given.setdefault("model", str(config["model_path"]))
    return Settings(**given)


def run_props(config: dict) -> int:
    """Run the pipeline over the configured records; returns the number that failed."""
    props = config.get("props") or {}
    steps = list(props.get("steps") or STEPS)
    bad = sorted(set(steps) - set(STEPS))
    if bad:
        raise ValueError(f"props.steps: unknown steps {', '.join(bad)}")
    settings = settings_from_config(config)
    records = _records(props)
    if not records:
        raise ValueError("props: no record directory found (set props.records or props.tree)")
    max_atoms = props.get("max_atoms")
    failed = []
    for rd in records:
        n = json.loads((rd / "record.json").read_text()).get("n_atoms", 0)
        if max_atoms and n > max_atoms:
            logging.info(f"[PROPS] {rd.name}: {n} atoms > max_atoms {max_atoms}, skipped")
            continue
        logging.info(f"[PROPS] {rd.name}: {n} atoms, steps {', '.join(steps)}")
        try:
            run_record(rd, steps, settings, cif=props.get("cif"), force=bool(props.get("force", False)),
                       log=logging.info)
        except Exception as exc:  # keep the batch going
            failed.append(rd.name)
            logging.error(f"[PROPS] {rd.name}: FAILED {type(exc).__name__}: {exc}")
    logging.info(f"[PROPS] {len(records)} records, {len(failed)} failed")
    return len(failed)
