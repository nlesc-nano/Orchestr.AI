# src/orchestr_ai/qd/__init__.py
"""
Ground-state properties of library quantum dots.

A standalone workflow, run over library records (`<id>/record.json` +
`start.xyz`), that relaxes each structure with MACE-MH-1 and computes
structural, vibrational and thermochemical properties, IR and Raman spectra
(MACE modes, g-xTB dipole and polarisability derivatives), stability against
bulk MA and molecular MX_q references and stepwise Z-type ligand detachment
(MACE-MH-1), and tight-binding electronic properties (GFN2-xTB).  Results go to `<id>/props/` next to the record; the webapp
ingest folds them into the library index.

    python -m orchestr_ai.qd run <record_dir> [--steps relax,structure,...,report]
    python -m orchestr_ai.qd batch <library tree> [--max-atoms N]
"""

SCHEMA_VERSION = 1
STEPS = ("relax", "structure", "hessian", "vibspec", "electronic", "stability", "detachment", "solvation", "sites", "report",
         "wigner", "md")
