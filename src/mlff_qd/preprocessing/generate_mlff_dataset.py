#!/usr/bin/env python3

import argparse
import logging

from mlff_qd.preprocessing.consolidate_ter import load_config, consolidate_dataset
from mlff_qd.utils.compact import create_stacked_xyz

	# --- Set up logging ---
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("mlff_dataset.log")])
logger = logging.getLogger(__name__)
def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified MLFF dataset generator (thin wrapper over consolidate_ter)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="preprocess_config.yaml",
        help="Path to YAML config (default: preprocess_config.yaml)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger.info(f"Loaded config: {args.config}")

    # --- Resolve dataset input: use input_file if present; otherwise build it from pos/frc via compact step ---
    ds = cfg.get("dataset", {})
    input_file = ds.get("input_file")
    pos_file   = ds.get("pos_file")
    frc_file   = ds.get("frc_file")
    prefix     = ds.get("output_prefix", "dataset")

    if not input_file or str(input_file).strip() == "":
        if pos_file and frc_file:
            out_hartree = "combined_pos_frc_hartree.xyz"
            out_ev      = "combined_pos_frc_ev.xyz"
            logger.info(f"No dataset.input_file given; building stacked XYZ via compact step using pos={pos_file}, frc={frc_file}")
            create_stacked_xyz(pos_file, frc_file, out_hartree, out_ev)
            
            # write back into cfg so consolidate_ter uses it transparently
            cfg.setdefault("dataset", {})["input_file"] = out_ev
            logger.info(f"Set dataset.input_file to: {out_ev}")
        else:
            raise ValueError(
                "Config must provide either dataset.input_file, or both dataset.pos_file and dataset.frc_file."
            )
    else:
        logger.info(f"Using existing dataset.input_file: {input_file}")

    consolidate_dataset(cfg)
    logger.info("Dataset consolidation complete.")

if __name__ == "__main__":
    main()
