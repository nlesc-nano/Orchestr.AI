#!/usr/bin/env python3

import argparse
import logging

from mlff_qd.preprocessing.consolidate_ter import load_config, consolidate_dataset

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
    consolidate_dataset(cfg)
    logger.info("Dataset consolidation complete.")

if __name__ == "__main__":
    main()
