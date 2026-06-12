#!/usr/bin/env python3

import argparse
import logging
import os
from orchestr_ai.preprocessing.consolidate_dataset import consolidate_dataset
from orchestr_ai.utils.compact import create_stacked_xyz
from orchestr_ai.utils.logging_utils import setup_logging
from orchestr_ai.utils.helpers import load_config_preproc, parse_args

setup_logging("data_preprocessing.log")
logger = logging.getLogger(__name__)

def main():
    args = parse_args(default="preprocess_config.yaml", description="Unified MLFF dataset generator")
    cfg = load_config_preproc(args.config)
    logger.info(f"Loaded config: {args.config}")

    # Resolve dataset inputs: prefer input_file; otherwise construct stacked XYZ from pos/frc.
    ds = cfg.get("dataset", {})

    # Pre-step: Merge and Center datasets
    merge_and_center_cfg = ds.get("merge_and_center", {})
    if merge_and_center_cfg.get("enabled", False):
        logger.info("Dataset merging and global centering is enabled.")
        from orchestr_ai.utils.merge_and_center import run_merge_and_center
        output_file = run_merge_and_center(merge_and_center_cfg, spin_state=ds.get("spin_state", "single"))
        
        if merge_and_center_cfg.get("only_merge_and_center", False):
            logger.info("only_merge_and_center is active. Exiting pipeline early.")
            return
            
        ds["input_file"] = output_file
        logger.info(f"Updated dataset.input_file to: {output_file}")

    input_file = ds.get("input_file")
    pos_file   = ds.get("pos_file")
    frc_file   = ds.get("frc_file")
    prefix     = ds.get("output_prefix", "dataset")

    spin_delta_cfg = ds.get("spin_delta_scaling", {})
    scaling_enabled = spin_delta_cfg.get("enabled", False)

    # Route 2: Direct Path
    if scaling_enabled and input_file and os.path.exists(input_file):
        logger.info(f"Spin delta scaling active with existing input file: {input_file}. Skipping consolidation.")
        from orchestr_ai.utils.preprocessing import compute_and_scale_delta_properties
        compute_and_scale_delta_properties(cfg, input_file)
        logger.info("Direct delta-scaling complete.")
        return

    # Route 1 / Standard path:
    if not input_file or str(input_file).strip() == "":
        if pos_file and frc_file:
            out_hartree = "combined_pos_frc_hartree.xyz"
            out_ev      = "combined_pos_frc_ev.xyz"
            logger.info(f"No dataset.input_file given; building stacked XYZ via compact step using pos={pos_file}, frc={frc_file}")
            create_stacked_xyz(pos_file, frc_file, out_hartree, out_ev)
            
            # Update config so consolidation uses the generated XYZ file.
            cfg.setdefault("dataset", {})["input_file"] = out_ev
            input_file = out_ev
            logger.info(f"Set dataset.input_file to: {out_ev}")
        else:
            raise ValueError(
                "Config must provide either dataset.input_file, or both dataset.pos_file and dataset.frc_file."
            )
    else:
        logger.info(f"Using existing dataset.input_file: {input_file}")

    consolidate_dataset(cfg)
    logger.info("Dataset consolidation complete.")

    # Route 1: Auto scaling after consolidation
    if scaling_enabled:
        logger.info("Executing automated delta-scaling step on the consolidated dataset.")
        from orchestr_ai.utils.preprocessing import compute_and_scale_delta_properties
        compute_and_scale_delta_properties(cfg, input_file)
        logger.info("Automated delta-scaling complete.")

if __name__ == "__main__":
    main()
