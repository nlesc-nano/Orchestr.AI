#!/usr/bin/env python3
import os
import re
import sys
import json
import math
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Parse MACE training log and compute estimated triplet errors.")
    parser.add_argument("--log", required=True, help="Path to MACE stdout/SLURM log file.")
    parser.add_argument("--metadata", default="mace_scale_metadata.json", help="Path to mace_scale_metadata.json.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.log):
        print(f"Error: Log file not found: {args.log}")
        sys.exit(1)
        
    if not os.path.exists(args.metadata):
        print(f"Error: Metadata file not found: {args.metadata}")
        sys.exit(1)
        
    with open(args.metadata, "r") as f:
        try:
            meta = json.load(f)
        except Exception as e:
            print(f"Error reading metadata file: {e}")
            sys.exit(1)
            
    k_E = meta.get("k_E")
    k_F = meta.get("k_F")
    if k_E is None or k_F is None:
        print("Error: Could not find 'k_E' or 'k_F' in metadata JSON.")
        sys.exit(1)
        
    with open(args.log, "r") as f:
        log_content = f.read()
        
    # Regex to find table row: e.g. |  train_delta  |          37.9      |        150.0    |        69.45     |
    row_pattern = re.compile(
        r"\|\s*(train_delta|train_singlet|valid_delta|valid_singlet)\s*\|\s*([\d\.\-]+)\s*\|\s*([\d\.\-]+)\s*\|\s*([\d\.\-]+)\s*\|"
    )
    
    results = {}
    for line in log_content.splitlines():
        match = row_pattern.search(line)
        if match:
            config_type, mae_e, mae_f, rel_f = match.groups()
            results[config_type] = {
                "mae_e": float(mae_e),
                "mae_f": float(mae_f),
                "rel_f": float(rel_f)
            }
            
    required = {"train_delta", "train_singlet", "valid_delta", "valid_singlet"}
    if not required.issubset(results.keys()):
        print("Error: Could not find all required table rows (train_delta, train_singlet, valid_delta, valid_singlet) in the log file.")
        sys.exit(1)
        
    # Calculations
    train_e_delta_phys = results["train_delta"]["mae_e"] / k_E
    train_f_delta_phys = results["train_delta"]["mae_f"] / k_F
    train_e_triplet_rss = math.sqrt(results["train_singlet"]["mae_e"]**2 + train_e_delta_phys**2)
    train_f_triplet_rss = math.sqrt(results["train_singlet"]["mae_f"]**2 + train_f_delta_phys**2)
    
    valid_e_delta_phys = results["valid_delta"]["mae_e"] / k_E
    valid_f_delta_phys = results["valid_delta"]["mae_f"] / k_F
    valid_e_triplet_rss = math.sqrt(results["valid_singlet"]["mae_e"]**2 + valid_e_delta_phys**2)
    valid_f_triplet_rss = math.sqrt(results["valid_singlet"]["mae_f"]**2 + valid_f_delta_phys**2)
    
    # Generate table
    summary = []
    summary.append("\n=== Estimated Physical Triplet Error Summary ===")
    summary.append(f"Using scaling factors from metadata: k_E = {k_E:.6f}, k_F = {k_F:.6f}")
    summary.append("+-----------------------+--------------------+-----------------+")
    summary.append("|      config_type      | MAE E / meV / atom | MAE F / meV / A |")
    summary.append("+-----------------------+--------------------+-----------------+")
    summary.append(f"| train_singlet         |          {results['train_singlet']['mae_e']:9.2f} |        {results['train_singlet']['mae_f']:8.2f} |")
    summary.append(f"| train_delta_physical  |          {train_e_delta_phys:9.2f} |        {train_f_delta_phys:8.2f} |")
    summary.append(f"| train_triplet_est     |          {train_e_triplet_rss:9.2f} |        {train_f_triplet_rss:8.2f} |")
    summary.append(f"| valid_singlet         |          {results['valid_singlet']['mae_e']:9.2f} |        {results['valid_singlet']['mae_f']:8.2f} |")
    summary.append(f"| valid_delta_physical  |          {valid_e_delta_phys:9.2f} |        {valid_f_delta_phys:8.2f} |")
    summary.append(f"| valid_triplet_est     |          {valid_e_triplet_rss:9.2f} |        {valid_f_triplet_rss:8.2f} |")
    summary.append("+-----------------------+--------------------+-----------------+")
    summary.append("")
    
    summary_str = "\n".join(summary)
    
    # Print table to console
    print(summary_str)
    
    # Append to the log file
    with open(args.log, "a") as f:
        f.write(summary_str)
    print(f"Appended triplet error summary to {args.log}")

if __name__ == "__main__":
    main()
