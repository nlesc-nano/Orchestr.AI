"""
create_plots.py

A command-line tool to generate plots from saved AL/UQ NPZ data files. This script
adds the project directory to PYTHONPATH, imports the necessary plotting functions from
plotting.py, and then processes each provided NPZ file based on the specified plot type.
"""

import argparse
import os
import sys
import traceback

# --- Add project directory to sys.path ---
# Ensure Python can find the other modules (like plotting.py)
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
    print(f"Added '{script_dir}' to sys.path")
# --- End path modification ---

# Import only the necessary plotting functions from plotting.py
try:
    from mlff_qd.postprocessing.plotting import (
        generate_uq_plots,
        generate_al_influence_plots,
        generate_al_traditional_plots
    )
    print("Successfully imported plotting functions from plotting.py")
except ImportError:
    print("Error: Could not import plotting functions from plotting.py.")
    print("Make sure plotting.py is in the same directory or your PYTHONPATH.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    traceback.print_exc()
    sys.exit(1)


def main():
    """
    Parses command-line arguments and calls the appropriate plotting functions based on
    the provided NPZ file(s) and the specified plot type.

    Usage examples:
        python create_plots.py uq_plot_data_eval_ensemble_ens10.npz --type uq
        python create_plots.py al_influence_plot_data.npz --type al_influence
    """
    parser = argparse.ArgumentParser(
        description="Generate plots from saved AL/UQ NPZ data files."
    )
    parser.add_argument(
        "npz_files",
        nargs='+',
        help=("Path(s) to the .npz file(s) containing plot data "
              "(e.g., uq_plot_data_eval_ensemble_ens10.npz, al_influence_plot_data.npz).")
    )
    parser.add_argument(
        "--type",
        choices=['uq', 'al_influence', 'al_traditional'],
        default='uq',
        help="Specify the type of plot data ('uq', 'al_influence', 'al_traditional'). Default is 'uq'."
    )
    # Optionally add more arguments (e.g., for an output directory)

    args = parser.parse_args()

    print(f"Processing {len(args.npz_files)} NPZ file(s) as type '{args.type}'...")

    # Select the plotting function based on the type argument
    if args.type == 'uq':
        plot_function = generate_uq_plots
    elif args.type == 'al_influence':
        plot_function = generate_al_influence_plots
    elif args.type == 'al_traditional':
        plot_function = generate_al_traditional_plots
    else:
        # This case is safeguarded by argparse choices
        print(f"Error: Unknown plot type '{args.type}'")
        sys.exit(1)

    successful_plots = 0
    for npz_file in args.npz_files:
        if not os.path.exists(npz_file):
            print(f"Error: File not found: {npz_file}. Skipping.")
            continue

        print(f"\nGenerating plots for: {os.path.basename(npz_file)}")
        try:
            if args.type == 'uq':
                # Simplified filename parsing for 'uq' plots
                parts = os.path.basename(npz_file) \
                    .replace('uq_plot_data_', '') \
                    .replace('.npz', '') \
                    .split('_')
                set_name = parts[0] if len(parts) > 0 else "UnknownSet"
                set_uq = parts[1] if len(parts) > 1 else "UnknownUQ"
                ensemble_size = None
                if "ens" in parts[-1]:
                    try:
                        ensemble_size = int(parts[-1].replace("ens", ""))
                    except ValueError:
                        pass
                # Call the UQ plotting function with parsed metadata
                plot_function(npz_file, set_name.capitalize(), set_uq, ensemble_size)
            else:
                # For active learning plots, only the file path is needed.
                plot_function(npz_file)

            successful_plots += 1
            print(f"Plots generated successfully from {os.path.basename(npz_file)}")
        except Exception as e:
            print(f"Error generating plots from {npz_file}: {e}")
            traceback.print_exc()

    print(f"\nFinished processing. Generated plots for {successful_plots}/{len(args.npz_files)} file(s).")


if __name__ == "__main__":
    main()

