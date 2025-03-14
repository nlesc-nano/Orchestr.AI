import argparse
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

results_folder = os.path.join("analysis", "results")  # Define results folder path globally
def validate_columns(df, required_cols, platform):
    """
    Check if the DataFrame has all required columns.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Error: Required columns missing for {platform}: {missing}")
        print("Check CSV file and platform selection.")
        sys.exit(1)

def save_and_show_plot(file_path, out_filename=None):
    """
    Saves the plot in the 'results' folder. Assumes 'results' folder exists.
    """
    if out_filename:
        if not out_filename.lower().endswith('.png'):
            out_filename += '.png'
        save_path = os.path.join(results_folder, os.path.basename(out_filename))
    else:
        base_name = os.path.basename(file_path)
        plot_filename = os.path.splitext(base_name)[0] + '_plot.png'
        save_path = os.path.join(results_folder, plot_filename)
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    # plt.show()

def plot_schnet(file_path, n_cols, out_filename=None):
    """Plot SchNet results."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at path: {file_path}")
        print("Please check the file path provided with the --file argument.")
        sys.exit(1)

    required_cols = ['Step', 'Metric', 'Value']
    validate_columns(df, required_cols, "SchNet")

    metric_pairs = [
        ("train_loss", "val_loss", "Loss"),
        ("train_energy_MAE", "val_energy_MAE", "Energy MAE"),
        ("train_forces_MAE", "val_forces_MAE", "Forces MAE")
    ]

    n_plots = len(metric_pairs)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 4 * n_rows))
    axs = axs.flatten()

    for i, (train_metric, val_metric, title) in enumerate(metric_pairs):
        train_df = df[df['Metric'] == train_metric]
        val_df = df[df['Metric'] == val_metric]

        axs[i].plot(train_df['Step'], train_df['Value'], label='Train', linestyle='-')
        axs[i].plot(val_df['Step'], val_df['Value'], label='Validation', linestyle=':')

        axs[i].set_title(f"SchNet - {title}")
        axs[i].set_xlabel('Step')
        axs[i].set_ylabel(title)
        axs[i].set_yscale('log')
        axs[i].legend()
        axs[i].grid(True)

    for j in range(n_plots, n_rows * n_cols):
        axs[j].axis('off')

    plt.tight_layout()
    save_and_show_plot(file_path, out_filename)

def plot_nequip(file_path, n_cols, out_filename=None):
    """Plot Nequip results."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at path: {file_path}")
        print("Please check the file path provided with the --file argument.")
        sys.exit(1)

    df.columns = df.columns.str.strip()

    required_cols = ['epoch', 'training_loss', 'validation_loss',
                      'training_f_mae', 'validation_f_mae',
                      'training_e_mae', 'validation_e_mae']
    validate_columns(df, required_cols, "Nequip")

    pairs = [
        ('training_loss', 'validation_loss', 'Total Loss'),
        ('training_f_mae', 'validation_f_mae', 'F MAE'),
        ('training_e_mae', 'validation_e_mae', 'E MAE'),
    ]

    n_plots = len(pairs)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 4 * n_rows))
    axs = axs.flatten()

    for i, (train_col, val_col, title) in enumerate(pairs):
        axs[i].plot(df['epoch'], df[train_col], label='Train', linestyle='-')
        axs[i].plot(df['epoch'], df[val_col], label='Validation', linestyle=':')
        axs[i].set_title(f"Nequip - {title}")
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(title)
        axs[i].legend()
        axs[i].grid()

    for j in range(n_plots, n_rows * n_cols):
        axs[j].axis('off')

    plt.tight_layout()
    save_and_show_plot(file_path, out_filename)

def main():
    parser = argparse.ArgumentParser(
        description="Plot results for SchNet or Nequip from a CSV file (plot saved in 'results' folder)."
    )
    parser.add_argument('--platform', required=True, choices=['schnet', 'nequip'],
                        help='Platform to plot results for: "schnet" or "nequip".')
    parser.add_argument('--file', required=True,
                        help='Path to CSV file with metrics (e.g., in "results" folder).') # Updated help text to suggest "results" folder
    parser.add_argument('--cols', type=int, default=2,
                        help='Number of columns for the subplot grid (default: 2)')
    parser.add_argument('--out', type=str, default=None,
                        help='Output plot file name (optional, .png, saved in "results" folder).') # Updated help text
    args = parser.parse_args()

    # Ensure 'results' folder exists at the start of main()
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        print(f"Created folder: {results_folder}")

    if args.platform.lower() == 'schnet':
        plot_schnet(args.file, args.cols, args.out)
    elif args.platform.lower() == 'nequip':
        plot_nequip(args.file, args.cols, args.out)
    else:
        sys.exit("Invalid platform specified. Use 'schnet' or 'nequip'.")

if __name__ == "__main__":
    main()