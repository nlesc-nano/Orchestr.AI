import argparse
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

def validate_columns(df, required_cols, platform):
    """
    Check if the DataFrame has all required columns.
    If not, print an error message and exit.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Error: The following required columns are missing for {platform} platform: {missing}")
        print("Please check that you are using the correct CSV file for the selected platform.")
        sys.exit(1)

def save_and_show_plot(file_path, out_filename=None):
    """
    Saves the current plot in the current working directory.
    If out_filename is provided, ensures it ends with '.png' and saves the file there.
    Otherwise, saves the plot with a filename derived from the CSV file name (with '_plot.png') 
    in the working directory.
    """
    if out_filename:
        if not out_filename.lower().endswith('.png'):
            out_filename += '.png'
        # Save in the current working directory using the basename of the provided out_filename
        save_path = os.path.join(os.getcwd(), os.path.basename(out_filename))
    else:
        base_name = os.path.basename(file_path)
        plot_filename = os.path.splitext(base_name)[0] + '_plot.png'
        save_path = os.path.join(os.getcwd(), plot_filename)
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    # plt.show()

def plot_schnet(file_path, n_cols, out_filename=None):
    """
    Plot SchNet results.
    :param file_path: Path to the CSV file.
    :param n_cols: Number of columns in the subplot grid.
    :param out_filename: Optional output file name for the saved plot.
    """
    df = pd.read_csv(file_path)
    required_cols = ['epoch', 'step', 'train_loss', 'val_loss',
                     'train_energy_MAE', 'val_energy_MAE',
                     'train_forces_MAE', 'val_forces_MAE']
    validate_columns(df, required_cols, "SchNet")
    
    df_cleaned = df.dropna(subset=['epoch'])
    df_cleaned['epoch'] = df_cleaned['epoch'].astype(int)
    val_steps = df_cleaned.groupby("epoch")["step"].first()
    
    pairs = [
        ("train_loss", "val_loss", "Loss"),
        ("train_energy_MAE", "val_energy_MAE", "Energy MAE"),
        ("train_forces_MAE", "val_forces_MAE", "Forces MAE")
    ]
    
    n_plots = len(pairs)
    n_rows = math.ceil(n_plots / n_cols)
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 4 * n_rows))
    axs = axs.flatten()
    
    step_ticks = np.linspace(df_cleaned["step"].min(), df_cleaned["step"].max(), num=10, dtype=int)
    
    for i, (train_col, val_col, title) in enumerate(pairs):
        axs[i].plot(df_cleaned["step"], df_cleaned[train_col], label="Train", linestyle="-")
        axs[i].plot(val_steps, df_cleaned.groupby("epoch")[val_col].first(), label="Validation", linestyle=":", color="red")
        # Update the title style to include the platform name.
        axs[i].set_title(f"SchNet - {title}")
        axs[i].set_xlabel("Steps")
        axs[i].set_ylabel(title)
        axs[i].set_yscale("log")
        axs[i].legend()
        axs[i].grid()
        axs[i].set_xticks(step_ticks)
    
    for j in range(n_plots, n_rows * n_cols):
        axs[j].axis("off")
    
    plt.tight_layout()
    save_and_show_plot(file_path, out_filename)

def plot_nequip(file_path, n_cols, out_filename=None):
    """
    Plot Nequip results.
    :param file_path: Path to the CSV file.
    :param n_cols: Number of columns in the subplot grid.
    :param out_filename: Optional output file name for the saved plot.
    """
    df = pd.read_csv(file_path)
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
        description="Plot results for SchNet or Nequip using a provided CSV file."
    )
    parser.add_argument('--platform', required=True, choices=['schnet', 'nequip'],
                        help='Platform to plot results for: "schnet" or "nequip".')
    parser.add_argument('--file', required=True,
                        help='Path to CSV file with metrics.')
    parser.add_argument('--cols', type=int, default=2,
                        help='Number of columns for the subplot grid (default: 2)')
    parser.add_argument('--out', type=str, default=None,
                        help='Output file name for the saved plot (must be .png).')
    args = parser.parse_args()
    
    if args.platform.lower() == 'schnet':
        plot_schnet(args.file, args.cols, args.out)
    elif args.platform.lower() == 'nequip':
        plot_nequip(args.file, args.cols, args.out)
    else:
        sys.exit("Invalid platform specified. Use 'schnet' or 'nequip'.")

if __name__ == "__main__":
    main()
