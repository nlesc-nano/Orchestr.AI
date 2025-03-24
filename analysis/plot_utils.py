# plot_utils.py
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

def custom_plot_schnet(df, n_cols, metrics_to_plot, train_color, val_color, log_scale, grid, fig_width, fig_height, title_inputs, main_title):
    metric_pairs = {
        "Loss": ("train_loss", "val_loss", "Loss"),
        "Energy MAE": ("train_energy_MAE", "val_energy_MAE", "Energy MAE"),
        "Forces MAE": ("train_forces_MAE", "val_forces_MAE", "Forces MAE")
    }
    selected_pairs = [metric_pairs[m] for m in metrics_to_plot]
    n_plots = len(selected_pairs)
    n_rows = math.ceil(n_plots / n_cols)
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width * n_cols, fig_height * n_rows))
    fig.suptitle(main_title, fontsize=16)
    axs = axs.flatten()
    
    for i, (train_metric, val_metric, default_title) in enumerate(selected_pairs):
        train_df = df[df['Metric'] == train_metric]
        val_df = df[df['Metric'] == val_metric]
        title = title_inputs.get(default_title, f"SchNet - {default_title}")
        axs[i].plot(train_df['Step'], train_df['Value'], label='Train', linestyle='-', color=train_color)
        axs[i].plot(val_df['Step'], val_df['Value'], label='Validation', linestyle=':', color=val_color)
        axs[i].set_title(title)
        axs[i].set_xlabel('Step')
        axs[i].set_ylabel(default_title)
        if log_scale:
            axs[i].set_yscale('log')
        if grid:
            axs[i].grid(True)
        axs[i].legend()
    
    for j in range(n_plots, n_rows * n_cols):
        axs[j].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, n_rows

def custom_plot_nequip(df, n_cols, metrics_to_plot, train_color, val_color, log_scale, grid, fig_width, fig_height, title_inputs, main_title):
    metric_pairs = {
        "Total Loss": ("training_loss", "validation_loss", "Total Loss"),
        "F MAE": ("training_f_mae", "validation_f_mae", "F MAE"),
        "E MAE": ("training_e_mae", "validation_e_mae", "E MAE")
    }
    selected_pairs = [metric_pairs[m] for m in metrics_to_plot]
    n_plots = len(selected_pairs)
    n_rows = math.ceil(n_plots / n_cols)
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width * n_cols, fig_height * n_rows))
    fig.suptitle(main_title, fontsize=16)
    axs = axs.flatten()
    
    for i, (train_col, val_col, default_title) in enumerate(selected_pairs):
        title = title_inputs.get(default_title, f"Nequip - {default_title}")
        axs[i].plot(df['epoch'], df[train_col], label='Train', linestyle='-', color=train_color)
        axs[i].plot(df['epoch'], df[val_col], label='Validation', linestyle=':', color=val_color)
        axs[i].set_title(title)
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(default_title)
        if log_scale:
            axs[i].set_yscale('log')
        if grid:
            axs[i].grid(True)
        axs[i].legend()
    
    for j in range(n_plots, n_rows * n_cols):
        axs[j].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, n_rows

def custom_plotly_schnet(df, n_cols, metrics_to_plot, train_color, val_color, log_scale, grid, fig_width, fig_height, title_inputs, main_title):
    metric_pairs = {
        "Loss": ("train_loss", "val_loss", "Loss"),
        "Energy MAE": ("train_energy_MAE", "val_energy_MAE", "Energy MAE"),
        "Forces MAE": ("train_forces_MAE", "val_forces_MAE", "Forces MAE")
    }
    selected_pairs = [metric_pairs[m] for m in metrics_to_plot]
    n_plots = len(selected_pairs)
    n_rows = math.ceil(n_plots / n_cols)
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[title_inputs.get(default_title, f"SchNet - {default_title}") for _, _, default_title in selected_pairs],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    for i, (train_metric, val_metric, default_title) in enumerate(selected_pairs):
        train_df = df[df['Metric'] == train_metric]
        val_df = df[df['Metric'] == val_metric]
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        
        fig.add_trace(
            go.Scatter(x=train_df['Step'], y=train_df['Value'], mode='lines', name='Train', line=dict(color=train_color)),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(x=val_df['Step'], y=val_df['Value'], mode='lines', name='Validation', line=dict(color=val_color, dash='dash')),
            row=row, col=col
        )
    
    # Update layout for all subplots
    for i in range(n_plots):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        fig.update_xaxes(title_text="Step", row=row, col=col)
        fig.update_yaxes(title_text=selected_pairs[i][2], type="log" if log_scale else "linear", row=row, col=col)
    
    fig.update_layout(
        title=main_title,
        showlegend=True,
        height=fig_height * n_rows * 100,
        width=fig_width * n_cols * 100,
        title_x=0.5
    )
    return fig, n_rows

def custom_plotly_nequip(df, n_cols, metrics_to_plot, train_color, val_color, log_scale, grid, fig_width, fig_height, title_inputs, main_title):
    metric_pairs = {
        "Total Loss": ("training_loss", "validation_loss", "Total Loss"),
        "F MAE": ("training_f_mae", "validation_f_mae", "F MAE"),
        "E MAE": ("training_e_mae", "validation_e_mae", "E MAE")
    }
    selected_pairs = [metric_pairs[m] for m in metrics_to_plot]
    n_plots = len(selected_pairs)
    n_rows = math.ceil(n_plots / n_cols)
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[title_inputs.get(default_title, f"Nequip - {default_title}") for _, _, default_title in selected_pairs],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    for i, (train_col, val_col, default_title) in enumerate(selected_pairs):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df[train_col], mode='lines', name='Train', line=dict(color=train_color)),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(x=df['epoch'], y=df[val_col], mode='lines', name='Validation', line=dict(color=val_color, dash='dash')),
            row=row, col=col
        )
    
    # Update layout for all subplots
    for i in range(n_plots):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        fig.update_xaxes(title_text="Epoch", row=row, col=col)
        fig.update_yaxes(title_text=selected_pairs[i][2], type="log" if log_scale else "linear", row=row, col=col)
    
    fig.update_layout(
        title=main_title,
        showlegend=True,
        height=fig_height * n_rows * 100,
        width=fig_width * n_cols * 100,
        title_x=0.5
    )
    return fig, n_rows