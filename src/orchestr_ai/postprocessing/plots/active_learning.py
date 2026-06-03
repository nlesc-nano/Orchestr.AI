import os
import traceback
import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(scores, title, xlabel, filename, bins=50, vline_values=None, vline_labels=None, vline_colors=None):
    """
    Plots and saves a histogram with optional vertical reference lines.

    Parameters:
        scores (np.ndarray): Data to plot.
        title (str): Plot title.
        xlabel (str): X-axis label.
        filename (str): Output filename.
        bins (int): Number of bins (default 50).
        vline_values (list): X-values for vertical lines.
        vline_labels (list): Corresponding labels.
        vline_colors (list): Colors for each vertical line.

    Returns:
        None
    """
    plt.figure(figsize=(8, 5))
    valid_scores = scores[~np.isnan(scores)]
    if len(valid_scores) == 0:
        print(f"Warning: No valid scores for histogram {filename}")
        plt.close()
        return
    plt.hist(valid_scores, bins=bins, alpha=0.7, color='teal', label='_nolegend_')
    all_handles, all_labels = [], []
    if vline_values and vline_labels and vline_colors and len(vline_values) == len(vline_labels) == len(vline_colors):
        for val, label, color in zip(vline_values, vline_labels, vline_colors):
            if np.isfinite(val):
                line = plt.axvline(val, color=color, linestyle='--', label=f'{label} ({val:.3f})')
                all_handles.append(line)
                all_labels.append(f'{label} ({val:.3f})')
            else:
                print(f"Warning: Skipping NaN/inf vline for {label} in {filename}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    if all_handles:
        plt.legend(handles=all_handles, labels=all_labels, fontsize=9)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Generated plot: {filename}")
    plt.close()


def plot_overall_score_vs_clusters(overall_score, hdbscan_cluster_labels, unique_eval_frames, selected_indices):
    """
    Plots overall score versus evaluation frame index, colored by HDBSCAN cluster.

    Parameters:
        overall_score (np.ndarray): Array of overall scores for evaluation frames.
        hdbscan_cluster_labels (np.ndarray): Cluster labels from HDBSCAN.
        unique_eval_frames (np.ndarray): Array of evaluation frame indices.
        selected_indices (list): List of frame indices selected by an AL procedure.

    Returns:
        None
    """
    print("Generating overall score vs. cluster plot...")
    plt.figure(figsize=(12, 7))
    x_values = np.arange(len(unique_eval_frames))
    unique_clusters = np.unique(hdbscan_cluster_labels)
    cmap = plt.cm.viridis
    if len(overall_score) != len(hdbscan_cluster_labels) or len(hdbscan_cluster_labels) != len(unique_eval_frames):
        print("Error plotting clusters: Length mismatch!")
        plt.close()
        return

    noise_color = 'grey'
    cluster_ids = sorted([c for c in unique_clusters if c >= 0])
    num_real_clusters = len(cluster_ids)
    cluster_colors = {cid: cmap(i / max(1, num_real_clusters - 1)) for i, cid in enumerate(cluster_ids)}
    cluster_colors[-1] = noise_color

    plotted_labels = set()
    for cluster_id in unique_clusters:
        mask = (hdbscan_cluster_labels == cluster_id)
        label_text = f'Cluster {cluster_id}' if cluster_id != -1 else 'Noise'
        color = cluster_colors[cluster_id]
        current_label = label_text if label_text not in plotted_labels else None
        plt.scatter(x_values[mask], overall_score[mask], color=color, alpha=0.6, label=current_label, s=20)
        if current_label:
            plotted_labels.add(current_label)

    frame_idx_to_pos = {f_idx: pos for pos, f_idx in enumerate(unique_eval_frames)}
    selected_pos_indices = [frame_idx_to_pos[idx] for idx in selected_indices if idx in frame_idx_to_pos]
    selected_mask = np.zeros(len(unique_eval_frames), dtype=bool)
    if selected_pos_indices:
        selected_mask[selected_pos_indices] = True
    if np.any(selected_mask):
        plt.scatter(x_values[selected_mask], overall_score[selected_mask],
                    c='black', marker='x', s=50, label='Selected', zorder=5)

    plt.xlabel('Evaluation Frame Index (Position)')
    plt.ylabel('Overall Score')
    plt.title('Overall Score vs. HDBSCAN Clustering')
    handles, labels_leg = plt.gca().get_legend_handles_labels()
    max_legend = 15
    if len(handles) > max_legend:
        step = max(1, len(handles) // max_legend)
        handles = handles[::step]
        labels_leg = labels_leg[::step]
    if handles:
        plt.legend(handles, labels_leg, bbox_to_anchor=(1.04, 1), loc='upper left', fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig('trad_overall_score_vs_clusters.png')
    plt.close()
    print("Generated overall score vs. cluster plot.")


def plot_rmse_vs_sorted_overall_score(overall_score, agg_error, frame_conf_labels, frame_error_labels):
    """
    Plots Aggregated Error (RMSE) versus sorted overall score, colored by confidence and error status.

    Parameters:
        overall_score (np.ndarray): Overall scores per evaluation frame.
        agg_error (np.ndarray): Aggregated error (RMSE) per frame.
        frame_conf_labels (list): Confidence labels per frame.
        frame_error_labels (list): Error labels per frame.

    Returns:
        None
    """
    print("Generating detailed RMSE vs sorted overall score plot...")
    plt.figure(figsize=(12, 7))
    valid_mask = ~np.isnan(overall_score) & ~np.isnan(agg_error)
    if not np.any(valid_mask):
        print("No valid data for RMSE vs Score plot.")
        plt.close()
        return
    overall_score_v = overall_score[valid_mask]
    agg_error_v = agg_error[valid_mask]
    frame_conf_labels_v = np.array(frame_conf_labels)[valid_mask]
    frame_error_labels_v = np.array(frame_error_labels)[valid_mask]
    sort_indices = np.argsort(overall_score_v)
    sorted_scores = overall_score_v[sort_indices]
    sorted_rmse = agg_error_v[sort_indices]
    sorted_conf = frame_conf_labels_v[sort_indices]
    sorted_err = frame_error_labels_v[sort_indices]
    masks = {
        'W/N': (sorted_conf == "within") & (sorted_err == "normal"),
        'W/H': (sorted_conf == "within") & (sorted_err == "high"),
        'Ov/N': (sorted_conf == "over") & (sorted_err == "normal"),
        'Ov/H': (sorted_conf == "over") & (sorted_err == "high"),
        'Un/N': (sorted_conf == "under") & (sorted_err == "normal"),
        'Un/H': (sorted_conf == "under") & (sorted_err == "high")
    }
    colors = {
        'W/N': 'grey',
        'W/H': 'orange',
        'Ov/N': 'deepskyblue',
        'Ov/H': 'mediumblue',
        'Un/N': 'lightcoral',
        'Un/H': 'firebrick'
    }
    alphas = {
        'W/N': 0.4,
        'W/H': 0.6,
        'Ov/N': 0.5,
        'Ov/H': 0.7,
        'Un/N': 0.5,
        'Un/H': 0.7
    }
    sizes = {
        'W/N': 15,
        'W/H': 25,
        'Ov/N': 20,
        'Ov/H': 30,
        'Un/N': 20,
        'Un/H': 30
    }
    full_labels = {
        'W/N': 'Within CI / Normal Error',
        'W/H': 'Within CI / High Error',
        'Ov/N': 'Overconfident / Normal Error',
        'Ov/H': 'Overconfident / High Error',
        'Un/N': 'Underconfident / Normal Error',
        'Un/H': 'Underconfident / High Error'
    }
    for key, mask in masks.items():
        if np.any(mask):
            plt.scatter(sorted_scores[mask], sorted_rmse[mask], c=colors[key],
                        alpha=alphas[key], s=sizes[key], label=full_labels[key])
    plt.xlabel('Sorted Overall Score')
    plt.ylabel('Aggregated Error (RMSE per Frame)')
    plt.title('RMSE vs. Sorted Overall Score (Colored by Confidence & Error Status)')
    plt.legend(fontsize=9, loc='best')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('trad_rmse_vs_sorted_overall_score.png')
    plt.close()
    print("Generated detailed RMSE vs sorted overall score plot.")


def plot_error_vs_uncertainty(agg_unc, agg_error, selected_indices, lower_threshold, upper_threshold, av_score, unique_eval_frames):
    """
    Plots Aggregated Error vs Aggregated Uncertainty on a log-log scale.

    Parameters:
        agg_unc (np.ndarray): Aggregated uncertainty per frame.
        agg_error (np.ndarray): Aggregated error (RMSE) per frame.
        selected_indices (list): List of selected frame indices.
        lower_threshold (float): Lower threshold for confidence.
        upper_threshold (float): Upper threshold for confidence.
        av_score (float): Average combined score.
        unique_eval_frames (np.ndarray): Array of evaluation frame indices.

    Returns:
        None
    """
    print("Generating error vs uncertainty plot...")
    plt.figure(figsize=(8, 6))
    valid_mask = ~np.isnan(agg_unc) & ~np.isnan(agg_error)
    if not np.any(valid_mask):
        print("No valid data for Error vs Uncertainty plot.")
        plt.close()
        return
    agg_unc_v = agg_unc[valid_mask]
    agg_error_v = agg_error[valid_mask]
    unique_eval_frames_v = unique_eval_frames[valid_mask]
    
    frame_idx_to_pos_v = {f_idx: pos for pos, f_idx in enumerate(unique_eval_frames_v)}
    selected_pos_indices = [frame_idx_to_pos_v[idx] for idx in selected_indices if idx in frame_idx_to_pos_v]
    selected_mask = np.zeros(len(unique_eval_frames_v), dtype=bool)
    if selected_pos_indices:
        selected_mask[selected_pos_indices] = True

    plt.scatter(agg_unc_v, agg_error_v, color='blue', alpha=0.3, label='All Eval Frames', s=15)
    plt.scatter(agg_unc_v[selected_mask], agg_error_v[selected_mask],
                color='orange', alpha=0.8, label='Selected Frames', s=30, edgecolors='k', lw=0.5)
    eps = 1e-9
    unc_min = np.nanmin(agg_unc_v[agg_unc_v > eps]) if np.any(agg_unc_v > eps) else eps
    unc_max = np.nanmax(agg_unc_v) if np.any(agg_unc_v > eps) else eps * 10
    unc_range = np.logspace(np.log10(unc_min), np.log10(unc_max), 200)
    ideal_line = unc_range
    lower_line = unc_range / max(eps, upper_threshold)
    upper_line = unc_range / max(eps, lower_threshold)
    plt.plot(unc_range, ideal_line, 'k--', label='Error = Unc', lw=1)
    plt.plot(unc_range, lower_line, 'r--', label=f'Underconf Thr ({upper_threshold:.2f})', lw=1)
    plt.plot(unc_range, upper_line, 'b--', label=f'Overconf Thr ({lower_threshold:.2f})', lw=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Aggregated Uncertainty (Mean per Frame)')
    plt.ylabel('Aggregated Error (Mean RMSE per Frame)')
    plt.title('Error vs. Uncertainty (Log-Log)')
    y_min = max(eps, np.nanmin(agg_error_v[agg_error_v > eps]) if np.any(agg_error_v > eps) else eps) * 0.5
    y_max = max(eps * 10, np.nanmax(agg_error_v)) * 2.0
    plt.ylim(max(y_min, 1e-7), y_max)
    plt.xlim(unc_min * 0.9, unc_max * 1.1)
    plt.fill_between(unc_range, upper_line, y_max, where=upper_line < y_max,
                     color='blue', alpha=0.1, label='Overconfident Region')
    plt.fill_between(unc_range, y_min, lower_line, where=lower_line > y_min,
                     color='red', alpha=0.1, label='Underconfident Region')
    plt.legend(fontsize=8)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('trad_error_vs_uncertainty_loglog.png')
    plt.close()
    print("Generated error vs uncertainty plot.")


# === New Plotting Functions for Active Learning Data ===

def generate_al_influence_plots(npz_path):
    """
    Generates plots specific to the Influence Active Learning method using saved NPZ data.

    Parameters:
        npz_path (str): Path to the NPZ file containing plot data.

    Returns:
        None
    """
    if not npz_path or not os.path.exists(npz_path):
        print(f"Warning: Influence AL plot data not found at {npz_path}. Skipping plots.")
        return

    print(f"\n--- Generating Plots for Influence AL from {npz_path} ---")
    try:
        data = np.load(npz_path)
        x_sorted_indices = np.argsort(data['X_valid'])
        x_plot = data['X_valid'][x_sorted_indices]
        y_calibrated_plot = data['y_calibrated_plot']
        corr_raw = data['corr_raw'].item()
        corr_calib = data['corr_calib'].item()

        plt.figure(figsize=(8, 6))
        plt.scatter(data['X_valid'], data['y_valid'], alpha=0.5, label='Raw Data', s=10)
        plt.plot(x_plot, y_calibrated_plot, color='red', linewidth=2, label='Isotonic Fit')
        plt.xlabel("Raw Uncertainty (Avg per Frame)")
        plt.ylabel("Actual Error (Avg per Frame)")
        title = f"Isotonic Regression Calibration\nCorr(Raw): {corr_raw:.3f}, Corr(Calib): {corr_calib:.3f}"
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.6)
        plot_filename = npz_path.replace("_plot_data.npz", "_calibration_plot.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved calibration plot: {plot_filename}")
    except Exception as e:
        print(f"Error generating influence AL plots: {e}")
        traceback.print_exc()
        plt.close("all")


def generate_al_traditional_plots(npz_path):
    """
    Generates plots specific to the Traditional Active Learning method using saved NPZ data.
    
    Parameters:
        npz_path (str): Path to the NPZ file with Traditional AL plot data.
    
    Returns:
        None
    """
    if not npz_path or not os.path.exists(npz_path):
        print(f"Warning: Traditional AL plot data not found at {npz_path}. Skipping plots.")
        return

    print(f"\n--- Generating Plots for Traditional AL from {npz_path} ---")
    try:
        data = np.load(npz_path)
        overall_score = data["overall_score"]
        agg_error = data["agg_error"]
        agg_unc = data["agg_unc"]
        combined_score = data["combined_score"]
        error_score = data["error_score"]
        frame_conf_labels = data["frame_conf_labels"]
        frame_error_labels = data["frame_error_labels"]
        hdbscan_cluster_labels = data["hdbscan_cluster_labels"]
        unique_eval_frames = data["unique_eval_frames"]
        selected_indices = data["selected_indices"]
        lower_thr_cs = data["lower_thr_cs"].item()
        upper_thr_cs = data["upper_thr_cs"].item()
        av_cs = data["av_cs"].item()
        z_threshold_high_es = data["z_threshold_high_es"].item()
        mean_es = data["mean_es"].item()
        std_es = data["std_es"].item()
        lower_perc_overall = np.percentile(overall_score[~np.isnan(overall_score)], 5)
        upper_perc_overall = np.percentile(overall_score[~np.isnan(overall_score)], 95)
        threshold_es_value = mean_es + z_threshold_high_es * std_es if std_es > 1e-9 else np.nan

        plot_dir = os.path.dirname(npz_path)
        base_filename = os.path.basename(npz_path).replace("_plot_data.npz", "")

        # Call the plotting helpers for Traditional AL.
        plot_overall_score_vs_clusters(overall_score, hdbscan_cluster_labels, unique_eval_frames, selected_indices)
        plot_rmse_vs_sorted_overall_score(overall_score, agg_error, frame_conf_labels, frame_error_labels)
        plot_error_vs_uncertainty(agg_unc, agg_error, selected_indices, lower_thr_cs, upper_thr_cs, av_cs, unique_eval_frames)
        plot_histogram(combined_score, 'Dist Combined Score (CS)', 'CS',
                       os.path.join(plot_dir, f"{base_filename}_cs_dist.png"),
                       vline_values=[lower_thr_cs, upper_thr_cs],
                       vline_labels=['LowCI', 'HighCI'],
                       vline_colors=['blue', 'red'])
        plot_histogram(error_score, 'Dist Error Score (ES)', 'ES',
                       os.path.join(plot_dir, f"{base_filename}_es_dist.png"),
                       vline_values=[threshold_es_value],
                       vline_labels=['HighErrThr'],
                       vline_colors=['red'])
        plot_histogram(overall_score, 'Dist Overall Score', 'Overall Score',
                       os.path.join(plot_dir, f"{base_filename}_overall_dist.png"),
                       vline_values=[lower_perc_overall, upper_perc_overall],
                       vline_labels=['5th %ile', '95th %ile'],
                       vline_colors=['purple', 'purple'])
        print("Traditional AL plot generation finished.")
    except KeyError as e_key:
        print(f"Error loading data from {npz_path}: Missing key {e_key}. Cannot generate plots.")
    except Exception as e:
        print(f"Error generating traditional AL plots: {e}")
        traceback.print_exc()
        plt.close("all")


