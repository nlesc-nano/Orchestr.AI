"""
uq_models.py

This module provides functions to train and use uncertainty quantification (UQ)
models for ML force-field residuals using PyCaret and Gaussian Mixture Models (GMMs).
It includes functions to train models for predicting residuals (energy and force)
and to predict uncertainties given new features. If PyCaret is unavailable, a warning
is issued and training/prediction functions return None.
"""

import numpy as np
import pandas as pd
import joblib
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import traceback

# Check for PyCaret installation
try:
    import pycaret
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False
    print("Warning: PyCaret not found. UQ model training/prediction (train_uq_models, predict_uncertainties) will be unavailable.")


def train_uq_models(features_val, energy_residuals_abs, force_residuals_mae, gpu_available=False):
    """
    Trains ML models (using PyCaret, typically with XGBoost) to predict residuals.
    
    This function sets up two regression experiments with PyCaret:
        - One for energy residuals.
        - One for force residuals.
    It compares and tunes models (restricting to XGBoost), finalizes the best model,
    and saves it to a pipeline file. If PyCaret is not installed, the function returns (None, None).
    
    Parameters:
        features_val (np.ndarray): Feature matrix used as predictors.
        energy_residuals_abs (np.ndarray): Absolute energy residuals (target for energy model).
        force_residuals_mae (np.ndarray): MAE of force residuals (target for force model).
        gpu_available (bool): If True, attempts to use GPU support in PyCaret.
    
    Returns:
        tuple: (best_model_e, best_model_f) for energy and force residuals, or (None, None) on failure.
    """
    if not PYCARET_AVAILABLE:
        print("Error: Cannot train UQ models because PyCaret is not installed.")
        return None, None

    from pycaret.regression import setup, compare_models, save_model, finalize_model, create_model

    best_model_e, best_model_f = None, None

    # --- Energy Model ---
    print("\n--- Training UQ Model for Energy Residuals ---")
    try:
        data_e = pd.DataFrame(
            features_val,
            columns=[f'feature_{i}' for i in range(features_val.shape[1])]
        )
        data_e['target'] = energy_residuals_abs
        print("Setting up PyCaret for energy residuals...")
        sys.stdout.flush()
        setup_kwargs_e = dict(
            data=data_e,
            target='target',
            session_id=123,
            train_size=0.8,
            fold=5,
            n_jobs=-1,
            verbose=False,
            log_experiment=False,
            html=False
        )
        if gpu_available:
            setup_kwargs_e['use_gpu'] = True
        try:
            reg_e = setup(**setup_kwargs_e)
            print("PyCaret setup complete.")
        except Exception as e_setup:
            print(f"[ERROR] PyCaret setup failed: {e_setup}")
            raise

        create_model_kwargs = {'tree_method': 'hist'}
        if gpu_available and getattr(reg_e, 'gpu_used', False):
            create_model_kwargs = {'tree_method': 'gpu_hist', 'device': 'cuda'}
        print(f"Creating XGBoost model ({'GPU' if gpu_available and getattr(reg_e, 'gpu_used', False) else 'CPU'})...")
        # Optionally create the model; here compare_models will handle tuning.
        print("Comparing/tuning XGBoost model for energy residuals...")
        sys.stdout.flush()
        best_model_e = compare_models(
            include=['xgboost'],
            sort='mae',
            n_select=1,
            verbose=True
        )
        if best_model_e:
            print("Finalizing and saving best energy residuals model...")
            final_model_e = finalize_model(best_model_e)
            save_model(final_model_e, 'best_model_energy_pipeline')
            print("Energy residuals model training completed.")
        else:
            print("Error: Failed to train/select energy model.")
    except Exception as e:
        print(f"Error during Energy UQ model training: {e}")
        traceback.print_exc()
        best_model_e = None

    # --- Force Model ---
    print("\n--- Training UQ Model for Force Residuals ---")
    try:
        data_f = pd.DataFrame(
            features_val,
            columns=[f'feature_{i}' for i in range(features_val.shape[1])]
        )
        data_f['target'] = force_residuals_mae
        print("Setting up PyCaret for force residuals...")
        sys.stdout.flush()
        setup_kwargs_f = dict(
            data=data_f,
            target='target',
            session_id=124,
            train_size=0.8,
            fold=5,
            n_jobs=-1,
            verbose=False,
            log_experiment=False,
            html=False
        )
        if gpu_available:
            setup_kwargs_f['use_gpu'] = True
        try:
            reg_f = setup(**setup_kwargs_f)
            print("PyCaret setup complete.")
        except Exception as e_setup:
            print(f"[ERROR] PyCaret setup failed: {e_setup}")
            raise

        create_model_kwargs_f = {'tree_method': 'hist'}
        if gpu_available and getattr(reg_f, 'gpu_used', False):
            create_model_kwargs_f = {'tree_method': 'gpu_hist', 'device': 'cuda'}
        print(f"Creating XGBoost model ({'GPU' if gpu_available and getattr(reg_f, 'gpu_used', False) else 'CPU'})...")
        print("Comparing/tuning XGBoost model for force residuals...")
        sys.stdout.flush()
        best_model_f = compare_models(
            include=['xgboost'],
            sort='mae',
            n_select=1,
            verbose=True
        )
        if best_model_f:
            print("Finalizing and saving best force residuals model...")
            final_model_f = finalize_model(best_model_f)
            save_model(final_model_f, 'best_model_force_pipeline')
            print("Force residuals model training completed.")
        else:
            print("Error: Failed to train/select force model.")
    except Exception as e:
        print(f"Error during Force UQ model training: {e}")
        traceback.print_exc()
        best_model_f = None

    return best_model_e, best_model_f


def predict_uncertainties(best_model_e, best_model_f, features):
    """
    Predicts uncertainties using pre-trained/loaded PyCaret models.

    Parameters:
        best_model_e: Pre-trained PyCaret regression model for energy residuals.
        best_model_f: Pre-trained PyCaret regression model for force residuals.
        features (np.ndarray): Feature matrix for prediction.

    Returns:
        tuple: (sigma_e_pred, sigma_f_pred) uncertainties for energy and force.
               Returns (None, None) if prediction fails.
    """
    if not PYCARET_AVAILABLE:
        print("Error: Cannot predict uncertainties because PyCaret is not installed.")
        return None, None

    from pycaret.regression import predict_model, load_model

    # Load models from saved pipelines if not provided.
    if best_model_e is None and os.path.exists('best_model_energy_pipeline.pkl'):
        try:
            print("Loading saved energy uncertainty model...")
            best_model_e = load_model('best_model_energy_pipeline', verbose=False)
        except Exception as e:
            print(f"Error loading energy model: {e}")
    if best_model_f is None and os.path.exists('best_model_force_pipeline.pkl'):
        try:
            print("Loading saved force uncertainty model...")
            best_model_f = load_model('best_model_force_pipeline', verbose=False)
        except Exception as e:
            print(f"Error loading force model: {e}")

    sigma_e_pred, sigma_f_pred = None, None
    if best_model_e is None and best_model_f is None:
        print("Error: No valid UQ models provided or loaded.")
        return None, None

    try:
        feature_names = [f'feature_{i}' for i in range(features.shape[1])]
        data = pd.DataFrame(features, columns=feature_names)
        print(f"Predicting uncertainties for {len(data)} data points...")
        if best_model_e:
            print("Predicting energy uncertainty...")
            pred_e = predict_model(best_model_e, data=data)
            label_col_e = 'prediction_label'
            if label_col_e in pred_e.columns:
                sigma_e_pred = pred_e[label_col_e].values
                sigma_e_pred = np.maximum(sigma_e_pred, 0)
                print(f"  Energy uncertainty shape: {sigma_e_pred.shape}, Mean: {np.nanmean(sigma_e_pred):.4f}")
            else:
                print(f"Warning: Column '{label_col_e}' not found in energy predictions.")
        if best_model_f:
            print("Predicting force uncertainty...")
            pred_f = predict_model(best_model_f, data=data)
            label_col_f = 'prediction_label'
            if label_col_f in pred_f.columns:
                sigma_f_pred = pred_f[label_col_f].values
                sigma_f_pred = np.maximum(sigma_f_pred, 0)
                print(f"  Force uncertainty shape: {sigma_f_pred.shape}, Mean: {np.nanmean(sigma_f_pred):.4f}")
            else:
                print(f"Warning: Column '{label_col_f}' not found in force predictions.")
    except Exception as e:
        print(f"Error during uncertainty prediction: {e}")
        traceback.print_exc()
        sigma_e_pred, sigma_f_pred = None, None

    return sigma_e_pred, sigma_f_pred


def effective_variance(gmm, X):
    """
    Computes effective variance for each sample in X based on the provided Gaussian Mixture Model.
    
    Parameters:
        gmm: A fitted GaussianMixture model.
        X (np.ndarray): Input features, must be 2D (n_samples, n_features).
    
    Returns:
        np.ndarray: Effective variance for each sample.
    """
    if not hasattr(gmm, 'covariances_') or gmm.covariances_ is None:
        return np.full(X.shape[0], np.nan)
    if X.ndim != 2 or X.shape[1] != gmm.n_features_in_:
        return np.full(X.shape[0], np.nan)
    d = X.shape[1]
    n_components = gmm.n_components
    try:
        if gmm.covariance_type == 'full':
            component_variances = np.array([np.trace(cov) / d for cov in gmm.covariances_])
        elif gmm.covariance_type == 'diag':
            component_variances = np.array([np.sum(cov) / d for cov in gmm.covariances_])
        elif gmm.covariance_type == 'tied':
            component_variances = np.full(n_components, np.trace(gmm.covariances_) / d)
        elif gmm.covariance_type == 'spherical':
            component_variances = gmm.covariances_
        else:
            return np.full(X.shape[0], np.nan)
        posterior = gmm.predict_proba(X)
        effective_var = np.dot(posterior, component_variances)
        return effective_var
    except Exception as e:
        print(f"Error calculating effective variance: {e}")
        return np.full(X.shape[0], np.nan)


def fit_gmm_and_compute_uncertainty(
    latent_train_list, latent_all_list,
    per_atom_latent_train_list, per_atom_latent_all_list,
    compute_per_atom_uncertainty=True,
    max_components=20,
    outlier_threshold=5.0,
    scaler_pf_path="gmm_scaler_pf.joblib", gmm_pf_path="gmm_model_pf.joblib",
    scaler_pa_path="gmm_scaler_pa.joblib", gmm_pa_path="gmm_model_pa.joblib"
):
    """
    Fits Gaussian Mixture Models (GMMs) to per-frame and per-atom latent features and computes uncertainties.
    
    Parameters:
        latent_train_list (list): List of per-frame latent features (training).
        latent_all_list (list): List of per-frame latent features (all frames).
        per_atom_latent_train_list (list): List of per-atom latent features (training).
        per_atom_latent_all_list (list): List of per-atom latent features (all frames).
        compute_per_atom_uncertainty (bool): Whether to compute per-atom uncertainties.
        max_components (int): Maximum number of GMM components to consider.
        outlier_threshold (float): Outlier threshold for filtering latent features.
        scaler_pf_path (str): File path to save/load per-frame scaler.
        gmm_pf_path (str): File path to save/load per-frame GMM.
        scaler_pa_path (str): File path to save/load per-atom scaler.
        gmm_pa_path (str): File path to save/load per-atom GMM.
    
    Returns:
        dict: Dictionary containing computed metrics and the fitted models, e.g.:
              { "train_ll": ..., "eval_ll": ..., "gmm_per_frame": ..., "scaler_per_frame": ...,
                "gmm_per_atom": ..., "scaler_per_atom": ..., ... }
    """
    results = {
        "train_ll": None, "eval_ll": None,
        "train_eff_var": None, "eval_eff_var": None,
        "mask_train_gmm": None, "mask_eval_gmm": None,
        "per_atom_ll_train": None, "per_atom_ll_eval": None,
        "per_atom_eff_var_train": None, "per_atom_eff_var_eval": None,
        "mask_per_atom_train_gmm": None, "mask_per_atom_eval_gmm": None,
        "gmm_per_frame": None, "scaler_per_frame": None,
        "gmm_per_atom": None, "scaler_per_atom": None
    }

    # --- Per-Frame Processing ---
    print("\n--- GMM: Processing Per-Frame Latent Features ---")
    try:
        if not latent_train_list or not latent_all_list:
            raise ValueError("Input per-frame latent lists empty.")
        latent_train_np = np.vstack(latent_train_list)
        latent_all_np = np.vstack(latent_all_list)
        n_train_frames = len(latent_train_list)
        n_all_frames = len(latent_all_list)
        frame_train_mask = np.array([True] * n_train_frames + [False] * (n_all_frames - n_train_frames))
        frame_eval_mask = ~frame_train_mask
        print(f"Per-Frame Shapes - Train: {latent_train_np.shape}, All: {latent_all_np.shape}")

        if os.path.exists(scaler_pf_path):
            scaler_pf = joblib.load(scaler_pf_path)
        else:
            scaler_pf = StandardScaler()
            scaler_pf.fit(latent_train_np)
            joblib.dump(scaler_pf, scaler_pf_path)
            print(f"Saved PF scaler: {scaler_pf_path}")
        results["scaler_per_frame"] = scaler_pf
        latent_all_np_transformed = scaler_pf.transform(latent_all_np)
        latent_train_tf = latent_all_np_transformed[frame_train_mask]
        latent_eval_tf = latent_all_np_transformed[frame_eval_mask]

        train_mean = np.mean(latent_train_tf, axis=0)
        train_std = np.std(latent_train_tf, axis=0) + 1e-9
        z_scores_train = np.abs((latent_train_tf - train_mean) / train_std)
        mask_train_gmm = np.all(z_scores_train <= outlier_threshold, axis=1)
        latent_train_filtered = latent_train_tf[mask_train_gmm]
        results["mask_train_gmm"] = mask_train_gmm

        z_scores_eval = np.abs((latent_eval_tf - train_mean) / train_std)
        mask_eval_gmm = np.all(z_scores_eval <= outlier_threshold, axis=1)
        results["mask_eval_gmm"] = mask_eval_gmm

        print(f"Filtered PF Train: {len(latent_train_filtered)}/{len(latent_train_tf)}")
        if len(latent_train_filtered) < max_components + 1:
            raise ValueError("Too few PF train points after filtering.")

        if os.path.exists(gmm_pf_path):
            gmm_pf = joblib.load(gmm_pf_path)
        else:
            print("Fitting new per-frame GMM...")
            bics, gmms = [], []
            max_k = min(max_components, len(latent_train_filtered) - 1)
            max_k = max(1, max_k)
            comp_range = range(1, max_k + 1)
            for n_comp in comp_range:
                try:
                    gmm = GaussianMixture(
                        n_components=n_comp,
                        covariance_type='full',
                        reg_covar=1e-6,
                        init_params='kmeans++',
                        n_init=5,
                        random_state=42,
                        max_iter=200
                    )
                    gmm.fit(latent_train_filtered)
                    bics.append(gmm.bic(latent_train_filtered))
                    gmms.append(gmm)
                except Exception as e_gmm_fit:
                    print(f"Warning: GMM fit PF n={n_comp} failed: {e_gmm_fit}")
                    bics.append(np.inf)
                    gmms.append(None)
            valid_indices = [i for i, g in enumerate(gmms) if g is not None]
            if not valid_indices:
                raise RuntimeError("All GMM fits failed.")
            bics = np.array(bics)[valid_indices]
            optimal_idx = np.argmin(bics)
            gmm_pf = gmms[valid_indices[optimal_idx]]
            optimal_n = comp_range[valid_indices[optimal_idx]]
            print(f"Selected Optimal PF GMM: n_components={optimal_n}")
            joblib.dump(gmm_pf, gmm_pf_path)
            print(f"Saved PF GMM: {gmm_pf_path}")
        results["gmm_per_frame"] = gmm_pf

        results["train_ll"] = gmm_pf.score_samples(latent_train_tf)
        results["eval_ll"] = gmm_pf.score_samples(latent_eval_tf)
        results["train_eff_var"] = effective_variance(gmm_pf, latent_train_tf)
        results["eval_eff_var"] = effective_variance(gmm_pf, latent_eval_tf)
        print(f"PF Train LL Mean: {np.nanmean(results['train_ll']):.4f}")
        print(f"PF Eval EffVar Mean: {np.nanmean(results['eval_eff_var']):.4f}")

    except Exception as e_pf:
        print(f"[ERROR] Per-frame GMM failed: {e_pf}")
        traceback.print_exc()

    # --- Per-Atom Processing ---
    if compute_per_atom_uncertainty:
        print("\n--- GMM: Processing Per-Atom Latent Features ---")
        try:
            if not per_atom_latent_train_list or not per_atom_latent_all_list:
                raise ValueError("Input per-atom latent lists empty.")
            per_atom_latent_train_np = np.vstack(
                [l for l in per_atom_latent_train_list if l is not None and l.ndim == 2 and l.shape[0] > 0]
            )
            per_atom_latent_all_np = np.vstack(
                [l for l in per_atom_latent_all_list if l is not None and l.ndim == 2 and l.shape[0] > 0]
            )
            if per_atom_latent_all_np.shape[0] == 0:
                raise ValueError("No valid per-atom features found.")

            atom_counts_all_valid = [
                l.shape[0] for l in latent_all_list
                if l is not None and l.ndim == 2 and l.shape[0] > 0
            ]
            frames_mask_all_valid = np.array(
                [l is not None and l.ndim == 2 and l.shape[0] > 0 for l in latent_all_list]
            )
            frame_train_mask_valid = frame_train_mask[frames_mask_all_valid]
            atom_mask_list_pa = [
                np.ones(c, dtype=bool) if fm else np.zeros(c, dtype=bool)
                for fm, c in zip(frame_train_mask_valid, atom_counts_all_valid)
            ]
            per_atom_train_mask_pa = np.concatenate(atom_mask_list_pa)
            per_atom_eval_mask_pa = ~per_atom_train_mask_pa
            print(f"Per-Atom Shapes - Train: {per_atom_latent_train_np.shape}, All: {per_atom_latent_all_np.shape}")

            if os.path.exists(scaler_pa_path):
                scaler_pa = joblib.load(scaler_pa_path)
            else:
                scaler_pa = StandardScaler()
                scaler_pa.fit(per_atom_latent_train_np)
                joblib.dump(scaler_pa, scaler_pa_path)
                print(f"Saved PA scaler: {scaler_pa_path}")
            results["scaler_per_atom"] = scaler_pa
            per_atom_latent_all_np_transformed = scaler_pa.transform(per_atom_latent_all_np)
            per_atom_train_tf = per_atom_latent_all_np_transformed[per_atom_train_mask_pa]
            per_atom_eval_tf = per_atom_latent_all_np_transformed[per_atom_eval_mask_pa]

            train_mean_pa = np.mean(per_atom_train_tf, axis=0)
            train_std_pa = np.std(per_atom_train_tf, axis=0) + 1e-9
            z_pa_train = np.abs((per_atom_train_tf - train_mean_pa) / train_std_pa)
            mask_per_atom_train_gmm = np.all(z_pa_train <= outlier_threshold, axis=1)
            per_atom_train_filtered = per_atom_train_tf[mask_per_atom_train_gmm]
            results["mask_per_atom_train_gmm"] = mask_per_atom_train_gmm

            z_pa_eval = np.abs((per_atom_eval_tf - train_mean_pa) / train_std_pa)
            mask_per_atom_eval_gmm = np.all(z_pa_eval <= outlier_threshold, axis=1)
            results["mask_per_atom_eval_gmm"] = mask_per_atom_eval_gmm

            print(f"Filtered PA Train: {len(per_atom_train_filtered)}/{len(per_atom_train_tf)}")
            if len(per_atom_train_filtered) < max_components + 1:
                raise ValueError("Too few training atoms after filtering.")

            if os.path.exists(gmm_pa_path):
                gmm_pa = joblib.load(gmm_pa_path)
            else:
                print("Fitting new per-atom GMM...")
                bics_pa, gmms_pa = [], []
                max_k_pa = min(max_components, len(per_atom_train_filtered) - 1)
                max_k_pa = max(1, max_k_pa)
                comp_range_pa = range(1, max_k_pa + 1)
                for n_comp in comp_range_pa:
                    try:
                        gmm = GaussianMixture(
                            n_components=n_comp,
                            covariance_type='full',
                            reg_covar=1e-6,
                            init_params='kmeans++',
                            n_init=5,
                            random_state=42,
                            max_iter=200
                        )
                        gmm.fit(per_atom_train_filtered)
                        bics_pa.append(gmm.bic(per_atom_train_filtered))
                        gmms_pa.append(gmm)
                    except Exception as e_gmm_fit_pa:
                        print(f"Warning: GMM fit PA n={n_comp} failed: {e_gmm_fit_pa}")
                        bics_pa.append(np.inf)
                        gmms_pa.append(None)
                valid_indices_pa = [i for i, g in enumerate(gmms_pa) if g is not None]
                if not valid_indices_pa:
                    raise RuntimeError("All PA GMM fits failed.")
                bics_pa = np.array(bics_pa)[valid_indices_pa]
                optimal_idx_pa = np.argmin(bics_pa)
                gmm_pa = gmms_pa[valid_indices_pa[optimal_idx_pa]]
                optimal_n_pa = comp_range_pa[valid_indices_pa[optimal_idx_pa]]
                print(f"Selected Optimal PA GMM: n_components={optimal_n_pa}")
                joblib.dump(gmm_pa, gmm_pa_path)
                print(f"Saved PA GMM: {gmm_pa_path}")
            results["gmm_per_atom"] = gmm_pa

            results["per_atom_ll_train"] = gmm_pa.score_samples(per_atom_train_tf)
            results["per_atom_ll_eval"] = gmm_pa.score_samples(per_atom_eval_tf)
            results["per_atom_eff_var_train"] = effective_variance(gmm_pa, per_atom_train_tf)
            results["per_atom_eff_var_eval"] = effective_variance(gmm_pa, per_atom_eval_tf)
            print(f"PA Train LL Mean: {np.nanmean(results['per_atom_ll_train']):.4f}")
            print(f"PA Eval EffVar Mean: {np.nanmean(results['per_atom_eff_var_eval']):.4f}")

        except Exception as e_pa:
            print(f"[ERROR] Per-atom GMM failed: {e_pa}")
            traceback.print_exc()

    return results

