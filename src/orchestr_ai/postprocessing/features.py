"""
features.py

This module computes descriptors and features for atomic configurations.
It uses SOAP from the dscribe library along with bond length and inter-feature
distances, and also performs scaling and PCA transformations. Additionally, it
can generate diagnostic plots for training mode.
"""

import hashlib
import os
import numpy as np
from ase.io import read

try:
    from dscribe.descriptors import SOAP
except ImportError:
    print("Warning: dscribe library not found. SOAP features cannot be computed.")
    SOAP = None  # Set to None if import fails

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import joblib


def compute_avg_bond_length(frames, cutoff=3.5):
    """
    Computes the average bond length for each frame.

    Parameters:
        frames (list of ase.Atoms): List of atomic configurations.
        cutoff (float): Maximum distance for bonds to be considered (default: 3.5 Å).
    
    Returns:
        np.ndarray: Array of average bond lengths with shape (n_frames, 1).
    """
    avg_bond_lengths = []
    for frame in frames:
        if len(frame) < 2:
            avg_bond_lengths.append(0.0)
            continue
        
        distances = frame.get_all_distances(mic=False)
        # Select only unique bond pairs (upper triangle without the diagonal)
        bond_indices = np.triu_indices_from(distances, k=1)
        bond_lengths = distances[bond_indices]
        valid_bonds = bond_lengths[bond_lengths < cutoff]
        if len(valid_bonds) > 0:
            avg_bond_lengths.append(np.mean(valid_bonds))
        else:
            avg_bond_lengths.append(0.0)
    
    return np.array(avg_bond_lengths).reshape(-1, 1)


def compute_features(all_frames, config, training_data_path=None, train_mask=None, eval_mask=None,
                     scaler_load_path=None, pca_load_path=None):
    """
    Computes features for a list of frames by combining SOAP descriptors, average bond lengths,
    and the minimum distances in PCA space.

    Parameters:
        all_frames (list of ase.Atoms): List of atomic configurations.
        config (dict): Configuration dictionary containing settings for SOAP, scaling, and PCA.
        training_data_path (str, optional): Path to training data (unused here).
        train_mask (np.ndarray, optional): Boolean array indicating training frames.
        eval_mask (np.ndarray, optional): Boolean array indicating evaluation frames.
        scaler_load_path (str, optional): File path for loading a pre-trained scaler.
        pca_load_path (str, optional): File path for loading a pre-trained PCA.
    
    Returns:
        tuple: A tuple containing:
            - features_all (np.ndarray): Combined features array.
            - min_distances_all (np.ndarray): Array of minimum distances in PCA space.
            - soap_pca_all (np.ndarray): PCA-transformed SOAP descriptors.
            - pca (PCA object): Fitted PCA object.
            - scaler (StandardScaler object): Fitted scaler object.
    """
    # Ensure SOAP library is available
    if SOAP is None:
        print("Error: dscribe not installed. Cannot compute SOAP features.")
        return None, None, None, None, None

    # --- SOAP Setup ---
    soap_config = config.get("eval", {}).get("SOAP", {})
    species = soap_config.get("species", [])
    if not species:
        print("Warning: SOAP 'species' not defined in config['eval']['SOAP']. Attempting to infer from first frame.")
        if all_frames:
            species = sorted(list(set(all_frames[0].get_chemical_symbols())))
        else:
            raise ValueError("Cannot infer SOAP species: No frames provided and not defined in config.")
        print(f"Inferred species: {species}")

    r_cut = soap_config.get("r_cut", 12.0)
    n_max = soap_config.get("n_max", 9)
    l_max = soap_config.get("l_max", 6)
    sigma = soap_config.get("sigma", 0.05)
    periodic = soap_config.get("periodic", False)
    sparse = soap_config.get("sparse", False)
    average = soap_config.get("average", "off")

    print(f"Initializing SOAP: species={species}, r_cut={r_cut}, n_max={n_max}, l_max={l_max}, average={average}")
    soap = SOAP(
        species=species, r_cut=r_cut, n_max=n_max, l_max=l_max, sigma=sigma,
        periodic=periodic, sparse=sparse, average=average
    )
    n_feat_soap = soap.get_number_of_features()

    # --- Compute SOAP Descriptors ---
    print(f"Computing SOAP descriptors for {len(all_frames)} frames...")
    n_jobs_soap = config.get("eval", {}).get("soap_n_jobs", 1)
    soap_raw_list = soap.create(all_frames, n_jobs=n_jobs_soap)

    soap_avg_all = []
    for i, s in enumerate(soap_raw_list):
        if s is None or s.shape[0] == 0:
            print(f"Warning: Frame {i} resulted in empty SOAP descriptor. Using zeros.")
            soap_avg_all.append(np.zeros(n_feat_soap))
        else:
            # If the SOAP descriptor is an array for all atoms, average over the atoms.
            if s.ndim == 2 and s.shape[0] > 0:
                soap_avg_all.append(np.mean(s, axis=0))
            # If already averaged (1D array) and matches expected feature length:
            elif s.ndim == 1 and s.shape[0] == n_feat_soap:
                soap_avg_all.append(s)
            else:
                print(f"Warning: Frame {i} SOAP descriptor has unexpected shape {s.shape}. Using zeros.")
                soap_avg_all.append(np.zeros(n_feat_soap))
    soap_avg_all = np.array(soap_avg_all)
    print(f"Averaged SOAP shape: {soap_avg_all.shape}")

    # --- Scale and PCA ---
    pca, scaler = None, None
    mode = "unknown"  # Track if we are in training or prediction mode.
    if scaler_load_path and pca_load_path and \
       os.path.exists(scaler_load_path) and os.path.exists(pca_load_path):
        mode = "prediction"
        print(f"Loading pre-trained Scaler from {scaler_load_path}")
        scaler = joblib.load(scaler_load_path)
        print(f"Loading pre-trained PCA from {pca_load_path}")
        pca = joblib.load(pca_load_path)
        soap_scaled_all = scaler.transform(soap_avg_all)
        soap_pca_all = pca.transform(soap_scaled_all)
    elif train_mask is not None:
        mode = "training"
        print("Fitting Scaler and PCA on training data...")
        scaler = StandardScaler()
        soap_avg_train = soap_avg_all[train_mask]
        if len(soap_avg_train) == 0:
            raise ValueError("Cannot fit Scaler/PCA: No training data.")
        soap_scaled_train = scaler.fit_transform(soap_avg_train)
        soap_scaled_all = scaler.transform(soap_avg_all)

        n_components_pca = config.get("eval", {}).get("pca_n_components", 0.99)
        pca = PCA(n_components=n_components_pca, svd_solver='auto')
        pca.fit(soap_scaled_train)
        print(f"PCA fitted: {pca.n_components_} components explain {sum(pca.explained_variance_ratio_):.4f} variance.")
        soap_pca_all = pca.transform(soap_scaled_all)

        # Save fitted models for future predictions.
        scaler_save_path = "soap_scaler.joblib"
        pca_save_path = "soap_pca.joblib"
        joblib.dump(scaler, scaler_save_path)
        joblib.dump(pca, pca_save_path)
        print(f"Saved Scaler to {scaler_save_path}, PCA to {pca_save_path}")
    else:
        raise ValueError("compute_features needs either train_mask or load paths for scaler/pca.")

    print(f"PCA features shape: {soap_pca_all.shape}")

    # --- Compute Minimum Distances in PCA Space ---
    print("Computing minimum distances in PCA space...")
    distances_pca = cdist(soap_pca_all, soap_pca_all, metric='euclidean')
    np.fill_diagonal(distances_pca, np.inf)
    min_distances_all = np.min(distances_pca, axis=1).reshape(-1, 1)
    print(f"Min distances shape: {min_distances_all.shape}")

    # --- Compute Average Bond Lengths ---
    print("Computing average bond lengths...")
    bond_len_cutoff = config.get("eval", {}).get("bond_length_cutoff", 3.5)
    bond_len_all = compute_avg_bond_length(all_frames, cutoff=bond_len_cutoff)
    print(f"Avg bond lengths shape: {bond_len_all.shape}")

    # --- Combine Features ---
    features_all = np.hstack([soap_pca_all, bond_len_all, min_distances_all])
    print(f"Combined features shape: {features_all.shape}")

    # --- Generate Diagnostic Plots (Training Mode Only) ---
    if mode == "training" and train_mask is not None and eval_mask is not None:
        print("Generating diagnostic plots...")
        diag_dir = "diagnostics"
        os.makedirs(diag_dir, exist_ok=True)
        try:
            plt.figure(figsize=(8, 6))
            plt.scatter(soap_pca_all[train_mask, 0], soap_pca_all[train_mask, 1],
                        alpha=0.5, label='Train', s=10)
            plt.scatter(soap_pca_all[eval_mask, 0], soap_pca_all[eval_mask, 1],
                        alpha=0.5, label='Eval', s=10)
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.title('PCA of Averaged SOAP Descriptors')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(diag_dir, "soap_pca_scatter.png"))
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.hist(min_distances_all[train_mask].flatten(), bins=50, alpha=0.5,
                     label='Train', density=True)
            plt.hist(min_distances_all[eval_mask].flatten(), bins=50, alpha=0.5,
                     label='Eval', density=True)
            plt.xlabel('Minimum SOAP Distance in PCA Space')
            plt.ylabel('Density')
            plt.title('Histogram of Minimum SOAP Distances')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(diag_dir, "hist_min_distances.png"))
            plt.close()
            print(f"Diagnostic plots saved to '{diag_dir}/'")
        except Exception as e_plot:
            print(f"Warning: Failed to generate diagnostic plots: {e_plot}")

    return features_all, min_distances_all, soap_pca_all, pca, scaler


# =============================================================================
# Active-learning feature spaces and batch selection (pool AL)
# =============================================================================
# eval options:
#   al_kernel   latent | soap | ntk_e | ntk_ef | quests   feature space of the batch redundancy
#   al_quality  sigma | pv                                ensemble sigmaF_mean in the score, or the kernel alone
#   al_selector greedy | lcmd                             batch rule
#   al_members  all | <index>                             ensemble members of latent / ntk_e / ntk_ef
#   ntk_probes  <int>                                     force probes of ntk_ef


def al_options(eval_cfg, framework):
    """Validated pool AL options (al_kernel, al_quality, al_selector) for the model framework."""
    from orchestr_ai.postprocessing.calculators.factory import normalize_framework

    kernel = str(eval_cfg.get("al_kernel", "latent")).lower()
    quality = str(eval_cfg.get("al_quality", "sigma")).lower()
    selector = str(eval_cfg.get("al_selector", "greedy")).lower()
    if (kernel not in ("latent", "soap", "ntk_e", "ntk_ef", "quests") or quality not in ("sigma", "pv")
            or selector not in ("greedy", "lcmd") or (kernel, selector) == ("quests", "lcmd")):
        raise ValueError(
            f"Unsupported pool AL options al_kernel={kernel}, al_quality={quality}, al_selector={selector}: "
            "al_kernel latent|soap|ntk_e|ntk_ef|quests, al_quality sigma|pv, al_selector greedy|lcmd "
            "(lcmd needs a frame-level kernel, not quests)."
        )
    if kernel in ("ntk_e", "ntk_ef") and normalize_framework(framework) == "nequip":
        raise ValueError(
            f"al_kernel={kernel} needs gradients w.r.t. the species embedding, but compiled NequIP/Allegro models are "
            "frozen (no trainable weights or embedding graph); use al_kernel latent, soap or quests."
        )
    return kernel, quality, selector


def member_features(kind, k, model_key, frames, compute, chunk=100):
    """Features of `frames` from ensemble member k, cached in ensemble_<kind>.npz (one file for all members, keyed by
    a hash of each frame).

    compute(idx) returns the features of frames[idx] that are not cached yet. Any frame set (training, calibration,
    pool) is computed once, and the file is saved after every chunk of frames, so long runs resume. model_key
    identifies the member's model and settings; on a mismatch that member's entries are recomputed.
    """
    path = f"ensemble_{kind}.npz"
    data = dict(np.load(path)) if os.path.exists(path) else {}
    store = {}
    if str(data.get(f"model_key_m{k}", "")) == model_key:
        store = dict(zip(data[f"keys_m{k}"], data[f"feat_m{k}"]))
    hashes = [hashlib.blake2b(a.numbers.tobytes() + a.positions.tobytes(), digest_size=16).hexdigest() for a in frames]
    todo = [i for i, h in enumerate(hashes) if h not in store]
    if todo:
        print(f"[AL features] {kind}, member {k}: {len(todo)} new frames")
    for b in range(0, len(todo), chunk):
        idx = todo[b:b + chunk]
        store.update(zip((hashes[i] for i in idx), compute(idx)))
        data.update({f"keys_m{k}": np.array(list(store)), f"feat_m{k}": np.array(list(store.values())),
                     f"model_key_m{k}": np.array(model_key)})
        np.savez(path, **data)
    return np.array([store[h] for h in hashes])


def ntk_frame_features(calc, frames, n_probes=16, batch_size=4):
    """NTK frame features w.r.t. the species-embedding weights W (Varga-Umbrich et al. 2026).

    dE/dW per frame (NTK-E) and, when n_probes > 0, the RMS over atoms and Cartesian components of dF/dW
    (force-aware NTK-EF), estimated with Rademacher probes that are identical for every frame of a given size, so
    their noise largely cancels between similar frames. W[z, c] enters only through the embedding output h0 of the
    atoms of species z, so both are per-species sums of per-atom gradients w.r.t. h0. calc.embedding_graph(frames)
    supplies the energies with their autograd graph (the framework-specific part).
    """
    import torch

    feats = []
    for b in range(0, len(frames), batch_size):
        chunk = frames[b:b + batch_size]
        with torch.set_grad_enabled(True):
            g = calc.embedding_graph(chunk)
            E, h0, n_species = g["energy"], g["h0"], g["n_species"]
            key = g["atom_frame"] * n_species + g["atom_species"]

            def per_species(x):
                out = torch.zeros(len(chunk) * n_species, x.shape[1], dtype=x.dtype, device=x.device)
                return out.index_add_(0, key, x).reshape(len(chunk), -1)

            phi = [per_species(torch.autograd.grad(E.sum(), h0, retain_graph=n_probes > 0)[0])]
            if n_probes:
                dEdR = torch.autograd.grad(E.sum(), g["positions"], create_graph=True)[0]
                sq = torch.zeros_like(phi[0])
                for s in range(n_probes):
                    z = torch.cat([
                        torch.randint(0, 2, (len(a), 3), generator=torch.Generator().manual_seed(s)) for a in chunk
                    ]).to(dEdR) * 2 - 1
                    sq += per_species(torch.autograd.grad(dEdR, h0, grad_outputs=z, retain_graph=True)[0]) ** 2
                n_at = torch.tensor([len(a) for a in chunk], dtype=sq.dtype, device=sq.device)[:, None]
                phi.append(torch.sqrt(sq / (n_probes * n_at)))
        feats.append(torch.cat(phi, dim=1).detach().cpu().numpy().astype(np.float64))
    return np.concatenate(feats)


def model_features(kind, frames, runner):
    """Frame features of `frames` from the ensemble members chosen by al_members ("all" concatenates the members,
    i.e. sums their kernels; an index keeps one).

    latent: each member's frame-mean invariant node features, from the same inference chain as the ensemble
    predictions (the ensemble passes store them, so they are rarely recomputed). ntk_e / ntk_ef: NTK features
    (ntk_frame_features). Cached in one file per kind (member_features).
    """
    from orchestr_ai.postprocessing.inference_runner import InferenceRunner

    n_probes = int(runner.eval_cfg.get("ntk_probes", 16)) if kind == "ntk_ef" else 0
    paths = runner._model_paths()
    members = runner.eval_cfg.get("al_members", "all")
    out = []
    for k in (range(len(paths)) if str(members).lower() == "all" else [int(members)]):
        calc = []

        def compute(idx):
            if not calc:
                calc.append(runner._calculator(paths[k]))
            chunk = [frames[i] for i in idx]
            if kind == "latent":
                return InferenceRunner(calc[0], runner.batch_size, None).run(frames=chunk)[2]
            return ntk_frame_features(calc[0], chunk, n_probes, runner.batch_size)

        key = runner._model_key(paths[k]) + (f"|probes={n_probes}" if n_probes else "")
        out.append(member_features(kind, k, key, frames, compute))
    return np.concatenate(out, axis=1)


def al_feature_space(eval_cfg, runner, labelled, train, pool, physical, quests_report=False):
    """Features of the labelled (calibration) frames, the training frames (reference) and the pool in the
    redundancy space of al_kernel, plus the QUESTS reference when it is needed.

    latent / ntk_e / ntk_ef: model features z-scored with the training statistics (frame features of one system are
    nearly parallel, so without this the informative variation sits far below any ridge lambda); constant channels,
    e.g. of species absent from the data, are dropped. The costly NTK features of the pool are returned as a function
    of the pool indices and only evaluated on the candidates. soap: frame-averaged SOAP; pool frames that fail the
    geometry checks stay at zero. quests: latent features for the frame-level diagnostics; its batch redundancy is
    per atom (quests_reference, quests_greedy).
    """
    kernel = str(eval_cfg.get("al_kernel", "latent")).lower()
    quests = None
    if kernel == "quests" or quests_report:
        quests = quests_reference(train, target=float(eval_cfg.get("candidate_tol", 0.01)))
    if kernel == "soap":
        species = sorted({s for a in labelled for s in a.get_chemical_symbols()})

        def soap(frames):
            return compute_soap_features(frames, species=species, r_cut=eval_cfg.get("soap_rcut", 6.0),
                                         n_max=eval_cfg.get("soap_nmax", 4), l_max=eval_cfg.get("soap_lmax", 4))[0]

        feat_ref = soap(train)
        feat_pool = np.zeros((len(pool), feat_ref.shape[1]))
        phys = np.flatnonzero(physical)
        if len(phys):
            feat_pool[phys] = soap([pool[i] for i in phys])
        return soap(labelled), feat_ref, feat_pool, quests

    kind = "latent" if kernel == "quests" else kernel
    feat_ref = model_features(kind, train, runner)
    keep = feat_ref.std(axis=0) > 0
    mu, sd = feat_ref[:, keep].mean(axis=0), feat_ref[:, keep].std(axis=0)

    def z(x):
        return (x[:, keep] - mu) / sd

    if kind in ("ntk_e", "ntk_ef"):
        feat_pool = lambda idx: z(model_features(kind, [pool[i] for i in idx], runner))
    else:
        feat_pool = z(model_features(kind, pool, runner)) if len(pool) else np.zeros((0, int(keep.sum())))
    return z(model_features(kind, labelled, runner)), z(feat_ref), feat_pool, quests


def compute_soap_features(frames, train_frames=None, species=None, r_cut=4.0, n_max=4, l_max=4):
    """
    Computes averaged SOAP descriptors for a list of ASE Atoms objects.
    
    Parameters:
        frames (list): List of ASE Atoms objects.
        train_frames (list, optional): List of training ASE Atoms objects to gather chemical symbols.
        species (list, optional): Predefined list of species (chemical symbols).
        r_cut (float): Cutoff radius in Angstrom. Default 4.0.
        n_max (int): Number of radial basis functions. Default 4.
        l_max (int): Maximum degree of spherical harmonics. Default 4.
        
    Returns:
        tuple: (features_array, species_list).
    """
    from dscribe.descriptors import SOAP
    
    # 1. Determine chemical species if not provided
    if species is None:
        species_set = set()
        for fr in frames:
            species_set.update(fr.get_chemical_symbols())
        if train_frames is not None:
            for fr in train_frames:
                species_set.update(fr.get_chemical_symbols())
        species = sorted(list(species_set))
        
    print(f"[SOAP] Computing descriptors for species: {species} (rcut={r_cut}, nmax={n_max}, lmax={l_max})")
    
    # 2. Construct SOAP descriptor
    soap = SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        periodic=False,     # Quantum dots in vacuum
        average="outer",    # Average SOAP over all atoms in the frame to get a per-frame descriptor
        sparse=False
    )
    
    # 3. Create SOAP vectors frame-by-frame to avoid multiprocessing hangs and show progress
    features_list = []
    n_frames = len(frames)
    print(f"[SOAP] Computing features sequentially for {n_frames} frames...")
    for idx, fr in enumerate(frames):
        try:
            feat = soap.create(fr)
            feat_arr = np.asarray(feat)
            if feat_arr.ndim > 1:
                feat_arr = feat_arr.ravel()
        except Exception as e:
            print(f"  -> [SOAP] Warning: Failed to compute SOAP for frame {idx + 1}/{n_frames}: {e}. Returning zeros.")
            feat_arr = np.zeros(soap.get_number_of_features())
        
        features_list.append(feat_arr)
        if (idx + 1) % max(1, n_frames // 10) == 0 or idx == n_frames - 1:
            print(f"  -> SOAP progress: {idx + 1}/{n_frames} frames completed...")
    
    features = np.vstack(features_list)
    
    # Make sure it's 2D array
    if features.ndim == 1:
        features = features.reshape(1, -1)
        
    return features, species


def greedy_posterior_variance(X, u, n_sel, relative=True):
    """Greedy log-det (posterior-variance) batch over candidates.

    X holds the candidate features whitened by the training posterior, so v0 = |x|^2 is the novelty
    w.r.t. the training set; every pick conditions v on itself (rank-1 update). relative=True scores
    u * sqrt(v / v0) (quality x diversity); relative=False scores v (pure posterior variance, u ignored).
    Returns the picks and each candidate's score (at pick time for the picks).
    """
    v0 = np.maximum(np.einsum("id,id->i", X, X), np.finfo(float).tiny)
    v = v0.copy()
    P = np.eye(X.shape[1])
    picks, acq = [], np.zeros(len(X))

    def score():
        return u * np.sqrt(np.clip(v, 0.0, None) / v0) if relative else np.clip(v, 0.0, None)

    for _ in range(n_sel):
        s = score()
        s[picks] = -np.inf
        r = int(np.argmax(s))
        picks.append(r)
        acq[r] = s[r]
        w = P @ X[r]
        den = 1.0 + X[r] @ w
        v -= (X @ w) ** 2 / den
        P -= np.outer(w, w) / den
    rest = np.setdiff1d(np.arange(len(X)), picks)
    acq[rest] = score()[rest]
    return picks, acq


def lcmd(feat_cand, feat_ref, w, n_sel):
    """LCMD (Holzmueller et al., JMLR 2023, TP mode) over candidates.

    The reference frames and the picks are cluster centres. Each step takes the cluster with the largest
    sum of squared weighted distances (w * distance to the nearest centre) and picks its farthest member.
    """
    n_ref = len(feat_ref)
    d2 = (feat_cand ** 2).sum(1)[:, None] + (feat_ref ** 2).sum(1)[None] - 2.0 * feat_cand @ feat_ref.T
    assign, dist2 = d2.argmin(axis=1), np.maximum(d2.min(axis=1), 0.0)
    picks, acq = [], np.zeros(len(feat_cand))
    for t in range(n_sel):
        wd = w * np.sqrt(dist2)
        mass = np.bincount(assign, weights=wd ** 2, minlength=n_ref + t)
        if mass.max() <= 0:
            break
        members = np.flatnonzero(assign == mass.argmax())
        r = int(members[np.argmax(wd[members])])
        picks.append(r)
        acq[r] = wd[r]
        d2r = ((feat_cand - feat_cand[r]) ** 2).sum(1)
        closer = d2r < dist2
        assign[closer], dist2[closer] = n_ref + t, d2r[closer]
    rest = np.setdiff1d(np.arange(len(feat_cand)), picks)
    acq[rest] = (w * np.sqrt(dist2))[rest]
    return picks, acq


def quests_reference(frames, k=32, cutoff=5.0, target=0.01, n_folds=5):
    """QUESTS descriptors (Schwalbe-Koda et al., Nat. Commun. 2025) of the reference atoms, per element,
    with one kernel bandwidth h per element.

    The descriptor ignores species, so atoms are only compared with reference atoms of their own element.
    h is set from the data: a fraction `target` of reference atoms comes out novel (dH > 0, kernel sum
    over the 32 nearest atoms of the other folds below 1) when contiguous folds of frames are held out.
    """
    from quests.descriptor import get_descriptors
    from sklearn.neighbors import NearestNeighbors

    X = get_descriptors(frames, k=k, cutoff=cutoff)
    sym = np.concatenate([a.get_chemical_symbols() for a in frames])
    fold = np.repeat(np.arange(len(frames)) * n_folds // len(frames), [len(a) for a in frames])
    ref, h = {}, {}
    for el in np.unique(sym):
        Xe, fe = X[sym == el], fold[sym == el]
        d2 = np.concatenate([
            NearestNeighbors(n_neighbors=32).fit(Xe[fe != f]).kneighbors(Xe[fe == f])[0] ** 2
            for f in range(n_folds)
        ])
        lo, hi = np.log(1e-4), np.log(1.0)
        for _ in range(40):   # the novel fraction falls monotonically with h
            mid = 0.5 * (lo + hi)
            novel = np.mean(np.exp(-0.5 * d2 / np.exp(2 * mid)).sum(axis=1) < 1.0)
            lo, hi = (mid, hi) if novel > target else (lo, mid)
        ref[el], h[el] = Xe, float(np.exp(hi))
    print("[QUESTS] Bandwidth per element (1/A): " + ", ".join(f"{el} {v:.4f}" for el, v in h.items()))
    return {"ref": ref, "h": h, "k": k, "cutoff": cutoff}


def quests_greedy(frames, u, atom_sigma, n_sel, quests, relative=True):
    """Greedy batch with per-atom QUESTS redundancy (al_kernel quests).

    A is each atom's kernel sum over the training atoms of its element plus the atoms already picked. An atom's
    distinctness min(1, 1/A) = min(1, exp(dH)) is 1 while it is novel (dH > 0) and smaller the better it is covered
    (its term in the QUESTS diversity D = log sum exp(dH)). relative=True scores u x the sigma^2-weighted mean
    distinctness of the frame's atoms, so low-sigma atoms (e.g. bulk solvent) barely count; relative=False scores the
    frame's largest dH (Yu, Lordi & Schwalbe-Koda, JCP 2026). Each pick adds its atoms to A.
    """
    from quests.descriptor import get_descriptors
    from quests.entropy import kernel_sum

    if not frames:
        return [], np.zeros(0)
    h_of = lambda el: quests["h"].get(el, float(np.median(list(quests["h"].values()))))
    X = get_descriptors(frames, k=quests["k"], cutoff=quests["cutoff"])
    sym = np.concatenate([a.get_chemical_symbols() for a in frames])
    fid = np.repeat(np.arange(len(frames)), [len(a) for a in frames])
    w = np.concatenate(atom_sigma) ** 2
    A = np.zeros(len(X))
    for el in np.unique(sym):
        if el in quests["ref"]:
            A[sym == el] = kernel_sum(X[sym == el], quests["ref"][el], h=h_of(el), batch_size=2000)

    def score():
        if relative:
            distinct = np.minimum(1.0, 1.0 / np.maximum(A, np.finfo(float).tiny))
            return u * np.bincount(fid, weights=w * distinct, minlength=len(frames)) / np.bincount(fid, weights=w, minlength=len(frames))
        dH = -np.log(np.maximum(A, np.finfo(float).tiny))
        return np.array([dH[fid == f].max() for f in range(len(frames))])

    picks, acq = [], np.zeros(len(frames))
    for _ in range(n_sel):
        s = score()
        s[picks] = -np.inf
        r = int(np.argmax(s))
        picks.append(r)
        acq[r] = s[r]
        for el in np.unique(sym[fid == r]):
            m = sym == el
            A[m] += kernel_sum(X[m], X[m & (fid == r)], h=h_of(el), batch_size=2000)
    rest = np.setdiff1d(np.arange(len(frames)), picks)
    acq[rest] = score()[rest]
    return picks, acq


def quests_coverage(frames, quests):
    """QUESTS coverage of `frames` by the training set: the share of atoms with dH > 0, per frame and per element."""
    from quests.descriptor import get_descriptors
    from quests.entropy import kernel_sum

    X = get_descriptors(frames, k=quests["k"], cutoff=quests["cutoff"])
    sym = np.concatenate([a.get_chemical_symbols() for a in frames])
    fid = np.repeat(np.arange(len(frames)), [len(a) for a in frames])
    novel = np.ones(len(X), dtype=bool)
    for el in np.unique(sym):
        if el in quests["ref"]:
            novel[sym == el] = kernel_sum(X[sym == el], quests["ref"][el], h=quests["h"][el], batch_size=2000) < 1.0
    per_frame = np.bincount(fid, weights=novel.astype(float)) / np.bincount(fid)
    return per_frame, {el: float(novel[sym == el].mean()) for el in np.unique(sym)}


def select_batch(quality, selector, kernel, n_sel, u, G, feat, feat_ref, frames=None, atom_sigma=None, quests=None):
    """DFT batch of n_sel candidates (indices into the candidate rows) and every candidate's score.

    selector greedy: log-det over the kernel posterior (G: candidate features whitened by the training posterior);
    quality sigma scores u * sqrt(v / v0) (u = sigmaF_mean: quality x diversity), pv scores v. Kernel quests swaps the
    frame-level redundancy for per-atom QUESTS coverage (frames, atom_sigma, quests). selector lcmd: LCMD over the
    feature vectors feat with the training frames feat_ref as initial centres, distances weighted by u (sigma) or not (pv).
    """
    w = u if quality == "sigma" else np.ones(len(u))
    if selector == "lcmd":
        return lcmd(feat, feat_ref, w, n_sel)
    if kernel == "quests":
        return quests_greedy(frames, w, atom_sigma, n_sel, quests, relative=quality == "sigma")
    return greedy_posterior_variance(G, w, n_sel, relative=quality == "sigma")
