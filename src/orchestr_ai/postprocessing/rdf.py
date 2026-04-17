"""
rdf.py

Handles geometric sanity checks and Radial Distribution Function (RDF) 
calculations to filter unphysical frames during active learning.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
from itertools import combinations
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components

def debug_plot_rdfs(reference_frames, rdf_thresholds, r_max=6.0, dr=0.02, outprefix="rdf_DEBUG"):
    """Saves RDF plots for all active species pairs."""
    print("[RDF] Debug plotting RDFs for each pair...")
    all_pairs = list(rdf_thresholds.keys())
    ref_pos  = [atoms.get_positions() for atoms in reference_frames]
    ref_syms = [atoms.get_chemical_symbols() for atoms in reference_frames]

    bins = np.arange(0.0, r_max + dr, dr)
    r_centers = 0.5 * (bins[:-1] + bins[1:])

    for pair in all_pairs:
        A, B = pair
        hist = np.zeros(len(bins) - 1, dtype=np.float64)

        for coords, syms in zip(ref_pos, ref_syms):
            idx_A = [ii for ii, s in enumerate(syms) if s == A]
            idx_B = [jj for jj, s in enumerate(syms) if s == B]
            if not idx_A or not idx_B: continue

            coords_A = np.asarray(coords)[idx_A]
            coords_B = np.asarray(coords)[idx_B]
            dAB = np.linalg.norm(coords_A[:,None,:] - coords_B[None,:,:], axis=2)

            if A == B:
                iu = np.triu_indices_from(dAB, k=1)
                d_flat = dAB[iu]
            else:
                d_flat = dAB.ravel()

            d_use = d_flat[d_flat < r_max]
            if d_use.size > 0:
                hist += np.histogram(d_use, bins=bins)[0]

        smooth = gaussian_filter1d(hist, sigma=2)
        r_soft, r_hard = rdf_thresholds[pair]

        plt.figure(figsize=(5,4))
        plt.plot(r_centers, smooth, label=f"g_{A}-{B}(r) (smoothed)")
        plt.axvline(r_soft, color='red', linestyle='--', label=f"r_soft={r_soft:.2f} Å")
        plt.axvline(r_hard, color='orange', linestyle=':', label=f"r_hard={r_hard:.2f} Å")
        plt.xlabel("r (Å)")
        plt.ylabel("arb. RDF units")
        plt.title(f"{A}-{B} RDF (validation set)")
        plt.xlim(0.0, r_max)
        plt.ylim(bottom=0.0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{outprefix}_{A}-{B}.png", dpi=200)
        plt.close()


def compute_rdf_thresholds_from_reference(reference_frames, cutoff=6.0, bins=300, stride=1, 
                                          min_peak_height_frac=0.05, min_peak_prominence_frac=0.10, 
                                          left_baseline_frac=0.05, beta=0.5, r_min_physical=1.0):
    """Infers r_soft and r_hard cutoffs for each element pair using trusted frames."""
    t0_all = time.time()
    ref_use = reference_frames[::stride]
    print(f"[RDF] Using {len(ref_use)}/{len(reference_frames)} reference frames (stride={stride}).")

    pair_dists = defaultdict(list)
    for fr in ref_use:
        pos, syms = fr.get_positions(), fr.get_chemical_symbols()
        for i, j in combinations(range(len(syms)), 2):
            rij = np.linalg.norm(pos[i] - pos[j])
            if rij <= cutoff:
                pair_dists[tuple(sorted((syms[i], syms[j])))].append(rij)

    thresholds, debug_rdfs = {}, {}
    r_edges = np.linspace(0.0, cutoff, bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    for pair, dlist in pair_dists.items():
        darray = np.asarray(dlist, dtype=np.float64)
        if darray.size < 20: continue

        hist, _ = np.histogram(darray, bins=r_edges, density=False)
        smooth = gaussian_filter1d(hist.astype(np.float64), sigma=2)
        debug_rdfs[pair] = {"r": r_centers.copy(), "g_smooth": smooth.copy()}

        locmax_idx = [k for k in range(1, len(smooth) - 1) if smooth[k] > smooth[k-1] and smooth[k] >= smooth[k+1]]
        locmax_idx = np.array(locmax_idx, dtype=int)
        if locmax_idx.size == 0: continue

        peak_vals = smooth[locmax_idx]
        global_max = peak_vals.max()

        height_mask = peak_vals >= (min_peak_height_frac * global_max)
        prominence_mask = np.array([(peak_vals[idx] - np.min(smooth[max(0, p - 5):min(len(smooth), p + 6)])) 
                                    >= (min_peak_prominence_frac * global_max) for idx, p in enumerate(locmax_idx)])
        
        good_mask = height_mask & prominence_mask & (r_centers[locmax_idx] > r_min_physical)
        good_peaks = locmax_idx[good_mask]

        if good_peaks.size == 0:
            fallback_idx = np.argmin(np.where(r_centers[locmax_idx] > r_min_physical, r_centers[locmax_idx], np.inf))
            chosen_peak_idx = locmax_idx[fallback_idx]
        else:
            chosen_peak_idx = good_peaks[np.argmin(r_centers[good_peaks])]

        peak_r, peak_val = r_centers[chosen_peak_idx], smooth[chosen_peak_idx]
        left_region = np.where((r_centers[:chosen_peak_idx] > r_min_physical) & (smooth[:chosen_peak_idx] < left_baseline_frac * peak_val))[0]

        if left_region.size > 0:
            r_soft = float(r_centers[left_region[-1]])
        else:
            r_soft = float(min(max(r_min_physical, r_centers[max(chosen_peak_idx - 1, 0)]), peak_r))

        r_hard = beta * r_soft
        thresholds[pair] = (r_soft, r_hard)

    print(f"[RDF] Done building thresholds for {len(thresholds)} pairs in {time.time() - t0_all:.2f}s total.")
    return thresholds


def fast_filter_by_rdf_kdtree(frames, rdf_thresholds, verbose=True):
    """Reject unphysical geometries using KDTree neighbor search."""
    n_frames = len(frames)
    ok_mask = np.ones(n_frames, dtype=bool)
    if not rdf_thresholds: return ok_mask

    r_hard_max = max(rh for _, rh in rdf_thresholds.values())
    n_rejected = 0

    for f_idx, atoms in enumerate(frames):
        pos, syms = atoms.get_positions(), atoms.get_chemical_symbols()
        if len(pos) < 2: continue

        tree = cKDTree(pos)
        close_pairs = tree.query_pairs(r_hard_max, output_type='ndarray')
        if close_pairs.size == 0: continue

        dists = np.linalg.norm(pos[close_pairs[:, 0]] - pos[close_pairs[:, 1]], axis=1)

        for (i, j), rij in zip(close_pairs, dists):
            pair = tuple(sorted((syms[i], syms[j])))
            if pair in rdf_thresholds and rij < rdf_thresholds[pair][1]:
                ok_mask[f_idx] = False
                n_rejected += 1
                break  

    if verbose:
        print(f"[RDF] Geometric sanity: {ok_mask.sum()}/{n_frames} frames OK. Rejected {n_rejected}.")
    return ok_mask


def plot_rdf_comparison(pair, rdf_ref, rdf_all, rdf_kept, rdf_thresholds, r_max=4.0, outprefix="rdf_poolcheck"):
    """Compares validation RDF against Pool RDF."""
    if pair not in rdf_ref: return

    r_ref, g_ref = rdf_ref[pair]["r"], rdf_ref[pair]["rdf"]
    r_all, g_all = rdf_all.get(pair, {}).get("r"), rdf_all.get(pair, {}).get("rdf")
    r_kept, g_kept = rdf_kept.get(pair, {}).get("r"), rdf_kept.get(pair, {}).get("rdf")
    r_soft, r_hard = rdf_thresholds.get(pair, (None, None))

    fig, ax = plt.subplots(figsize=(4,3), dpi=150)
    ax.plot(r_ref, g_ref, label="ref", linewidth=1.5)
    if r_all is not None: ax.plot(r_all, g_all, label="pool (all)", linestyle="--", linewidth=1.0)
    if r_kept is not None: ax.plot(r_kept, g_kept, label="pool (ok)", linestyle=":", linewidth=1.0)
    if r_soft: ax.axvline(r_soft, color="green", linestyle=":", label="r_soft")
    if r_hard: ax.axvline(r_hard, color="red", linestyle="--", label="r_hard")

    ax.set_xlim(0.0, r_max)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{outprefix}_{pair[0]}{pair[1]}.png", dpi=200)
    plt.close(fig)

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components

def fast_filter_connectivity_and_arms(frames, ok_mask, local_radius=None, margin=0.8, arm_tol=0.5, calib_frames=20, verbose=True):
    """
    Reject unphysical geometries where the cluster has fractured or formed 1D chains.
    Uses Graph Connectivity for detachments and Volumetric Spatial Density for arms.
    """
    n_frames = len(frames)

    if verbose:
        print("\n[AL] --- Calibrating Connectivity & Local Density (Arm) Filter ---")

    initial_nn_max = []
    initial_same_species_max = []

    # --- PASS 1: Calibrate Shell Boundaries ---
    for i in range(min(calib_frames, n_frames)):
        pos = frames[i].get_positions()
        syms = frames[i].get_chemical_symbols()
        if len(pos) < 2: continue

        tree = cKDTree(pos)

        # 1. First Shell (Bonded Pairs) -> Nearest Neighbor
        distances, indices = tree.query(pos, k=50) # Query enough neighbors to find the 2nd shell
        initial_nn_max.append(distances[:, 1].max())

        # 2. Second Shell (Non-Bonded Pairs) -> Nearest SAME-SPECIES Neighbor
        if local_radius is None:
            frame_max_same_species = 0.0
            for atom_idx, atom_sym in enumerate(syms):
                # Look through neighbors to find the first one with the identical chemical symbol
                for j in range(1, 50):
                    neighbor_idx = indices[atom_idx, j]
                    if syms[neighbor_idx] == atom_sym:
                        dist = distances[atom_idx, j]
                        if dist > frame_max_same_species:
                            frame_max_same_species = dist
                        break
            initial_same_species_max.append(frame_max_same_species)

    if not initial_nn_max:
        return ok_mask

    # Define the first-shell boundary (Detachment limit)
    base_of_first_peak = np.percentile(initial_nn_max, 99)
    max_allowed_bond = base_of_first_peak + margin

    # Define the second-shell boundary (Local Radius) based on non-bonded pairs
    if local_radius is None:
        base_of_nonbonded_peak = np.percentile(initial_same_species_max, 99)
        local_radius = base_of_nonbonded_peak + margin

    # --- PASS 2: Calibrate Baseline Volumetric Density ---
    initial_env_min = []
    for i in range(min(calib_frames, n_frames)):
        pos = frames[i].get_positions()
        if len(pos) < 2: continue

        tree = cKDTree(pos)
        # Count total atoms inside the discovered 2nd-shell sphere
        neighbor_counts = [len(n) for n in tree.query_ball_point(pos, r=local_radius)]
        initial_env_min.append(min(neighbor_counts))

    baseline_min_neighbors = int(np.median(initial_env_min))

    # --- SMART ARM TOLERANCE ---
    # Ensure arm_tol is a float so it triggers the percentage logic
    arm_tol_val = float(arm_tol)
    if arm_tol_val < 1.0:
        # If passed as a fraction (e.g., 0.5), require that percentage of the baseline
        min_allowed_neighbors = max(1, int(baseline_min_neighbors * arm_tol_val))
    else:
        # If passed as an integer > 1 (e.g., 2.0), subtract it directly
        min_allowed_neighbors = max(1, baseline_min_neighbors - int(arm_tol_val))

    if verbose:
        print(f"[AL] 1. Connectivity  | Cutoff: {max_allowed_bond:.2f} Å (Bonded peak base {base_of_first_peak:.2f} Å + {margin} Å)")
        if initial_same_species_max:
            print(f"[AL] 2. Local Density | Radius: {local_radius:.2f} Å (Non-bonded peak base {base_of_nonbonded_peak:.2f} Å + {margin} Å)")
        else:
            print(f"[AL] 2. Local Density | Radius: {local_radius:.2f} Å (Provided manually)")
        print(f"[AL] 3. Arm Threshold | Baseline corner has ~{baseline_min_neighbors} atoms within {local_radius:.2f} Å.")
        print(f"[AL]                  | Rejecting frames if any atom has < {min_allowed_neighbors} atoms in this radius (tol={arm_tol_val}).")

    # --- PASS 3: Apply Filters ---
    n_rejected_fracture = 0
    n_rejected_arm = 0

    for i, frame in enumerate(frames):
        if ok_mask[i]:
            pos = frame.get_positions()
            tree = cKDTree(pos)

            # Check 1: Graph Connectivity
            adj_matrix = tree.sparse_distance_matrix(tree, max_allowed_bond)
            n_components, labels = connected_components(csgraph=adj_matrix, directed=False)

            if n_components > 1:
                ok_mask[i] = False
                n_rejected_fracture += 1
                if verbose:
                    unique, counts = np.unique(labels, return_counts=True)
                    # Convert numpy int64 to standard Python ints for clean printing!
                    detached_sizes = [int(c) for c in sorted(counts)[:-1]]
                    print(f"Frame {i:4d} rejected: Cluster fractured! ({n_components} pieces. Chunk sizes: {detached_sizes})")
                continue

            # Check 2: Volumetric Local Density
            neighbor_counts = [len(n) for n in tree.query_ball_point(pos, r=local_radius)]
            min_atom_neighbors = min(neighbor_counts)

            if min_atom_neighbors < min_allowed_neighbors:
                ok_mask[i] = False
                n_rejected_arm += 1
                if verbose:
                    print(f"Frame {i:4d} rejected: Unphysical 'arm' detected! "
                          f"(Min neighbors in {local_radius:.2f} Å is {min_atom_neighbors}, required {min_allowed_neighbors})")

    if verbose:
        print(f"[Connectivity/Arm] Cohesion sanity: {ok_mask.sum()}/{n_frames} frames OK. "
              f"Rejected {n_rejected_fracture} (fractured), {n_rejected_arm} (arms).")

    return ok_mask

