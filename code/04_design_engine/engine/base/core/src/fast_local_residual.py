"""Process-local acceleration for frozen local-residual uncertainty.

The frozen ``LCMGP_without_head._local_residual_core`` computes the local
standard deviation once per candidate row using Python list comprehensions.
Its inputs and neighbour-selection semantics are retained exactly here; only
the final per-row standard-deviation reduction is expressed as one NumPy
matrix operation.  ``install_patch`` never modifies the frozen source file.

This module must be used only after the equivalence tests in
``tests/test_fast_local_residual_equivalence.py`` pass in the active runtime.
"""
from __future__ import annotations

import numpy as np


def local_residual_core_fast(
    Xs, X_ref, res_s, res_p, q=5, exclude_self=False,
    candidate_original_row_index=None, reference_original_row_index=None,
    exclude_same_original_row=False,
):
    """Numerically equivalent vectorised form of the frozen helper."""
    Xs = np.asarray(Xs, dtype=float)
    X_ref = np.asarray(X_ref, dtype=float)
    res_s = np.asarray(res_s, dtype=float).ravel()
    res_p = np.asarray(res_p, dtype=float).ravel()
    if Xs.ndim != 2:
        raise ValueError(f"Candidate feature matrix must be 2D; got shape {Xs.shape}.")
    if X_ref.ndim != 2:
        raise ValueError(f"OOS reference feature matrix must be 2D; got shape {X_ref.shape}.")
    if Xs.shape[1] != X_ref.shape[1]:
        raise ValueError(f"Candidate feature dimension {Xs.shape[1]} does not match OOS reference dimension {X_ref.shape[1]}.")
    if X_ref.shape[0] != res_s.shape[0] or X_ref.shape[0] != res_p.shape[0]:
        raise ValueError("OOS reference rows/features/residuals length mismatch.")
    candidate_orig = None
    ref_orig = None
    if candidate_original_row_index is not None:
        candidate_orig = np.asarray(candidate_original_row_index, dtype=int).ravel()
        if candidate_orig.shape[0] != Xs.shape[0]:
            raise ValueError("candidate_original_row_index length does not match candidate rows.")
    if reference_original_row_index is not None:
        ref_orig = np.asarray(reference_original_row_index, dtype=int).ravel()
        if ref_orig.shape[0] != X_ref.shape[0]:
            raise ValueError("reference_original_row_index length does not match reference rows.")
    if exclude_same_original_row and (candidate_orig is None or ref_orig is None):
        raise ValueError("exclude_same_original_row=True requires candidate and reference original-row indices.")
    if not np.isfinite(Xs).all():
        raise ValueError("Candidate feature matrix contains non-finite values.")
    q_req = int(q)
    if q_req < 2:
        raise ValueError("Local residual uncertainty requires q >= 2 to compute a sample standard deviation.")
    n_ref_before_filter = int(X_ref.shape[0])
    finite = np.isfinite(X_ref).all(axis=1) & np.isfinite(res_s) & np.isfinite(res_p)
    ref_original_idx = np.arange(n_ref_before_filter, dtype=int)[finite]
    if ref_orig is not None:
        ref_orig = ref_orig[finite]
    X_ref, res_s, res_p = X_ref[finite], res_s[finite], res_p[finite]
    if len(X_ref) < 2:
        raise ValueError("At least two finite OOS residual references are required.")
    d = np.sqrt(np.sum((Xs[:, None, :] - X_ref[None, :, :]) ** 2, axis=2))
    if exclude_self:
        if Xs.shape[0] != n_ref_before_filter:
            raise ValueError("exclude_self=True requires candidate rows to align one-to-one with the unfiltered OOS reference rows.")
        for row_idx in range(Xs.shape[0]):
            self_cols = np.where(ref_original_idx == row_idx)[0]
            if self_cols.size:
                d[row_idx, self_cols] = np.inf
    if exclude_same_original_row:
        for row_idx, original_idx in enumerate(candidate_orig):
            same_cols = np.where(ref_orig == int(original_idx))[0]
            if same_cols.size:
                d[row_idx, same_cols] = np.inf
    available = np.sum(np.isfinite(d), axis=1)
    min_available = int(np.min(available)) if available.size else 0
    if min_available < 2:
        raise ValueError("At least two valid non-self OOS neighbors are required for local residual uncertainty.")
    q_eff = int(min(q_req, min_available))
    nn_unsorted = np.argpartition(d, q_eff - 1, axis=1)[:, :q_eff]
    rows = np.arange(d.shape[0])[:, None]
    order = np.argsort(d[rows, nn_unsorted], axis=1)
    nn_idx = nn_unsorted[rows, order]
    nn_dist = d[rows, nn_idx]
    # This is the only algorithmic change: replace 2*N Python np.std calls
    # with two reductions over the identical (N, q_eff) selected entries.
    s_std = np.std(res_s[nn_idx], axis=1, ddof=1)
    p_std = np.std(res_p[nn_idx], axis=1, ddof=1)
    return {
        "q_neighbors": q_eff,
        "D_res_knn_mean": np.mean(nn_dist, axis=1),
        "D_res_knn_max": np.max(nn_dist, axis=1),
        "nearest_oos_indices_internal": nn_idx,
        "strength_residual_std_local_mpa": s_std,
        "logperm_residual_std_local": p_std,
    }


def install_patch(lcmgp_module):
    """Install in this Python process only; returns the patched function."""
    lcmgp_module._local_residual_core = local_residual_core_fast
    return local_residual_core_fast
