"""Vectorised, numerically-verified drop-in replacement for the gradation
descriptor computation in the frozen evaluator
(research/Stage1_CA_Aware_Optimizer_20260914/run_01/src/stage1_evaluator.py:
_grid_passing / _dpp_grid / descriptors_precise).

This module NEVER touches the frozen file on disk. It is meant to be
monkey-patched into a live process's `stage1_evaluator` module object only
(see fast_descriptors.install_patch), scoped to that process, so other runs
that import the frozen evaluator directly are completely unaffected.

Why this is safe:
- _grid_passing(f) is, for fixed bin edges, a LINEAR function of f (a
  constant 500x6 "fraction-passing" matrix times f), because the bin edges
  never depend on f -- only the masses do. The original computes this
  linear map via a 500x6 nested Python loop every call; here it is
  precomputed once and applied as a single matrix-vector product.
- The resulting passing curve is provably non-decreasing along the grid
  (each bin's own passing fraction is non-decreasing in threshold, and a
  nonnegative-weighted sum of non-decreasing functions is non-decreasing),
  so duplicate y-values always form contiguous runs. The original collapses
  duplicates via `for v in np.unique(p): xs.mean()`; here the same grouped
  mean is computed via run-boundary detection + np.add.reduceat, with no
  per-unique-value Python-level loop.
- descriptors_precise() recomputes _grid_passing twice per candidate (once
  each for the D10 and D60 percentile lookups) even though it does not
  depend on the percentile; here it is computed once and reused.

Every function here has a *_reference counterpart that is a straight,
unoptimised re-implementation used only by test_equivalence.py to prove
bit-for-bit (or float-tolerance) agreement with the frozen original before
any of this is wired into a live search.
"""
from __future__ import annotations

import numpy as np
from a3_quantiles import quantiles_batch  # sandbox A3 engine

# ---- bin geometry, copied verbatim from the frozen evaluator's _grid_passing
# (research/Stage1_CA_Aware_Optimizer_20260914/run_01/src/stage1_evaluator.py:58-76).
# Order matches f[0..5] / DIAM[0..5] / BINS in the frozen module.
_BIN_HI = np.array([25.0, 19.0, 12.5, 9.5, 6.3, 4.75], dtype=float)
_BIN_LO = np.array([19.0, 12.5, 9.5, 6.3, 4.75, 2.36], dtype=float)
_GRID_TH = np.geomspace(2.36, 25.0, 500)  # identical construction to the frozen _GRID_TH

_LOG_TH = np.log(_GRID_TH)
_LOG_LO = np.log(_BIN_LO)
_LOG_HI = np.log(_BIN_HI)

# FRAC_MATRIX[n, i] = fraction of bin i's mass counted as "passing" grid
# threshold n. Fixed constant, independent of any candidate's gradation.
_RATIO = (_LOG_TH[:, None] - _LOG_LO[None, :]) / (_LOG_HI[None, :] - _LOG_LO[None, :])
FRAC_MATRIX = np.clip(_RATIO, 0.0, 1.0)  # shape (500, 6)


def grid_passing_batch(G: np.ndarray) -> np.ndarray:
    """Vectorised _grid_passing for a batch. G: (N, 6) raw (unnormalised) mass
    fractions, any nonnegative scale. Returns (N, 500) percent-passing curves,
    numerically equivalent to calling the frozen _grid_passing(f) row-by-row.
    """
    G = np.asarray(G, dtype=float)
    tot = G.sum(axis=1, keepdims=True)
    f_norm = G / tot
    y = 100.0 * (f_norm @ FRAC_MATRIX.T) / f_norm.sum(axis=1, keepdims=True)
    return y


def _dedup_xy_single(y_row: np.ndarray, x: np.ndarray):
    """Collapse a non-decreasing y_row (paired with strictly increasing x)
    to (unique_y, mean_x_per_group), matching the frozen _dpp_grid's
    `for v in np.unique(p): xs.mean()` behaviour exactly, but via
    run-boundary detection since y_row is guaranteed non-decreasing."""
    change = np.empty(y_row.shape[0], dtype=bool)
    change[0] = True
    change[1:] = y_row[1:] != y_row[:-1]
    starts = np.flatnonzero(change)
    counts = np.diff(np.append(starts, y_row.shape[0]))
    sums = np.add.reduceat(x, starts)
    means = sums / counts
    uniq_y = y_row[starts]
    return uniq_y, means


def dpp_grid_batch(G: np.ndarray, percents) -> dict:
    """Vectorised replacement for descriptors_precise's two _dpp_grid calls
    (percent=10.0 and percent=60.0), computing _grid_passing ONCE per row
    (the frozen code recomputes it once per percentile, i.e. twice total).

    Returns {percent: (N,) array of interpolated log-threshold results,
    already exponentiated back to physical units, matching _dpp_grid's
    `float(np.exp(np.interp(...)))` return convention (nan where undefined).
    """
    G = np.asarray(G, dtype=float)
    n = G.shape[0]
    passing = grid_passing_batch(G)  # (N, 500), computed once, reused below
    finite_mask = np.isfinite(passing) & np.isfinite(_LOG_TH)[None, :] & (_GRID_TH > 0)[None, :]

    out = {p: np.full(n, np.nan, dtype=float) for p in percents}
    for row in range(n):
        mask = finite_mask[row]
        p = passing[row][mask]
        x = _LOG_TH[mask]
        if p.shape[0] == 0:
            continue
        up, ux = _dedup_xy_single(p, x)
        if up.shape[0] < 2:
            continue
        lo_p, hi_p = up.min(), up.max()
        for percent in percents:
            if percent < lo_p or percent > hi_p:
                continue
            out[percent][row] = float(np.exp(np.interp(percent, up, ux)))
    return out


# Pairwise log-diameter-ratio matrix for I_gap, upper triangle only (i<j),
# matching descriptors_precise's `for i in range(6): for j in range(i+1,6)`.
def _build_logratio_ut(diam: np.ndarray) -> np.ndarray:
    n = diam.shape[0]
    m = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = np.log(diam[i] / diam[j])
    return m


def descriptors_precise_batch(G: np.ndarray, diam: np.ndarray) -> dict:
    """Vectorised descriptors_precise for a whole batch at once.
    G: (N, 6) raw fractions (need not be pre-normalised -- descriptors_precise
    itself does not require sum==1, only _grid_passing normalises internally).
    diam: the frozen module's DIAM constant, passed in (never redefined here)
    so this can never silently drift from the frozen geometry.
    """
    G = np.asarray(G, dtype=float)
    diam = np.asarray(diam, dtype=float)
    was = G @ diam
    aag = np.sum(6.0 * G / diam[None, :], axis=1)
    logratio_ut = _build_logratio_ut(diam)
    ig = np.einsum('ni,nj,ij->n', G, G, logratio_ut)
    dpp = dpp_grid_batch(G, (10.0, 60.0))
    d10, d60 = dpp[10.0], dpp[60.0]
    with np.errstate(invalid="ignore", divide="ignore"):
        cu = np.where(d10 > 0, d60 / d10, np.nan)
    q = quantiles_batch(G)
    return {"WAS": was, "I_gap": ig, "Cu": cu, "Aag": aag, "q10": q[:, 0], "q25": q[:, 1], "q50": q[:, 2], "q75": q[:, 3], "q90": q[:, 4]}


def install_patch(stage1_evaluator_module):
    """Monkey-patch cached_descriptors on the given (already-imported)
    stage1_evaluator module object IN THIS PROCESS ONLY. Does not touch any
    file on disk. Intended for use after equivalence has been verified via
    test_equivalence.py in the same environment/process."""
    E = stage1_evaluator_module
    diam = E.DIAM
    cache: dict = {}

    def cached_descriptors_fast(g):
        key = tuple(np.round(np.asarray(g, dtype=float), 12))
        hit = cache.get(key)
        if hit is None:
            row = np.asarray(key, dtype=float)[None, :]
            batch = descriptors_precise_batch(row, diam)
            hit = {"WAS": float(batch["WAS"][0]), "I_gap": float(batch["I_gap"][0]),
                   "Cu": float(batch["Cu"][0]), "Aag": float(batch["Aag"][0]),
                   **{k: float(batch[k][0]) for k in ("q10", "q25", "q50", "q75", "q90")}}
            cache[key] = hit
        return hit

    E.cached_descriptors = cached_descriptors_fast
    return cached_descriptors_fast
