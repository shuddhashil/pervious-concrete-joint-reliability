"""Vectorised passing-size quantiles for model A3.

q_p = log2 size (mm) at passing fraction p for a six-bin gradation (bins listed coarse -> fine, 25-19 ... 4.75-2.36 mm),
mass uniform in log size within each bin, zero-mass bins skipped. This is EXACTLY the rule that produced the model's
training inputs (research/LCMGP_Retrain_20260930/retrain.py::quant); it is vectorised here only for speed and is checked
row-by-row against that loop in check_a3_quantiles.py.
"""
import numpy as np

ASC = np.log2(np.array([2.36, 4.75, 6.3, 9.5, 12.5, 19.0, 25.0]))
P_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90)
NAMES = ("q10", "q25", "q50", "q75", "q90")


def quantiles_batch(G, p_levels=P_LEVELS):
    """G: (N, 6) fractions (coarse -> fine). Returns (N, len(p_levels)) log2-mm quantiles."""
    G = np.asarray(G, dtype=float)
    if G.ndim == 1:
        G = G[None, :]
    G = G / G.sum(axis=1, keepdims=True)
    f = G[:, ::-1]
    F = np.concatenate([np.zeros((len(G), 1)), np.cumsum(f, axis=1)], axis=1)
    F[:, -1] = 1.0
    r = np.arange(len(G))
    out = np.empty((len(G), len(p_levels)))
    for j, p in enumerate(p_levels):
        k = np.clip((F < p).sum(axis=1), 1, 6)                  # np.searchsorted(F[r], p, side="left")
        for _ in range(6):                                       # skip zero-mass bins
            m = (k < 6) & ((F[r, np.minimum(k, 6)] - F[r, k - 1]) <= 0)
            if not m.any():
                break
            k = k + m
        lo, hi = F[r, k - 1], F[r, k]
        out[:, j] = ASC[k - 1] + (p - lo) / np.maximum(hi - lo, 1e-12) * (ASC[k] - ASC[k - 1])
    return out
