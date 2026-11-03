"""Ordered-bin Wasserstein-1 gradation support gate (TASK A2).

For gradation fractions g (6 bins ordered coarse to fine, sum 1) and a
training gradation t: W1(g,t) = sum_{k=1..5} |F_g(k)-F_t(k)| * w_k,
where F is the cumulative fraction up to bin k and
w_k = |ln(DIAM[k])-ln(DIAM[k+1])| (primary, log-diameter version).
Unit-spacing variant (w_k=1) reported for comparison.

Candidate statistic: W1_min(g) = min over unique training gradations t.
Threshold epsilon = 95th percentile of leave-study-out W1_min over unique
training gradations (90th/99th stored for sensitivity). Hard support gate:
violation = max(0, W1_min-epsilon)/epsilon, zero inside (support constraint,
not shape preference).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
TRAINING_CSV = os.path.join(RUN, "tables", "training_data_reconciled.csv")
BUNDLE_DIR = os.path.join(RUN, "state_support_gradation")
CONTRACT_PATH = os.path.join(RUN, "GRADATION_SUPPORT_CONTRACT.json")

DIAM = np.array([21.79449471770337, 15.411035007422441, 10.897247358851684,
                 7.736278174936576, 5.470374758643141, 3.348133808556641], dtype=float)
W_LOG = np.abs(np.log(DIAM[:-1]) - np.log(DIAM[1:]))
W_UNIT = np.ones(5)
BIN_COLS = ["R25-19mm", "R19-12.5mm", "R12.5-9.5mm", "R9.5-6.3mm", "R6.3-4.75mm", "R4.75-2.36mm"]


def w1_pairwise(G, T, w=W_LOG):
    G = np.asarray(G, float) / np.asarray(G, float).sum(axis=1, keepdims=True).clip(min=1e-12)
    T = np.asarray(T, float) / np.asarray(T, float).sum(axis=1, keepdims=True).clip(min=1e-12)
    Fg = np.cumsum(G[:, :5], axis=1)[:, None, :]
    Ft = np.cumsum(T[:, :5], axis=1)[None, :, :]
    return np.abs(Fg - Ft).sum(axis=-1) if w is None else (np.abs(Fg - Ft) * w).sum(axis=-1)


def w1_min(g, unique_T=None, w=W_LOG):
    g = np.asarray(g, float).reshape(1, -1)
    if unique_T is None:
        unique_T = load_bundle()[0]
    d = w1_pairwise(g, unique_T, w)
    return float(d.min())


def violation(g, epsilon=None):
    if epsilon is None:
        epsilon = load_contract()["epsilon_95"]
    m = w1_min(g)
    # Fail closed: a non-positive or non-finite threshold (or distance) can
    # never read as "inside support".  Previously NaN epsilon returned 0.0.
    if not (np.isfinite(m) and np.isfinite(epsilon) and epsilon > 0):
        return float("inf"), m
    return max(0.0, (m - epsilon) / epsilon), m


def load_bundle():
    df = pd.read_csv(os.path.join(BUNDLE_DIR, "unique_training_gradations.csv"))
    T = df[BIN_COLS].to_numpy(dtype=float)
    studies = df["study"].to_numpy()
    return T, studies


def load_contract():
    with open(CONTRACT_PATH) as f:
        return json.load(f)
