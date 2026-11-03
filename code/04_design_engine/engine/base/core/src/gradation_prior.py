"""Data-fitted gradation prior for target-agnostic adaptive initialisation.

Fits a logistic-normal (Gaussian in ALR coordinates) on unique training
gradations from training_data_reconciled.csv. Zero bins replaced by a small
documented epsilon before the transform and mapped back after sampling.
Allows exact zero bins in a fraction of samples so bins can be empty as in
the data. Used ONLY to propose starting points and diversity; not part of
the objective and adds no preference during search.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
TRAINING_CSV = os.path.join(RUN, "tables", "training_data_reconciled.csv")
BIN_COLS = ["R25-19mm", "R19-12.5mm", "R12.5-9.5mm", "R9.5-6.3mm", "R6.3-4.75mm", "R4.75-2.36mm"]
EPSILON = 1e-3
ZERO_BIN_FRACTION = 0.2


def load_unique_gradations():
    df = pd.read_csv(TRAINING_CSV, usecols=["Unnamed: 0"] + BIN_COLS)
    g = df[BIN_COLS].to_numpy(dtype=float)
    # normalise rows to sum 1 (data bins are percentages summing ~100)
    s = g.sum(axis=1, keepdims=True)
    s[s <= 0] = 1.0
    g = g / s
    # unique rows (rounded to avoid float noise)
    _, idx = np.unique(np.round(g, 9), axis=0, return_index=True)
    return g[np.sort(idx)], df["Unnamed: 0"].to_numpy()[np.sort(idx)]


def _alr(g, eps=EPSILON):
    g = np.clip(g, eps, None)
    g = g / g.sum(axis=-1, keepdims=True)
    return np.log(g[..., :-1] / g[..., -1:])


def _alr_inv(z):
    e = np.exp(z)
    denom = 1.0 + e.sum(axis=-1, keepdims=True)
    g = np.concatenate([e / denom, 1.0 / denom], axis=-1)
    return g / g.sum(axis=-1, keepdims=True)


def fit_prior():
    g, _ = load_unique_gradations()
    z = _alr(np.clip(g, EPSILON, None))
    return {"mean": z.mean(axis=0), "cov": np.cov(z.T) + 1e-6 * np.eye(5),
            "epsilon": EPSILON, "n_unique": int(g.shape[0])}


def sample_prior(rng, n, prior=None):
    if prior is None:
        prior = fit_prior()
    mean, cov = prior["mean"], prior["cov"]
    z = rng.multivariate_normal(mean, cov, size=n)
    g = _alr_inv(z)
    # allow exact zero bins in a fraction of samples (mask smallest bin)
    n_zero = int(round(n * ZERO_BIN_FRACTION))
    if n_zero > 0:
        for i in range(n_zero):
            j = int(rng.integers(0, 6))
            g[i, j] = 0.0
            s = g[i].sum()
            if s > 0:
                g[i] /= s
            else:
                g[i] = np.full(6, 1.0 / 6)
    return np.clip(g, 0.0, 1.0)
