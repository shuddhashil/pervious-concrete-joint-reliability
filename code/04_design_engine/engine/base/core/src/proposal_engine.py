"""Support-aware, target-independent proposal engine (blind-discovery repair).

Proposals may use ONLY target-independent information:

- physical bounds (solvers_common.BND / codec11);
- training-input recipes and their empirical geometry (input columns only --
  outcome targets such as strength/permeability are NEVER read);
- the Stage-17 support model and gradation support data (as fail-closed
  screens, not as proposal targets);
- random / low-discrepancy global coverage.

Do not use training outcome targets to smuggle in target-specific answers.

Mixture (with per-candidate ``source`` labels):
- ``global_sobol``: low-discrepancy full-domain safeguard;
- ``empirical_cluster``: perturbations around supported empirical recipe
  clusters using full covariance (whitened coordinates);
- ``boundary_atom``: explicit zero/near-zero sand, SP, and gradation bins
  where the training data support them;
- ``data_prior``: simplex-aware gradation draws via gradation_prior.

Simplex handling is always valid (renormalised, non-negative, sums to one).
When support cannot be established (missing training/support data), proposal
generation fails closed by raising instead of returning blind guesses.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    from scipy.stats import qmc
except Exception:  # pragma: no cover - bundled runtime includes scipy
    qmc = None

SCALAR_KEYS = ("C", "r", "A", "P", "S", "phi")
BOUNDARY_ATOM_EPS = 1e-3


def _bounds():
    import solvers_common as SC
    return SC.BND


def global_unit_design(n, dim, seed, method="sobol"):
    """Low-discrepancy global unit-cube design (full-domain safeguard)."""
    if n <= 0:
        return np.empty((0, dim), dtype=float)
    if method == "sobol" and qmc is not None:
        return np.asarray(qmc.Sobol(d=dim, scramble=True, seed=int(seed)).random(int(n)), dtype=float)
    rng = np.random.default_rng(int(seed))
    u = (np.arange(int(n), dtype=float)[:, None] + rng.random((int(n), dim))) / max(int(n), 1)
    for j in range(dim):
        rng.shuffle(u[:, j])
    return u


def empirical_recipe_geometry(training_csv=None, max_rows=5000):
    """Cluster centres + full covariances from training INPUTS only.

    Only recipe/input columns are read; outcome columns (strength,
    permeability, porosity targets) are never consulted, so no
    target-specific answer can leak through this path.
    """
    import solvers_common as SC
    if training_csv is None:
        import gradation_prior as GP
        training_csv = GP.TRAINING_CSV
    if not os.path.exists(str(training_csv)):
        raise RuntimeError("proposal engine fails closed: training recipe data not found")
    df = pd.read_csv(training_csv, nrows=max_rows)
    colmap = {"cem (kg/m3)": "C", "w/c": "r", "CA (kg/m3)": "A",
              "SP (kg/m3)": "P", "sand (kg/m3)": "S", "porosity": "phi"}
    avail = [c for c in colmap if c in df.columns]
    if not avail:
        raise RuntimeError("proposal engine fails closed: no recipe columns in training data")
    units = []
    for _, row in df[avail].dropna().iterrows():
        x = []
        ok = True
        for c in avail:
            k = colmap[c]
            lo, hi = SC.BND[k]
            v = float(row[c])
            if k == "A":
                import math
                x.append((math.log(max(v, 1e-9)) - SC.LNA_LO) / (SC.LNA_HI - SC.LNA_LO))
            else:
                x.append((v - lo) / (hi - lo))
            if not np.isfinite(x[-1]):
                ok = False
                break
        if ok:
            units.append(x)
    units = np.asarray(units, dtype=float)
    if len(units) < 8:
        raise RuntimeError("proposal engine fails closed: too few training recipes for geometry")
    # Deterministic small-k clustering in unit space (no outcome data used).
    k = min(6, max(2, len(units) // 50))
    rng = np.random.default_rng(94072021)
    centres = units[rng.choice(len(units), size=k, replace=False)]
    assign = np.zeros(len(units), dtype=int)
    for _ in range(25):
        d = ((units[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        assign = d.argmin(axis=1)
        for j in range(k):
            members = units[assign == j]
            if len(members):
                centres[j] = members.mean(axis=0)
    clusters = []
    for j in range(k):
        members = units[assign == j]
        if len(members) < 3:
            continue
        cov = np.cov(members.T) + 1e-4 * np.eye(members.shape[1])
        clusters.append({"centre": centres[j], "cov": cov,
                         "n_members": int(len(members)), "columns": [colmap[c] for c in avail]})
    if not clusters:
        raise RuntimeError("proposal engine fails closed: empirical clustering produced no basin")
    return clusters


def perturb_full_covariance(rng, clusters, n):
    """Whitened-coordinate perturbations around empirical clusters."""
    if n <= 0:
        return np.empty((0, len(SCALAR_KEYS)), dtype=float), []
    dim = len(SCALAR_KEYS)
    out, sources = [], []
    weights = np.array([c["n_members"] for c in clusters], dtype=float)
    weights /= weights.sum()
    counts = rng.multinomial(int(n), weights)
    for c, cnt in zip(clusters, counts):
        if cnt <= 0:
            continue
        # Full covariance (not independent raw-coordinate noise): sample in
        # whitened space via Cholesky so correlated recipe directions survive.
        try:
            chol = np.linalg.cholesky(c["cov"])
        except np.linalg.LinAlgError:
            chol = np.diag(np.sqrt(np.maximum(np.diag(c["cov"]), 1e-8)))
        z = rng.normal(size=(int(cnt), c["centre"].shape[0]))
        block = c["centre"][None, :] + z @ chol.T
        full = np.tile(0.5, (int(cnt), dim))
        for j, key in enumerate(c["columns"]):
            full[:, SCALAR_KEYS.index(key)] = block[:, j]
        out.append(np.clip(full, 0.0, 1.0))
        sources.extend(["empirical_cluster"] * int(cnt))
    if not out:
        return np.empty((0, dim), dtype=float), []
    return np.concatenate(out, axis=0), sources


def empirical_joint_geometry(training_csv=None, max_rows=5000):
    """Full 11-D recipe geometry, preserving scalar/gradation correlation."""
    import codec11 as K
    import gradation_prior as GP
    if training_csv is None:
        training_csv = GP.TRAINING_CSV
    df = pd.read_csv(training_csv, nrows=max_rows)
    scalar_cols = {"cem (kg/m3)": "C", "w/c": "r", "CA (kg/m3)": "A",
                   "SP (kg/m3)": "P", "sand (kg/m3)": "S", "porosity": "phi"}
    required = list(scalar_cols) + list(GP.BIN_COLS)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"proposal engine fails closed: missing joint recipe columns {missing}")
    units = []
    for _, row in df[required].dropna().iterrows():
        g = np.asarray([float(row[c]) for c in GP.BIN_COLS], dtype=float)
        if np.any(g < 0.0) or not np.isfinite(g).all() or g.sum() <= 0.0:
            continue
        g /= g.sum()
        cand = {key: float(row[col]) for col, key in scalar_cols.items()}
        cand["P"] = max(0.0, cand["P"])
        cand["S"] = max(0.0, cand["S"])
        cand["g"] = g
        try:
            x = np.asarray(K.encode11(cand, "coarse_first"), dtype=float)
        except Exception:
            continue
        if np.isfinite(x).all() and np.all((x >= 0.0) & (x <= 1.0)):
            units.append(x)
    units = np.asarray(units, dtype=float)
    if len(units) < 8:
        raise RuntimeError("proposal engine fails closed: too few complete joint recipes")
    k = min(8, max(2, len(units) // 40))
    rng = np.random.default_rng(94072022)
    centres = units[rng.choice(len(units), size=k, replace=False)].copy()
    assign = np.zeros(len(units), dtype=int)
    for _ in range(30):
        d = ((units[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        assign = d.argmin(axis=1)
        for j in range(k):
            members = units[assign == j]
            if len(members):
                centres[j] = members.mean(axis=0)
    clusters = []
    for j in range(k):
        members = units[assign == j]
        if len(members) < 3:
            continue
        clusters.append({"centre": centres[j],
                         "cov": np.cov(members.T) + 1e-5 * np.eye(11),
                         "n_members": int(len(members))})
    if not clusters:
        raise RuntimeError("proposal engine fails closed: no joint empirical clusters")
    return clusters


def perturb_joint_covariance(rng, clusters, n, scale=0.65):
    """Sample correlated full-recipe perturbations in encoded 11-D space."""
    if n <= 0:
        return np.empty((0, 11), dtype=float), []
    weights = np.asarray([c["n_members"] for c in clusters], dtype=float)
    weights /= weights.sum()
    counts = rng.multinomial(int(n), weights)
    blocks = []
    for cluster, count in zip(clusters, counts):
        if count <= 0:
            continue
        try:
            chol = np.linalg.cholesky(cluster["cov"])
        except np.linalg.LinAlgError:
            chol = np.diag(np.sqrt(np.maximum(np.diag(cluster["cov"]), 1e-8)))
        z = rng.normal(size=(int(count), 11))
        blocks.append(np.clip(cluster["centre"][None, :] + float(scale) * z @ chol.T,
                              0.0, 1.0))
    out = np.concatenate(blocks, axis=0) if blocks else np.empty((0, 11), dtype=float)
    return out, ["empirical_joint_cluster"] * len(out)


def boundary_atoms(rng, n, training_csv=None):
    """Zero/near-zero sand, SP, and gradation-bin atoms where data support them.

    Atoms are only emitted for coordinates that actually take zero values in
    the training inputs; otherwise the atom would propose unsupported space.
    """
    import gradation_prior as GP
    if training_csv is None:
        training_csv = GP.TRAINING_CSV
    df = pd.read_csv(training_csv, nrows=2000)
    support_zero = {}
    for col, key in (("sand (kg/m3)", "S"), ("SP (kg/m3)", "P")):
        if col in df.columns:
            support_zero[key] = bool((pd.to_numeric(df[col], errors="coerce").fillna(1.0) <= 0.0).any())
        else:
            support_zero[key] = False
    atoms, sources = [], []
    for _ in range(int(n)):
        u = rng.uniform(0.0, 1.0, size=len(SCALAR_KEYS))
        if support_zero.get("S") and rng.uniform() < 0.5:
            u[SCALAR_KEYS.index("S")] = 0.0
        if support_zero.get("P") and rng.uniform() < 0.5:
            u[SCALAR_KEYS.index("P")] = 0.0
        atoms.append(u)
        sources.append("boundary_atom")
    if not atoms:
        return np.empty((0, len(SCALAR_KEYS)), dtype=float), []
    return np.asarray(atoms, dtype=float), sources


def valid_simplex_gradations(rng, n):
    """Valid simplex gradations (Dirichlet mixture + data prior)."""
    import codec11 as K
    import gradation_prior as GP
    n = int(n)
    if n <= 0:
        return np.empty((0, 5), dtype=float), np.empty((0, 6), dtype=float)
    n1, n2 = int(n * 0.4), int(n * 0.3)
    n3 = n - n1 - n2
    parts = []
    if n1:
        parts.append(rng.dirichlet([1.0] * 6, size=n1))
    if n2:
        parts.append(rng.dirichlet([0.3] * 6, size=n2))
    if n3:
        try:
            parts.append(GP.sample_prior(rng, n3))
        except Exception:
            parts.append(rng.dirichlet([1.0] * 6, size=n3))
    g = np.concatenate(parts, axis=0)
    rng.shuffle(g)
    g = np.clip(g, 0.0, 1.0)
    g = g / g.sum(axis=1, keepdims=True).clip(min=1e-12)
    sticks = np.array([K.g_to_sticks(gg, "coarse_first") for gg in g])
    return sticks, g


def generate_proposals(n, seed, training_csv=None,
                       mix=(0.35, 0.35, 0.15, 0.15)):
    """Full target-independent proposal mixture with source labels.

    Returns (X11_unit, sources) where X11_unit is (n, 11) in codec11 unit
    coordinates.  ``mix`` = (global, empirical, boundary, prior-extra) and
    must sum to ~1.  Raises (fails closed) when support cannot be established.
    """
    rng = np.random.default_rng(int(seed))
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    w_g, w_e, w_b, _ = mix
    n_g = int(round(n * w_g))
    n_e = int(round(n * w_e))
    n_b = int(round(n * w_b))
    n_p = max(0, n - n_g - n_e - n_b)
    import codec11 as K
    import gradation_prior as GP
    blocks, sources = [], []

    def independent_block(count, scalar_seed, label):
        scalars = global_unit_design(count, 6, seed=scalar_seed)
        sticks, _ = valid_simplex_gradations(rng, count)
        return np.concatenate([scalars, sticks], axis=1), [label] * count

    xg, sg = independent_block(n_g, int(seed), "global_sobol")
    blocks.append(xg); sources.extend(sg)

    clusters = empirical_joint_geometry(training_csv=training_csv)
    xe, se = perturb_joint_covariance(rng, clusters, n_e)
    blocks.append(xe); sources.extend(se)

    scalar_atoms, atom_sources = boundary_atoms(rng, n_b, training_csv=training_csv)
    atom_sticks, atom_g = valid_simplex_gradations(rng, n_b)
    try:
        df = pd.read_csv(training_csv or GP.TRAINING_CSV, nrows=2000)
        zero_bins = [j for j, col in enumerate(GP.BIN_COLS)
                     if col in df.columns and
                     (pd.to_numeric(df[col], errors="coerce").fillna(1.0) <= 0.0).any()]
        for i in range(n_b):
            if not zero_bins:
                break
            j = int(rng.choice(zero_bins))
            atom_g[i, j] = 0.0 if rng.uniform() < 0.5 else BOUNDARY_ATOM_EPS
            atom_g[i] /= max(float(atom_g[i].sum()), 1e-12)
            atom_sticks[i] = K.g_to_sticks(atom_g[i], "coarse_first")
    except Exception:
        pass
    blocks.append(np.concatenate([scalar_atoms, atom_sticks], axis=1))
    sources.extend(atom_sources)

    xp, sp = independent_block(n_p, int(seed) + 999, "data_prior")
    blocks.append(xp); sources.extend(sp)
    X = np.concatenate(blocks, axis=0)
    if len(X) != n:
        raise RuntimeError(f"proposal engine produced {len(X)} rows; expected {n}")
    return np.clip(X, 0.0, 1.0), sources
