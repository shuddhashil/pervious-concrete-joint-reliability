"""Stage-17 data contract: authoritative 650 frame, recipe groups, representations.

Self-contained copy of the Stage-14/16 loader + grouping logic (credited).
Upstream files are only READ, never modified. CS/permeability columns are
never loaded anywhere in this folder (forbidden by contract).

Primary representation x_raw = [w/c, cem, SP, sand, b1..b6] with frozen
monotone transforms (log cem; log1p SP/sand/bins; raw w/c).
Sensitivity representation x9 = Stage-15 authoritative descriptors.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

# Read-only reuse of the Stage-14 authoritative loader (never modified).
S14_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage14")
if S14_SRC not in sys.path:
    sys.path.insert(0, S14_SRC)
import importlib.util as _ilu

_DC_SPEC = _ilu.spec_from_file_location(
    'stage14_data_contract',
    os.path.join(S14_SRC, 'data_contract.py'))
DC = _ilu.module_from_spec(_DC_SPEC)
sys.modules['stage14_data_contract'] = DC
_DC_SPEC.loader.exec_module(DC)  # noqa: E402  (Stage-14, read-only reuse)

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINS = ['R25-19mm', 'R19-12.5mm', 'R12.5-9.5mm', 'R9.5-6.3mm',
        'R6.3-4.75mm', 'R4.75-2.36mm']
R1_COLS = ['w/c', 'cem (kg/m3)', 'CA (kg/m3)', 'SP (kg/m3)', 'sand (kg/m3)',
           'WAS', 'I_gap', 'Cu', 'Aag(mm^-1)']
# Primary raw-bin representation columns (pre-transform raw frame fields).
XRAW_FIELDS = ['w/c', 'cem', 'SP', 'sand'] + [f'b{j}' for j in range(1, 7)]

# Columns that must never be loaded into support-model code.
FORBIDDEN_SUBSTRINGS = [
    'comp_converted', 'perm (mm/s)', 'ln k', 'lnk', 'strength',
    'compressive', 'cs (mpa)', 'cs_mpa', 'target', 'reliab', 'pass_fail',
    'outcome', 'c267', 'id-80', 'id-90',
]


def load_frame():
    """Authoritative 650-row frame with support-relevant columns only."""
    DC.apply_authoritative_contract()
    import LCMGP_without_head as L
    X10, _, _, _, info = L.load_and_prepare_data(DC.XLSX, return_metadata=True)
    assert X10.shape == (650, 10), X10.shape
    md = info['row_metadata']
    keep = md['source_dataframe_row_index'].to_numpy()
    xlsx = pd.read_excel(DC.XLSX)
    raw = xlsx.loc[keep].reset_index(drop=True)
    csv = pd.read_csv(DC.XLSX.replace('.xlsx', '.csv'))
    assert len(csv) == 650
    for col in ['cem (kg/m3)', 'porosity', 'CA (kg/m3)']:
        a = pd.to_numeric(raw[col], errors='coerce').to_numpy()
        b = pd.to_numeric(csv[col], errors='coerce').to_numpy()
        assert np.allclose(a, b, equal_nan=True), col
    fr = pd.DataFrame({
        'pos': np.arange(650, dtype=int),
        'excel_row': md['source_excel_row'].to_numpy(),
        'mix_id': md['mix_id'].astype(str).to_numpy(),
        'study': csv['_src_ffill'].astype(str).to_numpy(),
        'phi': pd.to_numeric(raw['porosity'], errors='coerce').to_numpy(float),
        'w/c': pd.to_numeric(raw['w/c'], errors='coerce').to_numpy(float),
        'cem': pd.to_numeric(raw['cem (kg/m3)'], errors='coerce').to_numpy(float),
        'water': pd.to_numeric(raw['water (kg/m3)'], errors='coerce').to_numpy(float),
        'CA': pd.to_numeric(raw['CA (kg/m3)'], errors='coerce').to_numpy(float),
        'sand': pd.to_numeric(raw['sand (kg/m3)'], errors='coerce').to_numpy(float),
        'SP': pd.to_numeric(raw['SP (kg/m3)'], errors='coerce').to_numpy(float),
    })
    G = raw[BINS].apply(pd.to_numeric, errors='coerce').to_numpy(float) / 100.0
    assert np.allclose(G.sum(1), 1.0, atol=1e-9), 'bin fractions must sum to one'
    for j in range(6):
        fr[f'f{j + 1}'] = G[:, j]
        fr[f'b{j + 1}'] = fr['CA'].to_numpy() * G[:, j]
    X10 = X10  # transformed, unstandardized, M1global-10D order
    for k, col in enumerate(R1_COLS):
        fr['R1_' + col] = X10[:, k + 1]
    assert np.isfinite(
        fr[[c for c in fr.columns if c.startswith('R1_')]].to_numpy()).all()
    assert np.isfinite(fr['phi'].to_numpy()).all()
    # Bin masses must sum back to total CA (contract check).
    bsum = fr[[f'b{j}' for j in range(1, 7)]].to_numpy().sum(1)
    assert np.allclose(bsum, fr['CA'].to_numpy(), rtol=1e-9), 'bins must sum to CA'
    return fr


def add_recipe_groups(fr, round_dec=4, near_t=0.1):
    """Nominal recipe groups (copy of Stage-16 logic, credited).

    Exact rounded pre-casting key, then complete-linkage merge of
    near-duplicates (diameter bounded; no single-linkage chaining).
    """
    from sklearn.cluster import AgglomerativeClustering
    key_cols = ['cem', 'water', 'CA', 'sand', 'SP'] + [f'b{j}' for j in range(1, 7)]
    M = fr[key_cols].to_numpy(float)
    keys = pd.DataFrame(np.round(M, round_dec)).astype(str).agg('|'.join, axis=1)
    n_exact = int(keys.nunique())
    mu, sd = M.mean(0), M.std(0, ddof=0) + 1e-12
    Z = (M - mu) / sd
    cl = AgglomerativeClustering(
        n_clusters=None, distance_threshold=float(near_t),
        linkage='complete').fit(Z)
    fr = fr.copy()
    fr['recipe_group'] = cl.labels_
    fr.attrs['n_exact_recipes'] = n_exact
    return fr


def build_x9(fr):
    """Sensitivity descriptor representation (already transformed upstream)."""
    cols = ['R1_' + c for c in R1_COLS]
    return fr[cols].to_numpy(float), cols


def build_xraw_raw(fr):
    """Primary raw-bin representation BEFORE transforms (for honesty checks)."""
    return fr[XRAW_FIELDS].to_numpy(float), list(XRAW_FIELDS)


def transform_xraw(Xraw):
    """Frozen monotone transforms: log cem; log1p SP/sand/b1..b6; raw w/c.

    Input column order: [w/c, cem, SP, sand, b1..b6].
    """
    Xraw = np.asarray(Xraw, float)
    assert Xraw.shape[1] == 10
    wc = Xraw[:, [0]]
    log_cem = np.log(Xraw[:, [1]])
    rest = np.log1p(Xraw[:, 2:10])
    assert np.isfinite(np.hstack([wc, log_cem, rest])).all()
    return np.hstack([wc, log_cem, rest])


XRAW_TCOLS = (['w/c', 'log_cem', 'log1p_SP', 'log1p_sand']
              + [f'log1p_b{j}' for j in range(1, 7)])


def fit_scaler(Xtr):
    m = Xtr.mean(0)
    s = Xtr.std(0, ddof=0)
    s = np.where(s == 0, 1.0, s)
    return m, s


def apply_scaler(X, m, s):
    return (np.asarray(X, float) - m) / s


def grouped_folds(groups, n_folds=5, seed=0):
    """Seeded size-balanced grouped folds (copy of Stage-16, credited)."""
    rng = np.random.default_rng(int(seed))
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    sizes = {g: int((groups == g).sum()) for g in uniq}
    order = sorted(uniq, key=lambda g: -sizes[g])
    folds = [[] for _ in range(n_folds)]
    loads = [0] * n_folds
    for g in order:
        f = int(np.argmin(loads))
        folds[f].append(g)
        loads[f] += sizes[g]
    rot = int(rng.integers(0, n_folds))
    folds = folds[rot:] + folds[:rot]
    g2f = {g: f for f, gs in enumerate(folds) for g in gs}
    return np.array([g2f[g] for g in groups], dtype=int)


def study_weights(studies):
    studies = np.asarray(studies)
    _, inv, cnt = np.unique(studies, return_inverse=True, return_counts=True)
    w = 1.0 / (len(cnt) * cnt[inv])
    return w / w.sum() * len(w)  # mean weight 1


def phi_grid(phi, step=0.1):
    """Global phi grid: observed range rounded outward (dataset constant)."""
    lo = float(np.floor(np.min(phi) * 10.0) / 10.0)
    hi = float(np.ceil(np.max(phi) * 10.0) / 10.0)
    g = np.arange(lo, hi + 0.5 * step, step)
    return g, lo, hi
