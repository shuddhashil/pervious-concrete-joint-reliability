"""Data contract for Stage14 porosity-mediated GP experiment.

Reuses the authoritative LCM-GP module for ALL transforms/kernels/scalers logic
so preprocessing, target transforms and scaling rules are identical to the
accepted M1global refit. This module never touches test rows for fitting.
"""
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_AUTHORITATIVE_MODULE = os.path.join(_REPO, "code", "02_forward_model")
if _AUTHORITATIVE_MODULE not in sys.path:
    sys.path.insert(0, _AUTHORITATIVE_MODULE)
import LCMGP_without_head as L  # noqa: E402

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XLSX = os.path.join(_REPO, "data", "data_with_passing_sizes.xlsx")
SPLITS_PATH = os.path.join(_REPO, "data", "splits_A3-1_remapped.json")

FEAT10 = ['porosity', 'w/c', 'cem (kg/m3)', 'CA (kg/m3)', 'SP (kg/m3)',
          'sand (kg/m3)', 'WAS', 'I_gap', 'Cu', 'Aag(mm^-1)']
FEAT_X9 = ['w/c', 'cem (kg/m3)', 'CA (kg/m3)', 'SP (kg/m3)', 'sand (kg/m3)',
           'WAS', 'I_gap', 'Cu', 'Aag(mm^-1)']
PHI_IDX_10 = 0
# Position of each x9 feature inside the 10D performance input.
X9_TO_10 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
CFG = {'kernel': 'rbf+rq', 'rank': 1, 'use_additive': True, 'add_kernel': 'rbf'}


def apply_authoritative_contract():
    """Pin the authoritative module to the M1global 10D contract (idempotent)."""
    L.FEATURES = list(FEAT10)
    L.CONT_IDXS = list(range(10))
    L.TRANSFORM_RECIPE = {
        'log_cols': ['cem (kg/m3)', 'CA (kg/m3)', 'Cu'],
        'log1p_cols': ['SP (kg/m3)', 'sand (kg/m3)', 'Aag(mm^-1)'],
        'raw_cols': ['porosity', 'w/c', 'WAS', 'I_gap'],
        'categorical_cols': [],
    }


def load_authoritative():
    """Load transformed 10D features + raw-scale targets + metadata.

    Returns dict with X10 (transformed, unstandardized), phi_raw, yS, yP,
    source_excel_row, mix_id. Row order/identities are the authoritative ones.
    """
    apply_authoritative_contract()
    out = L.load_and_prepare_data(XLSX, return_metadata=True)
    X, yS, yP, n0, info = out
    assert X.shape[1] == 10, X.shape
    assert len(X) == 650, len(X)
    meta = info['row_metadata']
    phi_raw = X[:, PHI_IDX_10].copy()  # NOTE: transformed; porosity transform is identity (raw)
    return {
        'X10': X, 'phi': X[:, PHI_IDX_10].copy(), 'phi_raw_check': phi_raw,
        'yS': yS, 'yP': yP, 'n0': n0,
        'prepared_row_index': meta['prepared_row_index'].to_numpy(),
        'source_excel_row': meta['source_excel_row'].to_numpy(),
        'mix_id': meta['mix_id'].to_numpy(),
    }


def load_splits():
    sp = json.load(open(SPLITS_PATH))
    splits = []
    for i in range(5):
        d = sp[f'split_{i}']
        splits.append({
            'split': i,
            'seed': d.get('seed'),
            'train_idx': np.asarray(d['train_idx'], dtype=int),
            'test_idx': np.asarray(d['test_idx'], dtype=int),
        })
    return splits


def fit_standardizers(X10_tr, Y_tr):
    """Fit train-only standardizers. Same formulas as XStandardizer/YStandardizer
    (per-column mean, population std, zero-std -> 1). Returns dict of params."""
    m_x = X10_tr.mean(0)
    s_x = X10_tr.std(0, ddof=0)
    s_x = np.where(s_x == 0, 1.0, s_x)
    m_y = Y_tr.mean(0)
    s_y = Y_tr.std(0, ddof=0)
    s_y = np.where(s_y == 0, 1.0, s_y)
    return {'x_mean': m_x, 'x_std': s_x, 'y_mean': m_y, 'y_std': s_y}


def standardize_X(X10, sc):
    return (X10 - sc['x_mean']) / sc['x_std']


def standardize_Y(Y, sc):
    return (Y - sc['y_mean']) / sc['y_std']


def unstandardize_Y(Ys, sc):
    return Ys * sc['y_std'] + sc['y_mean']
