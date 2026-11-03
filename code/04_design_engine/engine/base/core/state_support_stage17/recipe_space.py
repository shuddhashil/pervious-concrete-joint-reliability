"""Recipe-space interpolation gate (x only, never phi).

A conditional porosity density is meaningless for a recipe far outside the
observed recipe domain. The gate uses kth-nearest-neighbor distance in the
scaled recipe space, calibrated on grouped held-out real recipes (95th
percentile). Outside -> NO_LOCAL_EVIDENCE_FOR_RECIPE, no HDR emitted.
"""

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors


def fit_gate(X_fit, k=10):
    """Store FIT recipe matrix for kth-NN distance queries. x only."""
    nn = NearestNeighbors(n_neighbors=int(k), algorithm='auto')
    nn.fit(np.asarray(X_fit, float))
    return {'nn': nn, 'k': int(k), 'X_fit': np.asarray(X_fit, float)}


def kth_distances(gate, X_q):
    """kth-neighbor distance of each query row to the FIT set."""
    d, _ = gate['nn'].kneighbors(np.asarray(X_q, float))
    return d[:, -1]


def calibrate_threshold(X_sub, groups_sub, k=10, n_splits=5, pct=95.0):
    """Threshold = pct-th percentile of grouped held-out kth-NN distances.

    All inside the FIT-subset: GroupKFold splits by recipe group, so
    held-out distances mimic unseen-group queries. Uses x only.
    """
    X_sub = np.asarray(X_sub, float)
    groups_sub = np.asarray(groups_sub)
    gkf = GroupKFold(n_splits=n_splits)
    pooled = []
    for tri, vai in gkf.split(X_sub, groups=groups_sub):
        nn = NearestNeighbors(n_neighbors=int(k)).fit(X_sub[tri])
        d, _ = nn.kneighbors(X_sub[vai])
        pooled.append(d[:, -1])
    pooled = np.concatenate(pooled)
    return float(np.percentile(pooled, pct)), pooled


def apply_gate(gate, X_q, threshold):
    """Returns (outside: bool array, distances)."""
    d = kth_distances(gate, X_q)
    return d > float(threshold), d
