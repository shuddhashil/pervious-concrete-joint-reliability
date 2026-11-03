"""11-DOF codec: 6 normalized scalars + 5 stick-breaking gradation variables.

decode(x): x[0..5] scalars as legacy; x[6..10] sticks u in [0,1] mapped in the
configured bin ORDER: g_(1) = u1; R1 = 1-u1; g_(2) = u2*R1; ... g_(6) = R5.
Faces/vertices exact (u in {0,1}); simplex exact up to 1e-15 renormalization.
encode(cand): invert sticks along the same ORDER (0/0 remainders -> 0.0).
Two orders tested: 'coarse_first' (bins 0..5) and 'fine_first' (bins 5..0).
"""
import math

import numpy as np

import shim  # noqa: F401
import solvers_common as SC

KEYS = SC.KEYS
BND = SC.BND
LNA_LO, LNA_HI = SC.LNA_LO, SC.LNA_HI


def _order(name):
    return [0, 1, 2, 3, 4, 5] if name == "coarse_first" else [5, 4, 3, 2, 1, 0]


def sticks_to_g(u, order="coarse_first"):
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    assert u.shape == (5,)
    seq = _order(order)
    g = np.zeros(6)
    rem = 1.0
    for k in range(5):
        g[seq[k]] = u[k] * rem
        rem *= (1.0 - u[k])
    g[seq[5]] = rem
    s = g.sum()
    assert abs(s - 1.0) < 1e-9
    return g / s


def g_to_sticks(g, order="coarse_first"):
    g = np.asarray(g, dtype=float)
    assert np.all(g >= -1e-12) and abs(g.sum() - 1.0) < 1e-9
    g = np.clip(g, 0.0, 1.0)
    g = g / g.sum()
    seq = _order(order)
    u = np.zeros(5)
    rem = 1.0
    for k in range(5):
        u[k] = g[seq[k]] / rem if rem > 1e-15 else 0.0
        rem -= g[seq[k]]
        rem = max(rem, 0.0)
    return np.clip(u, 0.0, 1.0)


def decode11(x, order="coarse_first"):
    x = np.asarray(x, dtype=float)
    assert x.shape == (11,)
    v = {}
    for i, k in enumerate(KEYS):
        u = float(np.clip(x[i], 0.0, 1.0))
        lo, hi = BND[k]
        v[k] = float(np.exp(LNA_LO + u * (LNA_HI - LNA_LO))) if k == "A" else float(lo + u * (hi - lo))
    v["g"] = sticks_to_g(x[6:11], order)
    return v


def encode11(cand, order="coarse_first"):
    x = np.zeros(11)
    for i, k in enumerate(KEYS):
        lo, hi = BND[k]
        x[i] = ((math.log(float(cand["A"])) - LNA_LO) / (LNA_HI - LNA_LO) if k == "A"
                else (float(cand[k]) - lo) / (hi - lo))
    x[6:11] = g_to_sticks(np.asarray(cand["g"]), order)
    return np.clip(x, 0.0, 1.0)


def decode_legacy(x):
    return SC.decode(x)
