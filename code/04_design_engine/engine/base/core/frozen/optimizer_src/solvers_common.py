"""Shared solver coordinate mapping (identical for DE / DIRECT / local / tests).

Solver vector x in [0,1]^12:
  x[0..5] : normalized scalars [C, r, A(log-CA), P, S, phi]
  x[6..11]: simplex mass vector m in [0,1]^6 -> g = project_to_simplex(m)

Decode converts to physical raw units; encode inverts (modulo projection
many-to-one). All bounds from SEARCH_CONTRACT.json primary_domain.
"""
import json
import math
import os

import numpy as np

import stage1_evaluator as E

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(RUN, "SEARCH_CONTRACT.json")) as f:
    CONTRACT = json.load(f)

PD = CONTRACT["primary_domain"]
BND = {
    "C": (PD["C"]["lo"], PD["C"]["hi"]),
    "r": (PD["r_wc"]["lo"], PD["r_wc"]["hi"]),
    "A": (PD["A_CA"]["lo"], PD["A_CA"]["hi"]),
    "P": (PD["P_SP"]["lo"], PD["P_SP"]["hi"]),
    "S": (PD["S_sand"]["lo"], PD["S_sand"]["hi"]),
    "phi": (PD["phi_hardened_porosity"]["lo"], PD["phi_hardened_porosity"]["hi"]),
}
KEYS = ["C", "r", "A", "P", "S", "phi"]
C_MAX = BND["C"][1]
LNA_LO, LNA_HI = math.log(BND["A"][0]), math.log(BND["A"][1])


def decode(x):
    x = np.asarray(x, dtype=float)
    v = {}
    for i, k in enumerate(KEYS):
        u = float(np.clip(x[i], 0.0, 1.0))
        lo, hi = BND[k]
        if k == "A":
            v[k] = float(np.exp(LNA_LO + u * (LNA_HI - LNA_LO)))
        else:
            v[k] = float(lo + u * (hi - lo))
    g = E.project_to_simplex(np.clip(x[6:12], 0.0, 1.0))
    return {"C": v["C"], "r": v["r"], "A": v["A"], "P": v["P"],
            "S": v["S"], "phi": v["phi"], "g": g}


def encode(cand):
    x = np.zeros(12)
    for i, k in enumerate(KEYS):
        lo, hi = BND[k]
        if k == "A":
            x[i] = (math.log(float(cand["A"])) - LNA_LO) / (LNA_HI - LNA_LO)
        else:
            x[i] = (float(cand[k]) - lo) / (hi - lo)
    x[6:12] = np.asarray(cand["g"], dtype=float)
    return np.clip(x, 0.0, 1.0)


def decode_batch(X):
    return [decode(x) for x in np.asarray(X, dtype=float)]


def merit_batch(recs):
    return np.array([E.merit(r, C_MAX) for r in recs], dtype=float)
