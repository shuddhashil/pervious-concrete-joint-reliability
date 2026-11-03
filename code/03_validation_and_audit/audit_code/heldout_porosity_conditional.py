"""Held-out calibration audit re-run with porosity-conditional residual sampling.

Wraps final/review_hardening/heldout_reliability.py unchanged, replacing only the
step that turns a held-out row's prediction into a joint-pass probability P: the
standardised residual pairs are drawn from calibration-fold rows in the SAME
porosity tercile as the held-out row (tercile edges = quantiles of the bank's own
porosity, as in template/src/porosity_conditional_residual.py). Coverage of the
max-score screen (criterion C2) is left exactly as in the original.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heldout_reliability as H  # noqa: E402

N_BINS = 3
_PHI = {}


def _porosity(d):
    if "phi" not in _PHI:
        ref = pd.read_csv(H.MODELS / "oos_residual_reference_M1global.csv")
        assert np.array_equal(ref["original_row_index"].to_numpy(int), d.orig)
        phi = ref["xmodel_porosity"].to_numpy(float)
        edges = np.quantile(phi, np.linspace(0.0, 1.0, N_BINS + 1))
        edges[0] -= 1e-9
        _PHI["phi"], _PHI["bin"], _PHI["edges"] = phi, pd.cut(phi, edges, labels=False, include_lowest=True), edges
    return _PHI["bin"]


def crossfit_pairs(d, cfg, targets):
    b = _porosity(d)
    infl = H.ad_inflation(d, cfg)
    rows, cover = [], {p: [] for p in H.LEVELS}
    width = {p: [] for p in H.LEVELS}
    for h in sorted(np.unique(d.split)):
        te, cal = H.fold_masks(d, h)
        sS, sQ = H.scales(d, cfg, cal, infl)
        T = np.maximum((d.muS - d.yS) / sS, (d.muQ - d.yQ) / sQ)
        for p in H.LEVELS:  # unchanged implementation check (C2)
            qv = H.order_stat_q(T[cal], p)
            cover[p].append(float(np.mean(T[te] <= qv)))
            width[p].append(float(np.mean(qv * sS[te])))
        eS_all = (d.yS - d.muS) / sS
        eQ_all = (d.yQ - d.muQ) / sQ
        for k in range(N_BINS):
            ti = np.flatnonzero(te & (b == k))
            pool = cal & (b == k)
            if len(ti) == 0:
                continue
            eS, eQ = eS_all[pool], eQ_all[pool]
            Sreal = d.muS[ti, None] + sS[ti, None] * eS[None, :]
            Qreal = d.muQ[ti, None] + sQ[ti, None] * eQ[None, :]
            for tid, s_star, k_star in targets:
                P = ((Sreal >= s_star) & (Qreal >= math.log(k_star))).mean(1)
                O = ((d.yS[ti] >= s_star) & (d.yQ[ti] >= math.log(k_star))).astype(float)
                rows.append(pd.DataFrame({"fold": h, "row": ti, "study": d.study[ti], "porosity_bin": k,
                                          "n_pool": int(pool.sum()), "target_id": tid,
                                          "S_star": s_star, "K_star": k_star, "P": P, "O": O}))
    summary = {f"coverage_{p:.2f}": float(np.mean(cover[p])) for p in H.LEVELS}
    summary.update({f"mean_interval_halfwidth_strength_{p:.2f}": float(np.mean(width[p])) for p in H.LEVELS})
    return pd.concat(rows, ignore_index=True), summary


H.crossfit_pairs = crossfit_pairs

if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "results" / "A3-1_porosity_conditional"
    rc = H.main(["--out-dir", str(out), *sys.argv[1:]])
    pd.DataFrame({"edge": _PHI["edges"]}).to_csv(out / "porosity_tercile_edges.csv", index=False)
    sys.exit(rc)
