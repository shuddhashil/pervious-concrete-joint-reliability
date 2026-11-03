"""Deterministic bounded coordinate-poll local refinement (LOCAL_POLL_FALLBACK).

Genuine NOMAD/MADS is not installed in this environment (no pyNomad binary);
this fallback is ALWAYS labelled LOCAL_POLL_FALLBACK, never NOMAD/MADS.
Respects the same simplex projection, bounds, support, and paired screen.
Usage: python src/local_search.py --starts traces/starts_18_13.json --s 18 --k 13 --p 0.9 --out traces/local_18_13_p90
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage1_evaluator as E
import solvers_common as SC

METHOD_LABEL = "LOCAL_POLL_FALLBACK"
DIM = 12


def poll_refine(x0, s_star, k_star, p, bounds, max_evals=800, step0=0.1, step_min=1e-4):
    x = np.clip(np.asarray(x0, dtype=float), 0, 1)
    rec0 = E.evaluate(SC.decode(x), s_star, k_star, p=p, bounds=bounds, backend="fast")
    best_m = E.merit(rec0, SC.C_MAX)
    best_rec = rec0
    n = 1
    step = step0
    crossed = False
    trace = [{"n": n, "merit": best_m,
              "feasible": bool(best_rec["MODEL_SCREEN_FEASIBLE"]), "C": float(best_rec["C"])}]
    feas0 = bool(best_rec["MODEL_SCREEN_FEASIBLE"])
    while n < max_evals and step >= step_min:
        improved = False
        cand_X = []
        for d in range(DIM):
            for sgn in (1.0, -1.0):
                xp = x.copy()
                xp[d] = float(np.clip(xp[d] + sgn * step, 0, 1))
                cand_X.append(xp)
        recs = E.evaluate_batch([SC.decode(xp) for xp in cand_X], s_star, k_star,
                                p=p, bounds=bounds, backend="fast")
        n += len(cand_X)
        for xp, r in zip(cand_X, recs):
            m = E.merit(r, SC.C_MAX)
            if m < best_m - 1e-12:
                best_m, x, best_rec = m, xp.copy(), r
                improved = True
        if bool(best_rec["MODEL_SCREEN_FEASIBLE"]) != feas0:
            crossed = True
        trace.append({"n": n, "merit": best_m,
                      "feasible": bool(best_rec["MODEL_SCREEN_FEASIBLE"]), "C": float(best_rec["C"])})
        if not improved:
            step *= 0.5
    return {"x": [float(v) for v in x], "merit": float(best_m), "rec": best_rec,
            "n_eval": n, "crossed_feasibility": crossed, "method": METHOD_LABEL, "trace": trace}


def run(starts, s_star, k_star, p, out_prefix, max_evals=800):
    bounds = {k: SC.BND[k] for k in ["C", "r", "A", "P", "S", "phi"]}
    outs = []
    for i, x0 in enumerate(starts):
        o = poll_refine(x0, s_star, k_star, p, bounds, max_evals=max_evals)
        o["start_index"] = i
        outs.append(o)
    with open(out_prefix + ".best.json", "w") as f:
        json.dump({"method": METHOD_LABEL, "s_star": s_star, "k_star": k_star, "p": p,
                   "starts": outs}, f, indent=1, default=float)
    return outs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", required=True)
    ap.add_argument("--s", type=float, required=True)
    ap.add_argument("--k", type=float, required=True)
    ap.add_argument("--p", type=float, default=0.9)
    ap.add_argument("--out", required=True)
    ap.add_argument("--maxevals", type=int, default=800)
    a = ap.parse_args()
    with open(a.starts) as f:
        starts = json.load(f)
    outs = run(starts, a.s, a.k, a.p, a.out, max_evals=a.maxevals)
    for o in outs:
        print(f"start {o['start_index']}: merit={o['merit']:.3f} feasible={o['rec']['MODEL_SCREEN_FEASIBLE']} "
              f"C={o['rec']['C']:.2f} evals={o['n_eval']} crossed={o['crossed_feasibility']}")
