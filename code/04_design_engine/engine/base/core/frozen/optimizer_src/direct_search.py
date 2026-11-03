"""Bounded DIRECT challenger (scipy.optimize.direct), same domain + merit.

Usage: python src/direct_search.py --s 18 --k 13 --p 0.9 --maxfun 8000 --out traces/direct_18_13_p90
Challenger only; a bounded failure is not infeasibility evidence.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import direct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage1_evaluator as E
import solvers_common as SC

BOUNDS12 = [(0.0, 1.0)] * 12


def run(s_star, k_star, p, maxfun, out_prefix):
    bounds = {k: SC.BND[k] for k in ["C", "r", "A", "P", "S", "phi"]}
    E.get_bundle()
    E.evaluate({"C": 300.0, "r": 0.3, "A": 1500.0, "P": 2.0, "S": 100.0,
                "phi": 20.0, "g": np.full(6, 1 / 6)}, s_star, k_star, p=p)  # warm
    hist = {"n": [], "best": []}
    state = {"best": np.inf, "best_x": None, "n": 0}

    def func(x):
        rec = E.evaluate(SC.decode(np.asarray(x, dtype=float)), s_star, k_star,
                         p=p, bounds=bounds, backend="fast")
        m = E.merit(rec, SC.C_MAX)
        state["n"] += 1
        if m < state["best"]:
            state["best"] = m
            state["best_x"] = np.asarray(x, dtype=float).copy()
            state["best_rec"] = rec
        hist["n"].append(state["n"])
        hist["best"].append(state["best"])
        return m

    t0 = time.time()
    res = direct(func, BOUNDS12, maxfun=maxfun, locally_biased=True)
    dt = time.time() - t0
    best_cand = SC.decode(state["best_x"])
    out = {"s_star": s_star, "k_star": k_star, "p": p, "maxfun": maxfun,
           "n_eval": state["n"], "wall_s": dt, "best_merit": float(state["best"]),
           "scipy_success": bool(res.success), "scipy_nit": int(res.nit),
           "best_cand": {k: (float(v) if k != "g" else [float(x) for x in v])
                         for k, v in best_cand.items()},
           "best_rec": state["best_rec"]}
    with open(out_prefix + ".best.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    np.savez(out_prefix + ".trace.npz", n=np.array(hist["n"]), best=np.array(hist["best"]))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=float, required=True)
    ap.add_argument("--k", type=float, required=True)
    ap.add_argument("--p", type=float, default=0.9)
    ap.add_argument("--maxfun", type=int, default=8000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = run(a.s, a.k, a.p, a.maxfun, a.out)
    print(f"DIRECT {a.s}/{a.k} p={a.p}: merit={r['best_merit']:.3f} "
          f"feasible={r['best_rec']['MODEL_SCREEN_FEASIBLE']} C={r['best_rec']['C']:.2f} "
          f"evals={r['n_eval']} wall={r['wall_s']:.0f}s")
