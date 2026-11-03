"""Solver comparison tables + stability assessment (frozen tolerance rule)."""
import glob
import json
import os

import numpy as np
import pandas as pd

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

rows, seed_rows = [], []
for tag, s, k in [("18_13", 18, 13), ("15_8", 15, 8), ("12_5", 12, 5)]:
    de = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RUN, "traces", f"de_{tag}_p90_s*.best.json")))]
    d = json.load(open(os.path.join(RUN, "traces", f"direct_{tag}_p90.best.json")))
    l = json.load(open(os.path.join(RUN, "traces", f"local_{tag}_p90.best.json")))
    for r in de:
        b = r["best_rec"]
        seed_rows.append({"target": f"{s}/{k}", "solver": "DE", "seed": r["seed"],
                          "feasible": bool(b["MODEL_SCREEN_FEASIBLE"]), "C": round(b["C"], 2),
                          "support": round(b["support"], 3), "margin_S": round(b["margin_S"], 4),
                          "margin_Q": round(b["margin_Q"], 4), "n_eval": r["n_eval"]})
    de_feas = sorted([r["best_rec"]["C"] for r in de if r["best_rec"]["MODEL_SCREEN_FEASIBLE"]])
    de_best = min(de_feas) if de_feas else None
    loc_best = min([o["rec"]["C"] for o in l["starts"] if o["rec"]["MODEL_SCREEN_FEASIBLE"]], default=None)
    best = None
    for v in [de_best, loc_best]:
        if v is not None and (best is None or v < best):
            best = v
    tol = max(1.0, 0.005 * best) if best else None
    n_within = sum(1 for r in de if r["best_rec"]["MODEL_SCREEN_FEASIBLE"]
                   and r["best_rec"]["C"] <= best + tol + 1e-9) if best else 0
    # final-quarter improvement of the best seed's trace
    fq = None
    if de_best is not None:
        r0 = min([r for r in de if r["best_rec"]["MODEL_SCREEN_FEASIBLE"]],
                 key=lambda r: r["best_rec"]["C"])
        t = pd.read_csv(os.path.join(RUN, "traces", f"de_{tag}_p90_s{r0['seed']}.trace.csv"))
        q = len(t) // 4
        at_75 = t["best_merit"].iloc[3 * q - 1]
        at_100 = t["best_merit"].iloc[-1]
        fq = float(at_75 - at_100) if np.isfinite(at_75) else None
    stable = bool(best is not None and n_within >= 6 and (fq is not None and fq < tol))
    rows.append({"target": f"{s}/{k}", "DE_seeds": len(de),
                 "DE_feasible": sum(1 for r in de if r["best_rec"]["MODEL_SCREEN_FEASIBLE"]),
                 "DE_best_C": de_best, "DIRECT_feasible": bool(d["best_rec"]["MODEL_SCREEN_FEASIBLE"]),
                 "DIRECT_best": round(d["best_merit"], 3), "DIRECT_C": round(d["best_rec"]["C"], 2),
                 "LOCAL_best_C": loc_best, "LOCAL_improved": bool(loc_best is not None and de_best is not None
                                                                   and loc_best < de_best - 1e-9),
                 "best_C": best, "tol": round(tol, 3) if tol else None,
                 "seeds_within_tol": n_within, "finalQ_gain": round(fq, 3) if fq is not None else None,
                 "stable": stable})
    print(f"{s}/{k}: DE {sum(1 for r in de if r['best_rec']['MODEL_SCREEN_FEASIBLE'])}/{len(de)} feas, "
          f"best={best}, DIRECT_feas={bool(d['best_rec']['MODEL_SCREEN_FEASIBLE'])}, "
          f"LOCAL_best={loc_best}, within_tol={n_within}, finalQ={fq}, STABLE={stable}")
pd.DataFrame(rows).to_csv(os.path.join(RUN, "SOLVER_COMPARISON.csv"), index=False)
pd.DataFrame(seed_rows).to_csv(os.path.join(RUN, "tables", "de_seedwise.csv"), index=False)
