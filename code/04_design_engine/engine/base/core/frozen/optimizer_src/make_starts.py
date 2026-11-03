"""Assemble materially distinct local-refinement starts (frozen diversity rule)."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solvers_common as SC

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tv(g1, g2):
    return 0.5 * float(np.sum(np.abs(np.asarray(g1) - np.asarray(g2))))


def distinct(cands, ranges):
    kept = []
    for c in cands:
        ok = True
        for k in kept:
            grad = tv(c["g"], k["g"]) >= 0.05
            scal = any(abs(c[s] - k[s]) / (SC.BND[s][1] - SC.BND[s][0]) >= 0.10
                       for s in ["C", "r", "A", "P", "S", "phi"])
            if not (grad or scal):
                ok = False
                break
        if ok:
            kept.append(c)
    return kept


for tag, s, k in [("18_13", 18, 13), ("15_8", 15, 8), ("12_5", 12, 5)]:
    pool = []
    for f in sorted(glob.glob(os.path.join(RUN, "traces", f"de_{tag}_p90_s*.best.json"))):
        r = json.load(open(f))
        pool.append({"seed": f, "feas": r["best_rec"]["MODEL_SCREEN_FEASIBLE"],
                     "merit": r["best_merit"], "cand": r["best_cand"], "rec": r["best_rec"]})
    feas = sorted([x for x in pool if x["feas"]], key=lambda x: x["rec"]["C"])
    infeas = sorted([x for x in pool if not x["feas"]], key=lambda x: x["merit"])
    try:
        d = json.load(open(os.path.join(RUN, "traces", f"direct_{tag}_p90.best.json")))
        dextra = [{"seed": "direct", "feas": d["best_rec"]["MODEL_SCREEN_FEASIBLE"],
                   "merit": d["best_merit"], "cand": d["best_cand"], "rec": d["best_rec"]}]
    except FileNotFoundError:
        dextra = []
    ordered = feas + dextra + infeas
    ranges = None
    dcands = [{"g": np.asarray(x["cand"]["g"]), **{s2: x["cand"][s2] for s2 in ["C", "r", "A", "P", "S", "phi"]},
               "src": x} for x in ordered]
    kept = distinct(dcands, ranges)[:8]
    xs = [SC.encode({"C": c["C"], "r": c["r"], "A": c["A"], "P": c["P"], "S": c["S"], "phi": c["phi"],
                     "g": c["g"]}).tolist() for c in kept]
    meta = [{"src": os.path.basename(c["src"]["seed"]).replace(".best.json", "")
             if "de_" in c["src"]["seed"] else c["src"]["seed"],
             "feas": c["src"]["feas"], "C": c["src"]["rec"]["C"]} for c in kept]
    with open(os.path.join(RUN, "traces", f"starts_{tag}.json"), "w") as f:
        json.dump(xs, f)
    with open(os.path.join(RUN, "traces", f"starts_{tag}.meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(tag, f"{len(kept)} starts:", meta)
