"""Fixed-recipe porosity feasible sets: dense 1-D scan + bisection refinement.

For each shortlisted recipe (ingredients + gradation fixed), map
P_r = {phi in [phi_min, phi_max]: acceptance passes}. Allows disconnected
components; numerical set estimate, never a certified exhaustive union.
Usage: python src/porosity_sets.py
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage1_evaluator as E

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHI_LO, PHI_HI = 6.2, 38.02
N_SCAN = 2001
RES_PP = 0.01


def scan_recipe(base, s_star, k_star, p=0.9, chunk=256):
    phis = np.linspace(PHI_LO, PHI_HI, N_SCAN)
    recs = []
    for i0 in range(0, len(phis), chunk):
        cs = [{**base, "phi": float(ph)} for ph in phis[i0:i0 + chunk]]
        recs.extend(E.evaluate_batch(cs, s_star, k_star, p=p, bounds=None, backend="exact"))
    feas = np.array([r["MODEL_SCREEN_FEASIBLE"] for r in recs])
    return phis, recs, feas


def refine_edge(base, s_star, k_star, lo_feas, hi_feas, p=0.9):
    """Bisect between a feasible phi (lo_feas) and infeasible phi (hi_feas)."""
    a, b = float(lo_feas), float(hi_feas)
    fa = E.evaluate({**base, "phi": a}, s_star, k_star, p=p, backend="exact")["MODEL_SCREEN_FEASIBLE"]
    assert fa, "refine_edge called with infeasible lower point"
    while (b - a) > RES_PP:
        m = (a + b) / 2
        if E.evaluate({**base, "phi": m}, s_star, k_star, p=p, backend="exact")["MODEL_SCREEN_FEASIBLE"]:
            a = m
        else:
            b = m
    return a


def components(phis, feas):
    comps, i, n = [], 0, len(phis)
    while i < n:
        if feas[i]:
            j = i
            while j + 1 < n and feas[j + 1]:
                j += 1
            comps.append((i, j))
            i = j + 1
        else:
            i += 1
    return comps


if __name__ == "__main__":
    short = pd.read_csv(os.path.join(RUN, "CANDIDATE_SHORTLIST.csv"))
    short = short[short["feasible"]]
    comp_rows, curves = [], {}
    for _, row in short.iterrows():
        cid = row["cid"]
        base = {"C": row["C"], "r": row["r"], "A": row["A_CA"], "P": row["P"],
                "S": row["S"], "g": np.array([row[f"g{i}"] for i in range(1, 7)])}
        phis, recs, feas = scan_recipe(base, row["s_star"], row["k_star"])
        curves[cid] = pd.DataFrame({
            "phi": phis, "feasible": feas,
            "LB_S": [r["LB_S"] for r in recs], "LB_logQ": [r["LB_logQ"] for r in recs],
            "margin_S": [r["margin_S"] for r in recs], "margin_Q": [r["margin_Q"] for r in recs],
            "support": [r["support"] for r in recs]})
        comps = components(phis, feas)
        refined = []
        for (i, j) in comps:
            lo = refine_edge(base, row["s_star"], row["k_star"], phis[i], phis[i - 1]) if i > 0 else PHI_LO
            hi = refine_edge(base, row["s_star"], row["k_star"], phis[j], phis[j + 1]) if j < len(phis) - 1 else PHI_HI
            # note: right edge bisection needs feasible-left orientation; phis[j] feas, phis[j+1] infeas
            refined.append((lo, hi))
            comp_rows.append({"cid": cid, "s_star": row["s_star"], "k_star": row["k_star"],
                              "comp": len(refined), "phi_lo": round(lo, 2), "phi_hi": round(hi, 2),
                              "width_pp": round(hi - lo, 2),
                              "recipe_phi": round(float(row["phi_h"]), 4),
                              "recipe_inside": bool(lo - 1e-9 <= float(row["phi_h"]) <= hi + 1e-9)})
        print(f"{cid}: {len(comps)} component(s): {[(round(a,2), round(b,2)) for a, b in refined]}", flush=True)
    pd.DataFrame(comp_rows).to_csv(os.path.join(RUN, "POROSITY_FEASIBLE_SETS.csv"), index=False)
    for cid, df in curves.items():
        df.to_csv(os.path.join(RUN, "tables", f"porosity_curve_{cid}.csv"), index=False)
    print("wrote POROSITY_FEASIBLE_SETS.csv")
