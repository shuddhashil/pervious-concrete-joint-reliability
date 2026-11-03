"""Phase E: complete 5% six-bin lattice accounting (53,130 states).

Tier 1: 4 frozen anchors x all states (state accounting, NOT per-state scalar
optimization). Tier 2: <=128 promising states, scalar-only refinement with
gradation fixed (400 evals/state).
Usage: python src/lattice_audit.py
"""
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage1_evaluator as E
import solvers_common as SC
from lattice import enumerate_lattice, lattice_count

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOUNDS = {k: SC.BND[k] for k in ["C", "r", "A", "P", "S", "phi"]}


def load_anchors():
    with open(os.path.join(RUN, "traces", "local_18_13_p90.best.json")) as f:
        l18 = json.load(f)
    with open(os.path.join(RUN, "traces", "local_15_8_p90.best.json")) as f:
        l15 = json.load(f)
    # 12/5: all C=200 ties -> most robust = max min(normalized margins)
    best12, best12key = None, -np.inf
    for fp in sorted(os.listdir(os.path.join(RUN, "traces"))):
        if fp.startswith("de_12_5_p90_s") and fp.endswith(".best.json"):
            r = json.load(open(os.path.join(RUN, "traces", fp)))
            b = r["best_rec"]
            key = min(b["margin_S"] / 12.0, b["margin_Q"] / abs(math.log(5.0)),
                      b["support"] - 0.25)
            if key > best12key:
                best12key, best12 = key, r
    c18 = l18["starts"][0]["rec"]
    c15 = l15["starts"][0]["rec"]
    anchors = [
        {"id": "A18", "s": 18, "k": 13, "p": 0.9,
         "scal": {v: c18[v] for v in ["C", "r", "A", "P", "S", "phi"]}},
        {"id": "A15", "s": 15, "k": 8, "p": 0.9,
         "scal": {v: c15[v] for v in ["C", "r", "A", "P", "S", "phi"]}},
        {"id": "A12", "s": 12, "k": 5, "p": 0.9,
         "scal": {v: best12["best_rec"][v] for v in ["C", "r", "A", "P", "S", "phi"]}},
        {"id": "CTR", "s": 12, "k": 5, "p": 0.9,  # deterministic domain-center probe
         "scal": {"C": 450.07, "r": 0.30458031795, "A": float(np.exp((math.log(1041.32) + math.log(1800.0)) / 2)),
                  "P": 8.5, "S": 167.445, "phi": 22.11}},
    ]
    with open(os.path.join(RUN, "lattice", "anchors.json"), "w") as f:
        json.dump(anchors, f, indent=1)
    return anchors


def tier1(anchors, batch=512):
    assert lattice_count() == 53130
    states = enumerate_lattice()
    assert len(states) == 53130
    assert len({tuple(np.round(g, 10)) for g in states}) == 53130
    E.get_bundle()
    rows = []
    t0 = time.time()
    for a in anchors:
        n_sup, n_feas = 0, 0
        for i0 in range(0, len(states), batch):
            chunk = states[i0:i0 + batch]
            cands = [{**a["scal"], "g": g} for g in chunk]
            recs = E.evaluate_batch(cands, a["s"], a["k"], p=a["p"], bounds=BOUNDS, backend="fast")
            for j, r in enumerate(recs):
                rows.append({"state": i0 + j, "anchor": a["id"], "s": a["s"], "k": a["k"],
                             "feasible": bool(r["MODEL_SCREEN_FEASIBLE"]), "C": r["C"],
                             "support": r["support"], "margin_S": r["margin_S"],
                             "margin_Q": r["margin_Q"],
                             "viol": 0.0 if r["MODEL_SCREEN_FEASIBLE"] else float(
                                 max(0.0, -r["margin_S"]) / a["s"] + max(0.0, -r["margin_Q"]) / abs(math.log(a["k"]))
                                 + max(0.0, 0.25 - r["support"]) / 0.25)})
                n_sup += r["support"] >= 0.25
                n_feas += r["MODEL_SCREEN_FEASIBLE"]
        print(f"anchor {a['id']} ({a['s']}/{a['k']}): support-valid={n_sup} feasible={n_feas} t={time.time()-t0:.0f}s",
              flush=True)
    led = pd.DataFrame(rows)
    led.to_csv(os.path.join(RUN, "lattice", "tier1_ledger.csv"), index=False)
    return led


def scalar_refine(g_fixed, scal0, s_star, k_star, p, seed, iters=20, pop=20):
    rng = np.random.default_rng(seed)
    x0 = SC.encode({**scal0, "g": g_fixed})[:6]
    X = np.clip(x0 + rng.normal(0, 0.1, (pop, 6)), 0, 1)
    X[0] = x0
    best_m, best_x = np.inf, x0.copy()
    n = 0
    for _ in range(iters):
        cands = []
        for i in range(pop):
            d = {}
            for j, k in enumerate(SC.KEYS):
                u = float(np.clip(X[i, j], 0, 1))
                lo, hi = SC.BND[k]
                d[k] = float(np.exp(SC.LNA_LO + u * (SC.LNA_HI - SC.LNA_LO))) if k == "A" \
                    else float(lo + u * (hi - lo))
            d["g"] = g_fixed
            cands.append(d)
        recs = E.evaluate_batch(cands, s_star, k_star, p=p, bounds=BOUNDS, backend="fast")
        n += pop
        mer = SC.merit_batch(recs)
        idx = int(np.argmin(mer))
        if mer[idx] < best_m:
            best_m, best_x, best_rec = float(mer[idx]), X[idx].copy(), recs[idx]
        # mutate around best
        X = np.clip(best_x + rng.normal(0, 0.05, (pop, 6)), 0, 1)
        X[0] = best_x
    return {"merit": best_m, "rec": best_rec, "n_eval": n}


def tier2(led):
    # feasible-anchor states + support-valid close challengers (viol <= 0.15)
    feas_states = led[led["feasible"]].groupby(["state", "s", "k"]).agg(
        C=("C", "min")).reset_index().sort_values("C")
    chall = led[(~led["feasible"]) & (led["support"] >= 0.25) & (led["viol"] <= 0.15)]
    chall_states = chall.groupby(["state", "s", "k"]).agg(viol=("viol", "min")).reset_index(
        ).sort_values("viol")
    states = enumerate_lattice()
    picked, seen = [], set()
    for _, r in feas_states.iterrows():
        key = (int(r["state"]), r["s"], r["k"])
        if key not in seen:
            picked.append({"state": int(r["state"]), "s": r["s"], "k": r["k"], "kind": "feasible-anchor"})
            seen.add(key)
    for _, r in chall_states.iterrows():
        if len(picked) >= 128:
            break
        key = (int(r["state"]), r["s"], r["k"])
        if key not in seen:
            picked.append({"state": int(r["state"]), "s": r["s"], "k": r["k"], "kind": "close-challenger"})
            seen.add(key)
    picked = picked[:128]
    print(f"tier2: {len(picked)} states ({sum(1 for x in picked if x['kind']=='feasible-anchor')} feasible-anchor)",
          flush=True)
    anchors = {a["id"]: a for a in json.load(open(os.path.join(RUN, "lattice", "anchors.json")))}
    id2anchor = {"A18": (18, 13), "A15": (15, 8), "A12": (12, 5), "CTR": (12, 5)}
    results = []
    for t, p_ in enumerate(picked):
        g = states[p_["state"]]
        sub = led[(led["state"] == p_["state"]) & (led["s"] == p_["s"]) & (led["k"] == p_["k"])]
        arow = sub.iloc[int(np.argmax(sub["feasible"].to_numpy())) if sub["feasible"].any()
                        else int(np.argmin(sub["viol"].to_numpy()))]
        a = anchors[arow["anchor"]]
        o = scalar_refine(g, a["scal"], p_["s"], p_["k"], 0.9, seed=9000 + t)
        assert np.allclose(o["rec"]["g"], g, atol=1e-12)  # gradation fixed
        results.append({"state": p_["state"], "s": p_["s"], "k": p_["k"], "kind": p_["kind"],
                        "anchor": arow["anchor"], "feasible": bool(o["rec"]["MODEL_SCREEN_FEASIBLE"]),
                        "C": float(o["rec"]["C"]), "support": float(o["rec"]["support"]),
                        "margin_S": float(o["rec"]["margin_S"]), "margin_Q": float(o["rec"]["margin_Q"]),
                        "n_eval": o["n_eval"]})
        if (t + 1) % 16 == 0:
            print(f"  tier2 {t+1}/{len(picked)}", flush=True)
    pd.DataFrame(results).to_csv(os.path.join(RUN, "lattice", "tier2_results.csv"), index=False)
    return pd.DataFrame(results)


if __name__ == "__main__":
    anchors = load_anchors()
    print("anchors:", [(a["id"], a["s"], a["k"]) for a in anchors], flush=True)
    led = tier1(anchors)
    print(f"tier1 rows={len(led)} states=53130 x {len(anchors)} anchors", flush=True)
    print(f"support-valid anchors: {(led['support']>=0.25).sum()} feasible anchors: {led['feasible'].sum()}", flush=True)
    t2 = tier2(led)
    print(f"tier2 refined={len(t2)} feasible={t2['feasible'].sum()}", flush=True)
    if t2["feasible"].any():
        for (s, k), grp in t2[t2["feasible"]].groupby(["s", "k"]):
            print(f"  best refined lattice {s}/{k}: C={grp['C'].min():.2f}", flush=True)
