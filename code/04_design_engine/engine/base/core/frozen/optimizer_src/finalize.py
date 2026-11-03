"""Finalize: exact replay, shortlist, porosity sets, solver comparison, tables."""
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage1_evaluator as E
import solvers_common as SC

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOUNDS = {k: SC.BND[k] for k in ["C", "r", "A", "P", "S", "phi"]}
RNG = {"C": 500.14, "r": 0.2104291063, "A": 758.68, "P": 17.0, "S": 334.89, "phi": 31.82}


def tv(g1, g2):
    return 0.5 * float(np.sum(np.abs(np.asarray(g1, dtype=float) - np.asarray(g2, dtype=float))))


def load_de(tag):
    out = []
    for f in sorted(glob.glob(os.path.join(RUN, "traces", f"de_{tag}_p90_s*.best.json"))):
        r = json.load(open(f))
        out.append(r)
    return out


def gather(s, k, tag):
    cands = []
    for r in load_de(tag):
        cands.append({"src": f"de_seed{r['seed']}", "cand": r["best_cand"],
                      "fast": r["best_rec"], "seed": r["seed"], "kind": "DE"})
    lf = os.path.join(RUN, "traces", f"local_{tag}_p90.best.json")
    if os.path.exists(lf):
        l = json.load(open(lf))
        for o in l["starts"]:
            cands.append({"src": f"local#{o['start_index']}", "cand": o["x"] and SC.decode(np.array(o["x"])),
                          "fast": o["rec"], "seed": -1, "kind": "LOCAL_POLL_FALLBACK"})
    df_ = os.path.join(RUN, "traces", f"direct_{tag}_p90.best.json")
    if os.path.exists(df_):
        d = json.load(open(df_))
        cands.append({"src": "direct", "cand": d["best_cand"], "fast": d["best_rec"],
                      "seed": -1, "kind": "DIRECT"})
    return cands


def replay(cands, s, k, p=0.9):
    recs = E.evaluate_batch(
        [{kk: (np.asarray(v, dtype=float) if kk == "g" else float(v)) for kk, v in c["cand"].items()
          if kk in ["C", "r", "A", "P", "S", "phi", "g"]} for c in cands],
        s, k, p=p, bounds=BOUNDS, backend="exact")
    for c, r in zip(cands, recs):
        c["exact"] = r
        f = c["fast"]
        c["replay_diff"] = max(abs(f["mu_S"] - r["mu_S"]), abs(f["mu_logQ"] - r["mu_logQ"]),
                               abs(f["support"] - r["support"]))
    return cands


def shortlist_for(s, k, tag, cid_prefix):
    pool = replay(gather(s, k, tag), s, k)
    feas = sorted([c for c in pool if c["exact"]["MODEL_SCREEN_FEASIBLE"]],
                  key=lambda c: c["exact"]["C"])
    infeas = sorted([c for c in pool if not c["exact"]["MODEL_SCREEN_FEASIBLE"]],
                    key=lambda c: E.merit(c["exact"], SC.C_MAX))
    best_C = feas[0]["exact"]["C"] if feas else None
    window = max(5.0, 0.01 * best_C) if best_C else None
    short, notes = [], []
    def is_distinct(c, pool_):
        for q in pool_:
            grad = tv(c["cand"]["g"], q["cand"]["g"]) >= 0.05
            scal = any(abs(c["cand"][t] - q["cand"][t]) / RNG[t] >= 0.10
                       for t in ["C", "r", "A", "P", "S", "phi"])
            if not (grad or scal):
                return False
        return True
    if feas:
        for c in feas:
            if c["exact"]["C"] <= best_C + window + 1e-9:
                if is_distinct(c, short):
                    short.append(c)
                else:
                    notes.append(f"{c['src']} same-basin as {[q['src'] for q in short]}")
        # ensure a distinct-basin alternative is represented if one exists
        if len(short) < 2:
            for c in feas:
                if c not in short and is_distinct(c, short):
                    short.append(c)
                    extra = " (outside near-best window)" if c["exact"]["C"] > best_C + window + 1e-9 else ""
                    notes.append(f"{c['src']} added as distinct-basin alternative{extra}")
                    break
    near = infeas[:1]
    return short, near, pool, notes, best_C, window


if __name__ == "__main__":
    all_short, all_near, summ = [], [], []
    for (s, k, tag, pre) in [(18, 13, "18_13", "S1813"), (15, 8, "15_8", "S1508"), (12, 5, "12_5", "S1205")]:
        short, near, pool, notes, best_C, window = shortlist_for(s, k, tag, pre)
        for i, c in enumerate(short):
            c["cid"] = f"{pre}-{i+1:02d}"
            all_short.append((s, k, c))
        for c in near:
            c["cid"] = f"{pre}-NM"
            all_near.append((s, k, c))
        summ.append({"target": f"{s}/{k}", "n_pool": len(pool),
                     "n_feas_pool": sum(1 for c in pool if c["exact"]["MODEL_SCREEN_FEASIBLE"]),
                     "best_C": best_C, "window": window, "n_short": len(short), "notes": "; ".join(notes)})
        print(f"{s}/{k}: pool={len(pool)} feas={sum(1 for c in pool if c['exact']['MODEL_SCREEN_FEASIBLE'])} "
              f"best={best_C} short={[c['cid'] for c in short]}")
        for c in short:
            print(f"  {c['cid']} C={c['exact']['C']:.2f} r={c['exact']['r']:.4f} A={c['exact']['A']:.1f} "
                  f"P={c['exact']['P']:.2f} S={c['exact']['S']:.1f} phi={c['exact']['phi']:.2f} "
                  f"g={np.round(c['exact']['g'],3).tolist()} replay_diff={c['replay_diff']:.2e}")
    rows = []
    for (s, k, c) in all_short + all_near:
        r = c["exact"]
        rows.append({"cid": c["cid"], "s_star": s, "k_star": k, "kind": c["kind"], "src": c["src"],
                     "C": r["C"], "r": r["r"], "W": r["W"], "A_CA": r["A"], "P": r["P"], "S": r["S"],
                     "phi_h": r["phi"], **{f"g{i+1}": r["g"][i] for i in range(6)},
                     **{f"A_bin{i+1}": r["A_bin"][i] for i in range(6)},
                     "WAS": r["WAS"], "I_gap": r["I_gap"], "Cu": r["Cu"], "Aag": r["Aag"],
                     "mu_S": r["mu_S"], "mu_logQ": r["mu_logQ"], "k_pred": r["k_pred"],
                     "s_S_eff": r["s_S_eff"], "s_Q_eff": r["s_Q_eff"],
                     "LB_S": r["LB_S"], "LB_logQ": r["LB_logQ"],
                     "margin_S": r["margin_S"], "margin_Q": r["margin_Q"],
                     "support": r["support"], "D_AD": r["D_AD"], "AD_ratio": r["AD_ratio"],
                     "feasible": bool(r["MODEL_SCREEN_FEASIBLE"]),
                     "replay_diff": c["replay_diff"], "batch_ready": r["batch_ready"]})
    pd.DataFrame(rows).to_csv(os.path.join(RUN, "CANDIDATE_SHORTLIST.csv"), index=False)
    pd.DataFrame(summ).to_csv(os.path.join(RUN, "tables", "shortlist_summary.csv"), index=False)
    print("wrote CANDIDATE_SHORTLIST.csv with", len(rows), "rows")
