"""Robust alternatives + support-neighbor audit + robustness labels.

- Adds S1813-R (de_seed20260921, 275.19) and S1508-R (de_seed20260916, 210.78):
  min-C pool candidates with all margins above the frozen ambiguity bands.
- Robustness label per shortlisted row: SOLID (all margins > band) else THIN
  (passes exact replay but a margin sits inside the provisional numerical band:
  0.001 MPa CS / 0.0002 logQ / 1e-6 support).
- Nearest-train-neighbor audit in standardized support coordinates + ambiguous
  CA-label study flag (maguesvari2013 x24, andrew2010 x15).
Appends to CANDIDATE_SHORTLIST.csv / POROSITY_FEASIBLE_SETS.csv.
Usage: python src/robust_add.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage1_evaluator as E
import solvers_common as SC
import porosity_sets as PS

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAND_S, BAND_Q, BAND_H = 0.001, 0.0002, 1e-6
AMBIG = {"data source - 22 - maguesvari2013", "data source - 24 - andrew2010"}


def robustness(r):
    thin = (r["margin_S"] < BAND_S) or (r["margin_Q"] < BAND_Q) or ((r["support"] - 0.25) < BAND_H)
    return "THIN" if thin else "SOLID"


def neighbors(cand_rec_frame_row):
    import LCMGP_without_head as L
    b = E.get_bundle()
    obj = b["obj"]
    Xs = cand_rec_frame_row[E.EXPECTED_FEATURES].apply(pd.to_numeric, errors="coerce").copy()
    Xm = L._apply_feature_transforms(Xs)[E.EXPECTED_FEATURES].to_numpy(dtype=float)
    Xss = (Xm - np.asarray(obj["x_cont_mean"])) / np.where(np.asarray(obj["x_cont_std"]) == 0, 1,
                                                          np.asarray(obj["x_cont_std"]))
    T = obj["train_x"].float().numpy()
    d = np.sqrt(((Xss[:, None, :] - T[None, :, :]) ** 2).sum(axis=2))
    return d


if __name__ == "__main__":
    coh = pd.read_csv(r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\materials\data"
                      r"\reconciled_phase2v\phase1_reconciled_eligible.csv")
    specs = [("S1813-R", 18, 13, "traces/de_18_13_p90_s20260921.best.json"),
             ("S1508-R", 15, 8, "traces/de_15_8_p90_s20260916.best.json")]
    new_short, new_por = [], []
    frames = []
    for cid, s, k, fp in specs:
        r = json.load(open(os.path.join(RUN, fp)))
        cand = {kk: (np.asarray(v, dtype=float) if kk == "g" else float(v))
                for kk, v in r["best_cand"].items() if kk in ["C", "r", "A", "P", "S", "phi", "g"]}
        rec = E.evaluate(cand, s, k, p=0.9,
                         bounds={kk: SC.BND[kk] for kk in ["C", "r", "A", "P", "S", "phi"]},
                         backend="exact")
        assert rec["MODEL_SCREEN_FEASIBLE"]
        new_short.append({"cid": cid, "s_star": s, "k_star": k, "kind": "DE-ROBUST",
                          "src": os.path.basename(fp).replace(".best.json", ""),
                          "C": rec["C"], "r": rec["r"], "W": rec["W"], "A_CA": rec["A"],
                          "P": rec["P"], "S": rec["S"], "phi_h": rec["phi"],
                          **{f"g{i+1}": rec["g"][i] for i in range(6)},
                          **{f"A_bin{i+1}": rec["A_bin"][i] for i in range(6)},
                          "WAS": rec["WAS"], "I_gap": rec["I_gap"], "Cu": rec["Cu"], "Aag": rec["Aag"],
                          "mu_S": rec["mu_S"], "mu_logQ": rec["mu_logQ"], "k_pred": rec["k_pred"],
                          "s_S_eff": rec["s_S_eff"], "s_Q_eff": rec["s_Q_eff"],
                          "LB_S": rec["LB_S"], "LB_logQ": rec["LB_logQ"],
                          "margin_S": rec["margin_S"], "margin_Q": rec["margin_Q"],
                          "support": rec["support"], "D_AD": rec["D_AD"], "AD_ratio": rec["AD_ratio"],
                          "feasible": True, "replay_diff": 0.0, "batch_ready": rec["batch_ready"],
                          "robustness": robustness(rec)})
        base = {"C": rec["C"], "r": rec["r"], "A": rec["A"], "P": rec["P"], "S": rec["S"],
                "g": np.array(rec["g"])}
        phis, recs, feas = PS.scan_recipe(base, s, k)
        for (i, j) in PS.components(phis, feas):
            lo = PS.refine_edge(base, s, k, phis[i], phis[i - 1]) if i > 0 else PS.PHI_LO
            hi = PS.refine_edge(base, s, k, phis[j], phis[j + 1]) if j < len(phis) - 1 else PS.PHI_HI
            new_por.append({"cid": cid, "s_star": s, "k_star": k, "comp": 1,
                            "phi_lo": round(lo, 2), "phi_hi": round(hi, 2),
                            "width_pp": round(hi - lo, 2), "recipe_phi": round(float(rec["phi"]), 4),
                            "recipe_inside": bool(lo - 1e-9 <= float(rec["phi"]) <= hi + 1e-9)})
        frames.append(E.candidates_to_frame([{**{kk: cand[kk] for kk in ["C", "r", "A", "P", "S", "phi"]},
                                              "g": cand["g"]}]))
        print(f"{cid}: C={rec['C']:.2f} mS={rec['margin_S']:.3f} mQ={rec['margin_Q']:.3f} "
              f"sup={rec['support']:.3f} set=({new_por[-1]['phi_lo']},{new_por[-1]['phi_hi']})", flush=True)
    # neighbor audit for ALL shortlisted (old + new)
    short = pd.read_csv(os.path.join(RUN, "CANDIDATE_SHORTLIST.csv"))
    short = pd.concat([short, pd.DataFrame(new_short)], ignore_index=True)
    cand_frames = []
    for _, row in short.iterrows():
        cand_frames.append({"porosity": row["phi_h"], "w/c": row["r"], "cem (kg/m3)": row["C"],
                            "CA (kg/m3)": row["A_CA"], "SP (kg/m3)": row["P"], "sand (kg/m3)": row["S"],
                            "WAS": row["WAS"], "I_gap": row["I_gap"], "Cu": row["Cu"],
                            "Aag(mm^-1)": row["Aag"]})
    d = neighbors(pd.DataFrame(cand_frames))
    nn_rows = []
    for i, (_, row) in enumerate(short.iterrows()):
        idx = np.argsort(d[i])[:5]
        studies = [coh["_src_ffill"].iloc[j] for j in idx]
        n_amb = sum(1 for st in studies if st in AMBIG)
        nn_rows.append({"cid": row["cid"], "nn_dist": round(float(d[i][idx[0]]), 3),
                        "nn_row": int(idx[0]), "nn_study": studies[0],
                        "nn5_studies": "|".join(studies), "nn5_ambiguous": n_amb})
        if "robustness" not in short.columns or pd.isna(row.get("robustness", np.nan)):
            short.loc[i, "robustness"] = robustness(row)
    nn = pd.DataFrame(nn_rows)
    short = short.merge(nn, on="cid", how="left")
    short.to_csv(os.path.join(RUN, "CANDIDATE_SHORTLIST.csv"), index=False)
    por = pd.read_csv(os.path.join(RUN, "POROSITY_FEASIBLE_SETS.csv"))
    por = pd.concat([por, pd.DataFrame(new_por)], ignore_index=True)
    por.to_csv(os.path.join(RUN, "POROSITY_FEASIBLE_SETS.csv"), index=False)
    nn.to_csv(os.path.join(RUN, "tables", "neighbor_audit.csv"), index=False)
    print(nn.to_string())
    print("robustness:", short[["cid", "robustness"]].to_string(index=False))
