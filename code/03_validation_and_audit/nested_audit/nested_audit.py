"""2026-10-23 nested held-out reliability audit (forward-only, CPU, numpy). Same audit code as the paper's audit
(CalibrationAudit_A3-1_20261005/heldout_porosity_conditional.py + heldout_reliability.py, imported unchanged); only the error bank changes:
for outer group h, the bank rows of every other group j come from the nested fit that LEFT OUT groups {h, j} (Colab out_nested/pair_{h}{j}_side{j}.json),
so no bank entry used to test group h was produced by a model trained on group h.
usage: python nested_audit.py selftest                 (stand-ins = the paper's own near3icm fits; must reproduce the paper's audit exactly)
       python nested_audit.py run <out_nested_dir>     (real nested fits from Colab)"""
import dataclasses, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
AUDIT_CODE = REPO / "code" / "03_validation_and_audit" / "audit_code"
RESULTS = REPO / "results"
LAB = RESULTS / "forward_model_folds" / "lab_effect"
sys.path.insert(0, str(AUDIT_CODE))
import heldout_porosity_conditional as P
H = P.H; M = REPO / "inputs" / "nested_audit" / "base_LNI" / "core" / "models"; H.MODELS = M; H.load_data.__defaults__ = (M,)
HERE = Path(__file__).resolve().parent; MODE = sys.argv[1]


def bank_source(j, h):
    """arrays (orig_row, muS, rawS, muQ, rawQ) for the records of group j, as seen from outer group h"""
    if MODE == "selftest":
        f = LAB / f"labgrp_near3icm_split{j}.json"; d = json.load(open(f)); S, Q = d["y_std_raw_CS"], d["y_std_raw_logQ"]
    else:
        a, b = sorted((h, j)); d = json.load(open(Path(sys.argv[2]) / f"pair_{a}{b}_side{j}.json")); S, Q = d["y_std_icm_CS"], d["y_std_icm_logQ"]
    return np.asarray(d["original_row_index"], int), np.asarray(d["y_pred_CS"]), np.asarray(S), np.asarray(d["y_pred_logQ"]), np.asarray(Q)


d0 = H.load_data(); cfg = H.Config(); targets = H.target_grid(d0, None); b = P._porosity(d0); infl = H.ad_inflation(d0, cfg)
pos = {int(o): i for i, o in enumerate(d0.orig)}; rows, shifts = [], []
for h in sorted(np.unique(d0.split)):
    muS, rawS, muQ, rawQ = d0.muS.copy(), d0.rawS.copy(), d0.muQ.copy(), d0.rawQ.copy()
    for j in sorted(set(np.unique(d0.split)) - {h}):
        o, mS, sS_, mQ, sQ_ = bank_source(j, h); ix = np.array([pos[int(r)] for r in o]); assert (d0.split[ix] == j).all() and len(ix) == (d0.split == j).sum()
        muS[ix], rawS[ix], muQ[ix], rawQ[ix] = mS, sS_, mQ, sQ_
    dh = dataclasses.replace(d0, muS=muS, rawS=rawS, muQ=muQ, rawQ=rawQ)
    te, cal = H.fold_masks(dh, h); sS, sQ = H.scales(dh, cfg, cal, infl); sS0, sQ0 = H.scales(d0, cfg, H.fold_masks(d0, h)[1], infl)
    shifts.append(dict(fold=int(h), bank_rows=int(cal.sum()), bank_rmse_CS=float(np.sqrt(np.mean((dh.yS - dh.muS)[cal] ** 2))), bank_rmse_CS_paper=float(np.sqrt(np.mean((d0.yS - d0.muS)[cal] ** 2))),
                       bank_rmse_lnk=float(np.sqrt(np.mean((dh.yQ - dh.muQ)[cal] ** 2))), bank_rmse_lnk_paper=float(np.sqrt(np.mean((d0.yQ - d0.muQ)[cal] ** 2))),
                       median_scale_ratio_CS=float(np.median(sS[cal] / sS0[cal])), median_scale_ratio_lnk=float(np.median(sQ[cal] / sQ0[cal]))))
    eS_all, eQ_all = (dh.yS - dh.muS) / sS, (dh.yQ - dh.muQ) / sQ
    for k in range(P.N_BINS):
        ti = np.flatnonzero(te & (b == k)); pool = cal & (b == k)
        if len(ti) == 0: continue
        Sreal = dh.muS[ti, None] + sS[ti, None] * eS_all[pool][None, :]; Qreal = dh.muQ[ti, None] + sQ[ti, None] * eQ_all[pool][None, :]
        for tid, s_star, k_star in targets:
            Pp = ((Sreal >= s_star) & (Qreal >= math.log(k_star))).mean(1); O = ((dh.yS[ti] >= s_star) & (dh.yQ[ti] >= math.log(k_star))).astype(float)
            rows.append(pd.DataFrame({"fold": h, "row": ti, "study": dh.study[ti], "porosity_bin": k, "n_pool": int(pool.sum()), "target_id": tid, "S_star": s_star, "K_star": k_star, "P": Pp, "O": O}))
pairs = pd.concat(rows, ignore_index=True)
m = H.metrics(pairs["P"].to_numpy(), pairs["O"].to_numpy()); ci = H.cluster_bootstrap(pairs, 2000, 20260921); acc = pairs[pairs.P >= 0.9]
crit = {"C1a_slope_interval_overlaps_0.8_1.2": bool(ci["slope"]["lo95"] <= 1.2 and ci["slope"]["hi95"] >= 0.8), "C1b_accept_lower5_ge_0.85": bool(ci["accept_rate"]["lower5"] >= 0.85),
        "C1c_over_prediction_le_0.05": bool(m["accept_mean_predicted"] - m["accept_observed_rate"] <= 0.05)}
res = dict(mode=MODE, metrics=m, accept_studies=int(acc.study.nunique()), cluster_bootstrap=ci, criteria=crit, bank_shift_by_fold=shifts)
base = json.load(open(RESULTS / "audit" / "audit_LNI" / "calibration_summary.json"))["base_production"]; res["paper_audit"] = dict(metrics=base["metrics"], cluster_bootstrap=base["cluster_bootstrap"])
if MODE == "selftest":                                                      # control: must equal the paper's audit pair for pair
    ref = pd.read_csv(RESULTS / "audit" / "audit_LNI" / "heldout_pass_probability_pairs_base_production.csv"); j = pairs.merge(ref, on=["row", "target_id"], suffixes=("", "_ref"))
    res["selftest_max_abs_diff_P"] = float((j["P"] - j["P_ref"]).abs().max()); res["selftest_rows_matched"] = int(len(j)); res["selftest_rows_expected"] = int(len(ref))
out = HERE / f"nested_audit_{MODE}.json"; json.dump(res, open(out, "w"), indent=1)
pairs.to_csv(HERE / f"nested_audit_{MODE}_pairs.csv", index=False)
print(json.dumps({k: res[k] for k in res if k not in ("cluster_bootstrap", "paper_audit", "bank_shift_by_fold")}, indent=1))
print("slope CI", {k: round(v, 3) for k, v in ci["slope"].items()}, "| accept lower5", round(ci["accept_rate"]["lower5"], 3))
print("paper : slope", round(base["metrics"]["cal_slope"], 3), "accept n", base["metrics"]["n_accept"], "obs", round(base["metrics"]["accept_observed_rate"], 3), "pred", round(base["metrics"]["accept_mean_predicted"], 3), "lower5", round(base["cluster_bootstrap"]["accept_rate"]["lower5"], 3))
for s in shifts: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
