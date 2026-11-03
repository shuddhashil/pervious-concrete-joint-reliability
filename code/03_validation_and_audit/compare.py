"""Items 1-2 summary (2026-10-29). Predictive accuracy and calibration of MT vs IND on identical folds and adaptation data, and of the three
adaptation protocols, with whole-study (cluster) bootstrap intervals: studies are resampled with replacement, the SAME draw for both arms of a
difference. Reads out_mtind/*_fold*.json and the audit pairs; writes summary.json and prints the tables.
usage: python compare.py <out_mtind> <audit_dir> <legacy_out_lab_dir> <out_json>"""
import sys, os, json, math
import numpy as np, pandas as pd
from scipy.optimize import minimize
FO, AD, LEG, OUTJ = sys.argv[1:5]; NB = 2000; rng0 = 20261029
Z90 = 1.6448536269514722


def load(kind):
    recs = []
    for k in range(5):
        d = json.load(open(os.path.join(FO, f"{kind}_fold{k}.json")))
        for name, pr in d["protocols"].items():
            ev = np.asarray(pr["eval"], int); mu = np.asarray(pr["mu"])[ev]; sd = np.asarray(pr["sd_raw"])[ev]
            recs.append(pd.DataFrame(dict(kind=kind, protocol=name, fold=k, row=np.asarray(d["rows"])[ev], study=np.asarray(d["study"])[ev], yS=np.asarray(d["y_CS"])[ev],
                                          yQ=np.asarray(d["y_lnk"])[ev], muS=mu[:, 0], muQ=mu[:, 1], sS=sd[:, 0], sQ=sd[:, 1])))
    return pd.concat(recs, ignore_index=True)


def per_record(df):
    out = {}
    for o, y, m, s in (("CS", "yS", "muS", "sS"), ("lnk", "yQ", "muQ", "sQ")):
        e = df[y].to_numpy() - df[m].to_numpy(); sd = np.maximum(df[s].to_numpy(), 1e-9); z = e / sd
        out[f"se_{o}"] = e ** 2; out[f"cov90_{o}"] = (np.abs(z) <= Z90).astype(float)
        out[f"nll_{o}"] = 0.5 * np.log(2 * np.pi * sd ** 2) + 0.5 * z ** 2
        from scipy.stats import norm
        out[f"crps_{o}"] = sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
        out[f"y_{o}"] = df[y].to_numpy()
    return pd.DataFrame(out, index=df.index)


def stats(pr):
    r = {}
    for o in ("CS", "lnk"):
        r[f"rmse_{o}"] = float(np.sqrt(pr[f"se_{o}"].mean())); r[f"r2_{o}"] = float(1 - pr[f"se_{o}"].sum() / ((pr[f"y_{o}"] - pr[f"y_{o}"].mean()) ** 2).sum())
        for m in ("cov90", "nll", "crps"): r[f"{m}_{o}"] = float(pr[f"{m}_{o}"].mean())
    return r


def boot_diff(A, B, study):
    """A, B: per-record frames aligned row by row (same records); returns point estimate and 95 % study-bootstrap interval of stats(B) - stats(A)."""
    uq = np.unique(study); idx = {s: np.flatnonzero(study == s) for s in uq}; rng = np.random.default_rng(rng0); point = {k: stats(B)[k] - stats(A)[k] for k in stats(A)}; bs = {k: [] for k in point}
    for _ in range(NB):
        sel = np.concatenate([idx[s] for s in rng.choice(uq, len(uq), replace=True)]); a, b = stats(A.iloc[sel]), stats(B.iloc[sel])
        for k in bs: bs[k].append(b[k] - a[k])
    return {k: dict(diff=point[k], lo95=float(np.percentile(bs[k], 2.5)), hi95=float(np.percentile(bs[k], 97.5))) for k in point}


S = {}
DF = pd.concat([load("MT"), load("IND")], ignore_index=True)
prots = ["newlab", "near3"] + [f"fixed3_{r}" for r in range(5)]
# reproduction check of the paper's MT near-3 fits (refit here vs the 2026-10-21 files)
chk = []
for k in range(5):
    leg = json.load(open(os.path.join(LEG, f"labgrp_near3icm_split{k}.json"))); m = DF[(DF.kind == "MT") & (DF.protocol == "near3") & (DF.fold == k)]
    assert m["row"].tolist() == leg["original_row_index"]; chk.append(max(np.abs(m.muS - leg["y_pred_CS"]).max(), np.abs(m.muQ - leg["y_pred_logQ"]).max()))
S["MT_near3_refit_vs_paper_max_abs_mean_diff"] = float(max(chk))

# 1) MT vs IND under each protocol (identical records)
S["model_comparison"] = {}
for p in prots:
    a = DF[(DF.kind == "MT") & (DF.protocol == p)].sort_values("row"); b = DF[(DF.kind == "IND") & (DF.protocol == p)].sort_values("row"); assert a.row.tolist() == b.row.tolist()
    A, B = per_record(a.reset_index(drop=True)), per_record(b.reset_index(drop=True))
    S["model_comparison"][p] = dict(n_records=len(a), n_studies=int(a.study.nunique()), MT=stats(A), IND=stats(B), IND_minus_MT=boot_diff(A, B, a.study.to_numpy()))
# 2) adaptation protocols on common records (the fixed-set evaluation records of selection r), per model
S["protocol_comparison"] = {}
for kind in ("MT", "IND"):
    S["protocol_comparison"][kind] = []
    for r in range(5):
        fx = DF[(DF.kind == kind) & (DF.protocol == f"fixed3_{r}")].sort_values("row"); rows = fx.row.to_numpy(); ent = dict(selection=r, n_records=len(fx), n_studies=int(fx.study.nunique()))
        F = per_record(fx.reset_index(drop=True))
        for p in ("newlab", "near3"):
            o = DF[(DF.kind == kind) & (DF.protocol == p) & DF.row.isin(rows)].sort_values("row"); assert o.row.tolist() == rows.tolist(); O = per_record(o.reset_index(drop=True))
            ent[p] = stats(O); ent[f"fixed3_minus_{p}"] = boot_diff(O, F, fx.study.to_numpy())
        ent["fixed3"] = stats(F); S["protocol_comparison"][kind].append(ent)


# 3) calibration of joint pass probabilities (audit pairs), differences with study bootstrap
def logit_slope(P, O):
    x = np.log(np.clip(P, 1e-3, 1 - 1e-3) / (1 - np.clip(P, 1e-3, 1 - 1e-3))); A = np.column_stack([np.ones_like(x), x])
    f = lambda b: float(np.sum(np.logaddexp(0.0, A @ b) - O * (A @ b))); g = lambda b: A.T @ (1 / (1 + np.exp(-(A @ b))) - O)
    return float(minimize(f, np.array([0.0, 1.0]), jac=g, method="BFGS").x[1])


def cal(P, O):
    acc = P >= 0.90
    return dict(brier=float(np.mean((P - O) ** 2)), slope=logit_slope(P, O), n_accept=int(acc.sum()), accept_obs=float(O[acc].mean()) if acc.any() else np.nan,
                accept_gap=float(P[acc].mean() - O[acc].mean()) if acc.any() else np.nan)


def cal_diff(da, db, nb=500):
    m = da.merge(db, on=["row", "target_id"], suffixes=("_a", "_b")); assert len(m) == len(da) == len(db)
    st = m["study_a"].to_numpy(); uq = np.unique(st); idx = {s: np.flatnonzero(st == s) for s in uq}; rng = np.random.default_rng(rng0)
    Pa, Pb, O = m.P_a.to_numpy(), m.P_b.to_numpy(), m.O_a.to_numpy(); ca, cb = cal(Pa, O), cal(Pb, O); bs = {k: [] for k in ("brier", "slope", "accept_obs", "accept_gap")}
    for _ in range(nb):
        sel = np.concatenate([idx[s] for s in rng.choice(uq, len(uq), replace=True)]); a, b = cal(Pa[sel], O[sel]), cal(Pb[sel], O[sel])
        for k in bs: bs[k].append(b[k] - a[k])
    return dict(a=ca, b=cb, b_minus_a={k: dict(diff=cb[k] - ca[k], lo95=float(np.nanpercentile(bs[k], 2.5)), hi95=float(np.nanpercentile(bs[k], 97.5))) for k in bs})


rd = lambda n: pd.read_csv(os.path.join(AD, f"pairs_{n}.csv"))
S["calibration"] = {}
for p in ("newlab", "near3"):
    S["calibration"][f"{p}: IND(b) vs MT(a), paired residuals"] = cal_diff(rd(f"MT_{p}_paired"), rd(f"IND_{p}_paired"))
    for kind in ("MT", "IND"):
        S["calibration"][f"{p}: {kind} independent(b) vs paired(a) residuals"] = [cal_diff(rd(f"{kind}_{p}_paired"), rd(f"{kind}_{p}_indep{s}"), nb=200 if s else 500) for s in range(5)]
for r in range(5):
    S["calibration"][f"fixed3_{r}: IND(b) vs MT(a), paired residuals"] = cal_diff(rd(f"MT_fixed3_{r}_paired"), rd(f"IND_fixed3_{r}_paired"))
S["audit"] = json.load(open(os.path.join(AD, "audit_summary.json")))
json.dump(S, open(OUTJ, "w"), indent=1, default=float)

f3 = lambda x: f"{x:.3f}"
print("MT near-3 refit vs paper files, max |mean diff|:", S["MT_near3_refit_vs_paper_max_abs_mean_diff"])
for p, e in S["model_comparison"].items():
    d = e["IND_minus_MT"]; print(f"[{p:9s}] n {e['n_records']:3d} | CS RMSE MT {f3(e['MT']['rmse_CS'])} IND {f3(e['IND']['rmse_CS'])} diff {f3(d['rmse_CS']['diff'])} [{f3(d['rmse_CS']['lo95'])}, {f3(d['rmse_CS']['hi95'])}]"
          f" | lnk RMSE MT {f3(e['MT']['rmse_lnk'])} IND {f3(e['IND']['rmse_lnk'])} diff {f3(d['rmse_lnk']['diff'])} [{f3(d['rmse_lnk']['lo95'])}, {f3(d['rmse_lnk']['hi95'])}]"
          f" | CRPS CS diff {f3(d['crps_CS']['diff'])} lnk {f3(d['crps_lnk']['diff'])}")
for kind in ("MT", "IND"):
    for e in S["protocol_comparison"][kind]:
        print(f"[{kind} sel {e['selection']}] n {e['n_records']} | CS RMSE none {f3(e['newlab']['rmse_CS'])} fixed3 {f3(e['fixed3']['rmse_CS'])} near3 {f3(e['near3']['rmse_CS'])}"
              f" | lnk none {f3(e['newlab']['rmse_lnk'])} fixed3 {f3(e['fixed3']['rmse_lnk'])} near3 {f3(e['near3']['rmse_lnk'])}")
for k, v in S["calibration"].items():
    vv = v if isinstance(v, dict) else v[0]
    print(k, {m: (round(x["diff"], 4), round(x["lo95"], 4), round(x["hi95"], 4)) for m, x in vv["b_minus_a"].items()})
print("COMPARE DONE")
