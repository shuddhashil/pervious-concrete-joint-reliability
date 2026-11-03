"""Collect split-fit JSONs into comparison tables + hyperparameter inspection."""
import json, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
RUN = r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\research\Stage1_CA_Forward_Refit_20260913\run_01"
SF = os.path.join(RUN, "models", "splitfits"); TAB = os.path.join(RUN, "tables")
os.makedirs(TAB, exist_ok=True)
rec = []
for c in ["M0", "M1global", "M1agg"]:
    for s in range(5):
        d = json.load(open(os.path.join(SF, f"{c}_split{s}.json")))
        m = d["metrics"]
        rec.append(dict(config=c, split=s, rmseCS=m["rmse_strength"], r2CS=m["r2_strength"],
                        maeCS=m.get("mae_strength", np.nan), rmseQ=m["rmse_logperm"],
                        r2Q=m["r2_logperm"], maeQ=m.get("mae_logperm", np.nan),
                        nlpdCS=m["nlpd_strength_raw"], nlpdQ=m["nlpd_logperm_raw"],
                        cov90CS=m["cov90_strength_raw_gaussian"], cov90Q=m["cov90_logperm_raw_gaussian"],
                        minutes=d["minutes"]))
df = pd.DataFrame(rec)
df.to_csv(os.path.join(TAB, "lcmgp_split_metrics.csv"), index=False)
print(df.groupby("config")[["rmseCS", "r2CS", "maeCS", "rmseQ", "r2Q", "maeQ", "nlpdCS", "nlpdQ"]].agg(["mean", "std"]).round(4).to_string())
piv = df.pivot_table(index="split", columns="config", values=["rmseCS", "rmseQ", "r2CS", "r2Q"])
for cfg in ["M1global", "M1agg"]:
    for met in ["rmseCS", "rmseQ", "r2CS", "r2Q"]:
        d = piv[met][cfg] - piv[met]["M0"]
        print(f"d {cfg}-M0 {met}:", np.round(d.values, 4), "mean", round(float(d.mean()), 4))
# hyperparameters: CA lengthscales + task structure
FEAT10 = ['porosity', 'w/c', 'cem (kg/m3)', 'CA (kg/m3)', 'SP (kg/m3)', 'sand (kg/m3)', 'WAS', 'I_gap', 'Cu', 'Aag(mm^-1)']
for c in ["M0", "M1global", "M1agg"]:
    print("=" * 30, c)
    for s in range(5):
        d = json.load(open(os.path.join(SF, f"{c}_split{s}.json")))
        hd = d["hyper_detail"]
        ls_keys = [k for k in hd if "lengthscale" in k.lower()]
        print(f" split{s} n_ls_keys={len(ls_keys)}")
        for k in ls_keys:
            v = np.asarray(hd[k]).ravel()
            names = (FEAT10 if c != "M0" else ['porosity', 'w/c', 'cem (kg/m3)', 'SP (kg/m3)', 'sand (kg/m3)', 'WAS', 'I_gap', 'Cu', 'Aag(mm^-1)'])
            if len(v) == len(names):
                lab = "; ".join(f"{n}={x:.2f}" for n, x in zip(names, v))
                print(f"  {k}: {lab}")
            else:
                print(f"  {k} (dim {len(v)}):", np.round(v, 3))
        for k in hd:
            kl = k.lower()
            if ("task" in kl or "noise" in kl or "outputscale" in kl or ("alpha" in kl and "length" not in kl) or "mean" in kl):
                print(f"  {k}:", np.round(np.asarray(hd[k]).ravel(), 4))
print("COLLECT DONE")
