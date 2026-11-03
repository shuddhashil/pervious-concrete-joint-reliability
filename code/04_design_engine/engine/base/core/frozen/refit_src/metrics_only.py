import json, os
import numpy as np, pandas as pd
RUN = r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\research\Stage1_CA_Forward_Refit_20260913\run_01"
SF = os.path.join(RUN, "models", "splitfits")
df = pd.read_csv(os.path.join(RUN, "tables", "lcmgp_split_metrics.csv"))
print(df.groupby("config")[["rmseCS", "r2CS", "maeCS", "rmseQ", "r2Q", "maeQ", "nlpdCS", "nlpdQ"]].agg(["mean", "std"]).round(4).to_string())
piv = df.pivot_table(index="split", columns="config", values=["rmseCS", "rmseQ", "r2CS", "r2Q"])
for cfg in ["M1global", "M1agg"]:
    for met in ["rmseCS", "rmseQ", "r2CS", "r2Q"]:
        d = piv[met][cfg] - piv[met]["M0"]
        print(f"d {cfg}-M0 {met}:", np.round(d.values, 4), "mean", round(float(d.mean()), 4))
# CA lengthscale summary: kernels.0 (RBF global) and kernels.1 (RQ global) CA dim
FEAT10 = ['porosity', 'w/c', 'cem (kg/m3)', 'CA (kg/m3)', 'SP (kg/m3)', 'sand (kg/m3)', 'WAS', 'I_gap', 'Cu', 'Aag(mm^-1)']
for c in ["M1global", "M1agg"]:
    print("=" * 20, c, "CA lengthscales (constrained space)")
    for s in range(5):
        d = json.load(open(os.path.join(SF, f"{c}_split{s}.json")))
        hd = d["hyper_detail"]
        row = {}
        for k in ["shared_covar_module.data_covar_module.base_kernel.kernels.0.raw_lengthscale",
                  "shared_covar_module.data_covar_module.base_kernel.kernels.1.raw_lengthscale"]:
            v = np.asarray(hd[k]).ravel()
            # constrained ls = Interval(0.2,4) applied to raw; report raw and mapped approx via softplus-like? Interval constraint maps unconstrained->(0.2,4).
            # Just report the constrained value: gpytorch Interval transform. Approximate by loading? Use raw only + note.
            row[k.split(".")[-2] + "/" + k.split(".")[-1]] = round(float(v[3]), 3)
        print(f" split{s} raw CA ls:", row)
