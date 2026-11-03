"""Plots from saved tables only."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
RUN = r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\research\Stage1_CA_Forward_Refit_20260913\run_01"
TAB = os.path.join(RUN, "tables"); PL = os.path.join(RUN, "plots")
os.makedirs(PL, exist_ok=True)
m = pd.read_csv(os.path.join(TAB, "lcmgp_split_metrics.csv"))
piv = m.pivot_table(index="split", columns="config", values=["rmseCS", "rmseQ"])
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for i, met in enumerate(["rmseCS", "rmseQ"]):
    for c in ["M0", "M1global", "M1agg"]:
        ax[i].plot(range(5), piv[met][c], marker="o", label=c)
    ax[i].set_xticks(range(5)); ax[i].set_title(met); ax[i].legend()
plt.tight_layout(); plt.savefig(os.path.join(PL, "paired_rmse.png"), dpi=150); plt.close()
f = pd.read_csv(os.path.join(TAB, "joint_fold_metrics.csv"))
g = f.groupby("p")["joint_cov"].mean()
plt.figure(figsize=(6, 4))
plt.plot([0.8, 0.9, 0.95], [g.loc[0.8], g.loc[0.9], g.loc[0.95]], marker="o", label="M1global-10D cross-fitted")
plt.plot([0.8, 0.95], [0.8, 0.95], "k--", label="ideal")
plt.xlabel("nominal p"); plt.ylabel("held-out joint coverage"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(PL, "joint_coverage_10d.png"), dpi=150); plt.close()
sw = pd.read_csv(os.path.join(TAB, "ca_supported_sweeps.csv"))
sup = sw[sw["supported"]]
plt.figure(figsize=(7, 4))
plt.scatter(sup["CA"], sup["pred_CS"], s=8, alpha=0.5)
plt.xlabel("CA (kg/m3)"); plt.ylabel("pred CS (MPa) — supported sweep points")
plt.tight_layout(); plt.savefig(os.path.join(PL, "ca_supported_sweeps.png"), dpi=150); plt.close()
print("PLOTS DONE")
