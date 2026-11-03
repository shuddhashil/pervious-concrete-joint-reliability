"""Useful plots only: DE convergence, porosity margins, shortlist gradations."""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
for ax, (tag, s, k) in zip(axes, [("18_13", 18, 13), ("15_8", 15, 8), ("12_5", 12, 5)]):
    for f in sorted(glob.glob(os.path.join(RUN, "traces", f"de_{tag}_p90_s*.trace.csv"))):
        t = pd.read_csv(f)
        ax.plot(t["evals"], t["best_merit"], lw=0.8, alpha=0.7)
    ax.set_title(f"DE {s}/{k} @90 (8 seeds)")
    ax.set_xlabel("evaluations")
    ax.set_ylabel("best merit")
    ax.set_ylim(bottom=150)
axes[0].set_ylim(200, 750)
fig.tight_layout()
fig.savefig(os.path.join(RUN, "plots", "de_convergence.png"), dpi=120)

for cid in ["S1813-01", "S1813-R", "S1508-01", "S1508-R", "S1205-07"]:
    df = pd.read_csv(os.path.join(RUN, "tables", f"porosity_curve_{cid}.csv"))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(df["phi"], df["margin_S"], label="margin_S (MPa)")
    ax.plot(df["phi"], df["margin_Q"], label="margin_Q (log)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("required achieved hardened porosity (%)")
    ax.set_ylabel("joint margin")
    ax.set_title(f"{cid}: paired-screen margins vs porosity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RUN, "plots", f"porosity_{cid}.png"), dpi=120)

short = pd.read_csv(os.path.join(RUN, "CANDIDATE_SHORTLIST.csv"))
short = short[~short["cid"].str.endswith("NM")]
fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(short))
w = 0.12
for i in range(6):
    ax.bar(x + (i - 2.5) * w, short[f"g{i+1}"], width=w, label=f"bin{i+1}")
ax.set_xticks(x)
ax.set_xticklabels(short["cid"], rotation=30, ha="right", fontsize=8)
ax.set_ylabel("retained fraction")
ax.set_title("Shortlist gradations (six bins, coarse -> fine)")
ax.legend(fontsize=8, ncol=6)
fig.tight_layout()
fig.savefig(os.path.join(RUN, "plots", "shortlist_gradations.png"), dpi=120)
print("plots done")
