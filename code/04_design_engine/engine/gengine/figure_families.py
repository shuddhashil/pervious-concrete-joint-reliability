"""One figure: family centroid blends (mean six-bin gradation of the member blends) for the recommended k=3 and the alternative k=5; recorded-passer blends marked."""
import os, sys, json
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); r = json.load(open(os.path.join(ROOT, "stages", "01_families", "families_radii.json")))
bins = ["25-19", "19-12.5", "12.5-9.5", "9.5-6.3", "6.3-4.75", "4.75-2.36"]; col = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756", "#9d755d", "#72b7b2"]
fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, k in zip(axs, ("3", "5")):
    fams = r["families"][k]["families"]; w = 0.8 / len(fams)
    for i, f in enumerate(fams):
        ax.bar(np.arange(6) + i * w, f["centroid_blend"], w, color=col[i], label=f"family {i}: {f['n_blends']} blends, {f['n_rows']} rows" + (f", {len(f['passer_rows'])} passers" if f["passer_rows"] else ", no passer"))
    ax.set_xticks(np.arange(6) + 0.4 - w / 2); ax.set_xticklabels(bins); ax.set_xlabel("size bin (mm)"); ax.legend(fontsize=7.5, loc="upper right", frameon=False)
    ax.set_title(f"k = {k}" + ("  (recommended)" if k == "3" else "  (alternative, 4 families after merging a 3-blend cluster)"), fontsize=9)
axs[0].set_ylabel("mass fraction of coarse aggregate"); fig.suptitle("Family centroid blends (Ward on the five passing sizes, 83 recorded blends)", fontsize=10); fig.tight_layout()
os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True); fig.savefig(os.path.join(ROOT, "outputs", "family_centroid_blends.png"), dpi=140); print("saved")
