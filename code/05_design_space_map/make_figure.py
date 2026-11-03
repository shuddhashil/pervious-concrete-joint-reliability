"""Fig_design_space_map. Style copied from analysis/scripts/plot_relief_map.py of the manuscript project (Arial 9 pt, 5.6 x 4.3 in, dot style, log-K axis ticks).
Shaded = targets supported at R_LCB95 >= 0.90, thin outline = >= 0.95, optional dashed = >= 0.80; blank = outside the supported range. Dots = the 631 recorded mixtures (CS vs k). No 18/13 marker, no cement, no colour gradient."""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
SM = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ROOT = os.path.dirname(SM)
DATA = os.path.join(ROOT, "base", "core", "data_inputs", "phase1_reconciled_eligible.csv"); OUT = os.path.join(SM, "figures")
S = np.arange(5.0, 51.0, 1.0); K = np.geomspace(0.5, 30.0, 20); lk = np.log10(K)
g = pd.read_csv(os.path.join(SM, "grid_supported.csv")); F = {R: g.pivot(index="K_star", columns="S_star", values=f"supported_{R}").to_numpy(bool) for R in ("0.80", "0.90", "0.95")}   # rows = K ascending, cols = S ascending
x = pd.read_csv(DATA); assert len(x) == 631
x = x[x["perm (mm/s)"].between(K.min(), K.max()) & x.comp_converted.between(S.min(), S.max())]


def pad(M):                                   # replicate the edge row / column so the shaded area reaches the axis limits
    M = np.pad(M.astype(float), 1, mode="edge"); return M


Sp = np.r_[S[0] - 1, S, S[-1] + 1]; dl = lk[1] - lk[0]; Lp = np.r_[lk[0] - dl, lk, lk[-1] + dl]
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.linewidth": 0.6, "svg.fonttype": "none", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(5.6, 4.3)); FILL = "#b9cf8f"; EDGE = "#4f6b34"
ax.contourf(Sp, Lp, pad(F["0.90"]), levels=[0.5, 1.5], colors=[FILL], zorder=1)
ax.contour(Sp, Lp, pad(F["0.90"]), levels=[0.5], colors=[EDGE], linewidths=0.6, zorder=2)
ax.contour(Sp, Lp, pad(F["0.95"]), levels=[0.5], colors=["#1a1a1a"], linewidths=0.9, zorder=3)
ax.contour(Sp, Lp, pad(F["0.80"]), levels=[0.5], colors=["#6b6b6b"], linewidths=0.6, linestyles="--", zorder=2)
ax.scatter(x.comp_converted, np.log10(x["perm (mm/s)"]), s=3.5, color="#2b2b2b", alpha=0.35, lw=0, zorder=4)
ax.text(41.5, np.log10(1.6), "outside the\nsupported range", color="#555555", fontsize=8.5, ha="center", va="center", style="italic")
kt = [0.5, 1, 2, 5, 10, 20, 30]; ax.set_yticks(np.log10(kt)); ax.set_yticklabels([f"{t:g}" for t in kt]); ax.set_xlim(S.min(), S.max()); ax.set_ylim(lk[0], lk[-1])
ax.set_xlabel("Target compressive strength $S^*$ (MPa)"); ax.set_ylabel("Target permeability $K^*$ (mm/s)"); ax.tick_params(direction="out", length=3)
from matplotlib.lines import Line2D; from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=FILL, edgecolor=EDGE, lw=0.6, label="$R_{LCB95}\\geq 0.90$"), Line2D([0], [0], color="#1a1a1a", lw=0.9, label="$R_{LCB95}\\geq 0.95$"), Line2D([0], [0], color="#6b6b6b", lw=0.6, ls="--", label="$R_{LCB95}\\geq 0.80$"),
                   Line2D([0], [0], marker="o", color="none", markerfacecolor="#2b2b2b", alpha=0.6, markersize=3, label="measured mixtures")], loc="upper right", fontsize=7.5, frameon=True, framealpha=0.9, edgecolor="none", borderpad=0.5)
fig.tight_layout()
for ext in ("png", "pdf", "svg"): fig.savefig(os.path.join(OUT, f"Fig_design_space_map.{ext}"), dpi=300)
print("saved", os.path.join(OUT, "Fig_design_space_map.png"), "supported cells", {k: int(v.sum()) for k, v in F.items()}, "dots", len(x))
