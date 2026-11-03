"""Fig_design_space_map_v2. Same look as the v1 figure (Arial 9 pt, 5.6 x 4.3 in, dot style, log-K axis ticks), single level: shaded = targets supported at R_LCB95 >= 0.90, blank = outside the supported range,
dots = the 631 recorded mixtures (CS vs k; those inside the plotted axes). No 0.95 / 0.80 lines, no 18/13 marker, no cement information.
usage: make_figure_v2.py [grid csv (default grid_supported_v2.csv)] [column (default supported_0.90)] [output folder (default figures)] [file stem (default Fig_design_space_map_v2)]"""
import sys, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D; from matplotlib.patches import Patch
SM = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ROOT = os.path.dirname(SM)
csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SM, "grid_supported_v2.csv"); col = sys.argv[2] if len(sys.argv) > 2 else "supported_0.90"; OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(SM, "figures"); stem = sys.argv[4] if len(sys.argv) > 4 else "Fig_design_space_map_v2"
DATA = os.path.join(ROOT, "base", "core", "data_inputs", "phase1_reconciled_eligible.csv"); S = np.arange(5.0, 51.0, 1.0); K = np.geomspace(0.5, 30.0, 20); lk = np.log10(K)
g = pd.read_csv(csv); F = g.pivot(index="K_star", columns="S_star", values=col).to_numpy(bool)          # rows = K ascending, cols = S ascending
x = pd.read_csv(DATA); assert len(x) == 631; x = x[x["perm (mm/s)"].between(K.min(), K.max()) & x.comp_converted.between(S.min(), S.max())]
pad = lambda M: np.pad(M.astype(float), 1, mode="edge")                                                  # the shaded area reaches the axis limits
Sp = np.r_[S[0] - 1, S, S[-1] + 1]; dl = lk[1] - lk[0]; Lp = np.r_[lk[0] - dl, lk, lk[-1] + dl]
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.linewidth": 0.6, "svg.fonttype": "none", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(5.6, 4.3)); FILL = "#b9cf8f"; EDGE = "#4f6b34"
ax.contourf(Sp, Lp, pad(F), levels=[0.5, 1.5], colors=[FILL], zorder=1); ax.contour(Sp, Lp, pad(F), levels=[0.5], colors=[EDGE], linewidths=0.6, zorder=2)
ax.scatter(x.comp_converted, np.log10(x["perm (mm/s)"]), s=3.5, color="#2b2b2b", alpha=0.35, lw=0, zorder=4)
ax.text(float(os.environ.get("NOTE_X", 44.0)), np.log10(float(os.environ.get("NOTE_K", 4.0))), "outside the\nsupported range", color="#555555", fontsize=8.5, ha="center", va="center", style="italic")
kt = [0.5, 1, 2, 5, 10, 20, 30]; ax.set_yticks(np.log10(kt)); ax.set_yticklabels([f"{t:g}" for t in kt]); ax.set_xlim(S.min(), S.max()); ax.set_ylim(lk[0], lk[-1])
ax.set_xlabel("Target compressive strength $S^*$ (MPa)"); ax.set_ylabel("Target permeability $K^*$ (mm/s)"); ax.tick_params(direction="out", length=3)
ax.legend(handles=[Patch(facecolor=FILL, edgecolor=EDGE, lw=0.6, label="$R_{LCB95}\\geq 0.90$"), Line2D([0], [0], marker="o", color="none", markerfacecolor="#2b2b2b", alpha=0.6, markersize=3, label="measured mixtures")], loc="upper right", fontsize=7.5, frameon=True, framealpha=0.9, edgecolor="none", borderpad=0.5)
fig.tight_layout()
for ext in ("png", "pdf", "svg"): fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), dpi=300)
print("saved", os.path.join(OUT, stem + ".png"), "supported cells", int(F.sum()), "dots", len(x))
