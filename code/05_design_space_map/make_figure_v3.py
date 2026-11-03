"""Fig_design_space_map_v3. Same look as the v2 figure (Arial 9 pt, 5.6 x 4.3 in, log-K axis ticks, measured-mixture dots):
light green grid cells (one rectangle per S*/K* cell) = targets supported at R_LCB95 >= 0.90, darker cells inside = cells also backed by recorded mixtures from >= 2 studies,
a staircase line along the outer upper edge of the light-green cells, the six worked targets as filled (certified) / open (not certified) circles labelled S/K.
No cement information, no other reliability levels.
usage: make_figure_v3.py [grid csv (default grid_supported_v3.csv)] [supported column (default supported_final)] [data-backed column (default data_backed)] [output folder (default figures)] [file stem (default Fig_design_space_map_v3)]"""
import sys, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D; from matplotlib.patches import Patch
from matplotlib.collections import LineCollection; from matplotlib.colors import ListedColormap
SM = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ROOT = os.path.dirname(SM)
csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SM, "grid_supported_v3.csv"); col = sys.argv[2] if len(sys.argv) > 2 else "supported_final"
dcol = sys.argv[3] if len(sys.argv) > 3 else "data_backed"; OUT = sys.argv[4] if len(sys.argv) > 4 else os.path.join(SM, "figures"); stem = sys.argv[5] if len(sys.argv) > 5 else "Fig_design_space_map_v3"
DATA = os.path.join(ROOT, "base", "core", "data_inputs", "phase1_reconciled_eligible.csv"); S = np.arange(5.0, 51.0, 1.0); K = np.geomspace(0.5, 30.0, 20); lk = np.log10(K)
os.makedirs(OUT, exist_ok=True)
g = pd.read_csv(csv); F = g.pivot(index="K_star", columns="S_star", values=col).to_numpy(bool)          # rows = K ascending, cols = S ascending
DB = g.assign(_db=g[dcol].astype(str).str.lower().isin(("true", "1"))).pivot(index="K_star", columns="S_star", values="_db").to_numpy(bool) if dcol in g else np.zeros_like(F)
x = pd.read_csv(DATA); assert len(x) == 631; x = x[x["perm (mm/s)"].between(K.min(), K.max()) & x.comp_converted.between(S.min(), S.max())]
Se = np.r_[S[0] - 0.5, S + 0.5]                                                                            # cell edges: S +- 0.5 MPa
dl = lk[1] - lk[0]; Ke = np.r_[lk[0] - dl / 2, lk + dl / 2]                                               # K edges: log half-steps, same half-step at the ends
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.linewidth": 0.6, "svg.fonttype": "none", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(5.6, 4.3)); FILL = "#d5e2b8"; DARK = "#8aa85c"; EDGE = "#4f6b34"
ax.pcolormesh(Se, Ke, np.ma.masked_where(~F, F.astype(float)), cmap=ListedColormap([FILL]), shading="flat",
              edgecolors="none", antialiased=False, zorder=1)
if DB.any():
    ax.pcolormesh(Se, Ke, np.ma.masked_where(~DB, DB.astype(float)), cmap=ListedColormap([DARK]), shading="flat",
                  edgecolors="none", antialiased=False, zorder=1)
tops = np.array([int(np.flatnonzero(F[b, :]).max()) if F[b, :].any() else -1 for b in range(len(K))])     # top supported cell of every K* column
segs = []                                                                                                  # staircase: boundary of the union of light-green cells
for b in range(len(K)):
    if tops[b] < 0: continue
    xe = float(S[tops[b]]) + 0.5                                                                          # outer (top) edge of this column's highest supported cell
    segs.append([(xe, float(Ke[b])), (xe, float(Ke[b + 1]))])                                             # horizontal along that cell edge
    if b + 1 < len(K) and tops[b + 1] >= 0:
        segs.append([(xe, float(Ke[b + 1])), (float(S[tops[b + 1]]) + 0.5, float(Ke[b + 1]))])            # vertical step at the K-cell edge
if segs: ax.add_collection(LineCollection(segs, colors="#2f3d1f", linewidths=1.0, capstyle="butt", zorder=3))
ax.scatter(x.comp_converted, np.log10(x["perm (mm/s)"]), s=3.5, color="#2b2b2b", alpha=0.35, lw=0, zorder=4)
ax.text(float(os.environ.get("NOTE_X", 41.0)), np.log10(float(os.environ.get("NOTE_K", 8.0))), "outside the\nsupported range", color="#555555", fontsize=8.5, ha="center", va="center", style="italic", zorder=5)
TG = [((10.0, 8.0), True), ((14.0, 6.0), True), ((16.0, 10.0), True), ((20.0, 4.0), True), ((18.0, 13.0), False), ((25.0, 13.0), False)]   # (S*, K*), filled = engine-certified
LBL = {(18.0, 13.0): ((0, 7), "center"), (25.0, 13.0): ((6, 5), "left")}                                   # labels clear of markers, dots and the legend
for (s0, k0), ok in TG:
    ax.scatter([s0], [np.log10(k0)], s=30, facecolor="#111111" if ok else "none", edgecolor="#111111", linewidth=0.9, zorder=6)
    off, ha = LBL.get((s0, k0), ((4, 4), "left"))
    ax.annotate(f"{s0:g}/{k0:g}", (s0, np.log10(k0)), xytext=off, textcoords="offset points", fontsize=7.5, color="#111111", ha=ha, zorder=7)
kt = [0.5, 1, 2, 5, 10, 20, 30]; ax.set_yticks(np.log10(kt)); ax.set_yticklabels([f"{t:g}" for t in kt]); ax.set_xlim(S.min(), S.max()); ax.set_ylim(lk[0], lk[-1])
ax.set_xlabel("Target compressive strength $S^*$ (MPa)"); ax.set_ylabel("Target permeability $K^*$ (mm/s)"); ax.tick_params(direction="out", length=3)
ax.legend(handles=[Patch(facecolor=FILL, edgecolor=EDGE, lw=0.6, label="Certified at $R^* = 0.90$\n(with three trial batches)"),
                   Patch(facecolor=DARK, edgecolor=EDGE, lw=0.6, label="Certified and backed by recorded\nmixtures from at least two studies"),
                   Line2D([0], [0], marker="o", color="none", markerfacecolor="#2b2b2b", alpha=0.6, markersize=3, label="measured mixtures"),
                   Line2D([0], [0], marker="o", color="none", markerfacecolor="#111111", markeredgecolor="#111111", markersize=4.5, label="worked target, certified"),
                   Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#111111", markersize=4.5, label="worked target, not certified")],
          loc="lower right", fontsize=6.6, frameon=True, framealpha=0.9, edgecolor="none", borderpad=0.4,
          handlelength=1.6, handletextpad=0.6)
fig.tight_layout()
for ext in ("png", "pdf", "svg"): fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), dpi=300)
print("saved", os.path.join(OUT, stem + ".png"), "supported cells", int(F.sum()), "data-backed", int(DB.sum()), "dots", len(x))
