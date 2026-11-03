"""Alternative looks for the design-space map (mock-ups for the author to choose from; main thread, 2026-10-04). CPU only, reads grid_supported_v2.csv and the 631 recorded mixtures; writes figures/styles/*.png.
No 18/13 marker and no cement information (author pins). The supported region is the v2 region; only the presentation changes.
 A  two-tone: deep green = reliability >= 0.95 can be certified, light green = 0.90-0.95 (the 'how close to the edge' view), direct labels on the edge.
 B  property-space (Ashby style): measured mixtures coloured by porosity over the supported region.
 C  3D plateau: height = best certified reliability found for the target (supported cells coloured, others grey).
 D  reach: engine-supported region against the region reached by at least one recorded mixture (staircase of the data)."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa
SM = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ROOT = os.path.dirname(SM); OUT = os.path.join(SM, "figures", "styles"); os.makedirs(OUT, exist_ok=True)
S = np.arange(5.0, 51.0, 1.0); K = np.geomspace(0.5, 30.0, 20); lk = np.log10(K)
g = pd.read_csv(os.path.join(SM, "grid_supported_v2.csv")); F = g.pivot(index="K_star", columns="S_star", values="supported_0.90").to_numpy(bool)
V = pd.to_numeric(g["best_R_LCB95_stageB"], errors="coerce"); V = g.assign(v=V).pivot(index="K_star", columns="S_star", values="v").to_numpy(float)
HI = F & (np.nan_to_num(V, nan=0.0) >= 0.952)                                    # >= 0.95 certifiable (early-stopped at 0.952 in the search, so a lower bound)
x = pd.read_csv(os.path.join(ROOT, "base", "core", "data_inputs", "phase1_reconciled_eligible.csv")); assert len(x) == 631
x = x[x["perm (mm/s)"].between(K.min(), K.max()) & x.comp_converted.between(S.min(), S.max())]
pad = lambda M: np.pad(M.astype(float), 1, mode="edge"); Sp = np.r_[S[0] - 1, S, S[-1] + 1]; dl = lk[1] - lk[0]; Lp = np.r_[lk[0] - dl, lk, lk[-1] + dl]
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.linewidth": 0.6, "axes.edgecolor": "#444444", "text.color": "#222222", "axes.labelcolor": "#222222", "xtick.color": "#444444", "ytick.color": "#444444"})
LIGHT, DEEP, LINE = "#cfe3b0", "#6fa34d", "#2f5a22"; INK, MUTED = "#222222", "#6b6b6b"
kt = [0.5, 1, 2, 5, 10, 20, 30]


def base(ax, xlabel=True):
    ax.set_xlim(S.min(), S.max()); ax.set_ylim(lk[0], lk[-1]); ax.set_yticks(np.log10(kt)); ax.set_yticklabels([f"{t:g}" for t in kt]); ax.tick_params(direction="out", length=3)
    ax.set_xlabel("Target compressive strength $S^*$ (MPa)"); ax.set_ylabel("Target permeability $K^*$ (mm/s)")
    for s in ("top", "right"): ax.spines[s].set_visible(False)


def region(ax, fill=LIGHT, line=LINE, z=1, lw=1.2, halo=True):
    ax.contourf(Sp, Lp, pad(F), levels=[0.5, 1.5], colors=[fill], zorder=z)
    if halo: ax.contour(Sp, Lp, pad(F), levels=[0.5], colors=["white"], linewidths=lw + 2.2, zorder=z + 1)
    ax.contour(Sp, Lp, pad(F), levels=[0.5], colors=[line], linewidths=lw, zorder=z + 2)


# ---------------- A: two-tone -----------------
fig, ax = plt.subplots(figsize=(6.2, 4.6)); base(ax); region(ax, fill=LIGHT)
ax.contourf(Sp, Lp, pad(HI), levels=[0.5, 1.5], colors=[DEEP], zorder=2); ax.contour(Sp, Lp, pad(HI), levels=[0.5], colors=["white"], linewidths=1.0, zorder=3)
ax.scatter(x.comp_converted, np.log10(x["perm (mm/s)"]), s=3.2, color="#1f1f1f", alpha=0.28, lw=0, zorder=4)
ax.text(10, np.log10(0.8), "reliability 0.95 or better\ncan be certified here", color="white", fontsize=9, fontweight="bold", ha="left", va="center", zorder=6)
ax.annotate("edge: 0.90 can still be certified,\nbut with less room to spare", xy=(30.5, np.log10(5.0)), xytext=(36.5, np.log10(9.5)), fontsize=8, color=LINE, ha="left", va="center", arrowprops=dict(arrowstyle="-", color=LINE, lw=0.7), zorder=6)
ax.text(42.5, np.log10(3.2), "outside the\nsupported range", color=MUTED, fontsize=9.5, ha="center", va="center", style="italic")
ax.legend(handles=[Patch(facecolor=DEEP, label="$R_{LCB95}\\geq 0.95$ certifiable"), Patch(facecolor=LIGHT, edgecolor=LINE, lw=0.6, label="$R_{LCB95}\\geq 0.90$ certifiable"), Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f1f1f", alpha=0.5, markersize=3, label="measured mixtures")], loc="upper right", fontsize=7.5, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "A_two_tone.png"), dpi=220); plt.close(fig)

# ---------------- B: property space, dots by porosity -----------------
fig, ax = plt.subplots(figsize=(6.6, 4.6)); base(ax); region(ax, fill="#e4efd0", line=LINE, halo=False)
sc = ax.scatter(x.comp_converted, np.log10(x["perm (mm/s)"]), c=x["porosity"], s=11, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("blu", ["#a9cdf3", "#2a78d6", "#0d366b"]), lw=0.3, edgecolors="white", zorder=4, vmin=10, vmax=40)
cb = fig.colorbar(sc, ax=ax, pad=0.015, fraction=0.045); cb.set_label("Porosity of the measured mixture (%)"); cb.outline.set_linewidth(0.4)
ax.text(42.5, np.log10(3.2), "outside the\nsupported range", color=MUTED, fontsize=9.5, ha="center", va="center", style="italic")
ax.text(6, np.log10(0.62), "shaded = certifiable at $R_{LCB95}\\geq 0.90$", fontsize=8, color=LINE, ha="left", va="center")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "B_property_space_by_porosity.png"), dpi=220); plt.close(fig)

# ---------------- C: 3D plateau -----------------
Z = np.where(F, np.nan_to_num(V, nan=0.9), np.nan_to_num(V, nan=0.0)); Z = np.clip(Z, 0.0, 1.0); SS, LL = np.meshgrid(S, lk)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("g", ["#cfe3b0", "#6fa34d", "#2f5a22"]); fc = np.empty(Z.shape + (4,)); nz = np.clip((Z - 0.9) / 0.1, 0, 1)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]): fc[i, j] = cmap(nz[i, j]) if F[i, j] else (0.86, 0.86, 0.86, 1.0)
fig = plt.figure(figsize=(6.8, 5.0)); ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(SS, LL, Z, facecolors=fc, rstride=1, cstride=1, linewidth=0.15, edgecolor=(1, 1, 1, 0.35), shade=False, antialiased=True)
ax.set_xlabel("$S^*$ (MPa)", labelpad=4); ax.set_ylabel("$K^*$ (mm/s)", labelpad=4); ax.set_zlabel("Best certified reliability", labelpad=4); ax.set_yticks(np.log10([0.5, 1, 5, 20])); ax.set_yticklabels(["0.5", "1", "5", "20"]); ax.set_zlim(0, 1)
ax.view_init(elev=30, azim=-128); ax.xaxis.pane.set_alpha(0.0); ax.yaxis.pane.set_alpha(0.0); ax.zaxis.pane.set_alpha(0.0); ax.grid(False)
ax.text2D(0.03, 0.94, "green plateau: supported ($R_{LCB95}\\geq 0.90$)   grey: not supported", transform=ax.transAxes, fontsize=8, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "C_3D_plateau.png"), dpi=220); plt.close(fig)

# ---------------- D: reach of the data vs reach of the engine -----------------
cs, kk = x.comp_converted.to_numpy(), x["perm (mm/s)"].to_numpy(); xa = pd.read_csv(os.path.join(ROOT, "base", "core", "data_inputs", "phase1_reconciled_eligible.csv"))
cs_all, k_all = xa.comp_converted.to_numpy(float), xa["perm (mm/s)"].to_numpy(float); R = np.zeros((len(K), len(S)), bool)
for b, kv in enumerate(K):
    for a, sv in enumerate(S): R[b, a] = bool(np.any((cs_all >= sv) & (k_all >= kv)))
both, eng_only, data_only = int((F & R).sum()), int((F & ~R).sum()), int((~F & R).sum())
fig, ax = plt.subplots(figsize=(6.4, 4.6)); base(ax)
ax.contourf(Sp, Lp, pad(R), levels=[0.5, 1.5], colors=["#e3e3e3"], zorder=1); ax.contour(Sp, Lp, pad(R), levels=[0.5], colors=["#8a8a8a"], linewidths=0.9, linestyles=["--"], zorder=2)
ax.contourf(Sp, Lp, pad(F), levels=[0.5, 1.5], colors=[LIGHT], zorder=3, alpha=0.95); ax.contour(Sp, Lp, pad(F), levels=[0.5], colors=[LINE], linewidths=1.3, zorder=4)
ax.scatter(cs, np.log10(kk), s=3.2, color="#1f1f1f", alpha=0.3, lw=0, zorder=5)
ax.text(42.5, np.log10(3.2), "outside both", color=MUTED, fontsize=9, ha="center", va="center", style="italic")
ax.legend(handles=[Patch(facecolor=LIGHT, edgecolor=LINE, lw=0.8, label="certifiable by the inverse design ($R_{LCB95}\\geq 0.90$)"), Patch(facecolor="#e3e3e3", edgecolor="#8a8a8a", lw=0.8, ls="--", label="reached by at least one measured mixture"), Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f1f1f", alpha=0.5, markersize=3, label="measured mixtures")], loc="upper right", fontsize=7.2, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "D_reach_data_vs_engine.png"), dpi=220); plt.close(fig)
print(dict(cells=int(F.size), engine=int(F.sum()), data=int(R.sum()), both=both, engine_only=eng_only, data_only=data_only, deep=int(HI.sum())))
