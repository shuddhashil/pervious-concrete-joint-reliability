"""COPY of tone_down_2026-10-26/v6_new.py (2026-10-30): only the R_LCB95 axis labels changed to R_score (Figs 13, 14 of V4); writes to ./figs.
Results figures for the laboratory-effect engine (2026-10-22): forward model (Fig 6), reliability audit (Fig 8), cement by family (Fig 10),
reliability vs cement (Fig 12), porosity scatter (Fig 13). Same look as make_figures_v4.py. Reads forward_evidence_v3.json, designs_v3.json, the run tables
(RobustEngine_Results_20261022/tables) and the deployed engine's audit (audit_LNI). The design-space map (Fig 9) is made by make_figure_v3.py of the map run."""
import json, os, re, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figs")
RES = "C:/Users/Shuddhashil/Desktop/Mix Design Paper Rebuild/research"
CORE = RES + "/RobustEngine_20261022/GuidedEngine_20261003/base/core"
AUD = RES + "/RobustnessTests_20261021/results/audit_LNI"; LAB = RES + "/RobustnessTests_20261021/colab_lab/out_lab"
TAB = RES + "/RobustEngine_Results_20261022/tables"
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.linewidth": 0.6, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREEN, DARK = "#4c78a8", "#e0a458", "#7aa35f", "#2b2b2b"
FAM = ["#4c78a8", "#e0a458", "#7aa35f"]; FAMNAME = ["coarse", "well-graded", "fine single-size"]


def save(fig, stem):
    for ext in ("png", "pdf", "svg"): fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), dpi=300)
    plt.close(fig)


F = json.load(open(os.path.join(r"C:/Users/Shuddhashil/Desktop/Local-Agent-Team/Papers/MixDesign_Paper/analysis/results_section", "forward_evidence_v3.json"))); DJ = json.load(open(os.path.join(r"C:/Users/Shuddhashil/Desktop/Local-Agent-Team/Papers/MixDesign_Paper/analysis/results_section", "designs_v3.json")))
CERT4 = ["S10_K8", "S14_K6", "S16_K10", "S20_K4"]; ALL6 = CERT4 + ["S18_K13", "S25_K13"]
LABEL = {n: n.replace("S", "").replace("_K", "/") for n in ALL6}
RUN = {n: f"{n}_R0.90" for n in ALL6}; SCAN = {n: f"{TAB}/{n}_global_scan.csv" for n in ALL6}; SCAN["S18_K13"] = f"{TAB}/S18_K13_FULLGRID_global_scan.csv"

# ---------------- Figure 6: forward model ----------------
sp = [json.load(open(f"{LAB}/labrand_split{k}.json")) for k in range(5)]
d0 = pd.read_csv(CORE + "/data_inputs/phase1_reconciled_eligible.csv")
row = np.concatenate([s["original_row_index"] for s in sp]); phi_v = d0.porosity.to_numpy()[row] if "porosity" in d0 else None
V = dict(CS=(np.concatenate([s["y_true_CS"] for s in sp]), np.concatenate([s["y_pred_CS"] for s in sp])),
         Q=(np.concatenate([s["y_true_logQ"] for s in sp]), np.concatenate([s["y_pred_logQ"] for s in sp])))
edges = F["edges"]; band = np.digitize(phi_v, edges[1:3]); bl = [f"{edges[i]:.1f}\u2013{edges[i + 1]:.1f}%" for i in range(3)]; cols = [BLUE, ORANGE, GREEN]
fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.9), gridspec_kw=dict(width_ratios=[1, 1, 0.95])); VA = F["validation"]
for a, key, lab, lim, ann in [(ax[0], "CS", "CS (MPa)", (0, 46), f"mean of splits:\nRMSE {VA['rmse_strength'][0]:.2f}\n$R^2$ {VA['r2_strength'][0]:.3f}"),
                              (ax[1], "Q", "ln k (k in mm/s)", (-3, 4.5), f"mean of splits:\nRMSE {VA['rmse_logperm'][0]:.2f}\n$R^2$ {VA['r2_logperm'][0]:.3f}")]:
    for k in range(3):
        m = band == k; a.scatter(V[key][0][m], V[key][1][m], s=6, color=cols[k], alpha=0.6, lw=0, label=bl[k])
    a.plot(lim, lim, color="k", lw=0.8); a.set_xlim(lim); a.set_ylim(lim); a.set_xlabel("Measured " + lab); a.set_ylabel("Predicted " + lab)
    a.text(0.05, 0.95, ann, transform=a.transAxes, va="top", fontsize=8)
ax[0].legend(title="Porosity", title_fontsize=7.5, fontsize=7.5, loc="lower right", frameon=False, handletextpad=0.2, markerscale=1.6)
pb = F["pooled_bank"]; x = np.arange(3); w = 0.38
ax[2].bar(x - w / 2, [b["rmse_S"] / pb["rmse_S"] for b in F["bands"]], w, color="#555555", label="CS")
ax[2].bar(x + w / 2, [b["rmse_Q"] / pb["rmse_Q"] for b in F["bands"]], w, color="#aaaaaa", label="ln k")
ax[2].axhline(1, color="k", lw=0.7, ls="--"); ax[2].set_xticks(x); ax[2].set_xticklabels([s.replace("\u2013", "\u2013\n") for s in bl], fontsize=7.5)
ax[2].set_xlabel("Porosity band"); ax[2].set_ylabel("RMSE / pooled RMSE"); ax[2].legend(frameon=False, fontsize=7.5)
for a, t in zip(ax, "abc"): a.text(-0.02, 1.06, f"({t})", transform=a.transAxes, fontweight="bold", fontsize=10, va="bottom", ha="right")
fig.tight_layout(); save(fig, "Fig6_forward_model")

# ---------------- Figure 8: reliability audit ----------------
bins = pd.read_csv(AUD + "/reliability_bins_base_production.csv"); AJ = json.load(open(AUD + "/calibration_summary.json"))["base_production"]
fig = plt.figure(figsize=(6.8, 3.1)); gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.05], hspace=0.95, wspace=0.32)
a = fig.add_subplot(gs[:, 0]); a.axvspan(0.9, 1.0, color="#e6efd9", lw=0); a.plot([0, 1], [0, 1], color="k", lw=0.7)
a.errorbar(bins.mean_predicted, bins.observed_rate, yerr=[bins.observed_rate - bins.wilson_lo, bins.wilson_hi - bins.observed_rate], fmt="o", ms=4, color=BLUE, ecolor=BLUE, elinewidth=0.8, capsize=1.5)
a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_xlabel("Stated joint pass probability"); a.set_ylabel("Observed pass rate")
a.text(0.52, 0.06, "acceptance region\n$P\\geq0.90$", fontsize=7.5, color="#4f6b34", ha="left", va="bottom"); a.set_aspect("equal")
a.text(-0.02, 1.05, "(a)", transform=a.transAxes, fontweight="bold", fontsize=10, ha="right")
cb = AJ["cluster_bootstrap"]; mt = AJ["metrics"]; GOOD = "#d9ead3"
rows = [("Calibration slope", mt["cal_slope"], (cb["slope"]["lo95"], cb["slope"]["hi95"]), (0.8, 1.2), (0.6, 1.4), "interval overlaps 0.8\u20131.2"),
        ("Observed pass rate, $P\\geq0.90$", mt["accept_observed_rate"], (cb["accept_rate"]["lower5"], None), (0.85, 1.0), (0.80, 1.0), "lower 5% bound $\\geq$ 0.85"),
        ("Stated minus observed, $P\\geq0.90$", mt["accept_mean_predicted"] - mt["accept_observed_rate"], None, (-0.05, 0.05), (-0.05, 0.10), "difference $\\leq$ 0.05")]
for i, (lab, val, ci, bnd, lim, crit) in enumerate(rows):
    ax = fig.add_subplot(gs[i, 1]); ax.axvspan(bnd[0], bnd[1], color=GOOD, lw=0)
    if i == 0: ax.plot(ci, [0, 0], color=BLUE, lw=1.6)
    if i == 1: ax.plot([ci[0]], [0], marker="|", ms=11, mew=1.6, color=BLUE)
    ax.plot([val], [0], "o", color=BLUE, ms=5); ax.set_xlim(lim); ax.set_ylim(-1, 1); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_title(lab + ": " + crit, fontsize=7.5, loc="left", pad=2); ax.tick_params(axis="x", labelsize=7.5)
    if i == 0: ax.text(0.5, 0.2, "95% interval", transform=ax.transAxes, fontsize=7, color=BLUE, ha="center", va="bottom")
    if i == 1:
        ax.annotate("lower 5% bound", (ci[0], 0), xytext=(ci[0], 0.45), fontsize=7, color=BLUE, ha="center", va="bottom")
        ax.annotate("observed", (val, 0), xytext=(val, 0.45), fontsize=7, color=BLUE, ha="center", va="bottom")
fig.text(0.535, 0.955, "(b)", fontweight="bold", fontsize=10, ha="right"); fig.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.14)
save(fig, "Fig8_reliability_audit")

# ---------------- Figure 10: lowest certified cement by gradation family ----------------
fig, ax = plt.subplots(figsize=(7.0, 3.3)); xs = np.arange(len(ALL6)); off = [-0.22, 0, 0.22]
for i, n in enumerate(ALL6):
    rr = DJ[RUN[n]]["all"]
    for f in range(3):
        c = [r["C"] for r in rr if r["family"] == f]
        if c: ax.plot(xs[i] + off[f], min(c), marker="o", ms=7, mfc=FAM[f], mec=FAM[f], ls="none", zorder=3)
    if not rr: ax.text(xs[i], 300, "no accepted\ndesign", ha="center", va="center", fontsize=8, color="#555555", style="italic")
ax.set_xticks(xs); ax.set_xticklabels([LABEL[n] for n in ALL6]); ax.set_xlabel("Target $S^*$/$K^*$ (MPa / mm/s)"); ax.set_ylabel("Lowest accepted cement (kg/m$^3$)"); ax.set_ylim(200, 400)
h = [Line2D([0], [0], marker="o", color="none", mfc=FAM[f], mec=FAM[f], ms=6.5, label=FAMNAME[f].capitalize()) for f in range(3)]
ax.legend(handles=h, ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12), columnspacing=1.4, handletextpad=0.2)
fig.tight_layout(); save(fig, "Fig10_cement_by_family")

# ---------------- Figure 12: reliability vs cement (fine single-size family) ----------------
pal = ["#4c78a8", "#6baed6", "#7aa35f", "#e0a458", "#d95f02"]
fig, ax = plt.subplots(figsize=(6.2, 3.5))
for i, n in enumerate(CERT4 + ["S18_K13"]):
    s = pd.read_csv(SCAN[n]); s = s[s.family == 2].sort_values("cement")
    ax.plot(s.cement, s.best_LB_min_all, marker="o", ms=3.2, lw=1.1, color=pal[i], label=LABEL[n])
ax.axhline(0.90, color="k", lw=0.8, ls="--"); ax.text(500, 0.903, "$R^*=0.90$", ha="right", va="bottom", fontsize=8)
ax.set_xlabel("Cement content (kg/m$^3$)"); ax.set_ylabel("Best $R_{score}$ found at each cement level"); ax.set_ylim(0.0, 1.005); ax.set_xlim(235, 505)
ax.legend(title="$S^*$/$K^*$", title_fontsize=7.5, fontsize=7.5, frameon=False, ncol=1, loc="lower left", bbox_to_anchor=(1.0, 0.0))
fig.tight_layout(); save(fig, "Fig12_reliability_vs_cement")

# ---------------- Figure 13: porosity scatter, nominal and worst shift (certificates of the reported designs, from the run tables) ----------------
def certs(n):
    d = DJ[RUN[n]]["lowest"]; src = d["source"]; key = f"(s{d['seed']}, {d['parent']})"
    for l in open(f"{TAB}/{n}_TABLES.md", encoding="utf-8"):
        if l.startswith(f"| F{d['family']} | {src} | Tier 2") and key in l:
            v = {sd: tuple(map(float, re.search(rf"SD{sd}: ([0-9.]+) / ([0-9.]+)", l).groups())) for sd in ("1.0", "1.5", "2.0")}; return v
    raise SystemExit(f"no T1 row for {n} {key}")
C4 = {n: certs(n) for n in CERT4}; pass
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.2), sharey=True); sds = ["1.0", "1.5", "2.0"]
for i, n in enumerate(CERT4):
    for j in (0, 1): axs[j].plot([float(s_) for s_ in sds], [C4[n][s_][j] for s_ in sds], marker="o", ms=4, lw=1.2, color=pal[i], label=LABEL[n])
for j, t in enumerate(["(a) Nominal gradation", "(b) Worst 2% shift"]):
    axs[j].axhline(0.90, color="k", lw=0.8, ls="--"); axs[j].set_xticks([1.0, 1.5, 2.0]); axs[j].set_xlabel("Achieved-porosity scatter SD (points)"); axs[j].set_title(t, fontsize=9, loc="left", fontweight="bold")
axs[0].set_ylabel("Reliability $R_{score}$"); axs[1].legend(title="$S^*$/$K^*$", title_fontsize=7.5, fontsize=7.5, frameon=False, loc="lower left", bbox_to_anchor=(1.0, 0.0))
fig.tight_layout(); save(fig, "Fig13_porosity_scatter")
print(json.dumps(C4)); print("saved to", os.path.abspath(OUT))
