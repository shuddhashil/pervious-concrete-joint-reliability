"""All validation results so far in one table and one figure (no new model runs; reads saved predictions).
Inside the 631-record data: random 80/20 and study-grouped 5-fold (new lab / 3 nearest same-study records).
Outside studies (not in training): TiDE, RIENGG ( TiDE per-mould amounts), Mahalingam 2016, Meininger 1988 (Sonebi dropped by the author)."""
import json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUT = os.path.join(REPO, "results", "forward_model_folds", "lab_effect")
AN = os.path.join(REPO, "results")


def pool(prefix):
    a = [[], [], [], []]
    for i in range(5):
        d = json.load(open(f"{OUT}/{prefix}{i}.json"))
        for lst, key in zip(a, ("y_true_CS", "y_pred_CS", "y_true_logQ", "y_pred_logQ")): lst += d[key]
    return [np.array(x) for x in a]


def stats(y, p):
    e = p - y
    return dict(n=len(y), rmse=np.sqrt((e ** 2).mean()), mae=np.abs(e).mean(), r2=1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(), sd=y.std(ddof=1))


rows = []
def add(group, name, mode, cs, lk):
    rows.append(dict(group=group, data=name, mode=mode, n=cs["n"], cs_sd=cs["sd"], cs_rmse=cs["rmse"], cs_mae=cs["mae"], cs_r2=cs["r2"],
                     lnk_sd=lk["sd"], lnk_rmse=lk["rmse"], lnk_mae=lk["mae"], lnk_r2=lk["r2"]))

# inside the 631 records
yc, pc, yk, pk = pool("labrand_split"); add("Within the 631-record data", "Random 80/20 splits (5)", "lab seen in training", stats(yc, pc), stats(yk, pk))
BG = {}
for mode, pre in (("new lab", "labgrp_newlab_split"), ("3 nearest", "labgrp_near3_split")):
    yc, pc, yk, pk = pool(pre); BG[mode] = (yc, pc, yk, pk)
    add("Within the 631-record data", "Study-grouped 5-fold", mode, stats(yc, pc), stats(yk, pk))

# outside studies
own = pd.read_csv(os.path.join(AN, "outside_studies", "pred_mould.csv"))
bk = pd.read_csv(os.path.join(AN, "outside_studies", "pred_book1.csv")); bk = bk[bk.lab != "Sonebi"]
c = ["lab", "cs", "k", "cs_pred_newlab", "lnk_pred_newlab", "cs_pred_near3", "lnk_pred_near3"]
A = pd.concat([own[c], bk[c]], ignore_index=True)
NAMES = {"TiDE": "Ghosh et al. (2026)", "RIENGG": "Ghosh and Singh (2026a)", "Mahalingam": "Mahalingam and Mahalingam (2016)", "Meininger": "Meininger (1988)"}
for lab in ("TiDE", "RIENGG", "Mahalingam", "Meininger", "ALL"):
    x = A if lab == "ALL" else A[A.lab == lab]
    for mode, key in (("new lab", "newlab"), ("3 nearest", "near3")):
        add("Outside studies", "All four pooled" if lab == "ALL" else NAMES[lab], mode,
            stats(x.cs.values, x["cs_pred_" + key].values), stats(np.log(x.k.values), x["lnk_pred_" + key].values))
T = pd.DataFrame(rows)
T.round(3).to_csv(f"{HERE}/VALIDATION_TABLE.csv", index=False)

# markdown table
f2 = lambda v: f"{v:.2f}"; f3 = lambda v: f"{v:.3f}"
md = ["| Data | Mode | n | CS spread (SD, MPa) | CS RMSE | CS MAE | CS R2 | ln k spread (SD) | ln k RMSE | ln k MAE | ln k R2 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for g, sub in T.groupby("group", sort=False):
    md.append(f"| **{g}** | | | | | | | | | | |")
    for _, r in sub.iterrows():
        md.append(f"| {r.data} | {r['mode']} | {r.n} | {f2(r.cs_sd)} | {f2(r.cs_rmse)} | {f2(r.cs_mae)} | {f2(r.cs_r2)} | {f2(r.lnk_sd)} | {f3(r.lnk_rmse)} | {f3(r.lnk_mae)} | {f2(r.lnk_r2)} |")
open(f"{HERE}/VALIDATION_TABLE.md", "w", encoding="utf8").write("\n".join(md) + "\n")

# figure: measured vs predicted, rows = strength / permeability, columns = new lab / 3 nearest
COL = {"TiDE": "#E69F00", "RIENGG": "#009E73", "Mahalingam": "#0072B2", "Meininger": "#D55E00"}
MK = {"TiDE": "o", "RIENGG": "s", "Mahalingam": "^", "Meininger": "D"}
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e6e6e6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8})
fig, ax = plt.subplots(2, 2, figsize=(8.2, 7.4))
for j, (mode, key) in enumerate((("new lab", "newlab"), ("3 nearest", "near3"))):
    yc, pc, yk, pk = BG[mode]
    for i, (ytrue, ypred, label, lo, hi) in enumerate(((yc, pc, "compressive strength (MPa)", 0, 70), (np.exp(yk), np.exp(pk), "permeability k (mm/s)", 0.1, 60))):
        a = ax[i, j]; a.set_axisbelow(True); a.grid(color=GRID, lw=0.6)
        a.scatter(ytrue, ypred, s=9, color="#b8b8b8", lw=0, alpha=0.8, label="Study-grouped 5-fold, 631 records", zorder=1)
        for lab in COL:
            x = A[A.lab == lab]
            yt = x.cs if i == 0 else x.k
            yp = x["cs_pred_" + key] if i == 0 else np.exp(x["lnk_pred_" + key])
            a.scatter(yt, yp, s=26, marker=MK[lab], color=COL[lab], edgecolor="white", lw=0.6, label=NAMES[lab], zorder=3)
        a.plot([lo, hi], [lo, hi], color=INK, lw=0.9, ls="--", zorder=2)
        a.set_xlim(lo, hi); a.set_ylim(lo, hi)
        if i == 1: a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel("Measured " + label); a.set_ylabel("Predicted " + label)
        a.set_title(("New lab (no trial batches)" if j == 0 else "3 nearest same-study mixtures") if i == 0 else "", fontsize=9.5, color=INK, loc="left")
        a.text(0.03, 0.97, "abcd"[i * 2 + j] + ")", transform=a.transAxes, va="top", fontweight="bold", color=INK)
        for s in ("top", "right"): a.spines[s].set_visible(False)
        a.set_aspect("equal", adjustable="box")
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8.5, handletextpad=0.4, columnspacing=1.4, markerscale=1.2)
fig.tight_layout(rect=(0, 0.10, 1, 1))
for ext in ("png", "pdf", "svg"): fig.savefig(f"{HERE}/validation_pred_vs_measured.{ext}", dpi=300, bbox_inches="tight")
print(open(f"{HERE}/VALIDATION_TABLE.md").read())
