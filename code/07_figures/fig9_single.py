"""V5 Figure 9 (2026-10-30): COPY of figures/v4/fig14_alternatives.py with panel (a) only (panel (b) removed); same data.
Original: V4 Figure 14: held-out reliability audit under alternative trial-batch choices (a) and coupled vs independent models (b).
Data: SirReview_MTvsIND_20261029/work/audit/pairs_<model>_<protocol>_paired.csv (P = stated, O = observed)."""
import glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
A = "C:/Users/Shuddhashil/Desktop/Mix Design Paper Rebuild/research/SirReview_MTvsIND_20261029/work/audit"
plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
EDGES = [0, .1, .3, .5, .7, .8, .9, .95, 1.0001]
load = lambda pat: pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{A}/{pat}"))])
def curve(df, nmin=30):
    """Mean stated vs observed per bin; a bin with fewer than nmin comparisons is merged with the bin below it."""
    b = np.digitize(df.P, EDGES) - 1; xs, ys, acc = [], [], []
    for k in sorted(set(b), reverse=True):
        acc.append(df[b == k])
        part = pd.concat(acc)
        if len(part) >= nmin: xs.append(part.P.mean()); ys.append(part.O.mean()); acc = []
    if acc: part = pd.concat(acc); xs.append(part.P.mean()); ys.append(part.O.mean())
    return np.array(xs[::-1]), np.array(ys[::-1])
S = {"near_MT": load("pairs_MT_near3_paired.csv"), "near_IND": load("pairs_IND_near3_paired.csv"),
     "fix_MT": load("pairs_MT_fixed3_[0-4]_paired.csv"), "none_MT": load("pairs_MT_newlab_paired.csv")}
fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.4)); ax = [ax]
for a, series, ttl in ((ax[0], (("near_MT", "Three nearest records (adopted)", "#4e79a7", "o"), ("fix_MT", "Three records at random (five draws)", "#e39e4a", "s"),
                                ("none_MT", "No trial batches", "#8c8c8c", "^")), "(a)"),):
    a.axvspan(0.9, 1.0, color="#e5eedb", zorder=0); a.plot([0, 1], [0, 1], color="k", lw=0.9, zorder=1)
    for k, lab, c, m in series:
        x, y = curve(S[k]); a.plot(x, y, marker=m, color=c, lw=1.2, ms=6, label=lab, zorder=3)
    a.set(xlim=(0, 1), ylim=(0, 1), xlabel="Stated joint pass probability", ylabel="Observed pass rate")
    a.text(0.88, 0.06, "acceptance region\n$P \\geq$ 0.90", ha="right", color="#4f6b34")
    a.legend(loc="upper left", frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "Figure9_single.png"), dpi=300); print("saved")
for k, v in S.items(): print(k, len(v), [tuple(np.round(z, 3)) for z in zip(*curve(v))][-3:])
