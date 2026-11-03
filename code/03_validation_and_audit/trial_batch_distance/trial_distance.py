"""How close are the trial batches? For every held-out record, the distance to its three trial records
under nearest-3 and under each fixed-3 (random) selection, on the SAME 526 records (fixed-3 evaluation records). Distance = Euclidean in the
fold's standardised 11 inputs (the space nearest-3 is chosen in, mtind_fits.py line 132), plus plain-unit gaps in cement, porosity and w/c.
Reads the saved trial rows of work/kit/out_mtind/MT_fold*.json; no model run.
usage: python trial_distance.py <work/kit dir> <out_json>"""
import sys, os, json
KIT, OUTJ = sys.argv[1:3]; sys.path.insert(0, KIT); sys.dont_write_bytecode = True
import numpy as np
import robust_fits as RF
L = RF.L
feats, groups, X, yS, yP, st = RF.frame()
iC, iP, iW = feats.index("cem (kg/m3)"), feats.index("porosity"), feats.index("w/c")
cem = np.exp(X[:, iC])                                       # cement enters log-transformed
assert 100 < np.median(cem) < 600, np.median(cem)

rec = {p: [] for p in ["near3"] + [f"fixed3_{r}" for r in range(5)]}
for k in range(5):
    d = json.load(open(os.path.join(KIT, "out_mtind", f"MT_fold{k}.json"))); te = np.asarray(d["rows"]); pr = d["protocols"]
    sp = RF.grouped_splits(st)[f"split_{k}"]; tr = np.asarray(sp["train_idx"]); assert te.tolist() == list(sp["test_idx"])
    xsc = L.XStandardizer().fit(X[tr]); Xs = xsc.transform(X)    # same scaling as mtind_fits.run_fold

    def gaps(i, trial):
        t = np.asarray(trial); dist = np.sqrt(((Xs[t] - Xs[i]) ** 2).sum(1))
        return dict(row=int(i), study=st[i], d_mean=float(dist.mean()), d_max=float(dist.max()), cem=float(np.abs(cem[t] - cem[i]).mean()),
                    por=float(np.abs(X[t, iP] - X[i, iP]).mean()), wc=float(np.abs(X[t, iW] - X[i, iW]).mean()))
    near = {int(te[j]): pr["near3"]["trial_rows"][j] for j in range(len(te))}
    for r in range(5):
        for s_, trial in pr[f"fixed3_{r}"]["trial_rows"].items():
            for i in te[st[te] == s_]:
                if int(i) in trial: continue
                rec[f"fixed3_{r}"].append(gaps(i, trial))
                if r == 0: rec["near3"].append(gaps(i, near[int(i)]))

# typical spacing of recorded mixtures inside a study, for scale: distance from each record to its nearest other record of the same study
nn = []
for k in range(5):
    sp = RF.grouped_splits(st)[f"split_{k}"]; tr = np.asarray(sp["train_idx"]); te = np.asarray(sp["test_idx"]); Xs = L.XStandardizer().fit(X[tr]).transform(X)
    for i in te:
        o = te[(st[te] == st[i]) & (te != i)]
        if len(o): nn.append(float(np.sqrt(((Xs[o] - Xs[i]) ** 2).sum(1)).min()))

out = {}
for p, rows in rec.items():
    a = {m: np.array([x[m] for x in rows]) for m in ("d_mean", "d_max", "cem", "por", "wc")}
    out[p] = dict(n=len(rows), **{f"{m}_median": float(np.median(v)) for m, v in a.items()}, **{f"{m}_p90": float(np.percentile(v, 90)) for m, v in a.items()})
assert len({v["n"] for v in out.values()}) == 1, {p: v["n"] for p, v in out.items()}   # every protocol on the same 526 records
out["same_study_nearest_record_distance_median"] = float(np.median(nn))
json.dump(out, open(OUTJ, "w"), indent=1)
for p, v in out.items():
    if isinstance(v, dict):
        print(f"{p:9s} n {v['n']} | dist mean-of-3 median {v['d_mean_median']:.2f} (p90 {v['d_mean_p90']:.2f}) max-of-3 median {v['d_max_median']:.2f}"
              f" | |dCement| {v['cem_median']:.1f} kg/m3 | |dPorosity| {v['por_median']:.2f} pts | |dw/c| {v['wc_median']:.3f}")
print("median distance to nearest other record of same study:", round(out["same_study_nearest_record_distance_median"], 2))
