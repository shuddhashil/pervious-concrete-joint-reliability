"""Held-out coverage of the B8 conformal 90% porosity region, by the held-out row's observed porosity (631-record frame, 15 outer folds)."""
import numpy as np, pandas as pd
fr = pd.read_csv("outputs/frame_631.csv"); h = pd.read_csv("outputs/hdr_detail.csv")
print("folds present:", h.groupby(["seed", "fold"]).ngroups, "rows/fold method B8:", len(h[(h.method == "B8") & (h.version == "conf") & (h.level == 0.9)]))
d = h[(h.method == "B8") & (h.level == 0.9)].merge(fr[["pos", "phi", "study"]], on="pos")
zones = [("<15", -1, 15), ("15-20", 15, 20), ("20-30", 20, 30), ("30-35", 30, 35), (">35", 35, 99)]
rng = np.random.default_rng(0)
for ver in ("conf", "base"):
    x = d[d.version == ver]; print(f"\nB8 {ver} 90% region")
    print(f"all rows: n={len(x)} gate-passed={x.covered.notna().sum()} coverage={x.covered.mean():.3f}")
    for nm, lo, hi in zones + [("window 20-30", 20, 30)]:
        s = x[(x.phi >= lo) & (x.phi < hi if hi < 99 else x.phi >= lo)] if nm != "window 20-30" else x[(x.phi >= 20) & (x.phi <= 30)]
        if len(s) == 0: print(nm, "none"); continue
        st = s.study.unique(); idx = {u: np.flatnonzero(s.study.to_numpy() == u) for u in st}; s = s[s.covered.notna()]; c = s.covered.to_numpy(); bs = []; st = s.study.unique(); idx = {u: np.flatnonzero(s.study.to_numpy() == u) for u in st}
        for _ in range(2000):
            sel = np.concatenate([idx[u] for u in rng.choice(st, len(st))]); bs.append(c[sel].mean())
        print(f"{nm:>13}: n_obs={len(s):5d} rows={s.pos.nunique():4d} studies={len(st):2d} coverage={c.mean():.3f}  study-cluster 95% [{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]")
