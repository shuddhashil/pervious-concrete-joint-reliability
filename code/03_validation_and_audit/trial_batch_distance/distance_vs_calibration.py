"""Random-three audit pairs with stated P >= 0.90, split by how far the record's three random trial records were (mean standardised distance,
from trial_distance.py logic): observed vs stated pass rate per distance band. MT model, paired residuals, all five draws pooled.
usage: python distance_vs_calibration.py <work dir>"""
import sys, os, json
W = sys.argv[1]; sys.path.insert(0, os.path.join(W, "kit")); sys.dont_write_bytecode = True
import numpy as np, pandas as pd
import robust_fits as RF
L = RF.L; feats, groups, X, yS, yP, st = RF.frame()
rows = []
for k in range(5):
    d = json.load(open(os.path.join(W, "kit", "out_mtind", f"MT_fold{k}.json"))); te = np.asarray(d["rows"])
    tr = np.asarray(RF.grouped_splits(st)[f"split_{k}"]["train_idx"]); Xs = L.XStandardizer().fit(X[tr]).transform(X)
    for r in range(5):
        for s_, t in d["protocols"][f"fixed3_{r}"]["trial_rows"].items():
            for i in te[st[te] == s_]:
                if int(i) not in t: rows.append(dict(sel=r, row=int(i), dist=float(np.sqrt(((Xs[t] - Xs[i]) ** 2).sum(1)).mean())))
D = pd.DataFrame(rows)
P = pd.concat([pd.read_csv(os.path.join(W, "audit", f"pairs_MT_fixed3_{r}_paired.csv")).assign(sel=r) for r in range(5)])
M = P.merge(D, on=["sel", "row"]); assert len(M) == len(P), (len(M), len(P))
A = M[M.P >= 0.90]
for lo, hi in [(0, 1.2), (1.2, 2.0), (2.0, 3.0), (3.0, 99), (0, 2.0)]:
    b = A[(A.dist >= lo) & (A.dist < hi)]
    print(f"distance {lo}-{hi}: pairs {len(b):4d} records {b.row.nunique():3d} studies {b.study.nunique():2d} | observed {b.O.mean():.3f} stated {b.P.mean():.3f}")
