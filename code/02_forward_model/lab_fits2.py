"""Path 1 feasibility, second protocol set (2026-10-21). Refits the lab-effect model on study-grouped fold K exactly as lab_fits.labgrp (same seeds),
saves its state, then conditions on trial batches of the held-out laboratory (hyperparameters not refitted):
  near3  the 3 records of the same study NEAREST to the predicted record in standardised input space (trial batches made close to the design)
  rand5  5 random other records of the same study (seed 20261021)
usage: python lab_fits2.py all | K"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.dont_write_bytecode = True
import numpy as np, torch
import lab_fits as LF
RF, R, L = LF.RF, LF.R, LF.L
OUT = LF.OUT


def run(k):
    if os.path.exists(os.path.join(OUT, f"labgrp_rand5_split{k}.json")): return
    feats, groups, X, yS, yP, st = RF.frame(); Y = np.stack([yS, yP], 1); names = sorted(set(st)); sid = np.array([names.index(s) for s in st], float)
    sp = RF.grouped_splits(st)[f"split_{k}"]; tr, te = np.asarray(sp["train_idx"]), np.asarray(sp["test_idx"]); t0 = time.time()
    xsc, ysc, Z = LF.prep(X, Y, sid, tr); new = np.array([LF.NEW0 + names.index(s) for s in st[te]], float)
    model, lik = LF.fit(Z(tr, sid[tr]), ysc.transform(Y[tr]), k, f"labgrp{k}")
    torch.save({'model': model.state_dict(), 'lik': lik.state_dict(), 'x_mean': xsc.mean, 'x_std': xsc.std, 'y_mean': ysc.mean, 'y_std': ysc.std}, os.path.join(OUT, f"labgrp_state_split{k}.pt"))
    meta = dict(lab=LF.lab_B(model, ysc.std), studies=sp["studies"]); Xtr0, Ytr0 = Z(tr, sid[tr]), ysc.transform(Y[tr]); Xs = xsc.transform(X[te]); st_te = st[te]
    rng = np.random.default_rng(20261021 + 100 + k)
    for name in ("near3", "rand5"):
        mt, stt, rows = np.zeros((len(te), 2)), np.zeros((len(te), 2)), []
        for i in range(len(te)):
            pool = np.flatnonzero((st_te == st_te[i]) & (np.arange(len(te)) != i))
            if name == "near3": pick = pool[np.argsort(((Xs[pool] - Xs[i]) ** 2).sum(1), kind="stable")[:3]]
            else: pick = rng.choice(pool, size=min(5, len(pool)), replace=False)
            aug = te[pick]; model.set_train_data(inputs=LF.T_(np.vstack([Xtr0, Z(aug, new[pick])])), targets=LF.T_(np.vstack([Ytr0, ysc.transform(Y[aug])])), strict=False)
            mu, sg = L.predict(model, lik, LF.T_(Z(te[i:i + 1], new[i:i + 1])), fast_mode=R.FAST); m, s = LF.unstd(mu, sg, ysc); mt[i], stt[i] = m[0], s[0]; rows.append(aug.tolist())
        model.set_train_data(inputs=LF.T_(Xtr0), targets=LF.T_(Ytr0), strict=False)
        LF.record(f"labgrp_{name}_split{k}", Y[te], mt, stt, te, dict(meta, trial_rows=rows, minutes=(time.time() - t0) / 60))


if __name__ == "__main__":
    for k in (range(5) if sys.argv[1] == "all" else [int(sys.argv[1])]): run(k)
