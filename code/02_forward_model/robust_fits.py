"""Robustness-test fits for the A3-1 forward model. Deployed model/engine untouched.
Same 11 inputs, transforms, priors, bounds, kernel (retrain.py config Q5a3), Adam lr 0.02, 700 iters x 2 restarts per split fit and
900 iters x 2 restarts for a full-data fit, same seeds as retrain.py. Only the data/splits change:
  check      random split 0 of the paper (must reproduce ref/Q5a3_split0.json)
  grouped K  5 study-grouped folds (every record of a study in one fold; folds balanced by record count), all 631 records
  noli_full  full-data fit without Li 2021 (613 records)
  noli K     the paper's 5 random splits with the Li 2021 records removed
usage: python robust_fits.py all | check | grouped K | noli_full | noli K | summary
Outputs (resumable; finished fits are skipped): out/<job>.json (+ out/fulldata_Q5a3_noli.pt)."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.dont_write_bytecode = True
DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data")
import numpy as np, pandas as pd, torch
import retrain as R
L = R.L
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
LI = "data source - new - li 2021"


def frame():
    feats, groups = R.activate("Q5a3"); df = pd.read_excel(R.XLSX)
    df["study"] = df["Unnamed: 0"].where(df["Unnamed: 0"].astype(str).str.startswith("data source")).ffill()
    need = L.FEATURES + [L.STRENGTH_COL, L.PERM_COL]
    num = df[need].replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors="coerce"); keep = num.notna().all(axis=1) & num[L.PERM_COL].gt(0)
    sub = df.loc[keep].reset_index(drop=True); assert len(sub) == 631 and sub.study.notna().all()
    X, yS, yP, _ = L.load_and_prepare_data(R.XLSX); assert X.shape == (631, 11)
    assert np.allclose(np.asarray(yS, float), sub[L.STRENGTH_COL].astype(float).values), "row order differs from the model's"
    st = sub.study.values; assert (st == LI).sum() == 18, (st == LI).sum()
    return feats, groups, X, np.asarray(yS, float), np.asarray(yP, float), st


def grouped_splits(st):
    p = os.path.join(DATA, "splits_grouped5.json")
    if os.path.exists(p): return json.load(open(p))
    names, counts = np.unique(st, return_counts=True); order = np.argsort(-counts, kind="stable"); load = np.zeros(5); fold = {}
    for i in order: k = int(np.argmin(load)); fold[names[i]] = k; load[k] += counts[i]
    fs = np.array([fold[s] for s in st])
    out = {f"split_{k}": dict(studies=sorted(n for n in names if fold[n] == k), train_idx=np.flatnonzero(fs != k).tolist(), test_idx=np.flatnonzero(fs == k).tolist()) for k in range(5)}
    json.dump(out, open(p, "w"), indent=1); return out


T_ = lambda z: torch.as_tensor(z, dtype=L.DTYPE, device=L.DEVICE).contiguous()


def split_fit(job, X, yS, yP, tr, te, seed_k, extra):
    p = os.path.join(OUT, f"{job}.json")
    if os.path.exists(p): print("exists", p); return
    t0 = time.time(); L.set_seed(20260913 + seed_k)
    rec = L._prepare_split(X, yS, yP, tr, te)
    model, lik = L.train_lcmgp(T_(rec['Xtr_s']), T_(rec['Ytr_s']), R.CFG, R.ITERS, R.LR, R.RESTARTS, 12345 + seed_k, fast_mode=R.FAST, verbosity=0, label=job)
    mu_s, sig_s = L.predict(model, lik, T_(rec['Xte_s']), fast_mode=R.FAST)
    mean = mu_s * rec['y_std'] + rec['y_mean']; std_raw = L.apply_sigma_floors(sig_s * rec['y_std']); Yte = rec['Yte']
    yt_s, yt_p, yp_s, yp_p = Yte[:, 0], Yte[:, 1], mean[:, 0], mean[:, 1]; ss = np.maximum(std_raw[:, 0], 1e-12); sq = np.maximum(std_raw[:, 1], 1e-12)
    m = {'rmse_strength': L.rmse(yt_s, yp_s), 'r2_strength': L.r2_score(yt_s, yp_s), 'rmse_logperm': L.rmse(yt_p, yp_p), 'r2_logperm': L.r2_score(yt_p, yp_p),
         'cov90_strength_raw_gaussian': L.coverage(yt_s, yp_s, ss, 1.6449), 'cov90_logperm_raw_gaussian': L.coverage(yt_p, yp_p, sq, 1.6449)}
    json.dump(dict(job=job, config="Q5a3", features=list(L.FEATURES), n_train=len(tr), n_test=len(te), minutes=(time.time() - t0) / 60, metrics={k: float(v) for k, v in m.items()},
                   y_true_CS=yt_s.tolist(), y_pred_CS=yp_s.tolist(), y_std_raw_CS=ss.tolist(), y_true_logQ=yt_p.tolist(), y_pred_logQ=yp_p.tolist(), y_std_raw_logQ=sq.tolist(),
                   original_row_index=te.tolist(), hyper=R.hyper(model, lik), **extra), open(p, "w"))
    print(f"DONE {job} {(time.time()-t0)/60:.1f} min | CS RMSE {m['rmse_strength']:.3f} R2 {m['r2_strength']:.3f} | lnk RMSE {m['rmse_logperm']:.3f} R2 {m['r2_logperm']:.3f}", flush=True)


def run(job, k=None):
    feats, groups, X, yS, yP, st = frame(); SPL = json.load(open(R.SPLITS))
    if job == "check":
        sp = SPL["split_0"]; split_fit("check_split0", X, yS, yP, np.asarray(sp["train_idx"]), np.asarray(sp["test_idx"]), 0, {})
        a, b = json.load(open(os.path.join(OUT, "check_split0.json"))), json.load(open(os.path.join(HERE, "ref", "Q5a3_split0.json")))
        d = max(np.abs(np.subtract(a["y_pred_CS"], b["y_pred_CS"])).max(), np.abs(np.subtract(a["y_pred_logQ"], b["y_pred_logQ"])).max() * 10)
        print(f"CHECK vs deployed split 0: max |dCS| or 10*|dlnk| = {d:.4f} (ref RMSE {b['metrics']['rmse_strength']:.3f}, now {a['metrics']['rmse_strength']:.3f})", flush=True)
    elif job == "grouped":
        sp = grouped_splits(st)[f"split_{k}"]
        split_fit(f"grouped_split{k}", X, yS, yP, np.asarray(sp["train_idx"]), np.asarray(sp["test_idx"]), k, dict(studies=sp["studies"], study_of_test=list(st[sp["test_idx"]])))
    elif job == "noli":
        keep = np.flatnonzero(st != LI); pos = {r: i for i, r in enumerate(keep)}; sp = SPL[f"split_{k}"]
        tr = np.array([pos[r] for r in sp["train_idx"] if r in pos]); te = np.array([pos[r] for r in sp["test_idx"] if r in pos])
        split_fit(f"noli_split{k}", X[keep], yS[keep], yP[keep], tr, te, k, dict(original_row_index_631=keep[te].tolist(), kept_rows_631=keep.tolist()))
    elif job == "noli_full":
        p = os.path.join(OUT, "fulldata_Q5a3_noli.pt")
        if os.path.exists(p): print("exists", p); return
        keep = np.flatnonzero(st != LI); Xk = X[keep]; Yk = np.stack([yS[keep], yP[keep]], 1); t0 = time.time(); L.set_seed(20260913)
        xsc = L.XStandardizer().fit(Xk); ysc = L.YStandardizer().fit(Yk)
        model, lik = L.train_lcmgp(T_(xsc.transform(Xk)), T_(ysc.transform(Yk)), R.CFG, 900, 0.02, 2, 12345 + 999999, fast_mode=True, verbosity=0, label="full-noli")
        torch.save({'model': {k_: v.detach().cpu() for k_, v in model.state_dict().items()}, 'lik': {k_: v.detach().cpu() for k_, v in lik.state_dict().items()},
                    'features': feats, 'groups': groups, 'transform_recipe': L.TRANSFORM_RECIPE, 'cfg': R.CFG, 'x_mean': np.asarray(xsc.mean), 'x_std': np.asarray(xsc.std),
                    'y_mean': np.asarray(ysc.mean), 'y_std': np.asarray(ysc.std), 'X_model': Xk, 'Y': Yk, 'kept_rows_631': keep}, p)
        json.dump(R.hyper(model, lik), open(os.path.join(OUT, "fulldata_Q5a3_noli_hyper.json"), "w"), indent=1)
        print(f"DONE noli_full {(time.time()-t0)/60:.1f} min", flush=True)


def summary():
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".json") and "hyper" not in f:
            d = json.load(open(os.path.join(OUT, f))); m = d["metrics"]
            print(f"{f:24s} n_test {d['n_test']:4d} | CS RMSE {m['rmse_strength']:.3f} R2 {m['r2_strength']:.3f} | lnk RMSE {m['rmse_logperm']:.3f} R2 {m['r2_logperm']:.3f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "all":
        run("check")
        for k in range(5): run("grouped", k)
        run("noli_full")
        for k in range(5): run("noli", k)
        summary()
    elif a[0] == "summary": summary()
    else: run(a[0], int(a[1]) if len(a) > 1 else None)
