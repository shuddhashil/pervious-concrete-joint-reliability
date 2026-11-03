"""Authoritative LCM-GP refit: M0 (9D) / M1global (10D) / M1agg (10D+agg group).
One split-fit per invocation (checkpointed). Reuses the authoritative module
(model class, transforms, training, eval) via global patching only for the
feature contract. No other tuning differences.

Usage: python refit_lcmgp.py --config M0|M1global|M1agg --split 0..4
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch

sys.path.insert(0, r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\materials\code\forward_model")
import LCMGP_without_head as L

RUN = r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\research\Stage1_CA_Forward_Refit_20260913\run_01"
SF = os.path.join(RUN, "models", "splitfits"); os.makedirs(SF, exist_ok=True)
XLSX = r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\materials\data\reconciled_phase2v\phase1_reconciled_eligible.xlsx"
SPLITS = r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\materials\results\forward_model_audit\repeated_splits_indices.json"

FEAT9 = ['porosity', 'w/c', 'cem (kg/m3)', 'SP (kg/m3)', 'sand (kg/m3)', 'WAS', 'I_gap', 'Cu', 'Aag(mm^-1)']
FEAT10 = ['porosity', 'w/c', 'cem (kg/m3)', 'CA (kg/m3)', 'SP (kg/m3)', 'sand (kg/m3)', 'WAS', 'I_gap', 'Cu', 'Aag(mm^-1)']
CFG = {'kernel': 'rbf+rq', 'rank': 1, 'use_additive': True, 'add_kernel': 'rbf'}
ITERS, LR, RESTARTS, FAST = 700, 0.02, 2, True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=["M0", "M1global", "M1agg"])
    ap.add_argument("--split", required=True, type=int)
    a = ap.parse_args()
    L.set_seed(20260913 + a.split)
    if a.config in ("M1global", "M1agg"):
        L.FEATURES = list(FEAT10)
        L.CONT_IDXS = list(range(10))
        L.TRANSFORM_RECIPE = {'log_cols': ['cem (kg/m3)', 'CA (kg/m3)', 'Cu'],
                              'log1p_cols': ['SP (kg/m3)', 'sand (kg/m3)', 'Aag(mm^-1)'],
                              'raw_cols': ['porosity', 'w/c', 'WAS', 'I_gap'], 'categorical_cols': []}
        if a.config == "M1agg":
            _orig = L._additive_groups
            def _patched(_o=_orig):
                g = _o(); g['Aggregate'] = list(g['Aggregate']) + [FEAT10.index('CA (kg/m3)')]; return g
            L._additive_groups = _patched
    else:
        assert list(L.FEATURES) == FEAT9, "M0 feature contract changed!"
    X, yS, yP, n0 = L.load_and_prepare_data(XLSX)
    assert X.shape == ((650, 9) if a.config == "M0" else (650, 10)), X.shape
    sp = json.load(open(SPLITS))[f"split_{a.split}"]
    tr = np.asarray(sp["train_idx"], dtype=int); te = np.asarray(sp["test_idx"], dtype=int)
    t0 = time.time()
    rec = L._prepare_split(X, yS, yP, tr, te)
    Xt = torch.as_tensor(rec['Xtr_s'], dtype=L.DTYPE, device=L.DEVICE).contiguous()
    Yt = torch.as_tensor(rec['Ytr_s'], dtype=L.DTYPE, device=L.DEVICE).contiguous()
    Xte = torch.as_tensor(rec['Xte_s'], dtype=L.DTYPE, device=L.DEVICE).contiguous()
    model, lik = L.train_lcmgp(Xt, Yt, CFG, ITERS, LR, RESTARTS, 12345 + a.split,
                               fast_mode=FAST, verbosity=0, label=f"{a.config}_split{a.split}")
    mu_s, sig_s = L.predict(model, lik, Xte, fast_mode=FAST)
    mean = mu_s * rec['y_std'] + rec['y_mean']
    std_raw = L.apply_sigma_floors(sig_s * rec['y_std'])
    Yte = rec['Yte']
    yt_s, yt_p, yp_s, yp_p = Yte[:, 0], Yte[:, 1], mean[:, 0], mean[:, 1]
    ss = np.maximum(std_raw[:, 0], 1e-12); sp_ = np.maximum(std_raw[:, 1], 1e-12)
    metrics = {'rmse_strength': L.rmse(yt_s, yp_s), 'mae_strength': L.mae(yt_s, yp_s),
               'r2_strength': L.r2_score(yt_s, yp_s),
               'cov90_strength_raw_gaussian': L.coverage(yt_s, yp_s, ss, 1.6449),
               'cov95_strength_raw_gaussian': L.coverage(yt_s, yp_s, ss, 1.96),
               'nlpd_strength_raw': L.nlpd_gaussian(yt_s, yp_s, ss),
               'rmse_logperm': L.rmse(yt_p, yp_p), 'mae_logperm': L.mae(yt_p, yp_p),
               'r2_logperm': L.r2_score(yt_p, yp_p),
               'cov90_logperm_raw_gaussian': L.coverage(yt_p, yp_p, sp_, 1.6449),
               'cov95_logperm_raw_gaussian': L.coverage(yt_p, yp_p, sp_, 1.96),
               'nlpd_logperm_raw': L.nlpd_gaussian(yt_p, yp_p, sp_)}
    metrics['nlpd_joint_raw'] = 0.5 * (metrics['nlpd_strength_raw'] + metrics['nlpd_logperm_raw'])
    # hyperparameter snapshot from fitted state
    sd = {k: v.detach().cpu().to(torch.float64).numpy() for k, v in {**model.state_dict(), **lik.state_dict()}.items()}
    hyper = {k: (float(np.mean(v)) if v.size else None) for k, v in sd.items()}
    hyper_detail = {}
    for k, v in sd.items():
        kl = k.lower()
        if 'lengthscale' in kl or 'alpha' in kl or 'noise' in kl or 'outputscale' in kl or 'task' in kl or 'mean' in kl:
            hyper_detail[k] = np.asarray(v).ravel().tolist()
    torch.save({'model_state_dict': model.state_dict(), 'likelihood_state_dict': lik.state_dict()},
               os.path.join(SF, f"{a.config}_split{a.split}_state.pt"))
    dt = time.time() - t0
    pred = {'strength': {'y_true': yt_s, 'y_pred': yp_s, 'y_std_raw': ss},
            'perm': {'y_true': yt_p, 'y_pred': yp_p, 'y_std_raw': sp_},
            'meta': {'X_oos_model': rec['Xte_model'], 'original_row_index': te}}
    # hyperparameter snapshot: retrain-free — recover from a fresh probe? No:
    # eval_split does not return the model. Re-extract by refitting is wasteful.
    # Instead record metrics + preds + provenance here; hyperparams come from
    # the full-data fits and per-split state saved in step5. (Documented.)
    out = dict(config=a.config, split=a.split, n_train=len(tr), n_test=len(te),
               iters=ITERS, lr=LR, restarts=RESTARTS, restart_seed=12345 + a.split,
               fast_mode=FAST, dtype=str(L.DTYPE), device=str(L.DEVICE),
               minutes=dt / 60, metrics=metrics,
               y_true_CS=pred['strength']['y_true'].tolist(), y_pred_CS=pred['strength']['y_pred'].tolist(),
               y_std_raw_CS=pred['strength']['y_std_raw'].tolist(),
               y_true_logQ=pred['perm']['y_true'].tolist(), y_pred_logQ=pred['perm']['y_pred'].tolist(),
               y_std_raw_logQ=pred['perm']['y_std_raw'].tolist(),
               oos_model_features=pred['meta']['X_oos_model'].tolist(),
               original_row_index=pred['meta']['original_row_index'].tolist(),
               hyper_means=hyper, hyper_detail=hyper_detail)
    p = os.path.join(SF, f"{a.config}_split{a.split}.json")
    json.dump(out, open(p, "w"))
    print(f"DONE {a.config} split {a.split} in {dt/60:.1f} min | RMSE_CS={metrics['rmse_strength']:.3f} R2_CS={metrics['r2_strength']:.3f} RMSE_logQ={metrics['rmse_logperm']:.3f} R2_logQ={metrics['r2_logperm']:.3f}")

if __name__ == "__main__":
    main()
