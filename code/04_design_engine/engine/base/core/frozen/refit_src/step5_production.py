"""Step 5: rebuild 10D production pipeline for M1global (F10).
Full-data fit + pooled OOS + final scaler + calibration bundle + AD +
10D OCSVM/KDE support + versioned oracle .pt + deployment wrapper contract.
Mirrors the accepted main() construction; feature contract is the only change.
"""
import json, os, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch, joblib
sys.path.insert(0, r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\materials\code\forward_model")
sys.path.insert(0, r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\materials\code\feasibility")
import LCMGP_without_head as L
import ocsvm_kde_feasibility_without_head as F

RUN = r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\research\Stage1_CA_Forward_Refit_20260913\run_01"
SF = os.path.join(RUN, "models", "splitfits"); TAB = os.path.join(RUN, "tables"); MOD = os.path.join(RUN, "models")
FEAT10 = ['porosity', 'w/c', 'cem (kg/m3)', 'CA (kg/m3)', 'SP (kg/m3)', 'sand (kg/m3)', 'WAS', 'I_gap', 'Cu', 'Aag(mm^-1)']
RECIPE10 = {'log_cols': ['cem (kg/m3)', 'CA (kg/m3)', 'Cu'],
            'log1p_cols': ['SP (kg/m3)', 'sand (kg/m3)', 'Aag(mm^-1)'],
            'raw_cols': ['porosity', 'w/c', 'WAS', 'I_gap'], 'categorical_cols': []}
L.FEATURES = list(FEAT10); L.CONT_IDXS = list(range(10)); L.TRANSFORM_RECIPE = dict(RECIPE10)
CFG = {'kernel': 'rbf+rq', 'rank': 1, 'use_additive': True, 'add_kernel': 'rbf'}
XLSX = r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\materials\data\reconciled_phase2v\phase1_reconciled_eligible.xlsx"
SPLITS = r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\materials\results\forward_model_audit\repeated_splits_indices.json"

import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--resume", action="store_true")
ARGS = ap.parse_args()
FULL_STATE = os.path.join(MOD, "fulldata_M1global_state.pt")

t0 = time.time()
X, yS, yP, n0 = L.load_and_prepare_data(XLSX)
assert X.shape == (650, 10)
L.set_seed(20260913)
xsc = L.XStandardizer().fit(X); ysc = L.YStandardizer().fit(np.stack([yS, yP], 1))
Xall_s = xsc.transform(X); Yall = np.stack([yS, yP], 1); Yall_s = ysc.transform(Yall)
if ARGS.resume and os.path.exists(FULL_STATE):
    _st = torch.load(FULL_STATE, map_location=L.DEVICE, weights_only=True)
    Xt = torch.as_tensor(Xall_s, dtype=L.DTYPE, device=L.DEVICE).contiguous()
    Yt = torch.as_tensor(Yall_s, dtype=L.DTYPE, device=L.DEVICE).contiguous()
    lik = L._build_likelihood(); model = L.LCMGPModel(Xt, Yt, lik, CFG).to(device=L.DEVICE, dtype=L.DTYPE)
    model.load_state_dict(_st['model']); lik.load_state_dict(_st['lik'])
    Xt, Yt = Xt.detach().cpu(), Yt.detach().cpu()
    print("resumed full-data state from checkpoint")
else:
    Xt = torch.as_tensor(Xall_s, dtype=L.DTYPE, device=L.DEVICE).contiguous()
    Yt = torch.as_tensor(Yall_s, dtype=L.DTYPE, device=L.DEVICE).contiguous()
    model, lik = L.train_lcmgp(Xt, Yt, CFG, 900, 0.02, 2, 12345 + 999999, fast_mode=True, verbosity=1, label='full-data-M1global')
    print(f"full-data fit done in {(time.time()-t0)/60:.1f} min")
    torch.save({'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'lik': {k: v.detach().cpu() for k, v in lik.state_dict().items()}}, FULL_STATE)
sd = {k: v.detach().cpu().to(torch.float64).numpy() for k, v in {**model.state_dict(), **lik.state_dict()}.items()}
json.dump({k: np.asarray(v).ravel().tolist() for k, v in sd.items() if any(s in k.lower() for s in ['lengthscale', 'alpha', 'noise', 'outputscale', 'task', 'mean'])},
          open(os.path.join(TAB, "fulldata_hyperparams.json"), "w"), indent=1)
# task correlation from task kernel: K_task = f f' + diag(v)
ff = np.asarray(sd['shared_covar_module.task_covar_module.covar_factor']).ravel()
vv = np.exp(np.asarray(sd['shared_covar_module.task_covar_module.raw_var']).ravel()) if 'shared_covar_module.task_covar_module.raw_var' in sd else None
print("task covar_factor:", ff, "raw_var:", vv)

# pooled OOS from split fits (eval_split-equivalent records)
preds = []
SPL = json.load(open(SPLITS))
for s in range(5):
    d = json.load(open(os.path.join(SF, f"M1global_split{s}.json")))
    te = np.asarray(SPL[f"split_{s}"]["test_idx"])
    preds.append({'meta': {'original_row_index': te, 'X_oos_model': np.asarray(d['oos_model_features']),
                           'split': np.full(len(te), s), 'split_seed': np.full(len(te), 1000 + s)},
                  'strength': {'y_true': np.asarray(d['y_true_CS']), 'y_pred': np.asarray(d['y_pred_CS']),
                               'y_std_raw': np.asarray(d['y_std_raw_CS']), 'y_std': np.asarray(d['y_std_raw_CS']),
                               'residual': np.asarray(d['y_true_CS']) - np.asarray(d['y_pred_CS'])},
                  'perm': {'y_true': np.asarray(d['y_true_logQ']), 'y_pred': np.asarray(d['y_pred_logQ']),
                           'y_std_raw': np.asarray(d['y_std_raw_logQ']), 'y_std': np.asarray(d['y_std_raw_logQ']),
                           'residual': np.asarray(d['y_true_logQ']) - np.asarray(d['y_pred_logQ'])}})
pooled = L.concat_pooled(preds)
pooled = L.standardize_pooled_oos_features_with_final_scaler(pooled, xsc)
inv_err = L.check_oos_feature_standardization_invariant(pooled, xsc)
print("OOS standardization invariant max abs err:", inv_err)
bundle, calib_diag = L.build_calibration_bundle(pooled)
print("bundle global lower q CS:", bundle['global']['strength']['lower'], "Q:", bundle['global']['perm']['lower'])
oos_df = L.make_pooled_oos_residual_dataframe(pooled, calib_diag)
oos_df.to_csv(os.path.join(MOD, "oos_residual_reference_M1global.csv"), index=False)
ad_reference = L._compute_ad_reference(Xall_s)
print("AD tau:", ad_reference['tau_AD'])
json.dump(bundle, open(os.path.join(MOD, "calibration_bundle_M1global.json"), "w"), indent=1)

# 10D support (same procedure as accepted: auto nu CV + KDE grid CV)
ocsvm, nu_used, ocsvm_cv = F._fit_ocsvm(Xall_s, "auto", 20260913)
ocsvm_train_raw = ocsvm.decision_function(Xall_s).reshape(-1)
ocsvm_scale = float(np.std(ocsvm_train_raw)) or 1.0
kde, bw_used, kde_cv = F._fit_kde(Xall_s)
kde_train_log = kde.score_samples(Xall_s).reshape(-1); kde_log_max = float(np.max(kde_train_log))
artefact = {"ocsvm": ocsvm, "kde": kde, "ocsvm_scale": ocsvm_scale, "kde_log_max": kde_log_max,
            "x_cont_mean": np.asarray(xsc.mean, dtype=float), "x_cont_std": np.asarray(xsc.std, dtype=float),
            "features": list(FEAT10), "cont_idxs": list(range(10)), "transform_recipe": dict(RECIPE10),
            "nu_used": float(nu_used), "bandwidth_used": float(bw_used), "n_train": 650,
            "script_version": "2.0_no_head_CA10", "feature_dim": 10}
joblib.dump(artefact, os.path.join(MOD, "ocsvm_kde_feasibility_CA10.pkl"))
print(f"support: nu={nu_used} bw={bw_used:.4f} scale={ocsvm_scale:.6f}")
scores = F.score_feasibility(Xall_s, artefact)
comb = np.asarray(scores['combined_score'])
print("train combined score: mean", float(np.mean(comb)), "min", float(np.min(comb)),
      "frac>=0.25", float(np.mean(comb >= 0.25)))

save_obj = {'model_state_dict': model.state_dict(), 'likelihood_state_dict': lik.state_dict(),
    'train_x': Xt.detach().cpu(), 'train_y': Yt.detach().cpu(), 'features': list(FEAT10), 'cont_idxs': list(range(10)),
    'x_cont_mean': xsc.mean, 'x_cont_std': xsc.std, 'y_mean': ysc.mean, 'y_std': ysc.std,
    'transform_recipe': dict(RECIPE10), 'best_config': dict(CFG),
    'sigma_floor_s': L.SIGMA_FLOOR_S, 'sigma_floor_p': L.SIGMA_FLOOR_P,
    'calibration_bundle': bundle, 'deployment_seed': 20260913, 'n_total': 650,
    'dtype': str(L.DTYPE).replace('torch.', ''), 'fast_mode': True,
    'prediction_contract_version': 'empirical_local_regime_ad_v2_CA10',
    'official_prediction_function': 'predict_ca_oracle_dataframe (run wrapper; patches 10D contract then delegates)',
    'uncertainty_combine_rule': 'max', 'ad_reference': ad_reference,
    'ad_inflation_start': L.AD_INFLATION_START, 'ad_inflation_eta': L.AD_INFLATION_ETA, 'ad_inflation_cap': L.AD_INFLATION_CAP,
    'oos_residual_reference_available': True,
    'oos_residual_feature_columns': [c for c in oos_df.columns if c.startswith('xstd_')],
    'oos_residual_model_feature_columns': [c for c in oos_df.columns if c.startswith('xmodel_')],
    'oos_residual_features_std': oos_df[[c for c in oos_df.columns if c.startswith('xstd_')]].to_numpy(dtype=float),
    'oos_residual_features_model': oos_df[[c for c in oos_df.columns if c.startswith('xmodel_')]].to_numpy(dtype=float),
    'oos_residual_original_row_index': oos_df['original_row_index'].to_numpy(dtype=int),
    'oos_residual_split': oos_df['split'].to_numpy(dtype=int),
    'oos_residual_split_seed': oos_df['split_seed'].to_numpy(dtype=int),
    'oos_residual_strength': oos_df['strength_residual_mpa'].to_numpy(dtype=float),
    'oos_residual_logperm': oos_df['logperm_residual'].to_numpy(dtype=float),
    'oos_residual_strength_true': oos_df['strength_true_mpa'].to_numpy(dtype=float),
    'oos_residual_strength_pred': oos_df['strength_pred_mpa'].to_numpy(dtype=float),
    'oos_residual_logperm_true': oos_df['logperm_true'].to_numpy(dtype=float),
    'oos_residual_logperm_pred': oos_df['logperm_pred'].to_numpy(dtype=float),
    'local_residual_default_q_neighbors': 10,
    'feature_contract': 'M1global-10D [porosity, w/c, cement, CA(log), SP, sand, WAS, I_gap, Cu, Aag]',
    'ca_transform': 'natural log of CA (kg/m3), standardized with full-data training quantities; applied exactly once',
    'support_artefact': 'ocsvm_kde_feasibility_CA10.pkl (10D; threshold combined>=0.25)',
    'joint_bundle': 'PENDING_STEP6'}
torch.save(save_obj, os.path.join(MOD, "lcmgp_ca_oracle_full_data.pt"))
print("oracle saved. total minutes:", round((time.time() - t0) / 60, 1))
print("STEP5 DONE")
