"""Build the model + residual-bank files of a variant engine base (sandbox, 2026-10-21). Adapted from GuidedEngine_20261003/tools/build_base.py (A3-1 mode);
the bundle construction (calibration bundle, joint bundle, production scales, paired residual bank) is copied unchanged. Differences:
  G (by-study bank): same full-data model; the out-of-sample predictions come from 5 STUDY-GROUPED fits (Colab out/grouped_split*.json), 631 bank entries.
  N (no Li 2021):    full-data model fitted without Li 2021 (out/fulldata_Q5a3_noli.pt, 613 records); bank from the paper's 5 random splits minus Li 2021.
The support model (OCSVM + KDE) and every input-space gate are left as in base (copied); only models/ and tables/ bank files and the contract's bank block change.
usage: python build_variant.py G|N <variant_dir (copy of GuidedEngine base)> <colab_out_dir>"""
import os, sys, json, hashlib, time
import numpy as np, pandas as pd, torch
MODE, VB, FO = sys.argv[1], os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3]); T = os.path.join(VB, "core")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LCM = os.path.join(REPO, "inputs", "build_variant"); GE = os.path.join(REPO, "code", "04_design_engine", "engine")
INP = os.path.join(GE, "base", "core", "data_inputs"); CSV = os.path.join(INP, "data_with_passing_sizes_A3-1.csv")
COH = pd.read_csv(os.path.join(INP, "phase1_reconciled_eligible.csv")); LI = "data source - new - li 2021"
sys.path.insert(0, os.path.join(T, "frozen", "forward_model"))
import LCMGP_without_head as L
import a3_contract as A
A.install(L)
MOD = os.path.join(T, "models"); TAB = os.path.join(T, "tables"); XLSX = os.path.join(REPO, "data", "data_with_passing_sizes.xlsx")
pd.read_excel = lambda path, *a, **k: pd.read_csv(CSV)
t0 = time.time(); REP = {"mode": MODE}
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()

X631, yS631, yP631, _ = L.load_and_prepare_data(XLSX); assert X631.shape == (631, 11) and len(COH) == 631
studies = COH["_src_ffill"].to_numpy(); assert (studies == LI).sum() == 18
if MODE in ("G", "A", "L", "LT", "LR", "LN", "LNI"):
    # L / LT (Path 1 feasibility, AUDIT ONLY): bank from the lab-effect model's study-grouped predictions, new lab (L) or after 3 trial batches (LT).
    # The oracle written below is still the deployed model, so these variant folders must not be used for certificates.
    full = torch.load(os.path.join(LCM, "models", "fulldata_Q5a3.pt"), map_location="cpu", weights_only=False); ROWS = np.arange(631)
    pre = {"G": "grouped_split", "A": "grouped_split", "L": "labgrp_newlab_split", "LT": "labgrp_trial3_split", "LR": "labrand_split", "LN": "labgrp_near3_split", "LNI": "labgrp_near3icm_split"}[MODE]
    fits = [json.load(open(os.path.join(FO, f"{pre}{s}.json"))) for s in range(5)]; te_of = lambda d: np.asarray(d["original_row_index"], int)
    if MODE == "A":
        # A (author's choice 2026-10-21): a new laboratory first measures M trial batches and corrects its offset. For each held-out record, the offset of its
        # (held-out) study is estimated from M OTHER records of that study (random, fixed seed) as the mean residual in CS and ln k, and added to the prediction;
        # the bank therefore holds the error that remains after the correction, including the error of estimating the offset from M batches.
        M = int(os.environ.get("TRIAL_M", "3")); rng = np.random.default_rng(int(os.environ.get("TRIAL_SEED", "20261021"))); REP.update(trial_M=M)
        for d in fits:
            te = te_of(d); st = studies[te]; rS = np.subtract(d["y_true_CS"], d["y_pred_CS"]); rQ = np.subtract(d["y_true_logQ"], d["y_pred_logQ"])
            dS, dQ = np.zeros(len(te)), np.zeros(len(te))
            for i in range(len(te)):
                pool = np.flatnonzero((st == st[i]) & (np.arange(len(te)) != i)); pick = rng.choice(pool, size=min(M, len(pool)), replace=False)
                dS[i], dQ[i] = rS[pick].mean(), rQ[pick].mean()
            d["y_pred_CS"] = (np.asarray(d["y_pred_CS"]) + dS).tolist(); d["y_pred_logQ"] = (np.asarray(d["y_pred_logQ"]) + dQ).tolist()
elif MODE == "P":
    # P (Path 1 production): oracle = ICM with the lab-effect model's hyperparameters on lab-corrected targets (make_lab_oracle.py; its mean equals the lab
    # model's zero-offset mean exactly); bank = new-laboratory residuals after 3 trial batches of nearby mixes, SDs in this same engine convention.
    full = torch.load(os.path.join(os.path.dirname(FO), "lab_icm_oracle.pt"), map_location="cpu", weights_only=False); ROWS = np.arange(631)
    full = dict(full, model=full["model_icm"]); YS_CORR = np.asarray(full["Y_s_corrected"], float)
    fits = [json.load(open(os.path.join(FO, f"labgrp_near3icm_split{s}.json"))) for s in range(5)]; te_of = lambda d: np.asarray(d["original_row_index"], int)
else:
    full = torch.load(os.path.join(FO, "fulldata_Q5a3_noli.pt"), map_location="cpu", weights_only=False); ROWS = np.asarray(full["kept_rows_631"], int)
    assert np.array_equal(ROWS, np.flatnonzero(studies != LI))
    fits = [json.load(open(os.path.join(FO, f"noli_split{s}.json"))) for s in range(5)]; te_of = lambda d: np.asarray(d["original_row_index_631"], int)
assert list(full["features"]) == A.FEATURES
X = X631[ROWS]; Yall = np.stack([yS631[ROWS], yP631[ROWS]], 1); NROW = len(ROWS)
assert np.allclose(X, full["X_model"], atol=1e-12) and np.allclose(Yall, full["Y"])
xsc = L.XStandardizer().fit(X); ysc = L.YStandardizer().fit(Yall)
assert np.allclose(xsc.mean, full["x_mean"]) and np.allclose(xsc.std, full["x_std"]) and np.allclose(ysc.mean, full["y_mean"]) and np.allclose(ysc.std, full["y_std"])
Xall_s = xsc.transform(X); Yall_s = ysc.transform(Yall)
if MODE == "P": Yall_s = YS_CORR; REP["train_targets"] = "lab-corrected (y - posterior study offset), standardised with the original scaler"
Xt = torch.as_tensor(Xall_s, dtype=L.DTYPE, device=L.DEVICE).contiguous(); Yt = torch.as_tensor(Yall_s, dtype=L.DTYPE, device=L.DEVICE).contiguous()
lik = L._build_likelihood(); model = L.LCMGPModel(Xt, Yt, lik, A.CFG).to(device=L.DEVICE, dtype=L.DTYPE)
print(model.load_state_dict(full["model"])); lik.load_state_dict(full["lik"]); model.eval(); lik.eval()

# pooled out-of-sample predictions (row indices in the 631 frame throughout)
preds = []; TE = []
for s, d in enumerate(fits):
    te = te_of(d); TE.append(te); assert d["features"] == A.FEATURES and len(te) == len(d["y_true_CS"])
    assert np.allclose(d["y_true_CS"], yS631[te]) and np.allclose(d["y_true_logQ"], yP631[te]), "fit rows do not match the frame"
    preds.append({'meta': {'original_row_index': te, 'X_oos_model': X631[te], 'split': np.full(len(te), s), 'split_seed': np.full(len(te), 1000 + s)},
                  'strength': {'y_true': np.asarray(d['y_true_CS']), 'y_pred': np.asarray(d['y_pred_CS']), 'y_std_raw': np.asarray(d['y_std_raw_CS']),
                               'y_std': np.asarray(d['y_std_raw_CS']), 'residual': np.asarray(d['y_true_CS']) - np.asarray(d['y_pred_CS'])},
                  'perm': {'y_true': np.asarray(d['y_true_logQ']), 'y_pred': np.asarray(d['y_pred_logQ']), 'y_std_raw': np.asarray(d['y_std_raw_logQ']),
                           'y_std': np.asarray(d['y_std_raw_logQ']), 'residual': np.asarray(d['y_true_logQ']) - np.asarray(d['y_pred_logQ'])}})
NOOS = int(sum(len(t) for t in TE)); NSTUD = int(len(set(studies[np.concatenate(TE)])))
pooled = L.concat_pooled(preds); pooled = L.standardize_pooled_oos_features_with_final_scaler(pooled, xsc)
REP["oos_standardization_invariant_err"] = float(L.check_oos_feature_standardization_invariant(pooled, xsc))
bundle, calib_diag = L.build_calibration_bundle(pooled)
oos_df = L.make_pooled_oos_residual_dataframe(pooled, calib_diag); oos_df.to_csv(os.path.join(MOD, "oos_residual_reference_M1global.csv"), index=False)
ad_reference = L._compute_ad_reference(Xall_s); REP["tau_AD"] = float(ad_reference["tau_AD"])
json.dump(bundle, open(os.path.join(MOD, "calibration_bundle_M1global.json"), "w"), indent=1)

oc = lambda key: oos_df[[c for c in oos_df.columns if c.startswith(key)]].to_numpy(dtype=float)
save_obj = {'model_state_dict': model.state_dict(), 'likelihood_state_dict': lik.state_dict(), 'train_x': Xt.detach().cpu(), 'train_y': Yt.detach().cpu(),
    'features': list(A.FEATURES), 'cont_idxs': list(range(11)), 'x_cont_mean': xsc.mean, 'x_cont_std': xsc.std, 'y_mean': ysc.mean, 'y_std': ysc.std,
    'transform_recipe': dict(A.TRANSFORM_RECIPE), 'best_config': dict(A.CFG), 'sigma_floor_s': L.SIGMA_FLOOR_S, 'sigma_floor_p': L.SIGMA_FLOOR_P,
    'calibration_bundle': bundle, 'deployment_seed': 20260913, 'n_total': NROW, 'dtype': str(L.DTYPE).replace('torch.', ''), 'fast_mode': True,
    'prediction_contract_version': 'empirical_local_regime_ad_v2_CA10', 'official_prediction_function': 'predict_ca_oracle_dataframe (run wrapper; patches the A3 contract then delegates)',
    'uncertainty_combine_rule': 'max', 'ad_reference': ad_reference, 'ad_inflation_start': L.AD_INFLATION_START, 'ad_inflation_eta': L.AD_INFLATION_ETA,
    'ad_inflation_cap': L.AD_INFLATION_CAP, 'oos_residual_reference_available': True,
    'oos_residual_feature_columns': [c for c in oos_df.columns if c.startswith('xstd_')], 'oos_residual_model_feature_columns': [c for c in oos_df.columns if c.startswith('xmodel_')],
    'oos_residual_features_std': oc('xstd_'), 'oos_residual_features_model': oc('xmodel_'),
    'oos_residual_original_row_index': oos_df['original_row_index'].to_numpy(dtype=int), 'oos_residual_split': oos_df['split'].to_numpy(dtype=int),
    'oos_residual_split_seed': oos_df['split_seed'].to_numpy(dtype=int), 'oos_residual_strength': oos_df['strength_residual_mpa'].to_numpy(dtype=float),
    'oos_residual_logperm': oos_df['logperm_residual'].to_numpy(dtype=float), 'oos_residual_strength_true': oos_df['strength_true_mpa'].to_numpy(dtype=float),
    'oos_residual_strength_pred': oos_df['strength_pred_mpa'].to_numpy(dtype=float), 'oos_residual_logperm_true': oos_df['logperm_true'].to_numpy(dtype=float),
    'oos_residual_logperm_pred': oos_df['logperm_pred'].to_numpy(dtype=float), 'local_residual_default_q_neighbors': 10,
    'feature_contract': A.CONTRACT_NAME, 'ca_transform': 'natural log of CA (kg/m3), standardized with full-data training quantities; applied exactly once',
    'support_artefact': 'ocsvm_kde_feasibility_CA10.pkl (A3 11D; threshold combined>=0.25)', 'joint_bundle': 'PENDING_STEP6', 'variant': MODE}
assert save_obj['oos_residual_features_std'].shape == (NOOS, 11)

# joint bundle (step6), unchanged construction
recs = []
for s, d in enumerate(fits):
    for i, r in enumerate(TE[s]):
        recs.append(dict(split=s, row=int(r), study=studies[r], yCS=d['y_true_CS'][i], muCS=d['y_pred_CS'][i], rawCS=d['y_std_raw_CS'][i],
                         yQ=d['y_true_logQ'][i], muQ=d['y_pred_logQ'][i], rawQ=d['y_std_raw_logQ'][i], Xmodel=X631[r]))
OOS = pd.DataFrame(recs); Xstd = xsc.transform(np.stack(OOS['Xmodel'].to_numpy())); OOS['xs'] = list(Xstd)
dmat = np.sqrt(((Xstd[:, None, :] - Xall_s[None, :, :]) ** 2).sum(-1)); dmat.sort(axis=1)
D = dmat[:, :10].mean(1); tau = float(ad_reference['tau_AD']); infl = np.clip(1 + (D / tau - 0.75) / 0.25, 1, 3)
OOS['D_AD'] = D; OOS['AD_ratio'] = D / tau; OOS['infl'] = infl


def order_stat_q(scores, p):
    sc = np.sort(np.asarray(scores, dtype=float)); n = len(sc); k = int(np.ceil((n + 1) * p))
    if k > n or k < 1: return np.inf, k
    return float(sc[k - 1]), k


Xr = np.stack(OOS['xs'].to_numpy()); rs = (OOS['yCS'] - OOS['muCS']).to_numpy(); rq = (OOS['yQ'] - OOS['muQ']).to_numpy()
locp = L._local_residual_core(Xr, Xr, rs, rq, q=10, exclude_self=True, candidate_original_row_index=OOS['row'].to_numpy(dtype=int),
                              reference_original_row_index=OOS['row'].to_numpy(dtype=int), exclude_same_original_row=True)
sCS_p = np.maximum(OOS['rawCS'].clip(lower=1e-6), locp['strength_residual_std_local_mpa']) * OOS['infl']
sQ_p = np.maximum(OOS['rawQ'].clip(lower=1e-6), locp['logperm_residual_std_local']) * OOS['infl']
Tp = np.maximum((OOS['muCS'] - OOS['yCS']).to_numpy() / sCS_p, (OOS['muQ'] - OOS['yQ']).to_numpy() / sQ_p)
OOS['sCS'] = sCS_p; OOS['sQ'] = sQ_p; OOS['T'] = Tp
jb = {'method': 'paired_max_score', 'scope': f'pooled_oos_A3_11D_variant_{MODE}', 'independent_validation_estimate': False,
      'scales': 'production effective (max(raw,local10,same-row-excluded) x ADinfl)', 'levels': {}}
for p in [0.8, 0.9, 0.95]:
    q, k = order_stat_q(Tp, p); jb['levels'][str(p)] = {'q': float(q), 'k': int(k), 'n_cal': int(len(Tp))}
json.dump(jb, open(os.path.join(MOD, "joint_bundle_M1global.json"), "w"), indent=1); REP["joint_q"] = {p: jb['levels'][p]['q'] for p in jb['levels']}
OOS[['split', 'row', 'study', 'yCS', 'muCS', 'sCS', 'yQ', 'muQ', 'sQ', 'T', 'D_AD', 'AD_ratio', 'infl']].to_csv(os.path.join(MOD, "oos_production_scales_M1global.csv"), index=False)
save_obj['joint_bundle'] = jb; torch.save(save_obj, os.path.join(MOD, "lcmgp_ca_oracle_full_data.pt"))

# paired residual bank (unchanged construction)
res = pd.read_csv(os.path.join(MOD, "oos_residual_reference_M1global.csv")); scl = pd.read_csv(os.path.join(MOD, "oos_production_scales_M1global.csv"))
merged = res.merge(scl[["split", "row", "study"]], left_on=["split", "original_row_index"], right_on=["split", "row"], how="left"); assert int(merged["study"].isna().sum()) == 0
mf = res.merge(scl, left_on=["split", "original_row_index"], right_on=["split", "row"], how="left"); assert float((mf["strength_pred_mpa"] - mf["muCS"]).abs().max()) <= 1e-9
assert len(merged) == NOOS and merged["study"].nunique() == NSTUD
bank = pd.DataFrame({"study_id": merged["study"], "split": merged["split"], "original_row_index": merged["original_row_index"],
                     "y_strength": merged["strength_true_mpa"], "y_logk": merged["logperm_true"], "mu_strength": merged["strength_pred_mpa"], "mu_logk": merged["logperm_pred"],
                     "s_strength": merged["strength_std_oos_effective_mpa"], "s_logk": merged["logperm_std_oos_effective"],
                     "signed_standardised_strength_residual": merged["strength_residual_mpa"] / merged["strength_std_oos_effective_mpa"],
                     "signed_standardised_logk_residual": merged["logperm_residual"] / merged["logperm_std_oos_effective"],
                     "T_diagnostic_max_score": mf["T"], "support_metadata": "not_included_upstream;recomputable_from_xmodel_columns_if_needed"})
bp = os.path.join(TAB, "paired_residual_bank.csv"); bank.to_csv(bp, index=False)
np.savez(os.path.join(TAB, "residual_bank.npz"), study_id=bank["study_id"].to_numpy(), e_strength=bank["signed_standardised_strength_residual"].to_numpy(),
         e_logk=bank["signed_standardised_logk_residual"].to_numpy(), T=bank["T_diagnostic_max_score"].to_numpy())
cpath = os.path.join(T, "UNCERTAINTY_CONTRACT.json"); contract = json.load(open(cpath))
contract["paired_residual_bank"] = {"path": "tables/paired_residual_bank.csv", "sha256": sha(bp), "n_cal": NOOS, "n_studies": NSTUD, "note": f"variant {MODE} (RobustnessTests_20261021)"}
contract["model_contract_hashes"] = {"oracle_pt_sha256": sha(os.path.join(MOD, "lcmgp_ca_oracle_full_data.pt")), "joint_bundle_json_sha256": sha(os.path.join(MOD, "joint_bundle_M1global.json"))}
json.dump(contract, open(cpath, "w"), indent=2)
r = lambda a: float(np.sqrt(np.mean(a ** 2)))
REP.update(n_train=NROW, n_cal=NOOS, n_studies=NSTUD, bank_sha256=sha(bp), oos_rmse_CS=r(rs), oos_rmse_lnk=r(rq),
           std_resid_sd_CS=float(bank["signed_standardised_strength_residual"].std()), std_resid_sd_lnk=float(bank["signed_standardised_logk_residual"].std()),
           minutes=round((time.time() - t0) / 60, 2))
json.dump(REP, open(os.path.join(VB, f"BUILD_REPORT_{MODE}.json"), "w"), indent=1); print("BUILD DONE", json.dumps(REP))
