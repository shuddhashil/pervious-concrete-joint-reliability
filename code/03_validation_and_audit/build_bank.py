"""Model + residual-bank files of a variant engine base (2026-10-29). Port of RobustnessTests_20261021/build_variant.py mode P: the calibration bundle,
pooled out-of-sample reference, production scales, joint bundle and paired residual bank are built by the same code; only the 5 fold prediction
sets change. The oracle .pt keeps the paper engine's ICM state and support artefacts; its out-of-sample arrays, bundles and bank are replaced.
The engine's GP part for the IND model is supplied separately (design_eval.py), so an IND variant is used through that script only.
usage: python build_bank.py <engine_root> <variant_name> <kind MT|IND> <protocol newlab|near3|fixed3_r> <mtind_out_dir>
       python build_bank.py <engine_root> P_check legacy near3icm <dir with labgrp_near3icm_split*.json>   (must reproduce the paper bank)"""
import os, sys, json, hashlib, shutil, time
import numpy as np, pandas as pd, torch
ROOT, NAME, KIND, PROT, FO = os.path.abspath(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], os.path.abspath(sys.argv[5])
BASE = os.path.join(ROOT, "base"); VB = os.path.join(os.path.dirname(ROOT), "variants", NAME)
if os.path.exists(VB): shutil.rmtree(VB)
ign = shutil.ignore_patterns("__pycache__", "*.pyc")
shutil.copytree(os.path.join(BASE, "core"), os.path.join(VB, "core"), ignore=ign); shutil.copytree(os.path.join(BASE, "defengine"), os.path.join(VB, "defengine"), ignore=ign)
T = os.path.join(VB, "core"); MOD = os.path.join(T, "models"); TAB = os.path.join(T, "tables")
CSV = os.path.join(T, "data_inputs", "data_with_passing_sizes_A3-1.csv"); COH = pd.read_csv(os.path.join(T, "data_inputs", "phase1_reconciled_eligible.csv"))
sys.path.insert(0, os.path.join(T, "frozen", "forward_model"))
import LCMGP_without_head as L
import a3_contract as A
A.install(L)
pd.read_excel = lambda path, *a, **k: pd.read_csv(CSV)
t0 = time.time(); REP = {"variant": NAME, "kind": KIND, "protocol": PROT}
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()

X631, yS631, yP631, _ = L.load_and_prepare_data("data_with_passing_sizes.xlsx"); assert X631.shape == (631, 11) and len(COH) == 631
studies = COH["_src_ffill"].to_numpy()
if KIND == "legacy":
    fits = [json.load(open(os.path.join(FO, f"labgrp_{PROT}_split{s}.json"))) for s in range(5)]
else:
    fits = []
    for k in range(5):
        d = json.load(open(os.path.join(FO, f"{KIND}_fold{k}.json"))); pr = d["protocols"][PROT]; ev = np.asarray(pr["eval"], int)
        mu, sde = np.asarray(pr["mu"], float)[ev], np.asarray(d["sd_eng"], float)[ev]; assert (mu > -900).all()
        fits.append(dict(original_row_index=np.asarray(d["rows"], int)[ev].tolist(), y_true_CS=np.asarray(d["y_CS"])[ev].tolist(), y_true_logQ=np.asarray(d["y_lnk"])[ev].tolist(),
                         y_pred_CS=mu[:, 0].tolist(), y_pred_logQ=mu[:, 1].tolist(), y_std_raw_CS=sde[:, 0].tolist(), y_std_raw_logQ=sde[:, 1].tolist()))
te_of = lambda d: np.asarray(d["original_row_index"], int)
full = torch.load(os.path.join(MOD, "lcmgp_ca_oracle_full_data.pt"), map_location="cpu", weights_only=False)
xsc = L.XStandardizer().fit(X631); assert np.allclose(xsc.mean, full["x_cont_mean"]) and np.allclose(xsc.std, full["x_cont_std"])
Xall_s = xsc.transform(X631)

preds = []; TE = []
for s, d in enumerate(fits):
    te = te_of(d); TE.append(te); assert len(te) == len(d["y_true_CS"])
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
ad_reference = L._compute_ad_reference(Xall_s); assert abs(float(ad_reference["tau_AD"]) - float(full["ad_reference"]["tau_AD"])) < 1e-9
json.dump(bundle, open(os.path.join(MOD, "calibration_bundle_M1global.json"), "w"), indent=1)
oc = lambda key: oos_df[[c for c in oos_df.columns if c.startswith(key)]].to_numpy(dtype=float)
save_obj = dict(full)
save_obj.update({'calibration_bundle': bundle, 'oos_residual_feature_columns': [c for c in oos_df.columns if c.startswith('xstd_')],
    'oos_residual_model_feature_columns': [c for c in oos_df.columns if c.startswith('xmodel_')], 'oos_residual_features_std': oc('xstd_'), 'oos_residual_features_model': oc('xmodel_'),
    'oos_residual_original_row_index': oos_df['original_row_index'].to_numpy(dtype=int), 'oos_residual_split': oos_df['split'].to_numpy(dtype=int),
    'oos_residual_split_seed': oos_df['split_seed'].to_numpy(dtype=int), 'oos_residual_strength': oos_df['strength_residual_mpa'].to_numpy(dtype=float),
    'oos_residual_logperm': oos_df['logperm_residual'].to_numpy(dtype=float), 'oos_residual_strength_true': oos_df['strength_true_mpa'].to_numpy(dtype=float),
    'oos_residual_strength_pred': oos_df['strength_pred_mpa'].to_numpy(dtype=float), 'oos_residual_logperm_true': oos_df['logperm_true'].to_numpy(dtype=float),
    'oos_residual_logperm_pred': oos_df['logperm_pred'].to_numpy(dtype=float), 'variant': f"{NAME} ({KIND}/{PROT}, SirReview_MTvsIND_20261029)"})
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
jb = {'method': 'paired_max_score', 'scope': f'pooled_oos_A3_11D_variant_{NAME}', 'independent_validation_estimate': False,
      'scales': 'production effective (max(raw,local10,same-row-excluded) x ADinfl)', 'levels': {}}
for p in [0.8, 0.9, 0.95]:
    q, k = order_stat_q(Tp, p); jb['levels'][str(p)] = {'q': float(q), 'k': int(k), 'n_cal': int(len(Tp))}
json.dump(jb, open(os.path.join(MOD, "joint_bundle_M1global.json"), "w"), indent=1)
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
contract["paired_residual_bank"] = {"path": "tables/paired_residual_bank.csv", "sha256": sha(bp), "n_cal": NOOS, "n_studies": NSTUD, "note": f"variant {NAME} (SirReview_MTvsIND_20261029)"}
contract["model_contract_hashes"] = {"oracle_pt_sha256": sha(os.path.join(MOD, "lcmgp_ca_oracle_full_data.pt")), "joint_bundle_json_sha256": sha(os.path.join(MOD, "joint_bundle_M1global.json"))}
json.dump(contract, open(cpath, "w"), indent=2)
r = lambda a: float(np.sqrt(np.mean(a ** 2)))
REP.update(n_cal=NOOS, n_studies=NSTUD, bank_sha256=sha(bp), oos_rmse_CS=r(rs), oos_rmse_lnk=r(rq),
           std_resid_sd_CS=float(bank["signed_standardised_strength_residual"].std()), std_resid_sd_lnk=float(bank["signed_standardised_logk_residual"].std()), minutes=round((time.time() - t0) / 60, 2))
if NAME == "P_check":
    b0 = pd.read_csv(os.path.join(BASE, "core", "tables", "paired_residual_bank.csv")); num = [c for c in b0.columns if b0[c].dtype != object]
    REP["P_check_max_abs_diff_vs_paper_bank"] = float(np.abs(b0[num].to_numpy(float) - bank[num].to_numpy(float)).max()); REP["P_check_same_rows"] = bool((b0["study_id"] == bank["study_id"]).all())
json.dump(REP, open(os.path.join(VB, "BUILD_REPORT.json"), "w"), indent=1); print("BUILD DONE", json.dumps(REP), flush=True)
