"""Step 6+7: paired max-score joint screen from M1global production scales + target diagnostics.
Merges deployable joint bundle into the oracle .pt."""
import json, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch
sys.path.insert(0, r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\materials\code\forward_model")
import LCMGP_without_head as L

RUN = r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\research\Stage1_CA_Forward_Refit_20260913\run_01"
SF = os.path.join(RUN, "models", "splitfits"); TAB = os.path.join(RUN, "tables"); MOD = os.path.join(RUN, "models")
FEAT10 = ['porosity', 'w/c', 'cem (kg/m3)', 'CA (kg/m3)', 'SP (kg/m3)', 'sand (kg/m3)', 'WAS', 'I_gap', 'Cu', 'Aag(mm^-1)']
L.FEATURES = list(FEAT10); L.CONT_IDXS = list(range(10))
L.TRANSFORM_RECIPE = {'log_cols': ['cem (kg/m3)', 'CA (kg/m3)', 'Cu'],
                      'log1p_cols': ['SP (kg/m3)', 'sand (kg/m3)', 'Aag(mm^-1)'],
                      'raw_cols': ['porosity', 'w/c', 'WAS', 'I_gap'], 'categorical_cols': []}
XLSX = r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\materials\data\reconciled_phase2v\phase1_reconciled_eligible.xlsx"
SPLITS = r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\materials\results\forward_model_audit\repeated_splits_indices.json"
COH = pd.read_csv(r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\materials\data\reconciled_phase2v\phase1_reconciled_eligible.csv")
X, yS, yP, _ = L.load_and_prepare_data(XLSX)
SPL = json.load(open(SPLITS))
studies = COH["_src_ffill"].to_numpy()

# OOS table from split fits
recs = []
for s in range(5):
    d = json.load(open(os.path.join(SF, f"M1global_split{s}.json")))
    te = np.asarray(SPL[f"split_{s}"]["test_idx"])
    for i, r in enumerate(te):
        recs.append(dict(split=s, row=int(r), study=studies[r],
                         yCS=d['y_true_CS'][i], muCS=d['y_pred_CS'][i], rawCS=d['y_std_raw_CS'][i],
                         yQ=d['y_true_logQ'][i], muQ=d['y_pred_logQ'][i], rawQ=d['y_std_raw_logQ'][i],
                         Xmodel=np.asarray(d['oos_model_features'][i])))
OOS = pd.DataFrame(recs)
xsc = L.XStandardizer().fit(X)
Xstd = xsc.transform(np.stack(OOS['Xmodel'].to_numpy()))
OOS['xs'] = list(Xstd)
# final train reference for AD
Xtr_s = xsc.transform(X)
dmat = np.sqrt(((Xstd[:, None, :] - Xtr_s[None, :, :]) ** 2).sum(-1))
dmat.sort(axis=1)
D = dmat[:, :10].mean(1)
ad_ref = L._compute_ad_reference(Xtr_s); tau = float(ad_ref['tau_AD'])
infl = np.clip(1 + (D / tau - 0.75) / 0.25, 1, 3)
OOS['D_AD'] = D; OOS['AD_ratio'] = D / tau; OOS['infl'] = infl
print("AD tau(final,10D)=", round(tau, 4), "mean infl=", round(float(infl.mean()), 3))

def order_stat_q(scores, p):
    sc = np.sort(np.asarray(scores, dtype=float)); n = len(sc)
    k = int(np.ceil((n + 1) * p))
    if k > n or k < 1: return np.inf, k
    return float(sc[k - 1]), k

# local residuals per held-out split (reference = calibration rows only, same-row excluded)
OOS['locCS'] = np.nan; OOS['locQ'] = np.nan
per_split_loc = {}
for h in range(5):
    te_m = (OOS['split'] == h).to_numpy()
    te_rows = set(OOS[te_m]['row'].astype(int))
    cal_m = (~te_m) & (~OOS['row'].astype(int).isin(te_rows))
    ref_idx = np.flatnonzero(cal_m.to_numpy())
    Xr = np.stack(OOS.iloc[ref_idx]['xs'].to_numpy())
    rs = (OOS.iloc[ref_idx]['yCS'] - OOS.iloc[ref_idx]['muCS']).to_numpy()
    rq = (OOS.iloc[ref_idx]['yQ'] - OOS.iloc[ref_idx]['muQ']).to_numpy()
    cand = np.stack(OOS['xs'].to_numpy())
    loc = L._local_residual_core(cand, Xr, rs, rq, q=10, exclude_self=False,
                                 candidate_original_row_index=OOS['row'].to_numpy(dtype=int),
                                 reference_original_row_index=OOS.iloc[ref_idx]['row'].to_numpy(dtype=int),
                                 exclude_same_original_row=True)
    per_split_loc[h] = loc
    OOS.loc[te_m, 'locCS'] = loc['strength_residual_std_local_mpa'][te_m]
    OOS.loc[te_m, 'locQ'] = loc['logperm_residual_std_local'][te_m]
OOS['sCS'] = np.maximum(OOS['rawCS'].clip(lower=1e-6), OOS['locCS'].fillna(0)) * OOS['infl']
OOS['sQ'] = np.maximum(OOS['rawQ'].clip(lower=1e-6), OOS['locQ'].fillna(0)) * OOS['infl']
OOS['T'] = np.maximum((OOS['muCS'] - OOS['yCS']) / OOS['sCS'], (OOS['muQ'] - OOS['yQ']) / OOS['sQ'])
print("mean sCS:", round(float(OOS['sCS'].mean()), 3), "mean sQ:", round(float(OOS['sQ'].mean()), 4))

fold_rows, study_rows, diag_rows = [], [], []
TARGETS = [(18.0, np.log(13.0)), (12.0, np.log(5.0))]
for h in range(5):
    te_m = (OOS['split'] == h).to_numpy()
    te_rows = set(OOS[te_m]['row'].astype(int))
    cal_m = ((~te_m) & (~OOS['row'].astype(int).isin(te_rows))).to_numpy()
    loc = per_split_loc[h]
    sCS_h = np.maximum(OOS['rawCS'].clip(lower=1e-6).to_numpy(), loc['strength_residual_std_local_mpa']) * OOS['infl'].to_numpy()
    sQ_h = np.maximum(OOS['rawQ'].clip(lower=1e-6).to_numpy(), loc['logperm_residual_std_local']) * OOS['infl'].to_numpy()
    T_h = np.maximum((OOS['muCS'] - OOS['yCS']).to_numpy() / sCS_h, (OOS['muQ'] - OOS['yQ']).to_numpy() / sQ_h)
    Teval = OOS[te_m]
    sCS_e, sQ_e = sCS_h[te_m], sQ_h[te_m]
    Tcal = T_h[cal_m]
    for p in [0.8, 0.9, 0.95]:
        q, k = order_stat_q(Tcal, p)
        n_cal = int(cal_m.sum())
        LBcs = Teval['muCS'].to_numpy() - q * sCS_e; LBq = Teval['muQ'].to_numpy() - q * sQ_e
        yCS = Teval['yCS'].to_numpy(); yQ = Teval['yQ'].to_numpy()
        joint = ((yCS >= LBcs) & (yQ >= LBq)).astype(float)
        mcs = (yCS >= LBcs).astype(float); mq = (yQ >= LBq).astype(float)
        fold_rows.append(dict(held_split=h, n_cal=n_cal, p=p, k=k, q=round(float(q), 4),
                              joint_cov=round(float(joint.mean()), 4), marg_CS=round(float(mcs.mean()), 4),
                              marg_Q=round(float(mq.mean()), 4),
                              pen_CS_mean=round(float((q * sCS_e).mean()), 3),
                              pen_Q_mean=round(float((q * sQ_e).mean()), 4)))
        t2 = Teval.copy(); t2['j'] = joint
        for st, dd in t2.groupby('study'):
            study_rows.append(dict(held_split=h, p=p, study=st, n=len(dd), joint_cov=round(float(dd['j'].mean()), 4)))
        for (ss_, qq_) in TARGETS:
            obs = ((yCS >= ss_) & (np.exp(yQ) >= np.exp(qq_)))
            scr = ((LBcs >= ss_) & (LBq >= qq_))
            tp, fp, fn, tn = int((scr & obs).sum()), int((scr & ~obs).sum()), int(((~scr) & obs).sum()), int(((~scr) & (~obs)).sum())
            diag_rows.append(dict(held_split=h, p=p, s_star=ss_, k_star=round(float(np.exp(qq_)), 2),
                                  n_screened=int(scr.sum()), TP=tp, FP=fp, FN=fn, TN=tn,
                                  FAR_in=round(fp / max(1, tp + fp), 3), FRR=round(fn / max(1, tp + fn), 3)))
FOLD = pd.DataFrame(fold_rows); FOLD.to_csv(os.path.join(TAB, "joint_fold_metrics.csv"), index=False)
pd.DataFrame(study_rows).to_csv(os.path.join(TAB, "joint_study_metrics.csv"), index=False)
DIAG = pd.DataFrame(diag_rows); DIAG.to_csv(os.path.join(TAB, "TARGET_DIAGNOSTICS.csv"), index=False)
print(FOLD.to_string())
SM = pd.DataFrame(study_rows)
print(SM.groupby('p')['joint_cov'].agg(['mean', 'min', 'count']).round(4).to_string())
print(DIAG.groupby(['p', 's_star', 'k_star'])[['n_screened', 'TP', 'FP', 'FN']].sum().to_string())

# deployable pooled joint bundle (mirrors marginal-bundle deployment: pooled fit + cross-fitted evidence)
# pooled locals with full-pool reference, same-row excluded
Xr = np.stack(OOS['xs'].to_numpy()); rs = (OOS['yCS'] - OOS['muCS']).to_numpy(); rq = (OOS['yQ'] - OOS['muQ']).to_numpy()
locp = L._local_residual_core(Xr, Xr, rs, rq, q=10, exclude_self=True,
                              candidate_original_row_index=OOS['row'].to_numpy(dtype=int),
                              reference_original_row_index=OOS['row'].to_numpy(dtype=int),
                              exclude_same_original_row=True)
sCS_p = np.maximum(OOS['rawCS'].clip(lower=1e-6), locp['strength_residual_std_local_mpa']) * OOS['infl']
sQ_p = np.maximum(OOS['rawQ'].clip(lower=1e-6), locp['logperm_residual_std_local']) * OOS['infl']
Tp = np.maximum((OOS['muCS'] - OOS['yCS']).to_numpy() / sCS_p, (OOS['muQ'] - OOS['yQ']).to_numpy() / sQ_p)
OOS['sCS'] = sCS_p; OOS['sQ'] = sQ_p; OOS['T'] = Tp
jb = {'method': 'paired_max_score', 'scope': 'pooled_repeated_split_oos_M1global_10D',
      'independent_validation_estimate': False, 'scales': 'production effective (max(raw,local10,same-row-excluded) x ADinfl)',
      'levels': {}}
for p in [0.8, 0.9, 0.95]:
    q, k = order_stat_q(Tp, p)
    jb['levels'][str(p)] = {'q': float(q), 'k': int(k), 'n_cal': int(len(Tp))}
json.dump(jb, open(os.path.join(MOD, "joint_bundle_M1global.json"), "w"), indent=1)
print("deployable pooled q:", {p: jb['levels'][str(p)]['q'] for p in ['0.8', '0.9', '0.95']})
OOS[['split', 'row', 'study', 'yCS', 'muCS', 'sCS', 'yQ', 'muQ', 'sQ', 'T', 'D_AD', 'AD_ratio', 'infl']].to_csv(
    os.path.join(MOD, "oos_production_scales_M1global.csv"), index=False)
# merge into oracle pt
pt = os.path.join(MOD, "lcmgp_ca_oracle_full_data.pt")
obj = torch.load(pt, map_location="cpu", weights_only=False)
obj['joint_bundle'] = jb
torch.save(obj, pt)
print("joint bundle merged into oracle. STEP6 DONE")
