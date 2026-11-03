"""Step 3: supported CA-response audit with fitted M1global LCM-GP split models.
Plus ambiguous-study residual audit (M0 vs M1global)."""
import json, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch
sys.path.insert(0, r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\materials\code\forward_model")
import LCMGP_without_head as L

RUN = r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\research\Stage1_CA_Forward_Refit_20260913\run_01"
SF = os.path.join(RUN, "models", "splitfits"); TAB = os.path.join(RUN, "tables")
FEAT10 = ['porosity', 'w/c', 'cem (kg/m3)', 'CA (kg/m3)', 'SP (kg/m3)', 'sand (kg/m3)', 'WAS', 'I_gap', 'Cu', 'Aag(mm^-1)']
L.FEATURES = list(FEAT10); L.CONT_IDXS = list(range(10))
L.TRANSFORM_RECIPE = {'log_cols': ['cem (kg/m3)', 'CA (kg/m3)', 'Cu'],
                      'log1p_cols': ['SP (kg/m3)', 'sand (kg/m3)', 'Aag(mm^-1)'],
                      'raw_cols': ['porosity', 'w/c', 'WAS', 'I_gap'], 'categorical_cols': []}
CFG = {'kernel': 'rbf+rq', 'rank': 1, 'use_additive': True, 'add_kernel': 'rbf'}
XLSX = r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\materials\data\reconciled_phase2v\phase1_reconciled_eligible.xlsx"
SPLITS = r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\materials\results\forward_model_audit\repeated_splits_indices.json"
COH = pd.read_csv(r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\materials\data\reconciled_phase2v\phase1_reconciled_eligible.csv")
X, yS, yP, _ = L.load_and_prepare_data(XLSX)
CA_raw_all = np.exp(X[:, 3])
QS = np.quantile(CA_raw_all, [0.05, 0.25, 0.5, 0.75, 0.95])
print("CA sweep quantiles:", np.round(QS, 1))

sweep_rows = []
amb_rows = []
AMB_STUDIES = {"data source - 22 - maguesvari2013", "data source - 24 - andrew2010"}
studies = COH["_src_ffill"].to_numpy()
for s in range(5):
    sp = json.load(open(SPLITS))[f"split_{s}"]
    tr = np.asarray(sp["train_idx"]); te = np.asarray(sp["test_idx"])
    rec = L._prepare_split(X, yS, yP, tr, te)
    Xt = torch.as_tensor(rec['Xtr_s'], dtype=L.DTYPE, device=L.DEVICE).contiguous()
    Yt = torch.as_tensor(rec['Ytr_s'], dtype=L.DTYPE, device=L.DEVICE).contiguous()
    lik = L._build_likelihood(); model = L.LCMGPModel(Xt, Yt, lik, CFG).to(device=L.DEVICE, dtype=L.DTYPE)
    st = torch.load(os.path.join(SF, f"M1global_split{s}_state.pt"), map_location=L.DEVICE, weights_only=True)
    model.load_state_dict(st['model_state_dict']); lik.load_state_dict(st['likelihood_state_dict'])
    model.eval(); lik.eval()
    xsc_mean = rec['Xtr_s'].mean(0) * 0  # not needed; use module standardizer below
    # train scaler stats in MODEL space: recompute via XStandardizer fit on train model features
    xsc = L.XStandardizer().fit(X[tr]); ysc_mean, ysc_std = rec['y_mean'], rec['y_std']
    ad_ref = L._compute_ad_reference(rec['Xtr_s'])
    tau = float(np.asarray(ad_ref['tau_AD']).ravel()[0]) if 'tau_AD' in ad_ref else float(ad_ref.get('tau', np.nan))
    # anchors: 8 diverse train rows (spread over porosity + studies)
    por = X[tr][:, 0]; order = np.argsort(por)
    picks = np.linspace(0, len(tr) - 1, 8).astype(int)
    for a_i, anchor_pos in enumerate(order[picks]):
        grow = tr[anchor_pos]
        base_model = X[grow].copy()
        for q, cav in zip([5, 25, 50, 75, 95], QS):
            row = base_model.copy(); row[3] = np.log(cav)
            z = (row - xsc.mean) / np.where(xsc.std == 0, 1, xsc.std)
            with torch.no_grad():
                mu_s, sig_s = L.predict(model, lik,
                    torch.as_tensor(z[None, :], dtype=L.DTYPE, device=L.DEVICE).contiguous(), fast_mode=True)
            mu = mu_s * ysc_std + ysc_mean; sd = np.maximum(sig_s * ysc_std, 1e-12)
            d = float(np.sqrt(((rec['Xtr_s'] - z) ** 2).sum(1)).min())
            # kNN mean distance for AD ratio
            dd = np.sqrt(((rec['Xtr_s'] - z) ** 2).sum(1)); dd.sort()
            Dm = float(dd[:10].mean())
            sweep_rows.append(dict(split=s, anchor_row=int(grow), anchor_study=studies[grow],
                                    ca_q=q, CA=cav, pred_CS=float(mu[0, 0]), pred_logQ=float(mu[0, 1]),
                                    sd_CS=float(sd[0, 0]), sd_logQ=float(sd[0, 1]),
                                    dmin_train=float(d), D10mean=float(Dm), tau=float(tau),
                                    supported=bool(Dm / tau <= 1.0)))
    # ambiguous-study residual audit on test rows of this split
    d0 = json.load(open(os.path.join(SF, f"M0_split{s}.json"))); d1 = json.load(open(os.path.join(SF, f"M1global_split{s}.json")))
    for i, r in enumerate(te):
        amb = bool(studies[r] in AMB_STUDIES)
        amb_rows.append(dict(split=s, row=int(r), ambiguous=amb,
                             aeCS_M0=abs(d0['y_true_CS'][i] - d0['y_pred_CS'][i]),
                             aeCS_M1=abs(d1['y_true_CS'][i] - d1['y_pred_CS'][i]),
                             aeQ_M0=abs(d0['y_true_logQ'][i] - d0['y_pred_logQ'][i]),
                             aeQ_M1=abs(d1['y_true_logQ'][i] - d1['y_pred_logQ'][i])))
    del model, lik
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

SW = pd.DataFrame(sweep_rows); SW.to_csv(os.path.join(TAB, "ca_supported_sweeps.csv"), index=False)
print("supported fraction:", SW["supported"].mean(), "| n sweeps:", len(SW))
for (s, ar), d in SW.groupby(["split", "anchor_row"]):
    d = d.sort_values("CA")
    print(f"split{s} row{ar} {d['anchor_study'].iloc[0][:30]} sup={d['supported'].all()} dCS={d['pred_CS'].iloc[-1]-d['pred_CS'].iloc[0]:+.2f} dlogQ={d['pred_logQ'].iloc[-1]-d['pred_logQ'].iloc[0]:+.3f} sdCS {d['sd_CS'].iloc[0]:.2f}->{d['sd_CS'].iloc[-1]:.2f}")
AM = pd.DataFrame(amb_rows)
print(AM.groupby("ambiguous")[["aeCS_M0", "aeCS_M1", "aeQ_M0", "aeQ_M1"]].mean().round(4).to_string())
print("ambiguous rows total:", int(AM["ambiguous"].sum()))
AM.to_csv(os.path.join(TAB, "_ambiguous_residuals.csv"), index=False)
print("STEP3 DONE")
