"""Open the deployed A3-1 forward model (LCM-GP) without post-hoc explainers. Forward computations only; nothing is trained.

Reads the saved model, rebuilds it with the engine's own loader, then re-derives the posterior mean by hand so that every
prediction can be written exactly as
    prediction = constant + sum over the 631 recorded mixtures (weight x their measured CS and ln k)
               = constant + contribution of each additive kernel part.
Outputs: tables + report numbers in this folder (results.json, *.csv).
"""
import copy, json, os, sys
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild\research\RobustEngine_20261022\GuidedEngine_20261003\base\core"   # 2026-10-22: deployed laboratory-effect engine
sys.path.insert(0, os.path.join(CORE, "frozen", "forward_model")); sys.path.insert(0, os.path.join(CORE, "src"))
import LCMGP_without_head as L  # noqa: E402
import a3_contract as A3C  # noqa: E402
from a3_quantiles import quantiles_batch  # noqa: E402

L.DEVICE = torch.device("cpu")
A3C.install(L)
PT = os.path.join(CORE, "models", "lcmgp_ca_oracle_full_data.pt")
save, model, lik = L._load_saved_oracle(PT, L.DEVICE)
FEATS = list(save["features"]); xm, xs = np.asarray(save["x_cont_mean"]), np.asarray(save["x_cont_std"])
ym, ys = np.asarray(save["y_mean"], float), np.asarray(save["y_std"], float)
X = save["train_x"].double().numpy(); Y = save["train_y"].double().numpy(); n = len(X)
out = {}

# ---------- 1. hyperparameters, in plain units ----------
dc = model.shared_covar_module.data_covar_module
ks = dc.base_kernel.kernels
s = float(dc.outputscale)
B = model.shared_covar_module.task_covar_module.covar_matrix.to_dense().double().detach().numpy()
noise = lik.task_noises.double().detach().numpy()
c = np.array([float(model.mean_module.strength_const), float(model.mean_module.perm_const)])
names = ["RBF on recipe", "RQ on all 11 inputs", "Binder RBF (w/c, cement, SP)", "Filler RBF (porosity, sand)"]
comp = []
for name, k in zip(names, ks):
    ls = k.lengthscale.double().detach().numpy().ravel()
    dims = [int(d) for d in k.active_dims]
    comp.append(dict(name=name, dims=[FEATS[d] for d in dims], ls_std=ls.tolist(),
                     alpha=(float(k.alpha) if hasattr(k, "alpha") else None)))
out["hyper"] = dict(outputscale=s, B=B.tolist(), task_noise_std_units=noise.tolist(), const_std_units=c.tolist(),
                    latent_task_corr=float(B[0, 1] / np.sqrt(B[0, 0] * B[1, 1])), components=comp)


def phys_ls(feat, l_std):
    """Lengthscale (standardised) -> what one lengthscale means in the input's own units."""
    j = FEATS.index(feat); d = l_std * xs[j]
    if feat in ("cem (kg/m3)", "CA (kg/m3)"):
        return f"x{np.exp(d):.2f} (i.e. +{100*(np.exp(d)-1):.0f}%)"
    if feat in ("SP (kg/m3)", "sand (kg/m3)"):
        return f"{d:.2f} in ln(1+x)"
    if feat.startswith("q"):
        return f"x{2**d:.2f} on particle size"
    if feat == "porosity":
        return f"{d:.1f} % points"
    return f"{d:.3f}"


rows = []
for cpt in comp:
    for f, l in zip(cpt["dims"], cpt["ls_std"]):
        rows.append(dict(component=cpt["name"], input=f, lengthscale_std=round(l, 3), one_lengthscale=phys_ls(f, l),
                         at_lower_bound=bool(l < 0.205)))
pd.DataFrame(rows).to_csv(os.path.join(HERE, "lengthscales.csv"), index=False)

# ---------- 2. exact posterior by hand (float64) ----------
Xt = torch.tensor(X)


def kmats(A, Bx):
    """Per-component kernel matrices (unscaled) between A and Bx (standardised inputs)."""
    with torch.no_grad():
        return [k(torch.tensor(A), torch.tensor(Bx)).to_dense().double().numpy() for k in ksd]


ksd = [copy.deepcopy(k).double() for k in ks]   # double copies for the hand path; engine model untouched
Kc = kmats(X, X); Kx = s * sum(Kc)
K = np.kron(Kx, B) + np.kron(np.eye(n), np.diag(noise))                  # gpytorch order: point-major, task-minor
r = (Y - c).reshape(-1)
Kinv = np.linalg.inv(K)
alpha = Kinv @ r
A2 = alpha.reshape(n, 2)


def predict_hand(Xn):
    Kc_n = kmats(Xn, X); kx = s * sum(Kc_n)
    mu = c + kx @ A2 @ B.T                                                # task t: sum_i sum_u B[t,u] kx[i] alpha[i,u]
    parts = np.stack([s * kc @ A2 @ B.T for kc in Kc_n])                 # (component, point, task)
    W = np.kron(kx, B) @ Kinv                                             # (2m, 2n) smoother weights
    return mu, parts, W.reshape(len(Xn), 2, n, 2)


# check against the engine's own prediction on the training inputs (first 50) and designs later
mu_eng, _ = L.predict(model, lik, torch.tensor(X[:50]).float())
mu_h, _, _ = predict_hand(X[:50])
out["check_hand_vs_engine_max_abs_std_units"] = float(np.abs(mu_eng - mu_h).max())

# ---------- 3. data with study labels, matched to train_x ----------
raw = pd.read_csv(os.path.join(CORE, "data_inputs", "data_with_passing_sizes_A3-1.csv"))
raw["study"] = raw["Unnamed: 0"].where(raw["Unnamed: 0"].astype(str).str.startswith("data source")).ffill()
raw = raw[raw["perm (mm/s)"] > 0].reset_index(drop=True)


def to_std(df):
    Z = df[FEATS].astype(float).copy()
    for col in ("cem (kg/m3)", "CA (kg/m3)"):
        Z[col] = np.log(Z[col])
    for col in ("SP (kg/m3)", "sand (kg/m3)"):
        Z[col] = np.log1p(Z[col])
    return ((Z.values - xm) / xs)


Zr = to_std(raw)
D = ((Zr[:, None, :] - X[None, :, :]) ** 2).sum(-1)
match = D.argmin(0)                                                       # raw row for each training row
out["match_max_dist"] = float(D.min(0).max()); out["n_raw_after_perm_filter"] = len(raw)
meta = raw.iloc[match].reset_index(drop=True)
meta["CS_meas"] = Y[:, 0] * ys[0] + ym[0]; meta["lnk_meas"] = Y[:, 1] * ys[1] + ym[1]

# ---------- 4. the five designs (Table 3 of the Results draft) ----------
_dj = json.load(open(os.path.join(HERE, "..", "results_section", "designs_v3.json")))   # the four reported designs, exact nominal recipes from the run logs
_nd = [_dj[t]["lowest"]["nominal"] for t in ["S10_K8_R0.90", "S14_K6_R0.90", "S16_K10_R0.90", "S20_K4_R0.90"]]
des = pd.DataFrame({"target": ["10/8", "14/6", "16/10", "20/4"], "cem (kg/m3)": [d["C"] for d in _nd], "w/c": [d["r"] for d in _nd], "CA (kg/m3)": [d["A"] for d in _nd],
                    "sand (kg/m3)": [d["S"] for d in _nd], "SP (kg/m3)": [d["P"] for d in _nd], "porosity": [d["phi"] for d in _nd]})
bins = np.array([np.asarray(d["g"], float) * 100 for d in _nd])
q = quantiles_batch(bins)
for j, nm in enumerate(["q10", "q25", "q50", "q75", "q90"]):
    des[nm] = q[:, j]
Xd = to_std(des)
mu_d, parts_d, W_d = predict_hand(Xd)
mu_eng_d, sd_eng_d = L.predict(model, lik, torch.tensor(Xd).float())
out["check_designs_hand_vs_engine"] = float(np.abs(mu_eng_d - mu_d).max())

design_out = []
for i, t in enumerate(des["target"]):
    cs = mu_d[i, 0] * ys[0] + ym[0]; lnk = mu_d[i, 1] * ys[1] + ym[1]
    # contributions to the CS prediction (MPa) from each recorded mixture: weights on CS and on lnk targets
    wcs = W_d[i, 0, :, 0]; wq = W_d[i, 0, :, 1]
    contrib_cs = ys[0] * (wcs * (Y[:, 0] - c[0]) + wq * (Y[:, 1] - c[1]))
    wcs_k = W_d[i, 1, :, 0]; wq_k = W_d[i, 1, :, 1]
    contrib_k = ys[1] * (wcs_k * (Y[:, 0] - c[0]) + wq_k * (Y[:, 1] - c[1]))
    order = np.argsort(-np.abs(wcs))
    top = []
    for jx in order[:8]:
        m = meta.iloc[jx]
        top.append(dict(study=str(m["study"]).replace("data source - ", ""), ID=str(m["ID"]), weight_on_CS=round(float(wcs[jx]), 3),
                        CS_meas=round(float(m["CS_meas"]), 1), k_meas=round(float(np.exp(m["lnk_meas"])), 2),
                        cement=float(m["cem (kg/m3)"]), wc=float(m["w/c"]), porosity=round(float(m["porosity"]), 1),
                        sand=float(m["sand (kg/m3)"]),
                        bins=[round(float(m[b]), 1) for b in ["R25-19mm", "R19-12.5mm", "R12.5-9.5mm", "R9.5-6.3mm", "R6.3-4.75mm", "R4.75-2.36mm"]],
                        MPa_added=round(float(contrib_cs[jx]), 2)))
    by_study = (pd.DataFrame({"study": meta["study"], "c": contrib_cs, "w": wcs}).groupby("study")
                .agg(MPa=("c", "sum"), weight=("w", "sum")).sort_values("MPa", key=np.abs, ascending=False).head(6))
    design_out.append(dict(
        target=t, CS_pred_MPa=round(float(cs), 2), k_pred_mm_s=round(float(np.exp(lnk)), 2),
        CS_constant_MPa=round(float(c[0] * ys[0] + ym[0]), 2), k_constant_mm_s=round(float(np.exp(c[1] * ys[1] + ym[1])), 2),
        CS_parts_MPa={nm: round(float(parts_d[j, i, 0] * ys[0]), 2) for j, nm in enumerate(names)},
        lnk_parts={nm: round(float(parts_d[j, i, 1] * ys[1]), 3) for j, nm in enumerate(names)},
        sum_weights_on_CS=round(float(wcs.sum()), 3), sum_weights_on_lnk_for_CS=round(float(wq.sum()), 3),
        effective_n_records=round(float(1 / np.sum((wcs / wcs.sum()) ** 2)), 1),
        share_top10_abs_weight=round(float(np.sort(np.abs(wcs))[::-1][:10].sum() / np.abs(wcs).sum()), 3),
        top_records=top, by_study_MPa=by_study.round(3).reset_index().to_dict("records"),
        CS_from_k_targets_MPa=round(float(ys[0] * (wq * (Y[:, 1] - c[1])).sum()), 2)))
out["designs"] = design_out

# ---------- 5. exact local slopes of the model at all 631 recorded mixtures ----------
# derivative of the posterior mean w.r.t. each input, in physical units (autograd through the same formula)
Xr = torch.tensor(X, requires_grad=True)
A2t = torch.tensor(A2); Bt = torch.tensor(B)
kx = s * sum(k(Xr, Xt).to_dense() for k in ksd)
mu_r = torch.tensor(c) + kx @ A2t @ Bt.T
gcs = torch.autograd.grad(mu_r[:, 0].sum(), Xr, retain_graph=True)[0].numpy()
gk = torch.autograd.grad(mu_r[:, 1].sum(), Xr)[0].numpy()
phys = meta[FEATS].astype(float).values


def dphys(g, j, ysc):
    """d(output)/d(standardised input j) -> per physical unit."""
    d = g[:, j] * ysc / xs[j]
    f = FEATS[j]
    if f in ("cem (kg/m3)", "CA (kg/m3)"):
        d = d / phys[:, j]                       # d/dx = d/dln x * 1/x
    if f in ("SP (kg/m3)", "sand (kg/m3)"):
        d = d / (1 + phys[:, j])
    return d


steps = {"porosity": (1.0, "per +1 % point"), "w/c": (0.01, "per +0.01"), "cem (kg/m3)": (10, "per +10 kg/m3"),
         "CA (kg/m3)": (50, "per +50 kg/m3"), "SP (kg/m3)": (0.25, "per +0.25 kg/m3"), "sand (kg/m3)": (10, "per +10 kg/m3")}
slope_rows = []
for f, (h, lab) in steps.items():
    j = FEATS.index(f)
    a = dphys(gcs, j, ys[0]) * h; b = dphys(gk, j, ys[1]) * h
    slope_rows.append(dict(input=f, step=lab, CS_median_MPa=round(float(np.median(a)), 3), CS_share_negative=round(float((a < 0).mean()), 3),
                           lnk_median=round(float(np.median(b)), 4), k_median_pct=round(float(100 * (np.exp(np.median(b)) - 1)), 1),
                           lnk_share_positive=round(float((b > 0).mean()), 3)))
# gradation: shift of all five passing sizes together by +0.1 log2 (about 7% coarser)
jq = [FEATS.index(qq) for qq in ["q10", "q25", "q50", "q75", "q90"]]
a = sum(gcs[:, j] / xs[j] for j in jq) * ys[0] * 0.1; b = sum(gk[:, j] / xs[j] for j in jq) * ys[1] * 0.1
slope_rows.append(dict(input="all five passing sizes", step="+0.1 log2 (~7% coarser)", CS_median_MPa=round(float(np.median(a)), 3),
                       CS_share_negative=round(float((a < 0).mean()), 3), lnk_median=round(float(np.median(b)), 4),
                       k_median_pct=round(float(100 * (np.exp(np.median(b)) - 1)), 1), lnk_share_positive=round(float((b > 0).mean()), 3)))
wc = phys[:, FEATS.index("w/c")]; jw = FEATS.index("w/c")
for lo, hi in [(0, 0.30), (0.30, 9)]:
    msk = (wc >= lo) & (wc < hi); a = dphys(gcs, jw, ys[0])[msk] * 0.01; b = dphys(gk, jw, ys[1])[msk] * 0.01
    slope_rows.append(dict(input=f"w/c, records with {lo}<=w/c<{hi} (n={int(msk.sum())})", step="per +0.01",
                           CS_median_MPa=round(float(np.median(a)), 3), CS_share_negative=round(float((a < 0).mean()), 3),
                           lnk_median=round(float(np.median(b)), 4), k_median_pct=round(float(100 * (np.exp(np.median(b)) - 1)), 1),
                           lnk_share_positive=round(float((b > 0).mean()), 3)))
sd0 = phys[:, FEATS.index("sand (kg/m3)")] == 0
out["n_records_sand_zero"] = int(sd0.sum())
pd.DataFrame(slope_rows).to_csv(os.path.join(HERE, "local_slopes.csv"), index=False)
out["slopes"] = slope_rows

# ---------- 6. how much of the fitted surface each kernel part carries (over the 631 records) ----------
_, parts_tr, _ = predict_hand(X)
share = []
for j, nm in enumerate(names):
    share.append(dict(component=nm, CS_sd_MPa=round(float(np.std(parts_tr[j, :, 0]) * ys[0]), 2),
                      lnk_sd=round(float(np.std(parts_tr[j, :, 1]) * ys[1]), 3)))
out["component_spread_over_data"] = share
out["measured_sd"] = dict(CS_MPa=float(ys[0]), lnk=float(ys[1]))

# per-record slopes for the figure (porosity per +1 point, cement per +10 kg/m3)
jp, jc = FEATS.index("porosity"), FEATS.index("cem (kg/m3)")
pd.DataFrame({"dCS_dpor": dphys(gcs, jp, ys[0]), "dlnk_dpor": dphys(gk, jp, ys[1]),
              "dCS_dcem10": dphys(gcs, jc, ys[0]) * 10, "dlnk_dcem10": dphys(gk, jc, ys[1]) * 10}).to_csv(
    os.path.join(HERE, "slopes_per_record.csv"), index=False)

json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float)
print(json.dumps({k: out[k] for k in ["check_hand_vs_engine_max_abs_std_units", "check_designs_hand_vs_engine", "match_max_dist",
                                      "n_raw_after_perm_filter"]}, indent=1))
