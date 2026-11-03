"""Engine-format oracle from the lab-effect model (Path 1, 2026-10-22).
The lab-effect GP has training covariance K = K0 + Lab with K0 = Kx (x) B + noise (the deployed ICM structure) and Lab = 1[same study] (x) B_lab.
Its predictive mean for a laboratory with zero offset equals, EXACTLY, the ICM GP with the same hyperparameters trained on the lab-corrected targets
y - b_hat, b_hat = Lab K^-1 (y - m) (posterior mean of each study's offset):  K^-1 (y - m) = K0^-1 (y - m - b_hat).
So the engine keeps its ICM evaluators unchanged; the oracle = ICM model (lab-model hyperparameters, lab term removed) on corrected targets.
The ICM variance omits the posterior uncertainty of b_hat (slightly smaller than the lab model's); the gap is reported.
usage: python make_lab_oracle.py <lab .pt> <state .pt or ''> <out .pt> [train_rows.json]   (full fit: state from the .pt itself)"""
import os, sys, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "colab_kit"))
import numpy as np, torch
import lab_fits as LF
RF, R, L = LF.RF, LF.R, LF.L


def icm_from_lab(state, Xs, Y_s, sid):
    """state: lab-model state dict (model) + lik; Xs (n,11) standardised, Y_s (n,2) standardised, sid (n,) study index -> (icm model, lik, Ycorr_s, b_hat, check)"""
    T = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float64)
    L.DTYPE = torch.float64; L.DEVICE = torch.device("cpu")
    Z = T(np.column_stack([Xs, sid])); Yt = T(Y_s)
    lik = L._build_likelihood(); lab = LF.LabLCMGP(Z, Yt, lik, R.CFG).double(); lab.load_state_dict(state["model"]); lik.load_state_dict(state["lik"]); lab.eval(); lik.eval()
    with torch.no_grad():
        K0 = lab.shared_covar_module(Z).to_dense().numpy(); Lb = lab.lab_covar_module(Z).to_dense().numpy()
        noise = np.tile(lik.task_noises.detach().double().numpy(), len(Xs)); m = np.array([float(lab.mean_module.strength_const), float(lab.mean_module.perm_const)])
    r = (Y_s - m[None, :]).reshape(-1); K = K0 + Lb + np.diag(noise); a = np.linalg.solve(K, r); b_hat = (Lb @ a).reshape(-1, 2)
    Ycorr = Y_s - b_hat
    icm = LF.Base(T(Xs), T(Ycorr), lik, R.CFG).double()
    sd = {k: v for k, v in state["model"].items() if not k.startswith("lab_covar_module")}; icm.load_state_dict(sd); icm.eval()
    # exactness check on 200 random training-like points: ICM(corrected) mean vs lab model mean at a NEW lab
    rng = np.random.default_rng(0); q = Xs[rng.choice(len(Xs), 200)] + 0.3 * rng.standard_normal((200, Xs.shape[1]))
    with torch.no_grad():
        m_lab, s_lab = L.predict(lab, lik, T(np.column_stack([q, np.full(200, 99999.0)])), fast_mode=False)
        m_icm, s_icm = L.predict(icm, lik, T(q), fast_mode=False)
    Blab = lab.lab_covar_module.task_covar_module.covar_matrix.to_dense().detach().double().numpy()
    s_lab_nolab = np.sqrt(np.maximum(s_lab ** 2 - np.diag(Blab)[None, :], 1e-12))
    chk = dict(max_abs_mean_diff_std=float(np.abs(m_lab - m_icm).max()), sd_ratio_icm_over_lab_minus_labvar_median=np.median(s_icm / s_lab_nolab, 0).tolist(),
               sd_ratio_min=np.min(s_icm / s_lab_nolab, 0).tolist())
    return icm, lik, Ycorr, b_hat, chk


if __name__ == "__main__":
    lab_pt, out = sys.argv[1], sys.argv[3]
    full = torch.load(lab_pt, map_location="cpu", weights_only=False)
    feats, groups, X, yS, yP, st = RF.frame(); Y = np.stack([yS, yP], 1)
    assert np.allclose(X, full["X_model"]) and np.allclose(Y, full["Y"])
    Xs = (X - full["x_mean"]) / full["x_std"]; Y_s = (Y - full["y_mean"]) / full["y_std"]
    icm, lik, Ycorr, b_hat, chk = icm_from_lab(dict(model=full["model"], lik=full["lik"]), Xs, Y_s, full["study_index"])
    names = full["study_names"]; off = {names[int(j)]: (b_hat[full["study_index"] == j][0] * full["y_std"]).tolist() for j in np.unique(full["study_index"])}
    torch.save(dict(full, model_icm={k: v for k, v in icm.state_dict().items()}, Y_s_corrected=Ycorr, offsets_std=b_hat, check=chk), out)
    json.dump(dict(check=chk, lab=full["lab"], study_offsets_CS_mpa_lnk=off), open(out[:-3] + "_report.json", "w"), indent=1)
    print(json.dumps(chk)); print("offsets (CS MPa, ln k):", {k[-18:]: [round(v[0], 2), round(v[1], 2)] for k, v in sorted(off.items(), key=lambda kv: -kv[1][0])})
