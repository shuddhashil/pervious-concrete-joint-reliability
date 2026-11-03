"""Persistent-model fast backend for the Stage-1 evaluator.

Uses the IDENTICAL production ops as L.predict_saved_oracle_dataframe
(same save_obj, same helper functions, same argument values) but loads the
torch model once and computes only the columns the evaluator consumes.
Numerical equivalence with the exact reload path is verified in the
verification gate (tolerance documented in EVALUATOR_CONTRACT.md).
Final shortlist replay always uses the exact path.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(RUN, "frozen", "forward_model"))

import LCMGP_without_head as L  # noqa: E402
import a3_contract as A3C  # noqa: E402  (sandbox A3 engine)

_ORACLE = {}


def _load(pt_path):
    if pt_path not in _ORACLE:
        A3C.install(L)   # A3 kernel builder must be in place before the saved model is rebuilt
        save_obj, model, likelihood = L._load_saved_oracle(pt_path, L.DEVICE)
        feats = list(save_obj["features"])
        if feats != list(A3C.FEATURES) or "CA (kg/m3)" not in feats:
            raise RuntimeError(f"Refusing oracle with unexpected features={feats} (expected the A3 11D contract).")
        L.FEATURES = feats
        L.CONT_IDXS = list(save_obj["cont_idxs"])
        L.TRANSFORM_RECIPE = dict(save_obj["transform_recipe"])
        _ORACLE[pt_path] = (save_obj, model, likelihood)
    return _ORACLE[pt_path]


def predict_fast(pt_path, df_new):
    """Same prefix of predict_saved_oracle_dataframe up to effective scales."""
    save_obj, model, likelihood = _load(pt_path)
    Xs = L._prepare_prediction_features(df_new, save_obj)
    Xt_new = __import__("torch").as_tensor(Xs, dtype=L.DTYPE, device=L.DEVICE).contiguous()
    mu_s, sig_s = L.predict(model, likelihood, Xt_new, fast_mode=bool(save_obj.get("fast_mode", True)))
    y_mean = np.asarray(save_obj["y_mean"], dtype=float)
    y_std = np.asarray(save_obj["y_std"], dtype=float)
    mean = mu_s * y_std + y_mean
    std_raw = L.apply_sigma_floors(sig_s * y_std)
    local_residual_q = max(2, int(save_obj.get("local_residual_default_q_neighbors",
                                              L.LOCAL_RESIDUAL_K_NEIGHBORS)))
    local_info = L._prediction_local_residual_lookup(save_obj, Xs, int(local_residual_q))
    loc = local_info["loc"]
    local_s = local_info["strength_local"]
    local_p = local_info["logperm_local"]
    std_eff_s_pre = L._combine_std(std_raw[:, 0], local_s,
                                   save_obj.get("uncertainty_combine_rule", L.UNCERTAINTY_COMBINE_RULE))
    std_eff_p_pre = L._combine_std(std_raw[:, 1], local_p,
                                   save_obj.get("uncertainty_combine_rule", L.UNCERTAINTY_COMBINE_RULE))
    ad = L._ad_for_standardized_features(save_obj, Xs) if "ad_reference" in save_obj else {
        "D_AD": np.full(len(Xs), np.nan), "tau_AD": np.full(len(Xs), np.nan),
        "AD_ratio": np.full(len(Xs), np.nan), "inside_AD": np.full(len(Xs), False),
        "AD_k_neighbors": L.AD_K_NEIGHBORS}
    infl = L._ad_inflation_from_ratio(ad["AD_ratio"], eta=float(save_obj.get("ad_inflation_eta", L.AD_INFLATION_ETA)))
    return pd.DataFrame({
        "strength_pred_mpa": mean[:, 0], "logperm_pred": mean[:, 1],
        "perm_pred_mm_s": np.exp(mean[:, 1]),
        "strength_std_raw_diagnostic_mpa": std_raw[:, 0], "logperm_std_raw_diagnostic": std_raw[:, 1],
        "strength_residual_std_local_mpa": local_s, "logperm_residual_std_local": local_p,
        "strength_std_effective_mpa": std_eff_s_pre * infl,
        "logperm_std_effective": std_eff_p_pre * infl,
        "AD_uncertainty_inflation": infl,
        "D_AD": ad["D_AD"], "tau_AD": ad["tau_AD"], "AD_ratio": ad["AD_ratio"],
        "inside_AD": ad["inside_AD"],
    })
