"""Production evaluator for the CA-aware (10D M1global) oracle.
Patches the authoritative module's feature contract from the save_obj itself,
then delegates to predict_saved_oracle_dataframe. Refuses 9D artifacts loudly.
"""
import os, sys
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "forward_model"))
import LCMGP_without_head as L
import a3_contract as A3C  # sandbox A3 engine: 11-input contract + kernel builder

D_GEOM = np.array([21.7945, 15.4110, 10.8972, 7.7380, 5.4703, 3.3478])
BINS = ["R25-19mm", "R19-12.5mm", "R12.5-9.5mm", "R9.5-6.3mm", "R6.3-4.75mm", "R4.75-2.36mm"]

def _ensure_i_gap(df):
    if "I_gap" in df.columns:
        return df
    df = df.copy()
    B = df[BINS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    B = np.where(np.isnan(B), 0.0, B)
    rs = B.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
    F = B / rs
    ig = np.zeros(len(df))
    for i in range(6):
        for j in range(i + 1, 6):
            m = (F[:, i] > 0) & (F[:, j] > 0)
            ig[m] += F[m, i] * F[m, j] * np.log(D_GEOM[i] / D_GEOM[j])
    df["I_gap"] = ig
    return df

def _activate_10d_contract(pt_path):
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    feats = list(obj["features"])
    if feats != list(A3C.FEATURES) or "CA (kg/m3)" not in feats:
        raise RuntimeError(f"Refusing oracle with unexpected features={feats} (expected the A3 11D contract).")
    if "joint_bundle" not in obj or not isinstance(obj.get("joint_bundle"), dict) or "levels" not in obj.get("joint_bundle", {}):
        raise RuntimeError("Oracle lacks a merged joint_bundle; refusing to predict without the joint screen contract.")
    A3C.install(L)   # features, transforms, additive groups and the A3 kernel builder (needed before the model is rebuilt)
    L.FEATURES = feats
    L.CONT_IDXS = list(obj["cont_idxs"])
    L.TRANSFORM_RECIPE = dict(obj["transform_recipe"])
    return obj

def predict_ca_oracle_dataframe(pt_path, df_new, target_coverage=None):
    """Full production prediction (marginals + calibrated LBs + AD + local residuals)."""
    _activate_10d_contract(pt_path)
    df_new = _ensure_i_gap(df_new)
    return L.predict_saved_oracle_dataframe(pt_path, df_new, target_coverage=target_coverage)

def joint_screen_from_predictions(pred_df, pt_path, s_star, k_star, p=0.9):
    """Paired max-score joint acceptance from production prediction outputs.
    Returns boolean pass array. Raw Gaussian probability is never consulted."""
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    q = float(obj["joint_bundle"]["levels"][str(p)]["q"])
    muS = pred_df["strength_pred_mpa"].to_numpy(dtype=float)
    muQ = pred_df["logperm_pred"].to_numpy(dtype=float)
    sS = pred_df["strength_std_effective_mpa"].to_numpy(dtype=float)
    sQ = pred_df["logperm_std_effective"].to_numpy(dtype=float)
    return ((muS - q * sS >= float(s_star)) & (muQ - q * sQ >= np.log(float(k_star))))
