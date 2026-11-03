"""Authoritative Stage-1 candidate evaluator (CA-aware 10D M1global production system).

One evaluator for every solver (DE / DIRECT / local / lattice / tests).
Physical raw units at the boundary. CA enters in raw kg/m3 and is transformed
exactly once inside the production path (natural log + standardisation).

Provenance:
- Oracle/support/joint contract: research/Stage1_CA_Forward_Refit_20260913/run_01
  via src/predict_ca_oracle.py (refuses 9D loudly; joint_bundle merged in oracle).
- Descriptor convention: precise geometric diameters + 500-grid Cu, copied from
  research/Stage1_Gate_20260913/run_01/src/descriptors.py (frozen training
  convention). Explicit descriptors are supplied, bypassing the wrapper's
  rounded-diameter I_gap fallback. Accepted convention tolerance 8e-5 recorded
  in the run manifest (user decision: non-blocking).
- Support: models/ocsvm_kde_feasibility_CA10.pkl, threshold combined >= 0.25.
- Joint screen: paired max-score deployable q from the oracle joint_bundle.
"""
import math
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild"
REFIT_RUN = os.path.join(ROOT, "research", "Stage1_CA_Forward_Refit_20260913", "run_01")
ORACLE_PT = os.path.join(REFIT_RUN, "models", "lcmgp_ca_oracle_full_data.pt")
SUPPORT_PKL = os.path.join(REFIT_RUN, "models", "ocsvm_kde_feasibility_CA10.pkl")

sys.path.insert(0, os.path.join(REFIT_RUN, "src"))
sys.path.insert(0, os.path.join(ROOT, "materials", "code", "forward_model"))
sys.path.insert(0, os.path.join(ROOT, "materials", "code", "feasibility"))

import LCMGP_without_head as L  # noqa: E402
import ocsvm_kde_feasibility_without_head as F  # noqa: E402
from predict_ca_oracle import (  # noqa: E402
    _activate_10d_contract,
    predict_ca_oracle_dataframe,
)
from fast_backend import predict_fast  # noqa: E402

# ---- frozen production constants ----
EXPECTED_FEATURES = ["porosity", "w/c", "cem (kg/m3)", "CA (kg/m3)", "SP (kg/m3)",
                     "sand (kg/m3)", "WAS", "I_gap", "Cu", "Aag(mm^-1)"]
BINS = ["R25-19mm", "R19-12.5mm", "R12.5-9.5mm", "R9.5-6.3mm", "R6.3-4.75mm", "R4.75-2.36mm"]
SUPPORT_THRESHOLD = 0.25
I_GAP_CONVENTION_TOL = 8e-5

# Precise training-convention diameters (bounds.json bin_diameters_mm).
DIAM = np.array([21.79449471770337, 15.411035007422441, 10.897247358851684,
                 7.736278174936576, 5.470374758643141, 3.348133808556641], dtype=float)
_ASC_LOG_EDGES = np.log(np.array([2.36, 4.75, 6.3, 9.5, 12.5, 19.0, 25.0]))
_GRID_TH = np.geomspace(2.36, 25.0, 500)


def _grid_passing(f):
    f = np.asarray(f, dtype=float)
    f = f / f.sum()
    iv = [(float(l), float(u), float(m)) for m, (u, l) in
          zip(f, [(25.0, 19.0), (19.0, 12.5), (12.5, 9.5), (9.5, 6.3), (6.3, 4.75), (4.75, 2.36)]) if m > 0]
    tot = sum(m for _, _, m in iv)
    y = np.empty(500)
    for n, t in enumerate(_GRID_TH):
        ml = 0.0
        for lo, hi, m in iv:
            if t <= lo:
                fr = 0.0
            elif t >= hi:
                fr = 1.0
            else:
                fr = (math.log(t) - math.log(lo)) / (math.log(hi) - math.log(lo))
            ml += m * fr
        y[n] = 100.0 * ml / tot
    return y


def _dpp_grid(f, percent):
    passing = _grid_passing(f)
    th = _GRID_TH
    mask = np.isfinite(passing) & np.isfinite(th) & (th > 0)
    p = passing[mask]
    x = np.log(th[mask])
    up, ux = [], []
    for v in np.unique(p):
        xs = x[p == v]
        up.append(v)
        ux.append(xs.mean())
    up = np.asarray(up)
    ux = np.asarray(ux)
    if len(up) < 2 or percent < up.min() or percent > up.max():
        return float("nan")
    return float(np.exp(np.interp(percent, up, ux)))


def descriptors_precise(g):
    """Training-convention descriptors from six fractions (sum 1, >= 0)."""
    f = np.asarray(g, dtype=float)
    was = float(f @ DIAM)
    aag = float(np.sum(6.0 * f / DIAM))
    ig = 0.0
    for i in range(6):
        for j in range(i + 1, 6):
            if f[i] <= 0 or f[j] <= 0:
                continue
            ig += f[i] * f[j] * math.log(DIAM[i] / DIAM[j])
    d10 = _dpp_grid(f, 10.0)
    d60 = _dpp_grid(f, 60.0)
    cu = float(d60 / d10) if d10 > 0 else float("nan")
    return {"WAS": was, "I_gap": ig, "Cu": cu, "Aag": aag}


def project_to_simplex(m):
    """Euclidean projection of R^6 onto {g >= 0, sum g = 1} (Duchi et al.).

    Canonical solver->simplex map for ALL solvers. Hits faces/vertices exactly
    via clipping; many-to-one (5 effective DOF) by design.
    """
    v = np.asarray(m, dtype=float)
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, 7) > (css - 1.0))[0][-1]
    theta = (css[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


# ---- production bundle (loaded once) ----
_bundle = {}


def get_bundle():
    if _bundle:
        return _bundle
    obj = torch.load(ORACLE_PT, map_location="cpu", weights_only=False)
    feats = list(obj["features"])
    if len(feats) != 10 or "CA (kg/m3)" not in feats:
        raise RuntimeError(f"Refusing non-10D oracle: features={feats}. No silent 9D fallback.")
    if feats != EXPECTED_FEATURES:
        raise RuntimeError(f"Feature-order mismatch: {feats} vs {EXPECTED_FEATURES}")
    if "joint_bundle" not in obj or "levels" not in obj.get("joint_bundle", {}):
        raise RuntimeError("Oracle lacks a merged joint_bundle; refusing.")
    sup = joblib.load(SUPPORT_PKL)
    if list(sup["features"]) != EXPECTED_FEATURES:
        raise RuntimeError(f"Support feature mismatch: {sup['features']}")
    _activate_10d_contract(ORACLE_PT)  # patches L.FEATURES / CONT_IDXS / TRANSFORM_RECIPE, refuses 9D
    _bundle.update(obj=obj, support=sup)
    return _bundle


def joint_q(p):
    b = get_bundle()
    return float(b["obj"]["joint_bundle"]["levels"][str(p)]["q"])


_desc_cache = {}


def cached_descriptors(g):
    key = tuple(np.round(np.asarray(g, dtype=float), 12))
    hit = _desc_cache.get(key)
    if hit is None:
        hit = descriptors_precise(np.asarray(key, dtype=float))
        _desc_cache[key] = hit
    return hit


def candidates_to_frame(cands):
    """Physical candidate dicts -> production DataFrame (raw units, bins in %)."""
    rows = []
    for c in cands:
        g = np.asarray(c["g"], dtype=float)
        d = cached_descriptors(g)
        rows.append({
            "porosity": float(c["phi"]), "w/c": float(c["r"]),
            "cem (kg/m3)": float(c["C"]), "CA (kg/m3)": float(c["A"]),
            "SP (kg/m3)": float(c["P"]), "sand (kg/m3)": float(c["S"]),
            "WAS": d["WAS"], "I_gap": d["I_gap"], "Cu": d["Cu"], "Aag(mm^-1)": d["Aag"],
            BINS[0]: g[0] * 100.0, BINS[1]: g[1] * 100.0, BINS[2]: g[2] * 100.0,
            BINS[3]: g[3] * 100.0, BINS[4]: g[4] * 100.0, BINS[5]: g[5] * 100.0,
        })
    return pd.DataFrame(rows)


def _support_scores(df_feat):
    """Combined support scores via the 10D artifact (same path as verify_wrapper)."""
    b = get_bundle()
    sup = b["support"]
    obj = b["obj"]
    Xs = df_feat[EXPECTED_FEATURES].apply(pd.to_numeric, errors="coerce").copy()
    Xm = L._apply_feature_transforms(Xs)[EXPECTED_FEATURES].to_numpy(dtype=float)
    Xs_std = (Xm - np.asarray(obj["x_cont_mean"])) / np.where(np.asarray(obj["x_cont_std"]) == 0, 1, np.asarray(obj["x_cont_std"]))
    return F.score_feasibility(Xs_std, sup)["combined_score"]


def evaluate_batch(cands, s_star, k_star, p=0.9, bounds=None, backend="fast"):
    """Authoritative batch evaluation. Returns list of record dicts.

    cands: list of {C, r, A, P, S, phi, g(6)} in physical raw units.
    bounds: dict with keys C, r, A, P, S, phi -> (lo, hi); enforced for status.
    backend: 'fast' (persistent model, identical ops; for search) or 'exact'
      (reload path via predict_ca_oracle_dataframe; for final replay).
    """
    b = get_bundle()
    q = joint_q(p)
    q_star = math.log(float(k_star))  # exact single log transform
    frame = candidates_to_frame(cands)
    if backend == "exact":
        pred = predict_ca_oracle_dataframe(ORACLE_PT, frame)
    elif backend == "fast":
        pred = predict_fast(ORACLE_PT, frame)
    else:
        raise ValueError(f"unknown backend {backend}")
    sup_scores = np.asarray(_support_scores(frame), dtype=float)
    muS = pred["strength_pred_mpa"].to_numpy(dtype=float)
    muQ = pred["logperm_pred"].to_numpy(dtype=float)
    sS = pred["strength_std_effective_mpa"].to_numpy(dtype=float)
    sQ = pred["logperm_std_effective"].to_numpy(dtype=float)
    lbS = muS - q * sS
    lbQ = muQ - q * sQ
    recs = []
    for i, c in enumerate(cands):
        g = np.asarray(c["g"], dtype=float)
        d = cached_descriptors(g)
        in_bounds = True
        if bounds is not None:
            for key, bnd in (("C", bounds.get("C")), ("r", bounds.get("r")),
                             ("A", bounds.get("A")), ("P", bounds.get("P")),
                             ("S", bounds.get("S")), ("phi", bounds.get("phi"))):
                if bnd is None:
                    continue
                v = float(c["C"] if key == "C" else c["r"] if key == "r" else c["A"]
                          if key == "A" else c["P"] if key == "P" else c["S"] if key == "S" else c["phi"])
                if not (bnd[0] - 1e-9 <= v <= bnd[1] + 1e-9):
                    in_bounds = False
        simplex_ok = bool(np.all(g >= -1e-12) and abs(g.sum() - 1.0) <= 1e-9)
        finite_desc = bool(np.all(np.isfinite([d["WAS"], d["I_gap"], d["Cu"], d["Aag"]])))
        finite_pred = bool(np.all(np.isfinite([muS[i], muQ[i], sS[i], sQ[i], lbS[i], lbQ[i]])))
        sup = float(sup_scores[i])
        mS = float(lbS[i] - s_star)
        mQ = float(lbQ[i] - q_star)
        mH = float(sup - SUPPORT_THRESHOLD)
        feasible = bool(in_bounds and simplex_ok and finite_desc and finite_pred
                        and mH >= 0 and mS >= 0 and mQ >= 0)
        recs.append({
            "C": float(c["C"]), "r": float(c["r"]), "W": float(c["r"]) * float(c["C"]),
            "A": float(c["A"]), "P": float(c["P"]), "S": float(c["S"]),
            "phi": float(c["phi"]), "g": [float(x) for x in g],
            "A_bin": [float(c["A"]) * float(x) for x in g],
            "WAS": d["WAS"], "I_gap": d["I_gap"], "Cu": d["Cu"], "Aag": d["Aag"],
            "mu_S": float(muS[i]), "mu_logQ": float(muQ[i]), "k_pred": float(np.exp(muQ[i])),
            "s_S_raw": float(pred["strength_std_raw_diagnostic_mpa"].to_numpy()[i]),
            "s_Q_raw": float(pred["logperm_std_raw_diagnostic"].to_numpy()[i]),
            "s_S_eff": float(sS[i]), "s_Q_eff": float(sQ[i]),
            "D_AD": float(pred["D_AD"].to_numpy()[i]),
            "AD_ratio": float(pred["AD_ratio"].to_numpy()[i]),
            "AD_infl": float(pred["AD_uncertainty_inflation"].to_numpy()[i]),
            "support": sup, "q": q, "p_nominal": float(p),
            "LB_S": float(lbS[i]), "LB_logQ": float(lbQ[i]),
            "margin_S": mS, "margin_Q": mQ, "margin_H": mH,
            "s_star": float(s_star), "k_star": float(k_star), "q_star": q_star,
            "in_bounds": in_bounds, "simplex_ok": simplex_ok,
            "finite_desc": finite_desc, "finite_pred": finite_pred,
            "MODEL_SCREEN_FEASIBLE": feasible,
            "PHYSICAL_INPUTS_COMPLETE": False,
            "VOLUME_BASIS_CHECKED": False,
            "EXPERIMENTALLY_VERIFIED": False,
            "batch_ready": "NOT_YET_DECLARED",
            "proposal_kind": "porosity-conditioned computational recipe proposal",
        })
    return recs


def evaluate(cand, s_star, k_star, p=0.9, bounds=None, backend="fast"):
    return evaluate_batch([cand], s_star, k_star, p=p, bounds=bounds, backend=backend)[0]


# ---- solver merit (feasibility-first; final label always from the record) ----
def merit(rec, c_max):
    """Feasibility-first scalar merit (lower is better).

    Any feasible candidate outranks any infeasible one; among feasible, lower
    cement wins; among infeasible, merit guides toward feasibility. The exact
    MODEL_SCREEN_FEASIBLE label is never altered by this function.
    """
    if rec["MODEL_SCREEN_FEASIBLE"]:
        return float(rec["C"])
    viol = 0.0
    viol += max(0.0, -rec["margin_S"]) / max(1.0, abs(rec["s_star"]))
    viol += max(0.0, -rec["margin_Q"]) / max(0.5, abs(rec["q_star"]))
    viol += max(0.0, -rec["margin_H"]) / SUPPORT_THRESHOLD
    if not rec["finite_desc"] or not rec["finite_pred"]:
        viol += 10.0
    if not rec["in_bounds"] or not rec["simplex_ok"]:
        viol += 10.0
    return float(c_max) + 1.0 + viol
