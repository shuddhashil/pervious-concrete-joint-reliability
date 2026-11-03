"""Frozen Stage-17 direct local recipe-porosity state support (runtime module).

Nominal-only empirical state-admissibility constraint for the Stage-18
inverse optimizer. For a candidate recipe x and target hardened porosity phi:

  1. x passes the frozen x-only recipe gate (k=10 kth-NN, thr=3.346102); AND
  2. phi lies inside the frozen B8 stacked-ensemble conformalized 90% HDR.

Semantics (contract): empirical local state support, NOT physical
attainability. The B8 density is NEVER sampled as manufacturing uncertainty;
HDR mass is NEVER multiplied into performance reliability. No CS, permeability,
target, or outcome variable enters any calculation here.

Bundle: ../state_support_stage17/b8_fulldata_bundle.pkl (frozen 2026-09-19,
reconstructed from frozen Stage-17 sources; see STATE_SUPPORT_CONTRACT.json).
Self-contained: imports only numpy/sklearn/scipy + the bundled frozen sources.
"""
import os
import pickle
import sys

import numpy as np

# numpy>=2 renamed trapz->trapezoid; the frozen hdr.py uses np.trapezoid.
# Shim it for the bundled optimizer runtime (numpy 1.26) BEFORE importing hdr.
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
BUNDLE_DIR = os.path.join(RUN, "state_support_stage17")
if BUNDLE_DIR not in sys.path:
    sys.path.insert(0, BUNDLE_DIR)

import conformal_hcds as CH  # noqa: E402
import hdr as H  # noqa: E402
import recipe_space as RS  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

_BUNDLE = None
_GATE_NN = None

LEVELS = (0.8, 0.9, 0.95)
ALPHA_OF = {0.8: 0.2, 0.9: 0.1, 0.95: 0.05}
DEPLOY_LO, DEPLOY_HI = 20.0, 30.0


def _bundle():
    global _BUNDLE, _GATE_NN
    if _BUNDLE is None:
        with open(os.path.join(BUNDLE_DIR, "b8_fulldata_bundle.pkl"), "rb") as f:
            _BUNDLE = pickle.load(f)
        _GATE_NN = NearestNeighbors(n_neighbors=int(_BUNDLE["gate_k"]))
        _GATE_NN.fit(np.asarray(_BUNDLE["gate_X_fit"], float))
    return _BUNDLE


def candidate_to_xraw(cand):
    """Optimizer candidate -> frozen Stage-17 primary recipe coordinates.

    cand: {C, r, A, P, S, phi, g(6)}; x_raw = [r, C, P, S, b1..b6], b_j = A*g_j.
    """
    g = np.asarray(cand["g"], dtype=float)
    b = float(cand["A"]) * g
    return np.array([[float(cand["r"]), float(cand["C"]), float(cand["P"]),
                      float(cand["S"]), b[0], b[1], b[2], b[3], b[4], b[5]]])


def transform_xraw(Xraw):
    """Frozen monotone transforms: log cem; log1p SP/sand/b1..b6; raw w/c."""
    Xraw = np.asarray(Xraw, float)
    return np.hstack([Xraw[:, [0]], np.log(Xraw[:, [1]]),
                      np.log1p(Xraw[:, 2:10])])


def recipe_support_batch(Xraw_list):
    """x-only recipe gate for a batch of raw x_raw rows.

    Returns list of dicts: pass/fail, distance, threshold, normalized excess
    (continuous violation, >=0 iff FAIL), margin (thr-d), percentile.
    Invariant to phi by construction (phi never passed).
    """
    B = _bundle()
    Xraw = np.asarray(Xraw_list, float)
    Xt = transform_xraw(Xraw)
    Xs = (Xt - B["scaler_mean"]) / B["scaler_sd"]
    d, _ = _GATE_NN.kneighbors(Xs)
    d = d[:, -1]
    thr = float(B["gate_threshold"])
    held = np.asarray(B["gate_held"], float)
    out = []
    for di in d:
        excess = max(0.0, (di - thr) / thr)
        out.append({
            "recipe_pass": bool(di <= thr),
            "recipe_distance": float(di),
            "recipe_threshold": thr,
            "recipe_excess": float(excess),
            "recipe_margin": float(thr - di),
            "recipe_percentile": float((held <= di).mean() * 100.0),
        })
    return out


def recipe_support(x_raw):
    return recipe_support_batch([x_raw])[0]


def _stack_density(Xt_q, Xs_q, member_names=None, weights=None):
    """Frozen B8 stacked p(phi|x) on the frozen grid. (nq, G)."""
    B = _bundle()
    names = member_names or B["member_names"]
    w = np.asarray(B["weights"] if weights is None else weights, float)
    grid = B["phi_grid"]
    mats = [B["fitted_members"][nm].grid_density(Xt_q, Xs_q, None, grid)
            for nm in names]
    return sum(wi * np.asarray(mt, float) for wi, mt in zip(w, mats))


def porosity_density_batch(Xraw_list):
    """B8 conditional density rows on the frozen grid. (nq, G)."""
    B = _bundle()
    Xraw = np.asarray(Xraw_list, float)
    Xt = transform_xraw(Xraw)
    Xs = (Xt - B["scaler_mean"]) / B["scaler_sd"]
    return _stack_density(Xt, Xs)


def _intervals_from_mask(mask, grid):
    ivs, tot = H.mask_to_intervals(np.asarray(mask, bool), grid)
    return ivs, float(tot)


def _dist_to_intervals(phi, intervals):
    """(inside, dist): dist=0 if inside else min distance to nearest component."""
    if H.covers(intervals, phi):
        edge = min(min(abs(phi - lo), abs(phi - hi)) for lo, hi in intervals)
        return True, 0.0, -float(edge)
    if not intervals:
        return False, float("inf"), float("inf")
    d = min(max(lo - phi, phi - hi, 0.0) for lo, hi in intervals)
    return False, float(d), float(d)


def porosity_hdr_batch(Xraw_list, masses=(0.8, 0.9, 0.95), conformal=True, dens=None):
    """Base and/or conformal HDR interval unions. Gaps never bridged.

    ``dens``: optional precomputed ``porosity_density_batch(Xraw_list)`` (identical
    deterministic result; lets ``state_support_batch`` avoid computing the stacked
    density twice per candidate)."""
    B = _bundle()
    grid = B["phi_grid"]
    if dens is None:
        dens = porosity_density_batch(Xraw_list)
    out = []
    for row in dens:
        rec = {}
        for lv in masses:
            hb = H.density_to_hdr(row, grid, lv)
            rec[f"base{int(lv * 100)}"] = {
                "intervals": hb["intervals"],
                "total_measure": float(hb["total_measure"]),
                "n_modes": int(hb["n_modes"]),
            }
            if conformal:
                t = float(B["conformal_thresholds"][ALPHA_OF[lv]])
                ivs, tot = _intervals_from_mask(row >= t, grid)
                rec[f"conf{int(lv * 100)}"] = {
                    "intervals": ivs, "total_measure": tot,
                    "n_modes": len(ivs), "threshold": t,
                }
        out.append(rec)
    return out


def state_support_batch(cands):
    """Full nominal state-support diagnostic per candidate (descriptive)."""
    B = _bundle()
    grid = np.asarray(B["phi_grid"], float)
    step = float(B["phi_step"])
    Xraw = np.array([candidate_to_xraw(c)[0] for c in cands])
    gate = recipe_support_batch(Xraw)
    dens = porosity_density_batch(Xraw)
    hdrs = porosity_hdr_batch(Xraw, dens=dens)
    out = []
    for c, g, dn, hd in zip(cands, gate, dens, hdrs):
        phi = float(c["phi"])
        rec = dict(g)
        rec["target_phi"] = phi
        f_at = float(np.interp(phi, grid, dn))
        rec["density_at_target"] = f_at
        rec["density_rank"] = float((dn <= f_at).mean())
        rec["cdf_at_target"] = float(np.interp(
            phi, grid, H.conditional_cdf(dn, grid)))
        num = float(np.trapezoid(dn, dx=step))
        rec["density_integral"] = num
        for lv in LEVELS:
            tag = int(lv * 100)
            rec[f"base{tag}_intervals"] = hd[f"base{tag}"]["intervals"]
            rec[f"base{tag}_width"] = hd[f"base{tag}"]["total_measure"]
            rec[f"base{tag}_nmodes"] = hd[f"base{tag}"]["n_modes"]
            rec[f"base{tag}_inside"] = bool(
                H.covers(hd[f"base{tag}"]["intervals"], phi))
            rec[f"conf{tag}_intervals"] = hd[f"conf{tag}"]["intervals"]
            rec[f"conf{tag}_width"] = hd[f"conf{tag}"]["total_measure"]
            rec[f"conf{tag}_nmodes"] = hd[f"conf{tag}"]["n_modes"]
            inside, dist, signed = _dist_to_intervals(
                phi, hd[f"conf{tag}"]["intervals"])
            rec[f"conf{tag}_inside"] = inside
            rec[f"conf{tag}_dist"] = dist
            rec[f"conf{tag}_signed_dist"] = signed
        c90 = hd["conf90"]["intervals"]
        rec["conf90_measure"] = hd["conf90"]["total_measure"]
        rec["conf90_n_components"] = hd["conf90"]["n_modes"]
        if g["recipe_pass"]:
            inside, dist, signed = _dist_to_intervals(phi, c90)
            rec["phi_in_conf90"] = inside
            rec["phi_violation"] = 0.0 if inside else dist
            rec["phi_signed_dist"] = signed
        else:
            # No HDR is fabricated for a recipe outside the evidence domain;
            # the recipe-space violation governs.
            rec["phi_in_conf90"] = False
            rec["phi_violation"] = float("nan")
            rec["phi_signed_dist"] = float("nan")
        rec["deployment_ok"] = bool(DEPLOY_LO <= phi <= DEPLOY_HI)
        rec["empirical_pass"] = bool(
            g["recipe_pass"] and rec["phi_in_conf90"])
        rec["state_support_pass"] = bool(
            rec["empirical_pass"] and rec["deployment_ok"])
        out.append(rec)
    return out


def state_support(cand):
    return state_support_batch([cand])[0]


def violations(cand):
    """Continuous constraint violations for DE/NOMAD/Phase-D merit functions.

    Returns (recipe_violation, phi_violation), both 0.0 iff satisfied.
    recipe: normalized excess (d-thr)/thr beyond the frozen gate threshold.
    phi: distance in porosity percentage points to the nearest conf90
    component (nearest valid component when disconnected; gaps never bridged).
    A recipe-gate failure governs: phi violation is reported as nan and must
    be treated as dominated by the recipe violation by the caller.
    """
    r = state_support(cand)
    v_recipe = r["recipe_excess"]
    v_phi = 0.0 if r["phi_in_conf90"] else r["phi_violation"]
    return float(v_recipe), float(v_phi)


def b2_conf90_batch(Xraw_list):
    """Auditable runner-up B2 NNKCDE conf90 region (sensitivity only)."""
    B = _bundle()
    grid = np.asarray(B["phi_grid"], float)
    Xraw = np.asarray(Xraw_list, float)
    Xt = transform_xraw(Xraw)
    Xs = (Xt - B["scaler_mean"]) / B["scaler_sd"]
    mats = B["fitted_members"]["B2"].grid_density(Xt, Xs, None, grid)
    t = float(B["member_conformal_thresholds"]["B2"]["0.1"])
    out = []
    for row in np.asarray(mats, float):
        ivs, tot = _intervals_from_mask(row >= t, grid)
        out.append({"intervals": ivs, "total_measure": tot,
                    "n_modes": len(ivs), "threshold": t})
    return out


def manufacturing_cloud_support(nominal, scenarios, max_rows=2000):
    """MANUFACTURING_CLOUD_EMPIRICAL_SUPPORT_DIAGNOSTIC (descriptive only).

    Passes a fixed subset of ALREADY-SAMPLED manufacturing scenarios through
    Stage-17 support. scenarios: dict from
    uncertainty.apply_manufacturing_scenarios (must NOT be resampled here).
    Reports fractions; NEVER enters R_LCB95.
    """
    idx = np.arange(scenarios["C_actual"].shape[0])
    if len(idx) > max_rows:
        idx = idx[:max_rows]  # fixed prefix: deterministic, no resampling
    rows = []
    for i in idx:
        g = np.asarray(scenarios["g_actual"][i], float)
        rows.append({"C": float(scenarios["C_actual"][i]),
                     "r": float(scenarios["wc_actual"][i]),
                     "A": float(scenarios["A_actual"][i]),
                     "P": float(scenarios["SP_actual"][i]),
                     "S": float(scenarios["S_actual"][i]),
                     "phi": float(scenarios["phi_actual"][i]), "g": g})
    res = state_support_batch(rows)
    n = len(res)
    n_gate = sum(1 for r in res if r["recipe_pass"])
    n_phi = sum(1 for r in res if r["phi_in_conf90"])
    n_both = sum(1 for r in res if r["empirical_pass"])
    return {
        "n_diagnostic_scenarios": n,
        "fraction_recipe_supported": n_gate / n,
        "fraction_phi_in_conf90": n_phi / n,
        "fraction_empirically_supported": n_both / n,
        "fraction_outside_recipe_gate": 1.0 - n_gate / n,
        "fraction_outside_hdr": 1.0 - n_phi / n,
        "fraction_outside_either": 1.0 - n_both / n,
        "note": ("descriptive empirical-support diagnostic of an already-"
                 "sampled manufacturing cloud; not an attainability "
                 "probability and not used in R_LCB95"),
    }
