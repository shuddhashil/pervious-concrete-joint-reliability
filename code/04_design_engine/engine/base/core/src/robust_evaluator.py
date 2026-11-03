"""Robust joint-reliability evaluator: nominal design -> Monte-Carlo joint
reliability under manufacturing + paired predictive-residual uncertainty.

result = evaluate_robust(nominal_design, manufacturing_bank, residual_bank,
                          mode="paired_residual_mc", backend="fast")

Rebuild order per manufacturing scenario:
  perturbed masses/porosity (uncertainty.apply_manufacturing_scenarios)
  -> physical feature vector (10-D, same order as the frozen oracle)
  -> applicability-domain support check (frozen support model)
  -> model mean/scale evaluation (stage1_evaluator fast/exact backend)
  -> paired signed residual draw -> realised strength/logK
  -> joint pass/fail (out-of-support ALWAYS counts as failure)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage1_evaluator as E  # noqa: E402
import uncertainty as U  # noqa: E402
import fast_engine as FE  # noqa: E402
try:
    import runtime_targets as RT  # generated per run by run_stage1.py
except ImportError:  # template-level tests before run generation; a REAL import error (bad syntax, corrupted generated file) now propagates instead of silently falling back
    class RT:  # noqa: N801
        STRENGTH = 18.0
        PERMEABILITY = 13.0

LN13 = math.log(float(RT.PERMEABILITY))


def _sha256_text(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _candidates_frame_from_scenarios(mfg: dict) -> pd.DataFrame:
    """Build the production DataFrame (raw units, bins in %) for all n
    manufacturing scenarios at once, reusing stage1_evaluator's descriptor
    convention for each scenario's realised gradation."""
    n = mfg["C_actual"].shape[0]
    rows = []
    for i in range(n):
        g = mfg["g_actual"][i]
        d = E.cached_descriptors(g)
        rows.append({
            "porosity": float(mfg["phi_actual"][i]), "w/c": float(mfg["wc_actual"][i]),
            "cem (kg/m3)": float(mfg["C_actual"][i]), "CA (kg/m3)": float(mfg["A_actual"][i]),
            "SP (kg/m3)": float(mfg["SP_actual"][i]), "sand (kg/m3)": float(mfg["S_actual"][i]),
            "WAS": d["WAS"], "I_gap": d["I_gap"], "Cu": d["Cu"], "Aag(mm^-1)": d["Aag"],
            "q10": d["q10"], "q25": d["q25"], "q50": d["q50"], "q75": d["q75"], "q90": d["q90"],
        })
    return pd.DataFrame(rows)


def _support_scores_for_frame(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(E._support_scores(frame), dtype=float)


@dataclass
class RobustResult:
    fields: dict
    search_arrays: dict | None = None

    def __getitem__(self, k):
        return self.fields[k]

    def to_dict(self):
        return self.fields


def evaluate_robust(nominal_design: dict, manufacturing_bank: U.ManufacturingScenarioBank,
                     residual_bank: U.ResidualScenarioBank, mode: str = "paired_residual_mc",
                     backend: str = "fast", s_star: float = RT.STRENGTH,
                     k_star: float = RT.PERMEABILITY,
                     capture_search_arrays: bool = False
                     ) -> RobustResult:
    if manufacturing_bank.n != residual_bank.n:
        raise ValueError("manufacturing_bank.n must equal residual_bank.n (paired scenario rows)")
    n = manufacturing_bank.n

    E.get_bundle()  # ensure the 10D contract is activated before ANY predict_fast/exact call
    mfg = U.apply_manufacturing_scenarios(nominal_design, manufacturing_bank)
    if backend not in ("exact", "fast"):
        raise ValueError(f"unknown backend {backend}")

    # fast_engine: numerically equivalent re-expression of the frozen path
    # (self-checked; DISABLE_FAST_ENGINE=1 or a failed check falls back to it).
    eng = FE.get_engine(E) if backend == "fast" else None
    if eng is not None:
        muS, muQ, sS, sQ, support = eng.scenario_arrays(mfg)
    else:
        frame = _candidates_frame_from_scenarios(mfg)
        if backend == "exact":
            pred = E.predict_ca_oracle_dataframe(E.ORACLE_PT, frame)
        else:
            pred = E.predict_fast(E.ORACLE_PT, frame)
        support = _support_scores_for_frame(frame)
        muS = pred["strength_pred_mpa"].to_numpy(dtype=float)
        muQ = pred["logperm_pred"].to_numpy(dtype=float)
        sS = pred["strength_std_effective_mpa"].to_numpy(dtype=float)
        sQ = pred["logperm_std_effective"].to_numpy(dtype=float)

    if mode == "paired_residual_mc":
        S_real = muS + sS * residual_bank.e_strength
        logK_real = muQ + sQ * residual_bank.e_logk
    else:
        raise NotImplementedError(f"reliability mode {mode!r} not implemented in this evaluator "
                                   f"(nested_joint_lower_bound is implemented separately -- see "
                                   f"src/nested_lower_bound.py)")

    logk_star = math.log(float(k_star))  # use the PASSED target, never a silently hard-coded one
    in_support = support >= E.SUPPORT_THRESHOLD
    strength_ok = S_real >= s_star
    perm_ok = logK_real >= logk_star

    joint_pass = in_support & strength_ok & perm_ok  # out-of-support => failure, always
    strength_margin = S_real - s_star
    logk_margin = logK_real - logk_star
    joint_margin = np.minimum(strength_margin, logk_margin)  # for ranking only; pass/fail uses literal targets

    n_pass = int(joint_pass.sum())
    rhat = n_pass / n
    se = math.sqrt(max(rhat * (1 - rhat), 0.0) / n)  # exactly 0 at rhat in {0,1}, never an artificial floor

    support_fail = ~in_support
    strength_fail_only = (~strength_ok) & perm_ok & in_support
    perm_fail_only = strength_ok & (~perm_ok) & in_support
    both_fail = (~strength_ok) & (~perm_ok) & in_support

    def pct(arr, q):
        return float(np.percentile(arr, q))

    fields = {
        "nominal_cement": float(nominal_design["C"]),
        "n_scenarios": n,
        "joint_pass_count": n_pass,
        "joint_reliability_hat": rhat,
        "mc_standard_error": se,
        "support_fraction": float(in_support.mean()),
        "out_of_support_fraction": float(support_fail.mean()),

        "strength_margin_p01": pct(strength_margin, 1), "strength_margin_p05": pct(strength_margin, 5),
        "strength_margin_p10": pct(strength_margin, 10), "strength_margin_p50": pct(strength_margin, 50),
        "logk_margin_p01": pct(logk_margin, 1), "logk_margin_p05": pct(logk_margin, 5),
        "logk_margin_p10": pct(logk_margin, 10), "logk_margin_p50": pct(logk_margin, 50),
        "joint_margin_p01": pct(joint_margin, 1), "joint_margin_p05": pct(joint_margin, 5),
        "joint_margin_p10": pct(joint_margin, 10), "joint_margin_p50": pct(joint_margin, 50),

        "achieved_phi_p05": pct(mfg["phi_actual"], 5), "achieved_phi_p50": pct(mfg["phi_actual"], 50),
        "achieved_phi_p95": pct(mfg["phi_actual"], 95),

        "probability_strength_fail_only": float(strength_fail_only.mean()),
        "probability_perm_fail_only": float(perm_fail_only.mean()),
        "probability_both_fail": float(both_fail.mean()),
        "probability_support_fail": float(support_fail.mean()),

        "reliability_mode": mode,
        "uncertainty_contract_hash": manufacturing_bank.spec_hash,
        "model_hash": "1bf27004724bd311a5cfec98b4084d1063fe739704be2344616bf52698851a69",
        "residual_bank_hash": residual_bank.bank_hash(),
        "scenario_bank_hash": manufacturing_bank.bank_hash(),
        "backend": backend,
        "truncation_stats": manufacturing_bank.truncation_stats,
    }
    search_arrays = None
    if capture_search_arrays:
        # Search-only data.  These arrays are deliberately kept outside
        # ``fields`` so existing CSV/JSON outputs and final acceptance remain
        # byte-for-byte compatible with the authoritative result schema.
        search_arrays = {
            "strength_real": np.asarray(S_real, dtype=float),
            "logk_real": np.asarray(logK_real, dtype=float),
            "in_support": np.asarray(in_support, dtype=bool),
        }
    return RobustResult(fields, search_arrays=search_arrays)


def wilson_lower_bound(k: int, n: int, z: float = 1.6448536269514722) -> float:
    """One-sided 95% Wilson lower bound for a binomial proportion (z=1.6449
    is the one-sided 95% normal quantile)."""
    if n == 0:
        return 0.0
    phat = k / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return max(0.0, (center - margin) / denom)
