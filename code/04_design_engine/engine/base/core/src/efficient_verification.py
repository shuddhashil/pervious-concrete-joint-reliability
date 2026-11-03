"""Decisive independent final verification (Section 11 of the run_04 prompt).

Efficiency design: the expensive step (forward-model prediction at N
manufacturing scenarios) is run EXACTLY ONCE per candidate. The MC/Wilson
point-estimate component reuses one paired residual draw per scenario
(unchanged run_02/run_03 primary estimand). The calibration-uncertainty
component (run_04 REPAIR of run_03 Problem B) precomputes a per-residual-row
pass probability once (O(650 x N), not repeated per replicate) and then
draws EXACTLY 36 study IDs with replacement per bootstrap replicate --
see src/cluster_bootstrap.py and STUDY_CLUSTER_BOOTSTRAP_REPAIR.md.

Sequential escalation: N=20000 -> 50000 near the configured reliability threshold -> 100000
(if still within ~0.002 and the decision could flip).
"""
from __future__ import annotations

import math
import runtime_targets as RT
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import stage1_evaluator as E  # noqa: E402
import uncertainty as U  # noqa: E402
import robust_evaluator as RE  # noqa: E402
import cluster_bootstrap as CB  # noqa: E402
import fast_descriptors as FD  # noqa: E402
import fast_local_residual as FLR  # noqa: E402
import fast_engine as FE  # noqa: E402
import porosity_conditional_residual as PCR  # noqa: E402

if os.environ.get("DISABLE_FAST_DESCRIPTORS") != "1":
    FD.install_patch(E)
    print("[perf] fast_descriptors patch installed", flush=True)
if os.environ.get("DISABLE_FAST_LOCAL_RESIDUAL") != "1":
    FLR.install_patch(E.L)
    print("[perf] fast_local_residual patch installed", flush=True)

POROSITY_CONDITIONAL_CALIBRATION = os.environ.get("POROSITY_CONDITIONAL_CALIBRATION") == "1"
if POROSITY_CONDITIONAL_CALIBRATION:
    print("[perf] porosity-conditional (Mondrian) residual sampling ACTIVE for the "
          "Wilson/point-estimate term -- see template/src/porosity_conditional_residual.py. "
          "The study-cluster-bootstrap calibration term is unaffected.", flush=True)

CONTRACT = os.path.join(RUN, "UNCERTAINTY_CONTRACT.json")
BANK_CSV = os.path.join(RUN, "tables", "paired_residual_bank.csv")
S_STAR = RT.STRENGTH
K_STAR = RT.PERMEABILITY
LOGK_STAR = math.log(K_STAR)  # frozen target, explicit (run_04 Defect-C repair)


PREDICT_CHUNK_SIZE = 10000  # see PREDICTION_CHUNKING_NOTE.md: the frozen GP's
# fast_mode variance computation scales quadratically in the number of test
# points and runs out of memory (observed: OOM at N=50000, fine at N=20000).
# Chunking the SAME frame into independent batches of this size and
# concatenating results is mathematically exact (each test point's marginal
# mean/variance depends only on its own features and the FIXED training/AD
# reference data, never on other test points in the same call) -- verified
# in tests/test_prediction_chunking_equivalence.py by comparing chunked vs.
# unchunked predictions on an unchanged N=2000 case. No model weight, no
# prediction formula, and no frozen-model file is touched by this change.


def compute_scenario_predictions(nominal, mfg_bank):
    """One forward-model pass: returns (muS, muQ, sS, sQ, support, phi_actual) arrays.

    Internally chunked over PREDICT_CHUNK_SIZE rows to avoid the frozen
    model's O(N^2) memory blowup at large N; numerically identical to one
    unchunked call (see PREDICT_CHUNK_SIZE docstring above)."""
    E.get_bundle()
    mfg = U.apply_manufacturing_scenarios(nominal, mfg_bank)
    eng = FE.get_engine(E)
    if eng is not None:
        # fast_engine is row-independent and memory-bounded (its own 8192-row
        # chunks), numerically equivalent to the chunked frozen path below.
        muS, muQ, sS, sQ, support = eng.scenario_arrays(mfg)
        return muS, muQ, sS, sQ, support, mfg["phi_actual"]
    frame = RE._candidates_frame_from_scenarios(mfg)
    n = len(frame)
    mu_s_parts, mu_q_parts, s_s_parts, s_q_parts, support_parts = [], [], [], [], []
    for start in range(0, n, PREDICT_CHUNK_SIZE):
        chunk = frame.iloc[start:start + PREDICT_CHUNK_SIZE].reset_index(drop=True)
        pred = E.predict_fast(E.ORACLE_PT, chunk)
        mu_s_parts.append(pred["strength_pred_mpa"].to_numpy(dtype=float))
        mu_q_parts.append(pred["logperm_pred"].to_numpy(dtype=float))
        s_s_parts.append(pred["strength_std_effective_mpa"].to_numpy(dtype=float))
        s_q_parts.append(pred["logperm_std_effective"].to_numpy(dtype=float))
        support_parts.append(RE._support_scores_for_frame(chunk))
    muS = np.concatenate(mu_s_parts)
    muQ = np.concatenate(mu_q_parts)
    sS = np.concatenate(s_s_parts)
    sQ = np.concatenate(s_q_parts)
    support = np.concatenate(support_parts)
    return muS, muQ, sS, sQ, support, mfg["phi_actual"]


def _joint_pass(muS, muQ, sS, sQ, support, e_strength, e_logk, support_threshold):
    S_real = muS + sS * e_strength
    Q_real = muQ + sQ * e_logk
    in_support = support >= support_threshold
    return in_support & (S_real >= S_STAR) & (Q_real >= LOGK_STAR), S_real - S_STAR, Q_real - LOGK_STAR, in_support


def wilson_lower_bound(k, n, z=1.6448536269514722):
    if n == 0:
        return 0.0
    phat = k / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return max(0.0, (center - margin) / denom)


def decisive_verify(cid, nominal, n_scenarios, n_boot, seed):
    """Full decisive verification at a given (n_scenarios, n_boot). Cheap to
    call repeatedly at increasing n_scenarios since each call still only
    does ONE forward-model pass at that n_scenarios; the caller escalates by
    calling again with a larger n_scenarios, not by adding more bootstrap
    reps to the same forward pass (a larger N needs new manufacturing draws)."""
    t0 = time.time()
    spec = U.UncertaintySpec(CONTRACT)
    mfg_bank = U.build_manufacturing_bank(spec, n=n_scenarios, seed=seed)
    muS, muQ, sS, sQ, support, phi_actual = compute_scenario_predictions(nominal, mfg_bank)
    support_threshold = E.SUPPORT_THRESHOLD

    # primary point estimate: one residual draw per scenario (paired, iid bootstrap)
    # -- the FROZEN run_02/run_03 estimand, unchanged by the run_04 repair.
    if POROSITY_CONDITIONAL_CALIBRATION:
        res_bank = PCR.build_residual_bank_porosity_conditional(BANK_CSV, phi_actual, seed=seed + 1)
    else:
        res_bank = U.build_residual_bank(BANK_CSV, n=n_scenarios, seed=seed + 1, resample_mode="iid_bootstrap")
    joint_pass, s_margin, q_margin, in_support = _joint_pass(
        muS, muQ, sS, sQ, support, res_bank.e_strength, res_bank.e_logk, support_threshold)
    k = int(joint_pass.sum())
    rhat = k / n_scenarios
    wilson_lb = wilson_lower_bound(k, n_scenarios)

    # calibration-uncertainty bootstrap (run_04 REPAIR): the bootstrap UNIT is
    # the 36 source studies, never a scenario-level pseudo-study draw. The
    # expensive muS/muQ/sS/sQ/support arrays (already computed above) are
    # reused as-is; run_cluster_bootstrap() precomputes a per-row pass
    # probability ONCE (O(650 x n_scenarios)) and then every one of n_boot
    # replicates only draws 36 study indices and does an O(36) reweighted
    # sum -- no repeated forward-model calls, no repeated O(n_scenarios) work
    # per replicate.
    if POROSITY_CONDITIONAL_CALIBRATION:
        df, _porosity_bin, _ = PCR.filter_residual_bank_to_scenario_bin(BANK_CSV, phi_actual)
    else:
        df = pd.read_csv(BANK_CSV)
    cb = CB.run_cluster_bootstrap(muS, muQ, sS, sQ, support, support_threshold,
                                   df, S_STAR, LOGK_STAR, n_reps=n_boot, seed=seed + 2)
    calib_lb = cb["R_LCB95_cluster"]

    r_lcb95 = min(wilson_lb, calib_lb)

    def pct(a, q):
        return float(np.percentile(a, q))

    return {
        "cid": cid, "nominal_cement_kg_m3": float(nominal["C"]), "n_scenarios": n_scenarios,
        "n_bootstrap_reps": n_boot, "joint_pass_count": k, "Rhat_joint": rhat,
        "mc_wilson_lower_95": wilson_lb, "calib_bootstrap_R_LCB95": calib_lb, "R_LCB95_joint": r_lcb95,
        "support_fraction": float(in_support.mean()),
        "strength_margin_p05": pct(s_margin, 5), "strength_margin_p50": pct(s_margin, 50),
        "logk_margin_p05": pct(q_margin, 5), "logk_margin_p50": pct(q_margin, 50),
        "achieved_phi_p05": pct(phi_actual, 5), "achieved_phi_p50": pct(phi_actual, 50),
        "achieved_phi_p95": pct(phi_actual, 95),
        "probability_strength_fail_only": float(((s_margin < 0) & (q_margin >= 0) & in_support).mean()),
        "probability_perm_fail_only": float(((s_margin >= 0) & (q_margin < 0) & in_support).mean()),
        "probability_both_fail": float(((s_margin < 0) & (q_margin < 0) & in_support).mean()),
        "probability_support_fail": float((~in_support).mean()),
        "mfg_bank_hash": mfg_bank.bank_hash(), "res_bank_hash": res_bank.bank_hash(),
        "model_hash": "1bf27004724bd311a5cfec98b4084d1063fe739704be2344616bf52698851a69",
        "uncertainty_contract_hash": spec.spec_hash(),
        "n_studies_cluster_bootstrap": cb["n_studies"],
        "per_study_table": cb["per_study_table"],
        "boot_rhats": cb["boot_rhats"].tolist(),
        "elapsed_s": time.time() - t0,
    }


def sequential_decisive_verify(cid, nominal, seed, n_boot=100000, verbose=True):
    """N=20000 -> 50000 -> 100000 escalation. run_05 uses the STABILIZED
    100,000-replicate cluster bootstrap as the production default from the
    start (run_04's BOOTSTRAP_STABILIZATION_NOTE.md finding: 500 replicates
    is not numerically stable enough for a candidate near the 0.90 gate)."""
    r = decisive_verify(cid, nominal, 20000, n_boot, seed)
    if verbose:
        print(f"    [escalation N=20000] Rhat={r['Rhat_joint']:.4f} R_LCB95={r['R_LCB95_joint']:.4f} "
              f"gap={abs(r['R_LCB95_joint']-RT.RELIABILITY):.4f} elapsed={r['elapsed_s']:.1f}s", flush=True)
    if abs(r["R_LCB95_joint"] - RT.RELIABILITY) > 0.005:
        return r, "N=20000 decisive (gap > 0.005)"
    r2 = decisive_verify(cid, nominal, 50000, n_boot, seed + 100)
    if verbose:
        print(f"    [escalation N=50000] Rhat={r2['Rhat_joint']:.4f} R_LCB95={r2['R_LCB95_joint']:.4f} "
              f"gap={abs(r2['R_LCB95_joint']-RT.RELIABILITY):.4f} elapsed={r2['elapsed_s']:.1f}s", flush=True)
    if abs(r2["R_LCB95_joint"] - RT.RELIABILITY) > 0.002:
        return r2, "N=50000 decisive (gap > 0.002)"
    r3 = decisive_verify(cid, nominal, 100000, n_boot, seed + 200)
    return r3, "N=100000 (hard cap reached, still within 0.002 of threshold)"
