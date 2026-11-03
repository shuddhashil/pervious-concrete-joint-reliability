"""Phase A - full 11-DOF continuous DE search (run_05).

Adapted from run_04/src/robust_de.py (itself the repaired canonical
DE/rand/1/bin on the frozen stick11 codec11.py parameterisation, inherited
from run_02/run_03). The ONLY structural change from robust_de.py: this
version (a) retains EVERY evaluated candidate with Rhat>=SHORTLIST_RHAT_MIN
across all generations (not only the single running best, per Section 6 of
the driving prompt), and (b) is parameterised for a single seed per process
invocation so seeds can be run as independent parallel processes.

Objective/merit is feasibility-first and blind-discovery aware: unsupported
states cannot win; supported points above Rhat=0.90 minimize cement; below
0.90, reliability shortfall remains primary while normalized cement stays in
the score.  This preserves useful low-cement near-misses without weakening
the final decisive acceptance rule.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(RUN)))
sys.path.insert(0, HERE)
import codec11 as K  # noqa: E402
import solvers_common as SC  # noqa: E402
import stage1_evaluator as E  # noqa: E402
import uncertainty as U  # noqa: E402
import robust_evaluator as RE  # noqa: E402
import fast_descriptors as FD  # noqa: E402
import fast_local_residual as FLR  # noqa: E402
import stage17_state_support as S17  # noqa: E402  (Stage-18 nominal state-support gate)
import gradation_support as GS  # noqa: E402  (TASK A2 ordered-bin W1 gate)
try:
    import runtime_targets as RT  # generated per run by run_stage1.py
except ImportError:  # template-level tests before run generation; a REAL import error (bad syntax, corrupted generated file) now propagates instead of silently falling back
    class RT:  # noqa: N801
        STRENGTH = 18.0
        PERMEABILITY = 13.0
        RELIABILITY = 0.90
try:  # Blind-repair helpers: search-only guidance, never acceptance.
    import continuous_score as CS  # noqa: E402
    import batched_eval as BE  # noqa: E402
    import multifidelity as MF  # noqa: E402
except Exception:  # pragma: no cover
    CS = None
    BE = None
    MF = None

# Vectorised, numerically-verified replacement for the frozen evaluator's
# cached_descriptors (see test_fast_descriptors_equivalence.py -- 4026
# gradations incl. zero-mass boundary cases, max relative diff < 1e-9).
# Scoped to THIS process only via monkey-patch; the frozen file on disk is
# never modified, so run_04 and other consumers are unaffected.
if os.environ.get("DISABLE_FAST_DESCRIPTORS") != "1":
    FD.install_patch(E)
    print("[perf] fast_descriptors patch installed (vectorised gradation "
          "descriptors; verified numerically equivalent to the frozen "
          "implementation)", flush=True)

if os.environ.get("DISABLE_FAST_LOCAL_RESIDUAL") != "1":
    FLR.install_patch(E.L)
    print("[perf] fast_local_residual patch installed", flush=True)

CONTRACT = os.path.join(RUN, "UNCERTAINTY_CONTRACT.json")
BANK_CSV = os.path.join(RUN, "tables", "paired_residual_bank.csv")
TRAINING_CSV = os.path.join(RUN, "tables", "training_data_reconciled.csv")

REL_TARGET = float(RT.RELIABILITY)
# The 0.90 requirement is a discovery constraint, not a 0.92 cliff.  A
# safety margin belongs in final decisive verification, where its uncertainty
# is explicitly quantified; applying one as a hard DE threshold discarded
# low-cement near-misses before they could guide the blind search.
WORKING_MARGIN = 0.02
WORKING_TARGET = REL_TARGET
SHORTLIST_RHAT_MIN = max(0.0, REL_TARGET - 0.05)
ORDER = "coarse_first"
SEARCH_BANK_N = 512
SEARCH_BANK_SEED = 205609151
C_MAX = SC.BND["C"][1]
_GRADATION_CONTRACT = None


def load_seed_candidates():
    # Compatibility hook: the target-agnostic initializer owns all starts.
    # Empty dict prevents callers from injecting an external coordinate pool.
    return {}


def state_support_penalty(cand, return_details=False):
    """Stage-18 nominal state-support violation + TASK A2 W1 gradation gate.

    Returns (penalty, empirical_pass, deployment_ok). 0.0 penalty iff the
    candidate passes the recipe gate, phi-in-conf90-HDR, the 20-30%
    deployment domain, AND the ordered-bin W1 gradation gate (when enabled
    in GRADATION_SUPPORT_CONTRACT.json). W1 violation is zero inside the
    region (support constraint, not preference).
    """
    if os.environ.get("DISABLE_STAGE17_SUPPORT") == "1":
        details = {"recipe_gate": True, "porosity_hdr": True,
                   "deployment_window": True, "gradation_w1_gate": True,
                   "state_support": True}
        return (0.0, True, True, details) if return_details else (0.0, True, True)
    ss = S17.state_support(cand)
    v_r = float(ss["recipe_excess"])
    v_p_raw = float(ss["phi_violation"])
    v_p = 0.0 if (v_p_raw != v_p_raw or ss["phi_in_conf90"]) else v_p_raw / 10.0
    lo, hi = S17.DEPLOY_LO, S17.DEPLOY_HI
    v_d = 0.0 if ss["deployment_ok"] else min(
        abs(float(cand["phi"]) - lo), abs(float(cand["phi"]) - hi)) / 10.0
    v_w = 0.0
    try:
        global _GRADATION_CONTRACT
        if _GRADATION_CONTRACT is None:
            _GRADATION_CONTRACT = GS.load_contract()
        gc = _GRADATION_CONTRACT
        w1_min = float("nan")
        if gc.get("enabled", True) and os.environ.get("DISABLE_GRADATION_GATE") != "1":
            v_w, w1_min = GS.violation(np.asarray(cand["g"], float), gc.get("epsilon_95"))
    except Exception:
        # A failed scientific support check must never silently become a pass.
        details = {"recipe_gate": False, "porosity_hdr": False,
                   "deployment_window": False, "gradation_w1_gate": False,
                   "state_support": False, "support_check_error": True}
        return ((float("inf"), False, False, details) if return_details
                else (float("inf"), False, False))
    details = {
        "recipe_gate": bool(ss["recipe_pass"]),
        "porosity_hdr": bool(ss["phi_in_conf90"]),
        "deployment_window": bool(ss["deployment_ok"]),
        "gradation_w1_gate": bool(v_w <= 0.0),
        "state_support": bool(v_r <= 0.0 and v_p <= 0.0 and v_d <= 0.0 and v_w <= 0.0),
        "gradation_w1_min": float(w1_min),
    }
    value = (v_r + v_p + v_d + v_w, bool(ss["empirical_pass"]), bool(ss["deployment_ok"]), details)
    return value if return_details else value[:3]


def infeasible_search_merit(rhat, cvar, c_norm):
    """Reliability-primary merit below the provisional feasibility target."""
    viol = max(0.0, REL_TARGET - float(rhat))
    smooth_tie = (0.1 * min(10.0, max(0.0, -float(cvar)))
                  if np.isfinite(cvar) else 1.0)
    return C_MAX + 1000.0 * viol + smooth_tie + 0.01 * float(c_norm)


def robust_merit(x11, mfg_bank, res_bank):
    cand = K.decode11(x11, ORDER)
    lo_hi = SC.BND
    phys_ok = all(lo_hi[k][0] - 1e-9 <= cand[k] <= lo_hi[k][1] + 1e-9 for k in ("C", "r", "A", "P", "S", "phi"))
    if not phys_ok:
        viol = 0.0
        for k in ("C", "r", "A", "P", "S", "phi"):
            lo, hi = lo_hi[k]
            span = max(hi - lo, 1e-9)
            if cand[k] < lo:
                viol += (lo - cand[k]) / span
            elif cand[k] > hi:
                viol += (cand[k] - hi) / span
        return C_MAX + 2.0 + viol, 0.0, cand, False, 0.0

    # Stage-18: nominal state-support screen (cheap, ~0.1 s) BEFORE the
    # expensive robust evaluation. A state violation never minimizes cement;
    # the continuous penalty guides DE toward the supported region while the
    # robust reliability below is still evaluated so near-feasible basins
    # retain their Rhat signal for Phase-B diagnosis.
    state_pen, _, _ = state_support_penalty(cand)
    result = RE.evaluate_robust(
        cand, mfg_bank, res_bank, backend="fast",
        capture_search_arrays=CS is not None,
    )
    rhat = result["joint_reliability_hat"]
    support_fraction = result["support_fraction"]
    cvar = float("-inf")
    if CS is not None and result.search_arrays is not None:
        search = CS.search_score(
            result.search_arrays["strength_real"],
            result.search_arrays["logk_real"],
            RT.STRENGTH, RT.PERMEABILITY,
            result.search_arrays["in_support"],
        )
        cvar = float(search["cvar"])
    c_norm = (float(cand["C"]) - SC.BND["C"][0]) / max(SC.BND["C"][1] - SC.BND["C"][0], 1e-9)
    if state_pen > 0.0:
        # State-violating candidate: cement is NOT minimized; the merit stays
        # above C_max (so any state-supported reliability-feasible candidate
        # wins) while retaining the reliability gradient for diagnosis.
        viol = max(0.0, REL_TARGET - rhat)
        return C_MAX + 1000.0 + 100.0 * viol + state_pen, rhat, cand, True, support_fraction
    if rhat >= WORKING_TARGET:
        return float(cand["C"]), rhat, cand, True, support_fraction
    # Reliability remains the primary constraint below 0.90, but cement is
    # deliberately present in the score at every reliability level.  Thus a
    # supported 270-kg near-miss is retained on the frontier rather than
    # being erased until it crosses an arbitrary 0.92 threshold.
    # Literal reliability shortfall is the primary search coordinate.  The
    # smooth tail margin is only a bounded tie-breaker: a low-Rhat candidate
    # must never beat a substantially higher-Rhat candidate merely because
    # its failing scenarios miss by a smaller magnitude.
    return (infeasible_search_merit(rhat, cvar, c_norm),
            rhat, cand, True, support_fraction)


def robust_merit_batch(X, mfg_bank, res_bank):
    """Bounded-batch merit evaluation (scientifically identical, safe fallback).

    Candidates are processed in bounded chunks (STAGE1_EVAL_BATCH) to cap
    memory; on any failure the sequential path is used instead.  The merit
    itself is feasibility-first: unsupported designs sort above C_MAX, and
    below the working target reliability shortfall dominates while cement
    stays informative (feasibility archive and cement frontier are logged
    separately by the caller, never merged into one scalar that lets a
    low-cement hopeless candidate displace a more feasible one).
    """
    X = list(X)
    if BE is not None:
        try:
            return _robust_merit_batched(X, mfg_bank, res_bank)
        except Exception as ex:  # noqa: BLE001 - fall back safely
            print(f"[eval] batched path failed ({ex}); using sequential fallback", flush=True)
    return _robust_merit_sequential(X, mfg_bank, res_bank)


def _robust_merit_sequential(X, mfg_bank, res_bank):
    merits = np.empty(len(X))
    rhats = np.empty(len(X))
    supports = np.empty(len(X))
    cands = []
    for i, x in enumerate(X):
        m, r, c, ok, sup = robust_merit(x, mfg_bank, res_bank)
        merits[i] = m
        rhats[i] = r
        supports[i] = sup
        cands.append(c)
    return merits, rhats, supports, cands


def _robust_merit_batched(X, mfg_bank, res_bank):
    size = BE.batch_size()
    merits_list, rhats_list, supports_list, cands = [], [], [], []

    def one(x):
        return robust_merit(np.asarray(x, dtype=float), mfg_bank, res_bank)

    for i in range(0, len(X), size):
        for row in BE.batched_map(one, X[i:i + size], size=size):
            m, r, c, ok, sup = row
            merits_list.append(m)
            rhats_list.append(r)
            supports_list.append(sup)
            cands.append(c)
    return np.asarray(merits_list), np.asarray(rhats_list), np.asarray(supports_list), cands


def continuous_search_quality(rhats, cements, supported):
    """Smooth feasibility quality for ranking (search-only, never PASS).

    Uses the Rhat/cement Pareto geometry already established in
    adaptive_init: supported candidates rank by shortfall-then-cement, which
    is continuous in reliability and keeps low-cement near-misses visible.
    Literal Rhat is still logged alongside; final acceptance never reads
    this score.
    """
    import adaptive_init as AI  # noqa: E402
    rhats = np.asarray(rhats, dtype=float)
    cements = np.asarray(cements, dtype=float)
    supported = np.asarray(supported, dtype=bool)
    c_norm = (cements - SC.BND["C"][0]) / max(SC.BND["C"][1] - SC.BND["C"][0], 1e-9)
    quality = np.where(supported, -(np.maximum(0.0, REL_TARGET - rhats)) - 0.01 * c_norm,
                       float("-inf"))
    order, feas_rank = AI.feasibility_cement_frontier(quality, cements, supported)
    return quality, order, feas_rank


def build_init_population(rng, popsize, seed_cands, *, seed, n_scenarios):
    # TASK A: target-agnostic adaptive initialiser. No named seeds.
    # Use the support-aware, target-independent proposal mixture. seed_cands
    # is ignored (kept for signature compatibility) to guarantee no external
    # coordinate dependence.
    import adaptive_init as AI  # noqa: E402
    n0 = 4608  # former Stage-0 + Stage-0b total, now one labeled mixture
    stage0_seed = 205609100 + int(seed) % 1000
    X0, initial_sources = AI.blind_proposals(
        n0, stage0_seed, training_csv=TRAINING_CSV)
    # The former code discarded almost all 4,608 blind proposals by geometric
    # distance alone.  Screen every one, then spend larger banks only on a
    # diverse low-cement/reliability/support frontier.
    bank_cache = {}
    screen_records = []

    fidelity = ((32, 128, 512) if MF is None else MF.fidelity_schedule()[:3])
    scenario_schedule = tuple(min(int(n_scenarios), int(v)) for v in fidelity)
    stage_for_n = {n: i for i, n in enumerate(scenario_schedule)}

    def evaluate_screen(X, requested_n):
        n = min(int(requested_n), int(n_scenarios))
        if n not in bank_cache:
            spec = U.UncertaintySpec(CONTRACT)
            bank_seed = (SEARCH_BANK_SEED if MF is None else
                         MF.stage_seed(SEARCH_BANK_SEED, stage_for_n.get(n, 0)))
            bank_cache[n] = (
                U.build_manufacturing_bank(spec, n=n, seed=bank_seed),
                U.build_residual_bank(BANK_CSV, n=n, seed=bank_seed),
            )
        screen_mfg, screen_res = bank_cache[n]
        rhats = np.zeros(len(X), dtype=float)
        cements = np.zeros(len(X), dtype=float)
        supported = np.zeros(len(X), dtype=bool)
        for i, x in enumerate(X):
            cand = K.decode11(x, ORDER)
            cements[i] = float(cand["C"])
            try:
                pen, _, _, details = state_support_penalty(cand, return_details=True)
            except Exception:
                pen, details = float("inf"), {"recipe_gate": False,
                    "porosity_hdr": False, "deployment_window": False,
                    "gradation_w1_gate": False, "state_support": False}
            support_ok = bool(np.isfinite(pen) and pen <= 0.0)
            if support_ok:
                rr = RE.evaluate_robust(
                    cand, screen_mfg, screen_res, backend="fast",
                    capture_search_arrays=CS is not None,
                )
                rhats[i] = float(rr["joint_reliability_hat"])
                support_fraction = float(rr["support_fraction"])
                supported[i] = bool(support_fraction >= 1.0 - 1e-12)
                continuous_quality = float(rhats[i])
                if CS is not None and rr.search_arrays is not None:
                    continuous_quality = float(CS.search_score(
                        rr.search_arrays["strength_real"],
                        rr.search_arrays["logk_real"],
                        RT.STRENGTH, RT.PERMEABILITY,
                        rr.search_arrays["in_support"],
                    )["cvar"])
            else:
                support_fraction = float("nan")
                continuous_quality = float("-inf")
            desc = E.cached_descriptors(np.asarray(cand["g"], dtype=float))
            diagnostics = {
                "in_bounds": True,
                "support_score_pass": bool(support_fraction >= 1.0 - 1e-12) if np.isfinite(support_fraction) else False,
                "stage17_recipe_gate": bool(details.get("recipe_gate", False)),
                "gradation_w1_gate": bool(details.get("gradation_w1_gate", False)),
                "porosity_hdr": bool(details.get("porosity_hdr", False)),
                "deployment_window": bool(details.get("deployment_window", False)),
                "state_support": bool(details.get("state_support", False)),
                "reliability_gate": bool(rhats[i] >= REL_TARGET),
                "continuous_score": continuous_quality,
                "verified": False,
                "support_fraction": support_fraction,
                "WAS": desc["WAS"], "I_gap": desc["I_gap"],
                "Cu": desc["Cu"], "Aag": desc["Aag"],
                **{f"raw_{k}": float(cand[k]) for k in ("C", "r", "A", "P", "S", "phi")},
                **{f"g{j+1}": float(cand["g"][j]) for j in range(6)},
            }
            screen_records.append(diagnostics)
        return rhats, cements, supported, screen_records[-len(X):]

    candidate_log = []
    elites, adaptive_log = AI.screen_and_refine(
        X0, evaluate_screen, seed=seed,
        scenario_schedule=scenario_schedule,
        global_sampler=lambda n: AI.blind_proposals(
            n, stage0_seed + int(n) + int(seed), training_csv=TRAINING_CSV),
        candidate_log=candidate_log,
        initial_sources=initial_sources,
        target_reliability=REL_TARGET,
    )
    # The final DE population is selected from the screened frontier.  It is
    # deliberately diverse, not merely the best few copies of one basin.
    if len(elites) < popsize:
        raise RuntimeError("adaptive screening returned fewer candidates than DE population")
    # Preserve a substantial exploitation core before filling the remaining
    # slots by maximin diversity.  The former one-elite-plus-farthest rule
    # discarded almost every promising point in a narrow feasible basin.
    n_exploit = min(len(elites), max(4, (popsize + 1) // 2))
    idx = list(range(n_exploit))
    while len(idx) < popsize:
        d = np.min([np.linalg.norm(elites - elites[j], axis=1) for j in idx], axis=0)
        d[idx] = -1.0
        idx.append(int(np.argmax(d)))
    pop = elites[np.asarray(idx, dtype=int)]
    os.makedirs(os.path.join(RUN, "traces"), exist_ok=True)
    pd.DataFrame(pop).to_csv(os.path.join(RUN, "traces", f"adaptive_init_seed{seed}_population.csv"), index=False)
    pd.DataFrame(candidate_log).to_csv(
        os.path.join(RUN, "traces", f"adaptive_init_seed{seed}_candidates.csv"), index=False)
    pd.DataFrame(adaptive_log).to_csv(
        os.path.join(RUN, "traces", f"adaptive_init_seed{seed}_rounds.csv"), index=False)
    with open(os.path.join(RUN, "traces", f"adaptive_init_seed{seed}_summary.json"), "w") as f:
        json.dump({"seed": seed, "stage0_n": int(len(X0)), "selected_n": int(len(pop)),
                   "rounds": adaptive_log,
                   "candidate_count": int(len(candidate_log)),
                    "archive_policy": "persistent multimodal basins + feasibility/cement archives + fresh global quota",
                    "scalar_design": "labeled blind proposal mixture"}, f, indent=2)
    return np.clip(pop, 0, 1)


# jDE self-adaptive mutation constants (Brest et al. 2006). F in [F_L, F_L+F_U],
# CR in [0,1]; each individual carries its own (F, CR), re-randomised with
# probability tau1/tau2 each generation and kept only if the resulting trial
# is accepted -- so the search adapts its own explore/exploit balance from
# the fitness landscape it discovers, with no assumption about where the
# optimum is (works the same regardless of target strength/permeability).
JDE_F_L, JDE_F_U = 0.1, 0.9
JDE_TAU1, JDE_TAU2 = 0.1, 0.1
JDE_CR_DEFAULT = 0.9
JDE_F_DEFAULT = 0.5


def _replace_with_retry(src, dst, attempts=6, base_delay=0.25):
    """os.replace occasionally hits WinError 5 (Access is denied) on Windows
    when a real-time antivirus scanner briefly holds a lock on a just-written
    file -- observed killing seed 20260915's process outright at gen 41/50
    during this run. Retries with backoff rather than crashing a multi-hour
    unattended search over a transient lock."""
    last_err = None
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(base_delay * (2 ** attempt))
    raise last_err


def run_seed(seed, popsize, generations, mfg_bank, res_bank, seed_cands, out_dir, patience=25,
             shortlist_dir=None, adaptive=False):
    rng = np.random.default_rng(seed)
    ckpt_path = os.path.join(out_dir, f"de_full_seed{seed}.ckpt.npz")
    trace_path = os.path.join(out_dir, f"de_full_seed{seed}.trace.csv")
    shortlist_path = os.path.join(shortlist_dir or out_dir, f"de_shortlist_seed{seed}.csv")

    shortlist_rows = []
    candidate_trace_rows = []

    def maybe_shortlist(gen, X, merits, rhats, supports, cands):
        # Always retain the best reliability candidates from every evaluated
        # population, even when none clears the provisional threshold.  This
        # prevents a later merit-driven population from erasing the strongest
        # basin seen earlier in the run.
        top = set(np.argsort(-np.asarray(rhats, dtype=float))[:min(5, len(X))].tolist())
        for i in range(len(X)):
            if rhats[i] >= SHORTLIST_RHAT_MIN or i in top:
                c = cands[i]
                shortlist_rows.append({
                    "seed": seed, "gen": gen, "C": c["C"], "r": c["r"], "A": c["A"], "P": c["P"],
                    "S": c["S"], "phi": c["phi"], **{f"g{j+1}": c["g"][j] for j in range(6)},
                    "Rhat_search": rhats[i], "support_fraction": supports[i], "merit": merits[i],
                    "retention_reason": ("threshold" if rhats[i] >= SHORTLIST_RHAT_MIN
                                         else "top_reliability_in_population"),
                })

    def flush_shortlist_fallback():
        # Fail-safe: never leave an empty shortlist (crashes Phase B).
        # If nothing clears the threshold, retain top-200 by Rhat then merit.
        if shortlist_rows:
            return
        print(f"[seed {seed}] WARNING: no candidates >= {SHORTLIST_RHAT_MIN}; retaining top-K fallback", flush=True)

    t0 = time.time()
    if os.path.exists(ckpt_path):
        # np.load returns a lazy NpzFile that keeps the underlying zip file
        # handle open until closed. Left open (as before), the FIRST
        # checkpoint save after a resume fails on Windows with
        # PermissionError/WinError 5 on os.replace, because the rename
        # target still has an open handle from this load -- this is what
        # actually killed seed 20260915's process outright at gen 41/50
        # (not antivirus, as first suspected -- confirmed 100% reproducible
        # via a dedicated resume-then-save smoke test). The `with` block
        # closes the handle before any further checkpoint write happens.
        with np.load(ckpt_path, allow_pickle=True) as z:
            start_gen = int(z["gen"]) + 1
            pop = z["pop"].copy()
            merits = z["merits"].copy()
            rhats = z["rhats"].copy()
            best_x = z["best_x"].copy()
            best_merit = float(z["best_merit"])
            if adaptive:
                # Older non-adaptive checkpoints won't have F/CR arrays --
                # seed every individual with the jDE defaults and adapt on.
                F_pop = z["F_pop"].copy() if "F_pop" in z.files else np.full(popsize, JDE_F_DEFAULT)
                CR_pop = z["CR_pop"].copy() if "CR_pop" in z.files else np.full(popsize, JDE_CR_DEFAULT)
        trace = pd.read_csv(trace_path).to_dict("records") if os.path.exists(trace_path) else []
        if os.path.exists(shortlist_path):
            shortlist_rows = pd.read_csv(shortlist_path).to_dict("records")
        n_eval = popsize * (start_gen + 1)
    else:
        pop = build_init_population(rng, popsize, seed_cands, seed=seed,
                                    n_scenarios=mfg_bank.n)
        eval_t0 = time.perf_counter()
        merits, rhats, supports, cands0 = robust_merit_batch(pop, mfg_bank, res_bank)
        eval_elapsed = time.perf_counter() - eval_t0
        maybe_shortlist(-1, pop, merits, rhats, supports, cands0)
        candidate_trace_rows.extend([
            {"seed": seed, "gen": -1, "candidate_index": i,
             "n_scenarios": int(mfg_bank.n), "merit": float(merits[i]),
             "Rhat": float(rhats[i]), "support_fraction": float(supports[i]),
             "C": float(cands0[i]["C"]), "state_supported": bool(merits[i] < C_MAX),
             "accepted": True, "elapsed_eval_s": float(eval_elapsed),
             "cost_per_candidate_s": float(eval_elapsed / max(len(pop), 1))}
            for i in range(len(pop))])
        n_eval = popsize
        i = int(np.argmin(merits))
        best_merit, best_x = float(merits[i]), pop[i].copy()
        start_gen = 0
        trace = []
        if adaptive:
            F_pop = np.full(popsize, JDE_F_DEFAULT)
            CR_pop = np.full(popsize, JDE_CR_DEFAULT)

    idx_all = np.arange(popsize)
    stall = 0
    prev_best = best_merit
    initial_best_merit = float(best_merit)
    for gen in range(start_gen, generations):
        trial = np.empty_like(pop)
        if adaptive:
            F_trial = F_pop.copy()
            CR_trial = CR_pop.copy()
        for i in range(popsize):
            pool = idx_all[idx_all != i]
            r1, r2, r3 = rng.choice(pool, 3, replace=False)
            a, b_, c_ = pop[r1], pop[r2], pop[r3]
            if adaptive:
                if rng.uniform(0, 1) < JDE_TAU1:
                    F_trial[i] = JDE_F_L + rng.uniform(0, 1) * JDE_F_U
                if rng.uniform(0, 1) < JDE_TAU2:
                    CR_trial[i] = rng.uniform(0, 1)
                mut = np.clip(a + F_trial[i] * (b_ - c_), 0, 1)
                cross = rng.uniform(0, 1, 11) < CR_trial[i]
            else:
                jitter = rng.uniform(0.5, 1.0)
                mut = np.clip(a + 0.7 * jitter * (b_ - c_), 0, 1)
                cross = rng.uniform(0, 1, 11) < 0.9
            cross[rng.integers(0, 11)] = True
            trial[i] = np.where(cross, mut, pop[i])
        eval_t0 = time.perf_counter()
        tmer, trhat, tsupp, tcands = robust_merit_batch(trial, mfg_bank, res_bank)
        eval_elapsed = time.perf_counter() - eval_t0
        maybe_shortlist(gen, trial, tmer, trhat, tsupp, tcands)
        n_eval += popsize
        improve = tmer < merits
        candidate_trace_rows.extend([
            {"seed": seed, "gen": gen, "candidate_index": i,
             "n_scenarios": int(mfg_bank.n), "merit": float(tmer[i]),
             "Rhat": float(trhat[i]), "support_fraction": float(tsupp[i]),
             "C": float(tcands[i]["C"]), "state_supported": bool(tmer[i] < C_MAX),
             "accepted": bool(improve[i]), "elapsed_eval_s": float(eval_elapsed),
             "cost_per_candidate_s": float(eval_elapsed / max(len(trial), 1))}
            for i in range(len(trial))])
        pop[improve] = trial[improve]
        merits[improve] = tmer[improve]
        rhats[improve] = trhat[improve]
        if adaptive:
            # F/CR travel with the offspring only when the offspring wins;
            # a losing trial's (F, CR) are discarded along with it.
            F_pop[improve] = F_trial[improve]
            CR_pop[improve] = CR_trial[improve]
        i = int(np.argmin(merits))
        if merits[i] < best_merit:
            best_merit, best_x = float(merits[i]), pop[i].copy()

        feasible_mask = merits < C_MAX
        best_feasible_C = float(merits[feasible_mask].min()) if feasible_mask.any() else None
        best_near_feasible_R = float(rhats.max())
        try:
            _q, _order, _frank = continuous_search_quality(rhats, np.array([c["C"] for c in tcands]), tsupp >= 1.0 - 1e-12)
            best_feas_quality = float(np.max(_q[np.isfinite(_q)])) if np.any(np.isfinite(_q)) else float("-inf")
            n_feas_archive = int(np.sum(tsupp >= 1.0 - 1e-12))
        except Exception:
            best_feas_quality, n_feas_archive = float("nan"), int(np.sum(feasible_mask))

        trace.append({"gen": gen, "evals": n_eval, "n_candidates": len(trial),
                       "n_scenarios": int(mfg_bank.n), "best_merit": best_merit,
                       "best_feasible_C": best_feasible_C, "best_R": best_near_feasible_R,
                       "feasibility_archive_quality": best_feas_quality,
                       "feasibility_archive_size": n_feas_archive,
                       "accepted_count": int(np.sum(improve)),
                       "elapsed_eval_s": float(eval_elapsed),
                       "cost_per_candidate_s": float(eval_elapsed / max(len(trial), 1)),
                       "elapsed_s": time.time() - t0})
        print(f"[seed {seed}] gen {gen+1}/{generations} evals={n_eval} elapsed={time.time()-t0:.1f}s "
              f"best_feasible_C={best_feasible_C} best_R={best_near_feasible_R:.4f}", flush=True)

        tmp = ckpt_path[:-4] + ".tmp.npz"
        if adaptive:
            np.savez(tmp, gen=gen, pop=pop, merits=merits, rhats=rhats, best_x=best_x,
                      best_merit=best_merit, F_pop=F_pop, CR_pop=CR_pop)
        else:
            np.savez(tmp, gen=gen, pop=pop, merits=merits, rhats=rhats, best_x=best_x, best_merit=best_merit)
        _replace_with_retry(tmp, ckpt_path)
        pd.DataFrame(trace).to_csv(trace_path, index=False)
        pd.DataFrame(candidate_trace_rows).to_csv(
            os.path.join(shortlist_dir or out_dir, f"de_candidate_trace_seed{seed}.csv"), index=False)
        pd.DataFrame(shortlist_rows).to_csv(shortlist_path, index=False)

        if best_merit >= prev_best - 1e-9:
            stall += 1
        else:
            stall, prev_best = 0, best_merit
        if stall > patience:
            print(f"[seed {seed}] early stop: no improvement for {patience} generations", flush=True)
            break

    if not shortlist_rows:
        # Fallback: retain final population (top by Rhat) so Phase B never sees empty files.
        _, rhats_f, supp_f, cands_f = robust_merit_batch(pop, mfg_bank, res_bank)
        order = np.argsort(-rhats_f)[: min(200, len(pop))]
        for i in order:
            c = cands_f[int(i)]
            shortlist_rows.append({
                "seed": seed, "gen": generations, "C": c["C"], "r": c["r"], "A": c["A"], "P": c["P"],
                "S": c["S"], "phi": c["phi"], **{f"g{j+1}": c["g"][j] for j in range(6)},
                "Rhat_search": float(rhats_f[int(i)]), "support_fraction": float(supp_f[int(i)]),
                "merit": float(merits[int(i)]), "fallback_topK": True,
            })
        pd.DataFrame(shortlist_rows).to_csv(shortlist_path, index=False)

    best_cand = K.decode11(best_x, ORDER)
    pd.DataFrame(candidate_trace_rows).to_csv(
        os.path.join(shortlist_dir or out_dir, f"de_candidate_trace_seed{seed}.csv"), index=False)
    return {"seed": seed, "best_x": best_x.tolist(), "best_merit": best_merit,
            "best_cand": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in best_cand.items()},
            "n_eval": n_eval, "elapsed_s": time.time() - t0,
            "initial_best_merit": initial_best_merit,
            "final_best_merit": float(best_merit),
            "improved_over_initial": bool(best_merit < initial_best_merit - 1e-12),
            "candidate_trace": f"de_candidate_trace_seed{seed}.csv"}


def main(seed, popsize=25, generations=100, n_scenarios=SEARCH_BANK_N, adaptive=False):
    t0 = time.time()
    spec = U.UncertaintySpec(CONTRACT)
    mfg_bank = U.build_manufacturing_bank(spec, n=n_scenarios, seed=SEARCH_BANK_SEED)
    res_bank = U.build_residual_bank(BANK_CSV, n=n_scenarios, seed=SEARCH_BANK_SEED)
    print(f"[seed {seed}] Search bank: N={n_scenarios} mfg_hash={mfg_bank.bank_hash()[:12]} "
          f"res_hash={res_bank.bank_hash()[:12]} adaptive={adaptive}", flush=True)

    seed_cands = load_seed_candidates()
    out_dir = os.path.join(RUN, "checkpoints")
    shortlist_dir = os.path.join(RUN, "traces")
    os.makedirs(out_dir, exist_ok=True)

    result = run_seed(seed, popsize, generations, mfg_bank, res_bank, seed_cands, out_dir,
                       shortlist_dir=shortlist_dir, adaptive=adaptive)

    out_path = os.path.join(RUN, "traces", f"de_full_result_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump({"seed": seed, "popsize": popsize, "generations": generations, "n_scenarios": n_scenarios,
                    "adaptive": adaptive,
                    "mfg_bank_hash": mfg_bank.bank_hash(), "res_bank_hash": res_bank.bank_hash(),
                    "result": result, "elapsed_s": time.time() - t0}, f, indent=2, default=str)
    print(f"[seed {seed}] DONE elapsed={time.time()-t0:.1f}s best_C={result['best_cand']['C']:.3f} -> {out_path}",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--popsize", type=int, default=25)
    ap.add_argument("--generations", type=int, default=100)
    ap.add_argument("--n_scenarios", type=int, default=SEARCH_BANK_N)
    ap.add_argument("--adaptive", action="store_true",
                     help="jDE self-adaptive per-individual F/CR instead of the fixed jitter scheme")
    a = ap.parse_args()
    main(a.seed, a.popsize, a.generations, a.n_scenarios, adaptive=a.adaptive)
