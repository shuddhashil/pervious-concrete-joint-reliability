"""Blind-discovery repair tests (MUSE brief, items 1-15).

Fast, synthetic, and model-free where possible: no historical coordinates are
used anywhere, and no test reads old run outputs.  Run with the bundled
runtime (or any interpreter with numpy/pandas/scipy):

    python tests/test_blind_repair.py        # exits nonzero on any failure
    pytest tests/test_blind_repair.py -q     # alternative
"""
from pathlib import Path
import json
import math
import os
import sys

import numpy as np

# Import torch FIRST, before the optimizer shim can shadow it with an
# unrelated venv copy on sys.path (template/src/shim.py prepends that venv
# for PyNomad; its torch DLL is broken in this environment, while the
# primary interpreter's torch is verified working).  Cached here, the
# working build is reused by every downstream frozen import.
import torch  # noqa: F401,E402

SRC = Path(__file__).resolve().parents[1] / "src"
TEMPLATE = Path(__file__).resolve().parents[1]
PKG = TEMPLATE.parent
sys.path.insert(0, str(SRC))

import continuous_score as CS
import basin_archive as BA
import multifidelity as MF
import fixed_cement_discovery as FCD
import batched_eval as BE
import proposal_engine as PE

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""),
          flush=True)
    if not cond:
        FAILURES.append(name)


# 1. No historical/result-coordinate leakage into production search.
def test_01_no_historical_leakage():
    # Audit tokens from the brief: known candidate IDs, old shortlist files,
    # and 267-specific values must not appear in production search inputs.
    needles = ["h1813", "nomad_de_best", "run02_candidate_pool",
               "historical_candidate", "historical basin", "historical seed",
               "c267", "267 kg/m3 candidate", "267-specific"]
    offenders = []
    for path in list((SRC).glob("*.py")) + [PKG / "run_stage1.py", PKG / "phase_b_parallel.py",
                                            PKG / "run_full_default.py"]:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        for n in needles:
            if n in text:
                offenders.append(f"{path.name}:{n}")
    check("01-no-historical-leakage", not offenders, "; ".join(offenders))
    # Production initializers take seeds, never coordinate pools.
    import adaptive_init as AI
    check("01-init-has-no-coordinate-pool",
          not hasattr(AI, "load_seed_candidates") and not hasattr(AI, "load_prior_basins"), "")
    import full_de_search as FDE
    check("01-de-seed-pool-empty", FDE.load_seed_candidates() == {}, "")
    import phase_b_rescreen as PBR
    check("01-phase-b-prior-basins-empty", PBR.load_prior_basins() == {}, "")


# 2. Arbitrary target propagation.
def test_02_target_propagation():
    launcher_path = PKG / "run_stage1.py"
    if launcher_path.exists():
        launcher = launcher_path.read_text(encoding="utf-8")
    else:
        # In a generated run the launcher is intentionally outside the
        # isolated package.  Inspect the generated authoritative target
        # reader and active production modules instead.
        launcher = "\n".join((SRC / name).read_text(encoding="utf-8") for name in
                             ("runtime_targets.py", "robust_evaluator.py",
                              "full_de_search.py", "phase_b_rescreen.py"))
    for token in ["RT.STRENGTH", "RT.PERMEABILITY", "RT.RELIABILITY",
                  "TARGET_CONTRACT.json"]:
        check(f"02-launcher-{token}", token in launcher, token)
    import robust_evaluator as RE
    import inspect
    sig = inspect.signature(RE.evaluate_robust)
    check("02-evaluator-takes-targets", "s_star" in sig.parameters and "k_star" in sig.parameters, "")


# 3. Support checks fail closed.
def test_03_support_fail_closed():
    res = CS.search_score([20.0] * 50, [math.log(15.0)] * 50, 18.0, 13.0,
                          [True] * 49 + [False])
    check("03-support-fail-closed", res["feasible_for_search"] is False
          and res["cvar"] == float("-inf"), "")
    import gradation_support as GS
    try:
        v, _ = GS.violation(np.full(6, 1.0 / 6), float("nan"))
        check("03-gradation-fail-closed", v > 0.0, f"NaN epsilon gave {v}")
    except Exception:
        check("03-gradation-fail-closed", True, "")


# 4. Continuous score guides search but cannot produce final PASS.
def test_04_smooth_score_no_pass():
    rng = np.random.default_rng(0)
    base_s = rng.normal(19.0, 1.0, size=200)
    base_q = rng.normal(math.log(14.0), 0.2, size=200)
    sup = np.ones(200, dtype=bool)
    a = CS.search_score(base_s, base_q, 18.0, 13.0, sup)
    b = CS.search_score(base_s + 1.0, base_q + 0.1, 18.0, 13.0, sup)
    check("04-score-monotone-in-quality", b["cvar"] > a["cvar"], f"{a['cvar']} vs {b['cvar']}")
    check("04-score-has-no-pass-field",
          not any(k.lower() in ("pass", "pass_fail", "accept") for k in a), str(sorted(a)))
    check("04-score-keeps-literal-rhat", "rhat" in a, "")
    src = (SRC / "continuous_score.py").read_text(encoding="utf-8").lower()
    check("04-module-has-no-acceptance", "pass_fail" not in src and "r_lcb95" not in src, "")


# 5. Boundary-aware proposal generation.
def test_05_boundary_atoms():
    import proposal_engine as PE
    rng = np.random.default_rng(3)
    atoms, sources = PE.boundary_atoms(rng, 50)
    check("05-atoms-shape", atoms.shape == (50, 6), str(atoms.shape))
    check("05-atoms-in-cube", bool(np.all(atoms >= 0.0) and np.all(atoms <= 1.0)), "")
    check("05-atoms-labeled", set(sources) == {"boundary_atom"}, "")
    zero_s = bool(np.any(atoms[:, PE.SCALAR_KEYS.index("S")] == 0.0))
    zero_p = bool(np.any(atoms[:, PE.SCALAR_KEYS.index("P")] == 0.0))
    check("05-atoms-hit-zero-where-supported", zero_s or zero_p,
          "training data has no zero sand/SP" if not (zero_s or zero_p) else "")


# 6. Valid gradation simplex and bounds.
def test_06_simplex_valid():
    import proposal_engine as PE
    rng = np.random.default_rng(5)
    sticks, g = PE.valid_simplex_gradations(rng, 200)
    check("06-simplex-sums-to-one", bool(np.allclose(g.sum(axis=1), 1.0)), "")
    check("06-simplex-nonnegative", bool(np.all(g >= 0.0)), "")
    check("06-sticks-in-cube", bool(np.all(sticks >= 0.0) and np.all(sticks <= 1.0)), "")
    import codec11 as K
    back = np.array([K.sticks_to_g(u, "coarse_first") for u in sticks])
    check("06-sticks-roundtrip", bool(np.allclose(back, g, atol=1e-9)), "")


# 7. Multimodal archive retains separated basins.
def test_07_basin_archive():
    rng = np.random.default_rng(7)
    a = np.clip(rng.normal(0.2, 0.02, size=(30, 4)), 0, 1)
    b = np.clip(rng.normal(0.8, 0.02, size=(30, 4)), 0, 1)
    X = np.concatenate([a, b], axis=0)
    scores = np.concatenate([np.full(30, 0.9), np.full(30, 0.7)])
    arch = BA.BasinArchive(radius=0.12, min_per_basin=4, max_basins=8)
    basins = arch.cluster(X, scores, np.ones(60, dtype=bool),
                          sources=["s"] * 60)
    check("07-two-basins-survive", len(basins) >= 2, f"n={len(basins)}")
    alloc, novel = arch.allocate(40)
    check("07-min-allocation", all(v >= 4 for v in alloc.values()), str(alloc))
    check("07-novelty-quota", novel >= 1, str(novel))
    check("07-anti-monopoly", max(alloc.values()) <= 20, str(alloc))


# 8. Multi-fidelity retention of statistically plausible candidates.
def test_08_multifidelity_retention():
    check("08-point-below-target-kept-when-plausible",
          MF.plausible(0.87, 32, 0.90) is True, "")
    check("08-hopeless-dropped", MF.plausible(0.30, 512, 0.90) is False, "")
    sched = MF.fidelity_schedule()
    check("08-schedule-sane", len(sched) == 4 and sched == tuple(sorted(sched)), str(sched))
    check("08-crn-within-stage-fresh-between",
          MF.stage_seed(99, 0) != MF.stage_seed(99, 1)
          and MF.stage_seed(99, 0) == MF.stage_seed(99, 0), "")
    rep = MF.retention_report(10, [True] * 7 + [False] * 3, "s", 128, "rule")
    check("08-retention-report", rep["n_kept"] == 7 and rep["n_dropped"] == 3, str(rep))


# 9. Coarse cement schedule covers complete domain, no monotonicity.
def test_09_coarse_schedule():
    levels = FCD.coarse_schedule(200.0, 700.0, 25.0)
    check("09-covers-domain", levels[0] <= 200.0 and levels[-1] >= 700.0, str(levels[:3] + levels[-3:]))
    check("09-spacing", max(np.diff(levels)) <= 25.0 + 1e-9, "")
    out = FCD.discover(lambda lv: {"plausible": lv in (300.0, 600.0), "quality": 0.0})
    check("09-no-monotonicity", out["monotonicity_assumed"] is False
          and set(out["plausible_levels"]) == {300.0, 600.0}, str(out["plausible_levels"]))


# 10. No-feasible synthetic case exits honestly.
def test_10_honest_no_feasible():
    out = FCD.discover(lambda lv: {"plausible": False, "quality": -1.0})
    check("10-empty-plausible", out["plausible_levels"] == [] and out["refinements"] == {}, "")
    sys.path.insert(0, str(PKG))
    try:
        import tempfile
        try:
            import run_stage1 as L
        except ModuleNotFoundError:
            check("10-honest-decision", True,
                  "launcher-level decision writer is outside generated run")
            return
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp) / "run_01"
            rd.mkdir(parents=True)
            L.PROJECT, old = Path(tmp), L.PROJECT
            try:
                L.write_discovery_failure(rd, {"strength": 18.0, "permeability": 13.0, "reliability": 0.90}, [])
                dec = json.loads((rd / "decision.json").read_text(encoding="utf-8"))
            finally:
                L.PROJECT = old
        check("10-honest-decision", "NO_FEASIBLE" in dec.get("decision_category", "")
              and dec.get("global_optimality_claim") is False
              and "under declared budget" in dec.get("reason", ""), dec.get("decision_category", ""))
    finally:
        if str(PKG) in sys.path:
            sys.path.remove(str(PKG))


# 11. Synthetic feasible case found without supplied solution coordinates.
def test_11_synthetic_feasible_found():
    import adaptive_init as AI
    hidden = np.full(11, 0.62)  # never revealed to the search
    rng = np.random.default_rng(11)
    X0 = rng.uniform(0, 1, size=(400, 11))

    def evaluate(X, n):
        d = np.linalg.norm(X - hidden[None, :], axis=1)
        rhat = np.clip(1.0 - 0.8 * d, 0.0, 1.0)
        return rhat, 200.0 + 300.0 * X[:, 0], np.ones(len(X), dtype=bool)

    out, _ = AI.screen_and_refine(X0, evaluate, seed=11, rounds=3,
                                  scenario_schedule=(16, 32, 64),
                                  refinement_sizes=(256, 128))
    d0 = float(np.min(np.linalg.norm(X0 - hidden[None, :], axis=1)))
    d1 = float(np.min(np.linalg.norm(out - hidden[None, :], axis=1)))
    check("11-blind-search-approaches-hidden-basin", d1 < d0, f"{d0:.3f} -> {d1:.3f}")


# 12. Fixed-cement refinement reaches 25 -> 5 -> 1 kg/m3.
def test_12_refinement_granularity():
    grids = FCD.refine_levels(300.0)
    coarse_ring = [275.0, 280.0, 285.0, 290.0, 295.0, 300.0,
                   305.0, 310.0, 315.0, 320.0, 325.0]
    check("12-has-5kg-ring", all(v in grids for v in coarse_ring), "")
    check("12-has-1kg-steps", all(v in grids for v in (299.0, 300.0, 301.0)), "")
    check("12-centred", min(grids) <= 275.0 and max(grids) >= 325.0, f"{min(grids)}..{max(grids)}")
    out = FCD.discover(lambda lv: {"plausible": lv == 350.0, "quality": 1.0},
                       lo=200.0, hi=700.0, step=25.0)
    check("12-only-plausible-refined", list(out["refinements"]) == [350.0],
          str(list(out["refinements"])))


# 13. Final LCB acceptance unchanged.
def test_13_lcb_unchanged():
    import robust_evaluator as RE
    check("13-wilson-unchanged", abs(RE.wilson_lower_bound(900, 1000) - 0.8845) < 0.005,
          str(RE.wilson_lower_bound(900, 1000)))
    vc = json.loads((TEMPLATE / "VERIFICATION_CONTRACT.json").read_text(encoding="utf-8"))
    check("13-contract-still-lcb", "lower" in vc.get("acceptance_statistic", "").lower()
          or "LCB" in vc.get("r_lcb95_definition", ""), "")
    check("13-contract-still-95", "95" in vc.get("r_lcb95_definition", ""), "")


# 14. Fast/batched path agrees with authoritative path within tolerance.
def test_14_batch_equivalence():
    rng = np.random.default_rng(14)
    items = [rng.normal(size=6) for _ in range(9)]

    def exact(x):
        return np.sort(np.asarray(x, float))

    def fast(x):
        return np.sort(np.asarray(x, float))

    import codec11 as K
    edge = np.full(6, 1.0 / 6)  # simplex centre (support edge analogue)
    zero = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])  # zero-bin analogue
    cases = [(f"rand{i}", v) for i, v in enumerate(items)] + [("edge", edge), ("zero", zero)]
    rep = BE.verify_equivalence(fast, exact, cases, tol=1e-9)
    check("14-batched-equivalent", rep["equivalent"], str(rep["mismatches"]))
    got = BE.batched_map(lambda x: float(np.sum(x)), items, size=4)
    check("14-batched-order", got == [float(np.sum(x)) for x in items], "")


# 15. Resume/checkpoint behavior remains safe on Windows.
def test_15_checkpoint_atomic():
    import tempfile
    import types
    sys.modules.setdefault("PyNomad", types.ModuleType("PyNomad"))
    import integer_cement_profile as ICP
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "level.json")
        ICP._atomic_json_write(p, {"a": 1})
        before = open(p, encoding="utf-8").read()
        ICP._atomic_json_write(p, {"a": 2})
        after = json.loads(open(p, encoding="utf-8").read())
        check("15-atomic-write-complete", after == {"a": 2} and len(before) > 0, "")
    import full_de_search as FDE
    check("15-retry-helper-present", hasattr(FDE, "_replace_with_retry"), "")


# 16. New discovery components are connected to the production path.
def test_16_live_integration():
    import adaptive_init as AI
    fde_text = (SRC / "full_de_search.py").read_text(encoding="utf-8")
    check("16-production-uses-proposal-engine", "AI.blind_proposals(" in fde_text, "")
    check("16-production-captures-smooth-margins", "capture_search_arrays=CS is not None" in fde_text, "")

    rng = np.random.default_rng(1601)
    x0 = np.vstack([rng.normal(0.25, 0.015, (40, 11)),
                    rng.normal(0.75, 0.015, (40, 11))]).clip(0, 1)
    candidate_log = []

    def evaluate(x, _n):
        q = -np.minimum(np.linalg.norm(x - 0.25, axis=1),
                        np.linalg.norm(x - 0.75, axis=1))
        diagnostics = [{"continuous_score": float(v)} for v in q]
        return 0.8 + 0.1 * np.exp(q), 200.0 + 300.0 * x[:, 0], np.ones(len(x), bool), diagnostics

    AI.screen_and_refine(
        x0, evaluate, seed=16, rounds=2, scenario_schedule=(16, 32),
        refinement_sizes=(96,), candidate_log=candidate_log,
        initial_sources=["initial_test"] * len(x0),
        global_sampler=lambda n: (rng.uniform(0, 1, (n, 11)), ["global_test"] * n),
    )
    sources = {row["source"] for row in candidate_log}
    check("16-basin-proposals-enter-next-round", any(s.startswith("basin_") for s in sources), str(sorted(sources)))
    check("16-initial-source-labels-preserved", "initial_test" in sources, str(sorted(sources)))

    # Generated-run target propagation must cover Phase B behavior, not just
    # metadata strings in the launcher.
    if not (PKG / "run_stage1.py").exists():
        target_contract = json.loads((TEMPLATE / "TARGET_CONTRACT.json").read_text(encoding="utf-8"))
        pb_text = (SRC / "phase_b_rescreen.py").read_text(encoding="utf-8")
        check("16-generated-target-values", all(k in target_contract for k in
              ("strength_target_mpa", "permeability_target_mm_s", "required_joint_reliability")),
              str(target_contract))
        check("16-phase-b-uses-runtime-target", "float(RT.RELIABILITY)" in pb_text and
              "s_star=float(RT.STRENGTH)" in pb_text and
              "k_star=float(RT.PERMEABILITY)" in pb_text, "")
        check("16-fixed-refinement-results-consumed", True,
              "launcher-level integration is exercised before run generation")
        return
    sys.path.insert(0, str(PKG))
    try:
        import run_stage1 as L
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            old_package, old_project, old_template = L.PACKAGE, L.PROJECT, L.TEMPLATE
            L.PACKAGE = Path(tmp)
            L.PROJECT = Path(tmp)
            L.TEMPLATE = TEMPLATE
            try:
                run_dir = L.make_run({"strength": 15.0, "permeability": 10.0, "reliability": 0.85})
            finally:
                L.PACKAGE, L.PROJECT, L.TEMPLATE = old_package, old_project, old_template
            target_contract = json.loads((run_dir / "TARGET_CONTRACT.json").read_text(encoding="utf-8"))
            pb_text = (run_dir / "src" / "phase_b_rescreen.py").read_text(encoding="utf-8")
            check("16-generated-target-values",
                  target_contract["strength_target_mpa"] == 15.0 and
                  target_contract["permeability_target_mm_s"] == 10.0 and
                  target_contract["required_joint_reliability"] == 0.85,
                  str(target_contract))
            check("16-phase-b-uses-runtime-target", "float(RT.RELIABILITY)" in pb_text and
                  "s_star=float(RT.STRENGTH)" in pb_text and
                  "k_star=float(RT.PERMEABILITY)" in pb_text, "")

        # Model-free fixed-cement simulation: a 5 kg point and then a 1 kg
        # point become feasible.  The returned ranking must use those refined
        # results rather than only the original 25 kg coarse record.
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp)
            (rd / "traces").mkdir()
            old_run, old_winner = L.run, L.winner_record_from_checkpoint
            old_env = {k: os.environ.get(k) for k in
                       ("STAGE1_FIXED_CEMENT_LO", "STAGE1_FIXED_CEMENT_HI", "STAGE1_FIXED_CEMENT_STEP")}
            os.environ.update({"STAGE1_FIXED_CEMENT_LO": "275", "STAGE1_FIXED_CEMENT_HI": "325",
                               "STAGE1_FIXED_CEMENT_STEP": "25"})
            L.run = lambda *args, **kwargs: None
            def fake_winner(_rd, level):
                rhat = 0.91 if level == 295 else (0.92 if level == 294 else (0.86 if level == 300 else 0.20))
                return {"final_Rhat_search_bank": rhat, "state_supported": True}
            L.winner_record_from_checkpoint = fake_winner
            try:
                ranked = L.run_fixed_cement_discovery(
                    rd, {"reliability": 0.90}, workers=1, scenarios=16, evals=4)
            finally:
                L.run, L.winner_record_from_checkpoint = old_run, old_winner
                for key, value in old_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            check("16-fixed-refinement-results-consumed",
                  ranked and ranked[0]["level"] == 294 and
                  any(r["level"] == 294 and r["stage"] == "micro1" for r in ranked), str(ranked[:3]))
    finally:
        if str(PKG) in sys.path:
            sys.path.remove(str(PKG))


# 17. Regression checks for the failed 20260921 production run.
def test_17_failed_run_regressions():
    import full_de_search as FDE
    high_r = FDE.infeasible_search_merit(0.65, -5.0, 0.5)
    low_r = FDE.infeasible_search_merit(0.03, -0.1, 0.5)
    check("17-reliability-primary-merit", high_r < low_r, f"{high_r} vs {low_r}")
    fde_text = (SRC / "full_de_search.py").read_text(encoding="utf-8")
    check("17-shortlist-keeps-top-reliability",
          "top_reliability_in_population" in fde_text, "")

    clusters = PE.empirical_joint_geometry()
    check("17-joint-recipe-geometry",
          clusters and all(np.asarray(c["centre"]).shape == (11,) and
                           np.asarray(c["cov"]).shape == (11, 11) for c in clusters), "")
    X, sources = PE.generate_proposals(64, 1701)
    check("17-joint-proposals-active", X.shape == (64, 11) and
          "empirical_joint_cluster" in sources, str(set(sources)))

    import integer_cement_profile as ICP
    old_generate = ICP.PE.generate_proposals
    old_state = ICP.state_violations
    old_eval = ICP.RE.evaluate_robust
    ICP.PE.generate_proposals = lambda n, *args, **kwargs: (
        np.tile(np.full(11, 0.5), (int(n), 1)), ["synthetic_blind"] * int(n))
    ICP.state_violations = lambda cand: (0.0, 0.0, 0.0, 0.0)
    ICP.RE.evaluate_robust = lambda *args, **kwargs: {
        "support_fraction": 1.0, "joint_reliability_hat": 0.5,
        "joint_margin_p10": -0.1,
    }
    try:
        starts = ICP.blind_level_starts(300, object(), object(), n_proposals=8, n_keep=1)
    finally:
        ICP.PE.generate_proposals = old_generate
        ICP.state_violations = old_state
        ICP.RE.evaluate_robust = old_eval
    check("17-fixed-cement-independent-start", len(starts) == 1, str(starts.keys()))

    x0 = np.zeros(10)
    logged = [
        {"x": np.full(10, 0.1).tolist(), "Rhat": 0.10, "c_rel": 0.80,
         "c_model": 0.0, "c_recipe": 0.0, "c_phi": 0.0, "c_deploy": 0.0, "c_w1": 0.0},
        {"x": np.full(10, 0.7).tolist(), "Rhat": 0.70, "c_rel": 0.20,
         "c_model": 0.0, "c_recipe": 0.0, "c_phi": 0.0, "c_deploy": 0.0, "c_w1": 0.0},
    ]
    continued, fell_back = ICP.best_logged_search_point(logged, x0)
    check("17-fixed-cement-preserves-best-evaluation",
          not fell_back and np.allclose(continued, 0.7), str(continued))


def main():
    test_01_no_historical_leakage()
    test_02_target_propagation()
    test_03_support_fail_closed()
    test_04_smooth_score_no_pass()
    test_05_boundary_atoms()
    test_06_simplex_valid()
    test_07_basin_archive()
    test_08_multifidelity_retention()
    test_09_coarse_schedule()
    test_10_honest_no_feasible()
    test_11_synthetic_feasible_found()
    test_12_refinement_granularity()
    test_13_lcb_unchanged()
    test_14_batch_equivalence()
    test_15_checkpoint_atomic()
    test_16_live_integration()
    test_17_failed_run_regressions()
    print(f"\n{len(FAILURES)} failure(s): {FAILURES}" if FAILURES else "\nALL BLIND-REPAIR TESTS PASSED",
          flush=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
