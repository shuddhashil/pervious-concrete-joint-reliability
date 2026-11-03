"""Phase D - 1 kg/m3-resolution fixed-cement re-optimization profile
(SEARCH_CONTRACT.json phase_D_integer_cement_profile).

Adapted from run_04/src/fixed_cement_nomad.py: cement pinned EXACTLY per
tested integer level via fixed_cement_codec.py (wrapping the unchanged
codec11.py), all other 10 variables free and RE-OPTIMIZED at every level
via genuine NOMAD/MADS -- never copied unchanged from the continuous
result. Starts are the strongest distinct basins found in Phases A-C.
"""
from __future__ import annotations

import argparse
import runtime_targets as RT
import concurrent.futures as CF
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
import PyNomad  # noqa: E402
import stage1_evaluator as E  # noqa: E402
import fast_descriptors as FD  # noqa: E402
import fast_local_residual as FLR  # noqa: E402

if os.environ.get("DISABLE_FAST_DESCRIPTORS") != "1":
    FD.install_patch(E)
    print("[perf] fast_descriptors patch installed", flush=True)
if os.environ.get("DISABLE_FAST_LOCAL_RESIDUAL") != "1":
    FLR.install_patch(E.L)
    print("[perf] fast_local_residual patch installed", flush=True)
import uncertainty as U  # noqa: E402
import robust_evaluator as RE  # noqa: E402
import stage17_state_support as S17  # noqa: E402  (Stage-18 nominal state-support gate)
import gradation_support as GS  # noqa: E402
import codec11 as K  # noqa: E402
import proposal_engine as PE  # noqa: E402
from fixed_cement_codec import FixedCementCodec  # noqa: E402
from parallel_runtime import configure_parent_for_spawn, configure_single_thread_worker  # noqa: E402

CONTRACT = os.path.join(RUN, "UNCERTAINTY_CONTRACT.json")
BANK_CSV = os.path.join(RUN, "tables", "paired_residual_bank.csv")
REL_TARGET = RT.RELIABILITY
SEARCH_BANK_N = 512
SEARCH_BANK_SEED = 205609401
TRAINING_CSV = os.path.join(RUN, "tables", "training_data_reconciled.csv")
START_POLICY = "phaseC_plus_independent_blind_v2"


def _env_int(name, default, minimum=1):
    raw = os.environ.get(name, "").strip()
    return int(default) if not raw else max(int(minimum), int(raw))


def load_start_candidates(n_from_phase_c=4):
    """Feasible distinct basins from this run's Phase C archive only."""
    starts = {}

    phase_c_path = os.path.join(RUN, "tables", "CONTINUOUS_NOMAD_RUNS.csv")
    if os.path.exists(phase_c_path):
        pdf = pd.read_csv(phase_c_path)
        # This profile locates the MINIMUM feasible integer cement level.  The
        # Phase-C seeds must therefore cover the lowest-C feasible continuous
        # frontier, not merely the globally highest-Rhat designs (which can
        # sit far above the boundary and previously omitted C=263.22).
        state_ok = (pdf["state_supported"].astype(str) == "True") if "state_supported" in pdf else np.ones(len(pdf), dtype=bool)
        model_ok = (pd.to_numeric(pdf["support_fraction"], errors="coerce") >= 1.0 - 1e-12) if "support_fraction" in pdf else np.ones(len(pdf), dtype=bool)
        feasible = pdf[(pdf["final_Rhat"] >= REL_TARGET) & state_ok & model_ok]
        # A weak/infeasible point can guide Phase C diagnostics, but it is
        # never promoted to a fixed-cement final-profile start.
        if len(feasible) == 0:
            return starts
        pdf = feasible.sort_values(["C", "final_Rhat"], ascending=[True, False])
        n_added = 0
        for _, row in pdf.iterrows():
            if n_added >= n_from_phase_c:
                break
            tag = f"run05_phaseC_{row['tag']}"
            if tag in starts:
                continue
            g = [float(row[f"g{i}"]) for i in range(1, 7)]
            starts[tag] = dict(C=float(row["C"]), r=float(row["r"]), A=float(row["A"]), P=float(row["P"]),
                                S=float(row["S"]), phi=float(row["phi"]), g=g)
            n_added += 1
    return starts


def blind_level_starts(cement_level, screen_mfg, screen_res, n_proposals=None,
                       n_keep=None, seed=SEARCH_BANK_SEED):
    """Create independent target-agnostic starts for one fixed-C search.

    This path deliberately does not depend on Phase C feasibility.  Proposals
    use training INPUT geometry only; targets enter only in the subsequent
    screening/ranking, which is the intended inverse-search operation.
    """
    n_proposals = int(n_proposals or _env_int("STAGE1_FIXED_BLIND_PROPOSALS", 768, 64))
    n_keep = int(n_keep or _env_int("STAGE1_FIXED_BLIND_STARTS", 4, 1))
    X, sources = PE.generate_proposals(
        n_proposals, int(seed) + 1009 * int(round(float(cement_level))),
        training_csv=TRAINING_CSV,
        mix=(0.20, 0.60, 0.10, 0.10),
    )
    codec = FixedCementCodec(cement_level)
    screened = []
    for i, x in enumerate(X):
        cand = K.decode11(np.asarray(x, dtype=float), "coarse_first")
        cand["C"] = float(cement_level)
        c_recipe, c_phi, c_deploy, c_w1 = state_violations(cand)
        state_violation = float(c_recipe + c_phi + c_deploy + c_w1)
        if state_violation > 0.0 or not np.isfinite(state_violation):
            continue
        result = RE.evaluate_robust(cand, screen_mfg, screen_res, backend="fast")
        support_fraction = float(result["support_fraction"])
        if support_fraction < 1.0 - 1e-12:
            continue
        screened.append({
            "candidate": cand,
            "x10": np.asarray(codec.encode10(cand), dtype=float),
            "rhat": float(result["joint_reliability_hat"]),
            "margin_p10": float(result["joint_margin_p10"]),
            "source": str(sources[i]),
        })
    screened.sort(key=lambda row: (-row["rhat"], -row["margin_p10"]))
    chosen = []
    for row in screened:
        if not chosen or all(np.linalg.norm(row["x10"] - old["x10"]) >= 0.04 for old in chosen):
            chosen.append(row)
        if len(chosen) >= n_keep:
            break
    if len(chosen) < n_keep:
        used = {id(row) for row in chosen}
        chosen.extend(row for row in screened if id(row) not in used)
        chosen = chosen[:n_keep]
    starts = {
        f"blind_C{int(round(float(cement_level)))}_{j}_{row['source']}": row["candidate"]
        for j, row in enumerate(chosen)
    }
    print(f"  Blind fixed-C starts: {len(starts)}/{n_proposals} retained; "
          f"best screen Rhat={chosen[0]['rhat']:.4f}" if chosen else
          f"  Blind fixed-C starts: 0/{n_proposals} retained", flush=True)
    return starts


def state_violations(cand):
    """Stage-18 continuous state-support violations + TASK A2 W1 gate."""
    if os.environ.get("DISABLE_STAGE17_SUPPORT") == "1":
        return 0.0, 0.0, 0.0, 0.0
    ss = S17.state_support(cand)
    c_recipe = float(ss["recipe_excess"])
    v_p_raw = float(ss["phi_violation"])
    c_phi = 0.0 if (v_p_raw != v_p_raw or ss["phi_in_conf90"]) else v_p_raw
    c_deploy = 0.0 if ss["deployment_ok"] else min(
        abs(float(cand["phi"]) - S17.DEPLOY_LO),
        abs(float(cand["phi"]) - S17.DEPLOY_HI))
    gc = GS.load_contract()
    c_w1 = 0.0
    if gc.get("enabled", True) and os.environ.get("DISABLE_GRADATION_GATE") != "1":
        c_w1, _ = GS.violation(np.asarray(cand["g"], float), gc["epsilon_95"])
    return c_recipe, c_phi, c_deploy, c_w1


def make_bb(codec, mfg_bank, res_bank, log_rows):
    def bb(ev):
        try:
            xv = np.array([ev.get_coord(i) for i in range(ev.size())], dtype=float)
            cand = codec.decode10(xv)
            assert cand["C"] == codec.C_level, "cement drifted during decode -- codec bug"
            result = RE.evaluate_robust(cand, mfg_bank, res_bank, backend="fast")
            rhat = result["joint_reliability_hat"]
            support_fraction = float(result["support_fraction"])
            f = -rhat
            c_rel = REL_TARGET - rhat
            c_model = 1.0 - support_fraction
            c_recipe, c_phi, c_deploy, c_w1 = state_violations(cand)
            ev.setBBO(f"{f} {c_rel} {c_model} {c_recipe} {c_phi} {c_deploy} {c_w1}".encode("UTF-8"))
            log_rows.append({"x": xv.tolist(), "Rhat": rhat, "c_rel": c_rel,
                             "support_fraction": support_fraction, "c_model": c_model,
                             "c_recipe": c_recipe, "c_phi": c_phi,
                             "c_deploy": c_deploy, "c_w1": c_w1})
            return 1
        except Exception as ex:  # noqa: BLE001
            print("BB EVAL ERROR:", ex, flush=True)
            return 0
    return bb


def best_logged_search_point(log_rows, x0):
    """Keep the strongest evaluated scientific point when NOMAD finds no feasible point.

    This point is only a continuation/diagnostic result.  It is never labelled
    feasible unless the reliability, model-support, and state gates all pass.
    """
    if not log_rows:
        return np.asarray(x0, dtype=float), True
    constraint_names = ("c_model", "c_recipe", "c_phi", "c_deploy", "c_w1")
    supported = [row for row in log_rows
                 if all(float(row.get(name, float("inf"))) <= 0.0 for name in constraint_names)]
    if supported:
        row = max(supported, key=lambda item: (float(item["Rhat"]), -float(item["c_rel"])))
    else:
        def violation(item):
            return sum(max(0.0, float(item.get(name, float("inf")))) for name in constraint_names)
        row = min(log_rows, key=lambda item: (violation(item), -float(item["Rhat"])))
    return np.asarray(row["x"], dtype=float), False


def run_one_level(cement_level, mfg_bank, res_bank, starts, max_bb_eval=150):
    codec = FixedCementCodec(cement_level)
    results = {}
    for tag, cand in starts.items():
        x0 = np.clip(codec.encode10(cand), 0.0, 1.0)
        log_rows = []
        bb = make_bb(codec, mfg_bank, res_bank, log_rows)
        params = ["DIMENSION 10", "BB_OUTPUT_TYPE OBJ PB PB PB PB PB PB", f"MAX_BB_EVAL {max_bb_eval}", "DISPLAY_DEGREE 0"]
        t0 = time.time()
        result = PyNomad.optimize(bb, list(x0), [0.0] * 10, [1.0] * 10, params)
        elapsed = time.time() - t0
        # A NOMAD feasible point is authoritative.  Otherwise preserve the
        # best actually evaluated scientific point for discovery/continuation;
        # the gates below still prevent an infeasible point from passing.
        bf = result.get("x_best_feas") or []
        if len(bf):
            best_x = np.asarray(bf[0] if isinstance(bf[0], (list, tuple)) else bf, dtype=float)
            fell_back = False
            used_best_infeasible = False
        else:
            best_x, fell_back = best_logged_search_point(log_rows, x0)
            used_best_infeasible = not fell_back
        best_cand = codec.decode10(best_x)
        assert best_cand["C"] == cement_level
        final = RE.evaluate_robust(best_cand, mfg_bank, res_bank, backend="fast")
        final_support_fraction = float(final["support_fraction"])
        model_supported = final_support_fraction >= 1.0
        c_recipe, c_phi, c_deploy, c_w1 = state_violations(best_cand)
        state_supported = bool(model_supported and c_recipe <= 0.0 and c_phi <= 0.0
                               and c_deploy <= 0.0 and c_w1 <= 0.0)
        results[tag] = {
            "cement_level": cement_level, "start_tag": tag, "x0": x0.tolist(), "x_best": best_x.tolist(),
            "best_cand": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in best_cand.items()},
            "final_Rhat_search_bank": final["joint_reliability_hat"], "n_evals_used": len(log_rows),
            "final_support_fraction": final_support_fraction, "model_supported": model_supported,
            "c_recipe": c_recipe, "c_phi": c_phi, "c_deploy": c_deploy, "c_w1": c_w1,
            "state_supported": state_supported,
            "run_flag": result.get("run_flag"), "stop_reason": result.get("stop_reason"), "elapsed_s": elapsed,
            "fell_back_to_start": bool(fell_back),
            "used_best_infeasible": bool(used_best_infeasible),
            "phase_d_reoptimised": bool(len(log_rows) > 0),
        }
        print(f"  [{cement_level} kg/m3 | start={tag}] Rhat={final['joint_reliability_hat']:.4f} "
              f"evals={len(log_rows)} elapsed={elapsed:.1f}s", flush=True)
    return results


def _atomic_json_write(path, payload):
    """Write a completed level atomically; never expose a partial checkpoint."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)


def _run_one_start_worker(payload):
    """Spawn-safe one-start worker for a fixed cement level; never writes."""
    cement_level, tag, cand, n_scenarios, max_bb_eval, seed = payload
    configure_single_thread_worker()
    spec = U.UncertaintySpec(CONTRACT)
    mfg_bank = U.build_manufacturing_bank(spec, n=n_scenarios, seed=seed)
    res_bank = U.build_residual_bank(BANK_CSV, n=n_scenarios, seed=seed)
    E.get_bundle()
    return tag, run_one_level(cement_level, mfg_bank, res_bank, {tag: cand}, max_bb_eval)[tag]


def main(levels, n_scenarios=SEARCH_BANK_N, max_bb_eval=150, seed=SEARCH_BANK_SEED, parallel_workers=1):
    t0 = time.time()
    if parallel_workers <= 1:
        E.get_bundle()
    spec = U.UncertaintySpec(CONTRACT)
    mfg_bank = U.build_manufacturing_bank(spec, n=n_scenarios, seed=seed)
    res_bank = U.build_residual_bank(BANK_CSV, n=n_scenarios, seed=seed)
    print(f"Phase D search bank: N={n_scenarios} mfg_hash={mfg_bank.bank_hash()[:12]} "
          f"res_hash={res_bank.bank_hash()[:12]}", flush=True)

    phase_c_starts = load_start_candidates()
    screen_n = min(int(n_scenarios), _env_int("STAGE1_FIXED_BLIND_SCREEN_SCENARIOS", 64, 16))
    screen_mfg = U.build_manufacturing_bank(spec, n=screen_n, seed=seed + 700001)
    screen_res = U.build_residual_bank(BANK_CSV, n=screen_n, seed=seed + 700001)
    print(f"Phase D feasible Phase-C starts ({len(phase_c_starts)}): {list(phase_c_starts.keys())}", flush=True)

    all_results = {}
    for level in levels:
        print(f"\n=== Phase D fixed-cement level: {level} kg/m3 ===", flush=True)
        starts = dict(phase_c_starts)
        starts.update(blind_level_starts(level, screen_mfg, screen_res, seed=seed))
        if not starts:
            raise RuntimeError(f"No independent fixed-cement starts could be generated at C={level}")
        print(f"  Total starts for level {level}: {len(starts)}", flush=True)
        level_path = os.path.join(RUN, "checkpoints", f"integer_profile_level_{int(level)}.json")
        if os.path.exists(level_path):
            with open(level_path) as f:
                saved = json.load(f)
            if (saved.get("cement_level") == level and saved.get("n_scenarios") == n_scenarios
                    and saved.get("max_bb_eval") == max_bb_eval and saved.get("seed") == seed
                    and saved.get("start_policy") == START_POLICY):
                all_results[str(level)] = saved["results"]
                print(f"  Reusing completed checkpoint: {level_path}", flush=True)
                continue
            raise RuntimeError(f"Refusing incompatible existing checkpoint: {level_path}")
        if parallel_workers <= 1:
            level_results = run_one_level(level, mfg_bank, res_bank, starts, max_bb_eval)
        else:
            configure_parent_for_spawn()
            payloads = [(level, tag, cand, n_scenarios, max_bb_eval, seed) for tag, cand in starts.items()]
            level_results = {}
            print(f"  Running {len(payloads)} independent starts with {parallel_workers} single-thread workers", flush=True)
            with CF.ProcessPoolExecutor(max_workers=parallel_workers) as pool:
                futures = [pool.submit(_run_one_start_worker, p) for p in payloads]
                for future in CF.as_completed(futures):
                    tag, result = future.result()
                    level_results[tag] = result
        _atomic_json_write(level_path, {"cement_level": level, "n_scenarios": n_scenarios,
                                        "max_bb_eval": max_bb_eval, "seed": seed,
                                        "start_policy": START_POLICY,
                                        "mfg_bank_hash": mfg_bank.bank_hash(),
                                        "res_bank_hash": res_bank.bank_hash(), "results": level_results})
        all_results[str(level)] = level_results
        print(f"  Checkpointed completed level -> {level_path}", flush=True)

    # A later one-level boundary extension (for example, 260 after an
    # initial 261--265 run) must not discard the earlier completed profile
    # from the aggregate trace.  Reconstruct it from compatible atomic
    # checkpoints, which are the authoritative per-level records.
    aggregate_results = {}
    checkpoint_dir = os.path.join(RUN, "checkpoints")
    for name in os.listdir(checkpoint_dir):
        if not (name.startswith("integer_profile_level_") and name.endswith(".json")):
            continue
        with open(os.path.join(checkpoint_dir, name)) as f:
            saved = json.load(f)
        if (saved.get("n_scenarios") == n_scenarios and saved.get("max_bb_eval") == max_bb_eval
                and saved.get("seed") == seed and saved.get("start_policy") == START_POLICY):
            aggregate_results[str(saved["cement_level"])] = saved["results"]

    out_path = os.path.join(RUN, "traces", "integer_cement_profile_results.json")
    _atomic_json_write(out_path, {"levels": sorted(float(x) for x in aggregate_results), "n_scenarios": n_scenarios, "max_bb_eval": max_bb_eval,
                                  "mfg_bank_hash": mfg_bank.bank_hash(), "res_bank_hash": res_bank.bank_hash(),
                                  "results": aggregate_results, "elapsed_s": time.time() - t0})
    print(f"\nTotal Phase D elapsed: {time.time()-t0:.1f}s -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=float, nargs="+", required=True)
    ap.add_argument("--n_scenarios", type=int, default=SEARCH_BANK_N)
    ap.add_argument("--max_bb_eval", type=int, default=150)
    ap.add_argument("--parallel-workers", type=int, default=1,
                    help="Independent single-thread NOMAD starts; parent remains sole output writer.")
    a = ap.parse_args()
    main(a.levels, a.n_scenarios, a.max_bb_eval, parallel_workers=a.parallel_workers)
