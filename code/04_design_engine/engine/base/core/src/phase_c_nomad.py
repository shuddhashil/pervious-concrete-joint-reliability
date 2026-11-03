"""Phase C - genuine NOMAD/MADS local refinement over the FULL 11-D design
space (codec11.py directly, cement free -- no fixed_cement_codec wrapper).

Objective f=cement (minimize), constraint PB: c_rel=0.90-Rhat<=0, IDENTICAL
convention to run_04's inherited src/robust_nomad.py."""
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
sys.path.insert(0, HERE)
import PyNomad  # noqa: E402
import codec11 as K  # noqa: E402
import stage1_evaluator as E  # noqa: E402
import uncertainty as U  # noqa: E402
import robust_evaluator as RE  # noqa: E402
import fast_descriptors as FD  # noqa: E402
import fast_local_residual as FLR  # noqa: E402
import stage17_state_support as S17  # noqa: E402  (Stage-18 nominal state-support gate)
import gradation_support as GS  # noqa: E402
from parallel_runtime import configure_parent_for_spawn, configure_single_thread_worker  # noqa: E402

if os.environ.get("DISABLE_FAST_DESCRIPTORS") != "1":
    FD.install_patch(E)
    print("[perf] fast_descriptors patch installed", flush=True)
if os.environ.get("DISABLE_FAST_LOCAL_RESIDUAL") != "1":
    FLR.install_patch(E.L)
    print("[perf] fast_local_residual patch installed", flush=True)

CONTRACT = os.path.join(RUN, "UNCERTAINTY_CONTRACT.json")
BANK_CSV = os.path.join(RUN, "tables", "paired_residual_bank.csv")
ORDER = "coarse_first"
REL_TARGET = RT.RELIABILITY
N_STARTS_TARGET = 10
SEARCH_BANK_N = 1024
SEARCH_BANK_SEED = 205609301


def select_starts(n_starts=N_STARTS_TARGET):
    df = pd.read_csv(os.path.join(RUN, "tables", "CONTINUOUS_FRONTIER_SHORTLIST.csv"))
    # Stage-18: NOMAD starts must be state-supported whenever such basins
    # exist (the new hard constraint); fall back through progressively weaker
    # pools so a start is always available for diagnosis.
    if "s17_state_pass" in df.columns:
        # Phase C may inspect weaker rows for diagnosis if needed, but it
        # must first spend its optimisation budget on candidates that pass
        # every nominal support gate, including the W1 gradation contract.
        w1_pass = (df["gradation_w1_pass"].astype(str) == "True"
                   if "gradation_w1_pass" in df.columns
                   else pd.Series(False, index=df.index))
        sup = df[(df["s17_state_pass"].astype(str) == "True") & w1_pass]
        feasible = sup[sup["Rhat_rescreen"] >= REL_TARGET].copy()
        if len(feasible) == 0:
            feasible = sup.copy()
    else:
        feasible = pd.DataFrame()
    if len(feasible) == 0:
        feasible = df[df["Rhat_rescreen"] >= REL_TARGET].copy()
    if len(feasible) == 0:
        feasible = df.copy()  # fall back to best-available if nothing clears 0.90 on the rescreen bank
    feasible = feasible.sort_values(["Rhat_rescreen", "C"], ascending=[False, True])
    # Spread only this run's DE archive; no external coordinate source.
    # Multimodal discipline: keep several distinct basins with a minimum
    # allocation each (here: one NOMAD start per basin, up to n_starts), plus
    # an implicit novelty quota via the global Phase-A injections.  No single
    # broad mediocre basin may consume the whole start budget.
    de = feasible[feasible["source"] == "DE_dedup"]
    chosen = de.drop_duplicates(subset=["C", "r", "A", "P", "S", "phi"])
    chosen = chosen.head(n_starts)
    starts = {}
    for i, (_, row) in enumerate(chosen.iterrows()):
        tag = row["tag"] if pd.notna(row.get("tag")) else f"DE_dedup_{i}"
        g = [float(row[f"g{j}"]) for j in range(1, 7)]
        cand = dict(C=float(row["C"]), r=float(row["r"]), A=float(row["A"]), P=float(row["P"]),
                    S=float(row["S"]), phi=float(row["phi"]), g=g)
        starts[str(tag)] = K.encode11(cand, ORDER)
    return starts


def state_violations(cand):
    """Stage-18 continuous state-support violations (recipe, phi, deployment)
    plus TASK A2 W1 gradation gate as 4th constraint.

    All are <= 0-iff-satisfied NOMAD PB constraints (0.0 iff satisfied).
    """
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


def make_bb(mfg_bank, res_bank, log_rows):
    def bb(ev):
        try:
            xv = np.array([ev.get_coord(i) for i in range(ev.size())], dtype=float)
            cand = K.decode11(xv, ORDER)
            result = RE.evaluate_robust(cand, mfg_bank, res_bank, backend="fast")
            rhat = result["joint_reliability_hat"]
            c_rel = REL_TARGET - rhat
            c_recipe, c_phi, c_deploy, c_w1 = state_violations(cand)
            f = float(cand["C"])
            ev.setBBO(f"{f} {c_rel} {c_recipe} {c_phi} {c_deploy} {c_w1}".encode("UTF-8"))
            log_rows.append({"x": xv.tolist(), "C": f, "Rhat": rhat, "c_rel": c_rel,
                             "c_recipe": c_recipe, "c_phi": c_phi,
                             "c_deploy": c_deploy, "c_w1": c_w1})
            return 1
        except Exception as ex:  # noqa: BLE001
            print("BB EVAL ERROR:", ex, flush=True)
            return 0
    return bb


def run_start(tag, x0, mfg_bank, res_bank, max_bb_eval):
    log_rows = []
    bb = make_bb(mfg_bank, res_bank, log_rows)
    params = ["DIMENSION 11", "BB_OUTPUT_TYPE OBJ PB PB PB PB PB", f"MAX_BB_EVAL {max_bb_eval}", "DISPLAY_DEGREE 0"]
    t0 = time.time()
    result = PyNomad.optimize(bb, list(x0), [0.0] * 11, [1.0] * 11, params)
    elapsed = time.time() - t0
    # A non-feasible point is never reportable, but retaining the best one as
    # diagnostic search state is essential for blind discovery and restart.
    # It remains explicitly ineligible for Phase D/final selection.
    bf = result.get("x_best_feas") or []
    if len(bf):
        best_x = np.asarray(bf[0] if isinstance(bf[0], (list, tuple)) else bf, dtype=float)
        from_start = False
        diagnostic_x = best_x
    else:
        best_x = np.asarray(x0, dtype=float)
        from_start = True
        if log_rows:
            near = min(log_rows, key=lambda z: (max(0.0, z["c_recipe"]), max(0.0, z["c_phi"]),
                                                 max(0.0, z["c_deploy"]), max(0.0, z["c_w1"]),
                                                 max(0.0, z["c_rel"]), z["C"]))
            diagnostic_x = np.asarray(near["x"], dtype=float)
        else:
            diagnostic_x = best_x
    cand = K.decode11(best_x, ORDER)
    diagnostic_cand = K.decode11(diagnostic_x, ORDER)
    diagnostic_final = RE.evaluate_robust(diagnostic_cand, mfg_bank, res_bank, backend="fast")
    final = RE.evaluate_robust(cand, mfg_bank, res_bank, backend="fast")
    c_recipe, c_phi, c_deploy, c_w1 = state_violations(cand)
    return {
        "tag": tag, "x0": np.asarray(x0).tolist(), "x_best": best_x.tolist(),
        "x_best_ineligible_diagnostic": diagnostic_x.tolist(),
        "best_cand": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in cand.items()},
        "best_ineligible_diagnostic_cand": {k: (v.tolist() if hasattr(v, "tolist") else v)
                                             for k, v in diagnostic_cand.items()},
        "final_Rhat": final["joint_reliability_hat"], "final_support_fraction": final["support_fraction"],
        "c_recipe": c_recipe, "c_phi": c_phi, "c_deploy": c_deploy, "c_w1": c_w1,
        "state_supported": bool(c_recipe <= 0.0 and c_phi <= 0.0 and c_deploy <= 0.0 and c_w1 <= 0.0),
        "n_evals_used": len(log_rows), "run_flag": result.get("run_flag"),
        "stop_reason": result.get("stop_reason"), "elapsed_s": elapsed,
        "moved_from_start": (not from_start) and (not np.allclose(best_x, np.asarray(x0), atol=1e-9)),
        "fell_back_to_start": bool(from_start),
        "best_ineligible_diagnostic_Rhat": diagnostic_final["joint_reliability_hat"],
    }


def _run_start_worker(payload):
    """Spawn-safe worker: builds its own deterministic common-random bank.

    No worker writes results.  The parent is the sole writer after all starts
    finish, so concurrent starts cannot collide on a trace/checkpoint file.
    """
    tag, x0, n_scenarios, max_bb_eval = payload
    configure_single_thread_worker()
    spec = U.UncertaintySpec(CONTRACT)
    mfg_bank = U.build_manufacturing_bank(spec, n=n_scenarios, seed=SEARCH_BANK_SEED)
    res_bank = U.build_residual_bank(BANK_CSV, n=n_scenarios, seed=SEARCH_BANK_SEED)
    E.get_bundle()
    return run_start(tag, np.asarray(x0, dtype=float), mfg_bank, res_bank, max_bb_eval)


def main(max_bb_eval=150, n_scenarios=SEARCH_BANK_N, parallel_workers=1):
    t0 = time.time()
    if parallel_workers <= 1:
        E.get_bundle()
    spec = U.UncertaintySpec(CONTRACT)
    mfg_bank = U.build_manufacturing_bank(spec, n=n_scenarios, seed=SEARCH_BANK_SEED)
    res_bank = U.build_residual_bank(BANK_CSV, n=n_scenarios, seed=SEARCH_BANK_SEED)
    print(f"Phase C search bank: N={n_scenarios} mfg_hash={mfg_bank.bank_hash()[:12]} "
          f"res_hash={res_bank.bank_hash()[:12]}", flush=True)

    starts = select_starts()
    print(f"Phase C starts ({len(starts)}): {list(starts.keys())}", flush=True)

    results = {}
    if parallel_workers <= 1:
        for tag, x0 in starts.items():
            print(f"\n=== Phase C NOMAD start: {tag} ===", flush=True)
            r = run_start(tag, x0, mfg_bank, res_bank, max_bb_eval)
            results[tag] = r
            print(f"  {tag}: C_best={r['best_cand']['C']:.3f} Rhat={r['final_Rhat']:.4f} "
                  f"evals={r['n_evals_used']} moved={r['moved_from_start']} elapsed={r['elapsed_s']:.1f}s", flush=True)
    else:
        configure_parent_for_spawn()
        payloads = [(tag, x0.tolist(), n_scenarios, max_bb_eval) for tag, x0 in starts.items()]
        print(f"Running {len(payloads)} independent Phase C starts with {parallel_workers} single-thread workers", flush=True)
        with CF.ProcessPoolExecutor(max_workers=parallel_workers) as pool:
            futures = {pool.submit(_run_start_worker, p): p[0] for p in payloads}
            for future in CF.as_completed(futures):
                tag = futures[future]
                r = future.result()
                results[tag] = r
                print(f"  {tag}: C_best={r['best_cand']['C']:.3f} Rhat={r['final_Rhat']:.4f} "
                      f"evals={r['n_evals_used']} moved={r['moved_from_start']} elapsed={r['elapsed_s']:.1f}s", flush=True)

    out_path = os.path.join(RUN, "traces", "phase_c_nomad_results.json")
    with open(out_path, "w") as f:
        json.dump({"n_scenarios": n_scenarios, "max_bb_eval": max_bb_eval,
                    "mfg_bank_hash": mfg_bank.bank_hash(), "res_bank_hash": res_bank.bank_hash(),
                    "results": results, "elapsed_s": time.time() - t0}, f, indent=2, default=str)

    rows = []
    for basin_id, (tag, r) in enumerate(results.items()):
        c = r["best_cand"]
        rows.append({"tag": tag, "basin_id": int(basin_id), "C": c["C"], "r": c["r"], "A": c["A"], "P": c["P"], "S": c["S"], "phi": c["phi"],
                     **{f"g{i+1}": c["g"][i] for i in range(6)},
                     "final_Rhat": r["final_Rhat"], "support_fraction": r["final_support_fraction"],
                     "c_recipe": r.get("c_recipe", 0.0), "c_phi": r.get("c_phi", 0.0),
                      "c_deploy": r.get("c_deploy", 0.0), "c_w1": r.get("c_w1", 0.0),
                      "reliability_gate": bool(r["final_Rhat"] >= REL_TARGET),
                      "verified": False,
                     "state_supported": r.get("state_supported", True),
                     "n_evals_used": r["n_evals_used"], "moved_from_start": r["moved_from_start"],
                     "run_flag": r["run_flag"], "stop_reason": r["stop_reason"], "elapsed_s": r["elapsed_s"]})
    df = pd.DataFrame(rows).sort_values(["final_Rhat", "C"], ascending=[False, True])
    df.to_csv(os.path.join(RUN, "tables", "CONTINUOUS_NOMAD_RUNS.csv"), index=False)
    print(f"\nTotal Phase C elapsed: {time.time()-t0:.1f}s -> {out_path}")
    print(df.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_bb_eval", type=int, default=150)
    ap.add_argument("--n_scenarios", type=int, default=SEARCH_BANK_N)
    ap.add_argument("--parallel-workers", type=int, default=1,
                    help="Independent single-thread NOMAD starts; parent remains sole output writer.")
    a = ap.parse_args()
    main(a.max_bb_eval, a.n_scenarios, a.parallel_workers)
