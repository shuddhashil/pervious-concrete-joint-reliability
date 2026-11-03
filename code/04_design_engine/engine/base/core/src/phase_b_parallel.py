"""Parallel Phase-B rescreen with deterministic banks and parent-only output."""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import phase_b_rescreen as B
import parallel_runtime as PR

_MFG_BANK = None
_RES_BANK = None


def _init_worker() -> None:
    global _MFG_BANK, _RES_BANK
    PR.configure_single_thread_worker()
    B.E.get_bundle()
    spec = B.U.UncertaintySpec(B.CONTRACT)
    _MFG_BANK = B.U.build_manufacturing_bank(spec, n=B.RESCREEN_N, seed=B.RESCREEN_SEED)
    _RES_BANK = B.U.build_residual_bank(B.BANK_CSV, n=B.RESCREEN_N, seed=B.RESCREEN_SEED)


def _evaluate(payload: dict) -> dict:
    cand = payload.pop("cand")
    result = B.RE.evaluate_robust(cand, _MFG_BANK, _RES_BANK, backend="fast")
    rhat = result["joint_reliability_hat"]
    sup = result["support_fraction"]
    # Stage-18: nominal state-support record + failure taxonomy per candidate.
    s17 = B.s17_record(cand)
    return {
        **payload,
        **B.cand_flat(cand),
        "Rhat_rescreen": rhat,
        "support_fraction": sup,
        **s17,
        "failure_reason": B.failure_reason(s17, sup, rhat),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None,
                        help="Developer smoke-test only; production must omit this option.")
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    t0 = time.perf_counter()
    PR.configure_parent_for_spawn()
    de_shortlist = B.load_all_de_shortlist()
    deduped = B.dedup(de_shortlist)
    payloads = []
    for _, row in deduped.iterrows():
        payloads.append({
            "cand": B._cand_from_row(row), "source": "DE_dedup", "tag": None,
            "Rhat_search_original": row["Rhat_search"], "seed": row["seed"], "gen": row["gen"],
        })
    if args.limit is not None:
        payloads = payloads[:args.limit]

    print(f"Loaded {len(de_shortlist)} Phase-A rows; retained {len(deduped)} deduplicated rows; "
          f"evaluating {len(payloads)} candidates with {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        rows = list(pool.map(_evaluate, payloads, chunksize=1))

    out = B.pd.DataFrame(rows).sort_values(["Rhat_rescreen", "C"], ascending=[False, True])
    out_path = os.path.join(B.RUN, "tables", "CONTINUOUS_FRONTIER_SHORTLIST.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} rows). Elapsed {time.perf_counter()-t0:.1f}s", flush=True)
    print(out[["source", "tag", "C", "Rhat_rescreen", "support_fraction"]].head(20).to_string())


if __name__ == "__main__":
    main()
