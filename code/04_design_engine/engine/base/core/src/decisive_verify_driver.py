"""Shared decisive-verification driver for run_05 (used for both the
continuous frontier candidates in Section 7 and the integer profile in
Section 8/Phase D). Seed scheme per VERIFICATION_CONTRACT.json is derived
from the stable candidate identifier, rather than cement alone.  That avoids
reusing a verification stream for two distinct candidates at the same (or
nearest-integer) cement level."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
with open(os.path.join(RUN, "TARGET_CONTRACT.json"), encoding="utf-8") as _f:
    RELIABILITY = float(json.load(_f)["required_joint_reliability"])
sys.path.insert(0, HERE)
import efficient_verification as EV  # noqa: E402
import stage17_state_support as S17  # noqa: E402
import gradation_support as GS  # noqa: E402


def seed_base_for(candidate_id: str) -> int:
    """Return a reproducible 63-bit seed base for one named candidate.

    Candidate IDs are required by the CLI and are part of the verification
    trace.  The enormous seed space prevents ordinary collisions; the caller
    additionally checks all existing trace files before an expensive run.
    Reserving 201 following values leaves the N=20k, 50k, and 100k streams
    (base/base+100/base+200) disjoint for each candidate.
    """
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("candidate_id must be a non-empty stable string")
    ceiling = (1 << 63) - 1
    span = ceiling - 1_000_000 - 200
    digest = hashlib.sha256(candidate_id.encode("utf-8")).digest()
    return 1_000_000 + (int.from_bytes(digest[:8], "big") % span)


def _assert_no_seed_collision(cid: str, seed: int) -> None:
    """Fail closed if an existing decisive trace has this stream for another ID."""
    trace_dir = os.path.join(RUN, "traces")
    if not os.path.isdir(trace_dir):
        return
    for name in os.listdir(trace_dir):
        if not (name.startswith("decisive_verification_") and name.endswith(".json")):
            continue
        with open(os.path.join(trace_dir, name), encoding="utf-8") as f:
            prior = json.load(f)
        if prior.get("seed_base") == seed and prior.get("cid") != cid:
            raise RuntimeError(
                f"seed collision with {name}: choose a distinct stable candidate ID"
            )


def support_recheck(cand: dict) -> dict:
    """Fail-closed nominal support recheck required before final acceptance."""
    ss = S17.state_support(cand)
    gc = GS.load_contract()
    w1_violation, w1_min = GS.violation(np.asarray(cand["g"], dtype=float), gc["epsilon_95"])
    state_ok = bool(ss["empirical_pass"] and ss["deployment_ok"] and w1_violation <= 0.0)
    return {
        "stage17_empirical_pass": bool(ss["empirical_pass"]),
        "stage17_deployment_ok": bool(ss["deployment_ok"]),
        "gradation_w1_min": float(w1_min),
        "gradation_w1_epsilon_95": float(gc["epsilon_95"]),
        "gradation_w1_pass": bool(w1_violation <= 0.0),
        "nominal_state_support_pass": state_ok,
    }


def decisive_verify_candidate(cid, cand, out_tag):
    """cand: dict with C, r, A, P, S, phi, g(6). Writes
    traces/decisive_verification_{out_tag}.json and returns the result dict."""
    seed = seed_base_for(cid)
    _assert_no_seed_collision(cid, seed)
    t0 = time.time()
    result, note = EV.sequential_decisive_verify(cid, cand, seed=seed, n_boot=100000)
    support = support_recheck(cand)
    result["escalation_note"] = note
    result["cand"] = cand
    result["nominal_support_recheck"] = support
    result["PASS_FAIL"] = "PASS" if (
        result["R_LCB95_joint"] >= RELIABILITY
        and result["support_fraction"] >= 1.0 - 1e-12
        and support["nominal_state_support_pass"]
    ) else "FAIL"
    result["seed_base"] = seed
    elapsed = time.time() - t0
    print(f"[{out_tag}] C={cand['C']:.3f} Rhat={result['Rhat_joint']:.4f} "
          f"R_LCB95={result['R_LCB95_joint']:.4f} ({result['PASS_FAIL']}) "
          f"n_scenarios={result['n_scenarios']} note={note} elapsed={elapsed:.1f}s", flush=True)

    out_path = os.path.join(RUN, "traces", f"decisive_verification_{out_tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cid", required=True)
    ap.add_argument("--out_tag", required=True)
    ap.add_argument("--C", type=float, required=True)
    ap.add_argument("--r", type=float, required=True)
    ap.add_argument("--A", type=float, required=True)
    ap.add_argument("--P", type=float, required=True)
    ap.add_argument("--S", type=float, required=True)
    ap.add_argument("--phi", type=float, required=True)
    ap.add_argument("--g", type=float, nargs=6, required=True)
    a = ap.parse_args()
    cand = dict(C=a.C, r=a.r, A=a.A, P=a.P, S=a.S, phi=a.phi, g=list(a.g))
    decisive_verify_candidate(a.cid, cand, a.out_tag)
