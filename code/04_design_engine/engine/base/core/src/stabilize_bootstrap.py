"""Bootstrap-stability correction (post-review finding, documented in
BOOTSTRAP_STABILIZATION_NOTE.md). The 500-replicate cluster bootstrap
adopted per the driving prompt's stated minimum turned out to have enough
Monte Carlo noise in its 5th percentile (sd across independent seeds
~0.0014 at 500 reps for a candidate this close to 0.90) that the
PASS/FAIL classification for 265 kg/m3 depended on which bootstrap seed
happened to be used -- confirmed by re-drawing the SAME already-computed
per-study pass probabilities (no GP recomputation) with many independent
seeds and much larger replicate counts.

This script re-derives calib_bootstrap_R_LCB95 (and hence R_LCB95_joint)
for each existing decisive_verification_{level}.json using
N_BOOT_STABILIZED=100,000 replicates and a dedicated, documented seed --
reusing the ALREADY-COMPUTED per_study_table (study_id, n_rows, mean_p_r)
saved by efficient_verification.py, so NO forward-model (GP) prediction is
re-run. This is a legitimate refinement within the already-adopted
cluster-bootstrap method (Section 5 of the driving prompt requires "at
least 500" replicates, not exactly 500), not a redesign of the estimator.

Original 500-replicate numbers are preserved in the output under
`original_500rep_*` keys for audit -- nothing is silently overwritten.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import cluster_bootstrap as CB  # noqa: E402

N_BOOT_STABILIZED = 100000
SEED_BASE = 700000  # dedicated, documented, disjoint from every search/verification seed used so far


def stabilize_one(level):
    path = os.path.join(RUN, "traces", f"decisive_verification_{int(level)}.json")
    with open(path) as f:
        d = json.load(f)

    pst = d["per_study_table"]
    n_rows = np.array(pst["n_rows"], dtype=float)
    mean_p_r = np.array(pst["mean_p_r"], dtype=float)
    study_p_sum = n_rows * mean_p_r
    n_studies = len(n_rows)
    assert n_studies == d["n_studies_cluster_bootstrap"] == 36

    seed = SEED_BASE + int(level)
    rng = np.random.default_rng(seed)
    counts = CB.sample_study_bootstrap_counts(n_studies, n_studies, N_BOOT_STABILIZED, rng)
    boot = CB.cluster_bootstrap_reliability_fast(study_p_sum, n_rows, counts)
    stabilized_cluster_lb = float(np.percentile(boot, 5))

    d["original_500rep_calib_bootstrap_R_LCB95"] = d["calib_bootstrap_R_LCB95"]
    d["original_500rep_R_LCB95_joint"] = d["R_LCB95_joint"]
    d["original_n_bootstrap_reps"] = d["n_bootstrap_reps"]
    d["original_PASS_FAIL"] = d["PASS_FAIL"]

    d["calib_bootstrap_R_LCB95"] = stabilized_cluster_lb
    d["n_bootstrap_reps"] = N_BOOT_STABILIZED
    d["bootstrap_stabilization_seed"] = seed
    d["R_LCB95_joint"] = min(d["mc_wilson_lower_95"], stabilized_cluster_lb)
    d["PASS_FAIL"] = "PASS" if d["R_LCB95_joint"] >= 0.90 else "FAIL"
    d["boot_rhats"] = boot.tolist()  # replace with the stabilized replicate distribution
    d["stabilization_note"] = ("calib_bootstrap_R_LCB95/R_LCB95_joint/PASS_FAIL recomputed with "
                                f"{N_BOOT_STABILIZED} cluster-bootstrap replicates (seed {seed}), reusing "
                                "the already-computed per_study_table -- no GP forward pass re-run. "
                                "See BOOTSTRAP_STABILIZATION_NOTE.md. Original 500-rep values preserved "
                                "under original_500rep_* keys.")

    with open(path, "w") as f:
        json.dump(d, f, indent=2, default=str)

    print(f"C={level}: 500rep_R_LCB95={d['original_500rep_R_LCB95_joint']:.5f} ({d['original_PASS_FAIL']}) "
          f"-> {N_BOOT_STABILIZED}rep_R_LCB95={d['R_LCB95_joint']:.5f} ({d['PASS_FAIL']})")
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=float, nargs="+", required=True)
    a = ap.parse_args()
    for lvl in a.levels:
        stabilize_one(lvl)
