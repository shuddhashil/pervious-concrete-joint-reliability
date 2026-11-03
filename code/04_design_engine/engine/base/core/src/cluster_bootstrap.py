"""Correct source-study cluster bootstrap (run_04 repair of run_03 Problem B).

The bootstrap UNIT is the 36 source studies backing
tables/paired_residual_bank.csv, never a scenario-level pseudo-study draw.

Design (see STUDY_CLUSTER_BOOTSTRAP_REPAIR.md for the full derivation and
equivalence proof):

1. For a fixed candidate and a fixed manufacturing-scenario bank of size N
   (mu_S, mu_logQ, s_S, s_logQ, support arrays, computed ONCE), precompute a
   per-residual-row pass probability `p_r` for every one of the 650 rows in
   the residual bank: `p_r = mean over the N scenarios of joint_pass when
   EVERY scenario uses that single row's paired residual`. This costs
   650 x N vectorized evaluations, done once, never repeated inside the
   bootstrap loop.

2. Aggregate `p_r` by study: `study_p_sum[s] = sum_{r in study s} p_r`,
   `study_n_rows[s] = count of rows in study s`. Both are 36-length arrays.

3. For each of >=500 bootstrap replicates, draw EXACTLY 36 study IDs with
   replacement (uniform over the 36 observed studies) and compute

     Rhat_b = (sum_s count_s * study_p_sum[s]) / (sum_s count_s * study_n_rows[s])

   where `count_s` is how many times study s was drawn in that replicate.
   This is the row-weighted mean of `p_r` over the literal multiset formed
   by concatenating each selected study's COMPLETE residual-row cluster
   (with multiplicity when a study is drawn more than once) -- proved
   algebraically equivalent to literal enumeration in
   tests/test_cluster_bootstrap_studies.py, and far cheaper (O(36) per
   replicate instead of enumerating hundreds of rows).

`R_LCB95_cluster` is the empirical 5th percentile of the >=500 `Rhat_b`
values. This is a FINITE-STUDY uncertainty calculation around the frozen
cross-fitted residual model (repeated random row splits, same-row excluded), not independent external validation.
"""
from __future__ import annotations

import numpy as np


def per_row_pass_probability(muS, muQ, sS, sQ, support, support_threshold,
                              e_strength_rows, e_logk_rows, s_star, logk_star):
    """p_r[j] = fraction of the N scenarios (muS, muQ, sS, sQ, support) that
    pass when EVERY scenario is paired with residual row j's fixed
    (e_strength_rows[j], e_logk_rows[j]).

    Vectorized over the N scenarios for each of the R rows in a Python loop
    over R (R=650 for the authoritative bank) -- O(R x N) total, done once.
    """
    in_support = support >= support_threshold
    R = len(e_strength_rows)
    p_r = np.empty(R, dtype=float)
    for j in range(R):
        S_real = muS + sS * e_strength_rows[j]
        Q_real = muQ + sQ * e_logk_rows[j]
        joint = in_support & (S_real >= s_star) & (Q_real >= logk_star)
        p_r[j] = joint.mean()
    return p_r


def study_aggregates(study_id_per_row: np.ndarray, p_r: np.ndarray):
    """Returns (studies, study_p_sum, study_n_rows) -- three length-K arrays
    (K = number of distinct studies), in the same (sorted) study order."""
    studies = np.unique(study_id_per_row)
    study_p_sum = np.array([p_r[study_id_per_row == s].sum() for s in studies], dtype=float)
    study_n_rows = np.array([int((study_id_per_row == s).sum()) for s in studies], dtype=int)
    return studies, study_p_sum, study_n_rows


def sample_study_bootstrap_counts(n_studies_total: int, n_draw: int, n_reps: int,
                                   rng: np.random.Generator) -> np.ndarray:
    """Draw, for each of n_reps replicates, exactly n_draw study indices with
    replacement (uniform over [0, n_studies_total)) and return the resulting
    (n_reps, n_studies_total) integer COUNT matrix (each row sums to n_draw).

    This is the ONLY randomness in the cluster bootstrap: it samples the
    STUDY units, never scenario-level pseudo-studies, and is decoupled from
    the size N of the manufacturing-scenario bank.
    """
    idx = rng.integers(0, n_studies_total, size=(n_reps, n_draw))
    # one bincount over row-offset indices == the per-replicate np.add.at loop
    # (identical integer counts; ~100x faster at n_reps=100000)
    offsets = (np.arange(n_reps, dtype=np.int64) * n_studies_total)[:, None]
    counts = np.bincount((idx + offsets).ravel(), minlength=n_reps * n_studies_total)
    return counts.reshape(n_reps, n_studies_total).astype(np.int64, copy=False)


def cluster_bootstrap_reliability_fast(study_p_sum: np.ndarray, study_n_rows: np.ndarray,
                                        counts: np.ndarray) -> np.ndarray:
    """Vectorized (fast/reweighted) cluster-bootstrap reliability per
    replicate: Rhat_b = (counts @ study_p_sum) / (counts @ study_n_rows).

    counts: (n_reps, K) integer matrix from sample_study_bootstrap_counts.
    Returns: (n_reps,) array of Rhat_b values.
    """
    total_pass_weight = counts @ study_p_sum
    total_rows = counts @ study_n_rows
    return total_pass_weight / total_rows


def cluster_bootstrap_reliability_literal(p_r: np.ndarray, study_id_per_row: np.ndarray,
                                           studies: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Slow, literal reference implementation: for each replicate, EXPLICITLY
    build the multiset of row indices (concatenating each selected study's
    full row cluster, repeated by its draw count) and average p_r over that
    literal list. O(bootstrap-multiset-size) per replicate -- used only to
    prove equivalence with cluster_bootstrap_reliability_fast on small/toy
    examples, never in production (too slow at 650 rows x 500+ reps x large
    per-study clusters, and mathematically unnecessary once the fast
    identity is proved).
    """
    n_reps, K = counts.shape
    out = np.empty(n_reps, dtype=float)
    rows_by_study = {s: np.flatnonzero(study_id_per_row == s) for s in studies}
    for b in range(n_reps):
        multiset_rows = []
        for k in range(K):
            c = int(counts[b, k])
            if c == 0:
                continue
            rows = rows_by_study[studies[k]]
            for _ in range(c):
                multiset_rows.extend(rows.tolist())
        multiset_rows = np.asarray(multiset_rows, dtype=int)
        out[b] = p_r[multiset_rows].mean()
    return out


def run_cluster_bootstrap(muS, muQ, sS, sQ, support, support_threshold,
                           residual_df, s_star, logk_star, n_reps, seed) -> dict:
    """Full pipeline: precompute p_r once, aggregate by study, draw exactly
    36 studies per replicate (>= n_reps times), return the R_LCB95_cluster
    (5th percentile) plus the full replicate array and study-level table for
    reporting (tables/PER_STUDY_RELIABILITY.csv).
    """
    e_s = residual_df["signed_standardised_strength_residual"].to_numpy(dtype=float)
    e_q = residual_df["signed_standardised_logk_residual"].to_numpy(dtype=float)
    study_id_per_row = residual_df["study_id"].to_numpy()

    p_r = per_row_pass_probability(muS, muQ, sS, sQ, support, support_threshold,
                                    e_s, e_q, s_star, logk_star)
    studies, study_p_sum, study_n_rows = study_aggregates(study_id_per_row, p_r)
    n_studies = len(studies)

    rng = np.random.default_rng(seed)
    counts = sample_study_bootstrap_counts(n_studies, n_studies, n_reps, rng)
    boot_rhats = cluster_bootstrap_reliability_fast(study_p_sum, study_n_rows, counts)
    r_lcb95_cluster = float(np.percentile(boot_rhats, 5))

    per_study_table = {
        "study_id": studies.tolist(),
        "n_rows": study_n_rows.tolist(),
        "mean_p_r": (study_p_sum / study_n_rows).tolist(),
    }
    return {
        "n_studies": n_studies,
        "n_bootstrap_reps": n_reps,
        "boot_rhats": boot_rhats,
        "R_LCB95_cluster": r_lcb95_cluster,
        "per_study_table": per_study_table,
        "p_r": p_r,
    }
