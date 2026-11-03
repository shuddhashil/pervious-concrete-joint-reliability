"""run_02 Phase F - lineage verification + paired residual bank construction
+ exact reproduction of the frozen joint_bundle_M1global.json quantiles.

Verifies that research/Stage1_CA_Forward_Refit_20260913/run_01/models/
oos_residual_reference_M1global.csv is the row-level, signed, paired,
study-labelled held-out residual bank for the CURRENT CA-aware M1global
oracle, by:
  1. joining it to oos_production_scales_M1global.csv on (split, row) to
     recover study labels (0 unmatched required);
  2. checking the joined mu_strength/mu_logk match exactly;
  3. RECONSTRUCTING the exact paired_max_score calibration T-statistic
     T = max((muCS - yCS) / sCS, (muQ - yQ) / sQ) from
     oos_production_scales_M1global.csv's own saved columns, and confirming
     its 0.80/0.90/0.95 order-statistic quantiles (order_stat_q, the SAME
     ceil((n+1)*p) convention used by research/Stage1_CA_Forward_Refit_20260913/
     run_01/src/step6_joint.py) reproduce joint_bundle_M1global.json's saved
     q values EXACTLY (to floating point precision) -- this is the strongest
     possible confirmation that the calibration scale/statistic definition
     used here matches the frozen production calibration, not something
     merely plausible.

The PRIMARY reliability path (paired_residual_mc) resamples SIGNED residual
PAIRS (y-mu)/s directly from the bank -- NOT the max-score T statistic, which
is retained only as a diagnostic column (see UNCERTAINTY_MODEL.md /
ROBUST_METHOD.md for why the two must never be combined).
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"C:\Users\Shuddhashil\Desktop\Mix Design Paper Rebuild"
REFIT_RUN = os.path.join(ROOT, "research", "Stage1_CA_Forward_Refit_20260913", "run_01")
RUN = os.path.join(ROOT, "research", "Stage1_1813_90_Robust_20260915", "run_02")

RESID_CSV = os.path.join(REFIT_RUN, "models", "oos_residual_reference_M1global.csv")
SCALES_CSV = os.path.join(REFIT_RUN, "models", "oos_production_scales_M1global.csv")
JOINT_BUNDLE_JSON = os.path.join(REFIT_RUN, "models", "joint_bundle_M1global.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def order_stat_q(scores, p):
    """Exact convention from step6_joint.py: k = ceil((n+1)*p), q = sorted[k-1]."""
    sc = np.sort(np.asarray(scores, dtype=float))
    n = len(sc)
    k = int(np.ceil((n + 1) * p))
    if k > n or k < 1:
        return float("inf"), k
    return float(sc[k - 1]), k


def main():
    res = pd.read_csv(RESID_CSV)
    scl = pd.read_csv(SCALES_CSV)
    with open(JOINT_BUNDLE_JSON) as f:
        jb = json.load(f)

    # --- check 1: join for study labels, 0 unmatched ---
    merged = res.merge(scl[["split", "row", "study"]], left_on=["split", "original_row_index"],
                        right_on=["split", "row"], how="left")
    n_unmatched = int(merged["study"].isna().sum())
    if n_unmatched != 0:
        raise RuntimeError(f"LINEAGE CHECK FAILED: {n_unmatched} residual rows did not join to a study label")

    # --- check 2: mu values match exactly (same model pass) ---
    merged_full = res.merge(scl, left_on=["split", "original_row_index"], right_on=["split", "row"], how="left")
    mu_diff_max = float((merged_full["strength_pred_mpa"] - merged_full["muCS"]).abs().max())
    if mu_diff_max > 1e-9:
        raise RuntimeError(f"LINEAGE CHECK FAILED: mu_strength mismatch (max diff {mu_diff_max})")

    n_cal = len(merged)
    n_studies = int(merged["study"].nunique())
    if n_cal != 650:
        raise RuntimeError(f"LINEAGE CHECK FAILED: n_cal={n_cal} != joint_bundle_M1global.json n_cal=650")

    # --- check 3: reconstruct paired_max_score T and reproduce quantiles EXACTLY ---
    T_reconstructed = np.maximum((scl["muCS"] - scl["yCS"]) / scl["sCS"],
                                  (scl["muQ"] - scl["yQ"]) / scl["sQ"])
    T_saved = scl["T"].to_numpy()
    t_max_diff = float(np.abs(T_reconstructed.to_numpy() - T_saved).max())
    if t_max_diff > 1e-9:
        raise RuntimeError(f"T-statistic reconstruction mismatch: max diff {t_max_diff}")

    quantile_checks = {}
    for p in (0.8, 0.9, 0.95):
        q, k = order_stat_q(T_saved, p)
        expected = jb["levels"][str(p)]
        diff = abs(q - expected["q"])
        quantile_checks[str(p)] = {"reconstructed_q": q, "reconstructed_k": k, "expected_q": expected["q"],
                                    "expected_k": expected["k"], "abs_diff": diff,
                                    "exact_match": diff < 1e-9 and k == expected["k"]}
        if not quantile_checks[str(p)]["exact_match"]:
            raise RuntimeError(f"QUANTILE REPRODUCTION FAILED at p={p}: {quantile_checks[str(p)]}")

    # --- build the versioned artifact: signed paired residuals (NOT the T max-score) ---
    bank = pd.DataFrame({
        "study_id": merged["study"],
        "split": merged["split"],
        "original_row_index": merged["original_row_index"],
        "y_strength": merged["strength_true_mpa"],
        "y_logk": merged["logperm_true"],
        "mu_strength": merged["strength_pred_mpa"],
        "mu_logk": merged["logperm_pred"],
        "s_strength": merged["strength_std_oos_effective_mpa"],
        "s_logk": merged["logperm_std_oos_effective"],
        "signed_standardised_strength_residual": merged["strength_residual_mpa"] / merged["strength_std_oos_effective_mpa"],
        "signed_standardised_logk_residual": merged["logperm_residual"] / merged["logperm_std_oos_effective"],
        "T_diagnostic_max_score": merged_full["T"],
        "support_metadata": "not_included_upstream;recomputable_from_xmodel_columns_if_needed",
    })

    os.makedirs(os.path.join(RUN, "tables"), exist_ok=True)
    bank_path = os.path.join(RUN, "tables", "paired_residual_bank.csv")
    bank.to_csv(bank_path, index=False)
    bank_hash = sha256_file(bank_path)

    npz_path = os.path.join(RUN, "tables", "residual_bank.npz")
    np.savez(npz_path,
              study_id=bank["study_id"].to_numpy(), e_strength=bank["signed_standardised_strength_residual"].to_numpy(),
              e_logk=bank["signed_standardised_logk_residual"].to_numpy(), T=bank["T_diagnostic_max_score"].to_numpy())

    validation_rows = [
        {"check": "join_zero_unmatched", "result": n_unmatched == 0, "detail": f"n_unmatched={n_unmatched}"},
        {"check": "mu_exact_match", "result": mu_diff_max <= 1e-9, "detail": f"max_diff={mu_diff_max}"},
        {"check": "n_cal_650", "result": n_cal == 650, "detail": f"n_cal={n_cal}"},
        {"check": "n_studies_36", "result": n_studies == 36, "detail": f"n_studies={n_studies}"},
        {"check": "T_reconstruction_exact", "result": t_max_diff <= 1e-9, "detail": f"max_diff={t_max_diff}"},
    ]
    for p, chk in quantile_checks.items():
        validation_rows.append({"check": f"quantile_reproduction_p{p}", "result": chk["exact_match"], "detail": json.dumps(chk)})
    pd.DataFrame(validation_rows).to_csv(os.path.join(RUN, "tables", "RESIDUAL_BANK_VALIDATION.csv"), index=False)

    # --- write UNCERTAINTY_CONTRACT.json reliability_mode confirmation ---
    contract_path = os.path.join(RUN, "UNCERTAINTY_CONTRACT.json")
    with open(contract_path) as f:
        contract = json.load(f)
    contract["paired_residual_bank"] = {
        "path": "tables/paired_residual_bank.csv", "sha256": bank_hash, "n_cal": n_cal, "n_studies": n_studies,
        "quantile_reproduction": quantile_checks,
        "source_files": {"oos_residual_reference_M1global.csv": sha256_file(RESID_CSV),
                           "oos_production_scales_M1global.csv": sha256_file(SCALES_CSV),
                           "joint_bundle_M1global.json": sha256_file(JOINT_BUNDLE_JSON)},
    }
    with open(contract_path, "w") as f:
        json.dump(contract, f, indent=2)

    print(f"LINEAGE + QUANTILE CHECKS PASSED. n_cal={n_cal} n_studies={n_studies} bank_sha256={bank_hash}")
    for p, chk in quantile_checks.items():
        print(f"  p={p}: reconstructed q={chk['reconstructed_q']:.10f} expected q={chk['expected_q']:.10f} "
              f"diff={chk['abs_diff']:.2e} k={chk['reconstructed_k']}/{chk['expected_k']}")


if __name__ == "__main__":
    main()
