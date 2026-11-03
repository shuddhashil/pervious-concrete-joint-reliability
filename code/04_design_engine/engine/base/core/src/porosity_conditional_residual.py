"""Porosity-conditional (Mondrian) residual sampling for the decisive-verification
Wilson/point-estimate term.

Why this exists: the pooled residual bank draws a paired (strength, log-permeability)
residual uniformly from all 650 training rows, regardless of the candidate's own
achieved porosity for that scenario. But the model is known to be less well
calibrated at low porosity specifically. Pooling residuals
across porosity hides this: a low-porosity scenario borrows "average" residual
behaviour from better-behaved porosity ranges, understating its true risk.

This module stratifies the residual bank into porosity bins (terciles by default, sized so no bin falls below about 180 rows), and draws each scenario's
residual ONLY from the bin matching that scenario's own achieved porosity.

Explicitly out of scope (not modified by this module): the study-cluster bootstrap
calibration term in cluster_bootstrap.py, which stratifies by STUDY, a different
and orthogonal axis. R_LCB95 = min(wilson_lb, calib_lb) -- this module only changes
the residuals feeding the Wilson-bound / point-estimate side.

Gated by the POROSITY_CONDITIONAL_CALIBRATION=1 environment variable at the call
site (efficient_verification.py); default behaviour is completely unchanged unless
that flag is set, so this is purely additive and safe to leave in place.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

from uncertainty import ResidualScenarioBank  # noqa: E402  (same package, sibling module)

MIN_BIN_N = 180  # F-013's stated floor; asserted, never silently violated


def _reference_csv_path(bank_csv_path: str) -> str:
    """paired_residual_bank.csv lives in <run>/tables/; the porosity reference
    (xmodel_porosity per original_row_index/split) lives in <run>/models/."""
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(bank_csv_path)))
    return os.path.join(run_dir, "models", "oos_residual_reference_M1global.csv")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_binned_bank(bank_csv_path: str, n_bins: int):
    """Load the residual bank, attach porosity, and cut it into n_bins terciles
    (by default) of the bank's OWN porosity distribution. Returns
    (bank_df_with_porosity_and_bin, bin_edges, bank_sha256)."""
    ref_path = _reference_csv_path(bank_csv_path)
    if not os.path.isfile(ref_path):
        raise FileNotFoundError(
            f"porosity-conditional sampling requires the porosity reference file, "
            f"not found at {ref_path!r} (expected alongside {bank_csv_path!r})")
    bank = pd.read_csv(bank_csv_path)
    ref = pd.read_csv(ref_path)
    if "xmodel_porosity" not in ref.columns:
        raise KeyError("oos_residual_reference file lacks 'xmodel_porosity'")
    merged = bank.merge(ref[["split", "original_row_index", "xmodel_porosity"]],
                         on=["split", "original_row_index"], how="left")
    n_missing = int(merged["xmodel_porosity"].isna().sum())
    if n_missing:
        raise RuntimeError(
            f"{n_missing} of {len(merged)} residual-bank rows failed to join a "
            f"porosity value -- refusing to silently drop or impute them")
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = merged["xmodel_porosity"].quantile(quantiles).to_numpy()
    edges[0] -= 1e-9  # include the exact minimum in bin 0 under right-closed cuts
    merged["_bin"] = pd.cut(merged["xmodel_porosity"], bins=edges, labels=False, include_lowest=True)
    counts = merged["_bin"].value_counts().sort_index()
    for b, n in counts.items():
        if n < MIN_BIN_N:
            raise RuntimeError(
                f"porosity bin {b} has only {n} rows (< MIN_BIN_N={MIN_BIN_N}); "
                f"refusing an under-populated Mondrian bin -- reduce n_bins")
    return merged, edges, _sha256_file(bank_csv_path)


def build_residual_bank_porosity_conditional(bank_csv_path: str, phi_actual: np.ndarray,
                                              seed: int, n_bins: int = 3) -> ResidualScenarioBank:
    """Drop-in replacement for uncertainty.build_residual_bank(..., resample_mode="iid_bootstrap")
    that conditions each scenario's residual draw on that scenario's OWN achieved
    porosity (phi_actual), sampled with replacement from the matching porosity bin
    only. n = len(phi_actual). Never mixes bins; never falls back to pooled sampling
    silently -- any data problem raises rather than degrading quietly.
    """
    phi_actual = np.asarray(phi_actual, dtype=float)
    n = len(phi_actual)
    merged, edges, bank_sha = _load_binned_bank(bank_csv_path, n_bins)

    scenario_bin = np.digitize(phi_actual, edges[1:-1], right=True)
    scenario_bin = np.clip(scenario_bin, 0, n_bins - 1)

    pool_by_bin = {b: merged.index[merged["_bin"] == b].to_numpy() for b in range(n_bins)}
    rng = np.random.default_rng(seed)
    idx = np.empty(n, dtype=int)
    for b, pool in pool_by_bin.items():
        mask = scenario_bin == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        idx[mask] = pool[rng.integers(0, len(pool), size=cnt)]

    sub = merged.iloc[idx]
    return ResidualScenarioBank(
        n=n, seed=seed, row_indices=idx, study_id=sub["study_id"].to_numpy(),
        e_strength=sub["signed_standardised_strength_residual"].to_numpy(dtype=float),
        e_logk=sub["signed_standardised_logk_residual"].to_numpy(dtype=float),
        bank_csv_sha256=bank_sha, resample_mode=f"porosity_conditional_iid_bootstrap_{n_bins}bins",
    )


MIN_STUDIES_FOR_CLUSTER_BOOTSTRAP = 10  # below this, between-study variance is not
# meaningfully estimable; refuse rather than silently return an unstable/miscounted bound


def filter_residual_bank_to_scenario_bin(bank_csv_path: str, phi_actual: np.ndarray, n_bins: int = 3):
    """For the study-cluster-bootstrap calibration term (cluster_bootstrap.py), which
    is not itself per-scenario the way the Wilson term is: restrict the 650-row
    residual bank down to only the rows sharing the SAME porosity bin as this
    candidate's own achieved-porosity scenarios (bin membership decided by the
    MEDIAN of phi_actual, since manufacturing variability is tight -- ~1 pp SD --
    around one nominal target, so scenarios overwhelmingly sit in one bin; see the
    achieved_phi_p05/p50/p95 diagnostic already reported alongside R_LCB95).

    This changes the effective number of studies contributing to the bootstrap unit
    (only studies with rows in this porosity bin count) -- the honest consequence
    Mondrian calibration is meant to surface, not a bug: a design in a
    thinly-populated porosity zone SHOULD get a wider, more uncertain bound.

    Returns (filtered_df, bin_index, bin_edges). Raises if the filtered set has
    fewer than MIN_STUDIES_FOR_CLUSTER_BOOTSTRAP distinct studies -- refuses to
    report a between-study bound built on too few studies rather than understate
    the resulting uncertainty.
    """
    phi_actual = np.asarray(phi_actual, dtype=float)
    merged, edges, _ = _load_binned_bank(bank_csv_path, n_bins)
    median_phi = float(np.median(phi_actual))
    b = int(np.clip(np.digitize([median_phi], edges[1:-1], right=True)[0], 0, n_bins - 1))
    filtered = merged[merged["_bin"] == b].reset_index(drop=True)
    n_studies = filtered["study_id"].nunique()
    if n_studies < MIN_STUDIES_FOR_CLUSTER_BOOTSTRAP:
        raise RuntimeError(
            f"porosity bin {b} (median candidate porosity {median_phi:.2f}) has only "
            f"{n_studies} distinct studies (< MIN_STUDIES_FOR_CLUSTER_BOOTSTRAP="
            f"{MIN_STUDIES_FOR_CLUSTER_BOOTSTRAP}) -- refusing an unstable cluster bootstrap "
            f"rather than silently reporting one built on too few studies")
    return filtered, b, edges
