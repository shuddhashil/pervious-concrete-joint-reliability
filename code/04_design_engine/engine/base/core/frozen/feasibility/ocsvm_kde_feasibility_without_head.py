"""Train and use an OCSVM + KDE soft feasibility filter.

This script builds a continuous applicability-domain score in the exact
no-head standardised feature space used by the LCMGP oracle. It does not
retrain the LCMGP model and does not read DATASET.xlsx for the training
feature matrix. The nine-feature training matrix is recovered directly from
the oracle's saved ``train_x`` tensor.

When imported, the public API at the bottom can be used by inverse13.py:

    load_feasibility_filter(...)
    prepare_features_for_filter(...)
    score_feasibility(...)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.svm import OneClassSVM


FEATURES = [
    "porosity",
    "w/c",
    "cem (kg/m3)",
    "SP (kg/m3)",
    "sand (kg/m3)",
    "WAS",
    "I_gap",
    "Cu",
    "Aag(mm^-1)",
]

CONT_IDXS = list(range(len(FEATURES)))

TRANSFORM_RECIPE = {
    "log_cols": ["cem (kg/m3)", "Cu"],
    "log1p_cols": ["SP (kg/m3)", "sand (kg/m3)", "Aag(mm^-1)"],
    "raw_cols": ["porosity", "w/c", "WAS", "I_gap"],
}

DEFAULT_ORACLE = "lcmgp_without_head_oracle_full_data.pt"
DEFAULT_OUTPUT = "ocsvm_kde_feasibility_without_head.pkl"
DEFAULT_DIAGNOSTICS = "ocsvm_kde_diagnostics_without_head.csv"
DEFAULT_SCORE_BATCH_SIZE = 50_000
DEFAULT_AD_K_NEIGHBORS = 10
NU_GRID = [0.01, 0.05, 0.10, 0.15, 0.20]
KDE_BANDWIDTH_GRID = np.logspace(-1, 1, 20)
EPS = 1e-12


def _as_numpy_1d(value: Any, name: str, expected_len: int | None = None) -> np.ndarray:
    """Convert an oracle metadata array to a one-dimensional numpy array."""

    arr = np.asarray(value, dtype=float).reshape(-1)
    if expected_len is not None and arr.size != expected_len:
        raise ValueError(
            f"Expected {name} to contain {expected_len} values, got {arr.size}."
        )
    return arr


def _load_oracle(oracle_path: str | os.PathLike[str]) -> dict:
    """Load the LCMGP oracle file without changing any model weights."""

    path = Path(oracle_path)
    if not path.exists():
        raise FileNotFoundError(f"Oracle file not found: {path}")
    save_obj = torch.load(path, weights_only=False, map_location="cpu")
    if not isinstance(save_obj, dict):
        raise TypeError(f"Expected oracle file {path} to contain a dict.")
    return save_obj


def _extract_train_x_std(save_obj: dict) -> np.ndarray:
    """Return the oracle's saved standardised training matrix as float numpy."""

    if "train_x" not in save_obj:
        raise KeyError("Oracle save object is missing required key 'train_x'.")
    train_x = save_obj["train_x"]
    if hasattr(train_x, "detach"):
        train_x = train_x.detach().cpu().numpy()
    X_train_std = np.asarray(train_x, dtype=float)
    if X_train_std.ndim != 2:
        raise ValueError(
            f"Expected train_x to be a 2-D matrix, got shape {X_train_std.shape}."
        )
    if X_train_std.shape[1] != len(FEATURES):
        raise ValueError(
            f"Expected train_x to have {len(FEATURES)} features, "
            f"got {X_train_std.shape[1]}."
        )
    finite_rows = np.isfinite(X_train_std).all(axis=1)
    if not np.all(finite_rows):
        dropped = int((~finite_rows).sum())
        print(f"Warning: dropping {dropped} non-finite rows from train_x.")
        X_train_std = X_train_std[finite_rows]
    if len(X_train_std) < 5:
        raise ValueError("Need at least 5 finite training rows for this filter.")
    return X_train_std.astype(float, copy=False)


def _validate_oracle_metadata(save_obj: dict) -> None:
    """Fail early if the oracle metadata no longer matches this feature contract."""

    saved_features = save_obj.get("features")
    if saved_features is not None and list(saved_features) != FEATURES:
        raise ValueError(
            "Oracle feature order does not match the expected LCMGP feature order."
        )

    saved_cont_idxs = save_obj.get("cont_idxs")
    if saved_cont_idxs is not None and list(saved_cont_idxs) != CONT_IDXS:
        raise ValueError("Oracle continuous indices do not match CONT_IDXS.")

    _as_numpy_1d(save_obj.get("x_cont_mean"), "x_cont_mean", len(CONT_IDXS))
    _as_numpy_1d(save_obj.get("x_cont_std"), "x_cont_std", len(CONT_IDXS))


def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid used to map OCSVM distances into [0, 1]."""

    z = np.asarray(x, dtype=float)
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _select_ocsvm_nu_cv(X_train_std: np.ndarray, seed: int) -> tuple[float, pd.DataFrame]:
    """Choose nu by 5-fold CV, treating validation training points as in-distribution."""

    n_splits = min(5, len(X_train_std))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows: list[dict[str, float]] = []

    for nu in NU_GRID:
        fold_scores = []
        fold_coverages = []
        for train_idx, val_idx in cv.split(X_train_std):
            model = OneClassSVM(kernel="rbf", nu=float(nu), gamma="scale")
            model.fit(X_train_std[train_idx])
            pred = model.predict(X_train_std[val_idx])
            y_true = np.ones(len(val_idx), dtype=int)
            y_pred = (pred == 1).astype(int)
            fold_scores.append(float(f1_score(y_true, y_pred, zero_division=0)))
            fold_coverages.append(float(np.mean(y_pred == 1)))
        rows.append(
            {
                "nu": float(nu),
                "mean_f1": float(np.mean(fold_scores)),
                "std_f1": float(np.std(fold_scores)),
                "mean_validation_coverage": float(np.mean(fold_coverages)),
            }
        )

    results = pd.DataFrame(rows)
    best_idx = int(results["mean_f1"].idxmax())
    return float(results.loc[best_idx, "nu"]), results


def _fit_ocsvm(
    X_train_std: np.ndarray, nu_arg: str, seed: int
) -> tuple[OneClassSVM, float, pd.DataFrame]:
    """Fit the final One-Class SVM on the full oracle training matrix."""

    if str(nu_arg).strip().lower() == "auto":
        try:
            nu_used, cv_results = _select_ocsvm_nu_cv(X_train_std, seed)
            print(f"Selected OCSVM nu={nu_used:.3g} by 5-fold CV.")
        except Exception as exc:  # pragma: no cover - defensive CLI fallback
            nu_used = 0.05
            cv_results = pd.DataFrame(
                [{"nu": nu_used, "mean_f1": np.nan, "std_f1": np.nan}]
            )
            print(
                "Warning: OCSVM nu CV failed; falling back to nu=0.05. "
                f"Reason: {exc}"
            )
    else:
        nu_used = float(nu_arg)
        if not (0.0 < nu_used <= 1.0):
            raise ValueError("--nu must be 'auto' or a float in the interval (0, 1].")
        cv_results = pd.DataFrame(
            [{"nu": nu_used, "mean_f1": np.nan, "std_f1": np.nan}]
        )
        print(f"Using user-specified OCSVM nu={nu_used:.3g}.")

    ocsvm = OneClassSVM(kernel="rbf", nu=nu_used, gamma="scale")
    ocsvm.fit(X_train_std)
    return ocsvm, nu_used, cv_results


def _fit_kde(X_train_std: np.ndarray) -> tuple[KernelDensity, float, pd.DataFrame]:
    """Fit KDE after selecting Gaussian bandwidth by held-out log-likelihood."""

    n_splits = min(5, len(X_train_std))
    search = GridSearchCV(
        KernelDensity(kernel="gaussian"),
        {"bandwidth": KDE_BANDWIDTH_GRID},
        cv=n_splits,
    )
    search.fit(X_train_std)

    bandwidth_used = float(search.best_params_["bandwidth"])
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth_used)
    kde.fit(X_train_std)

    cv_results = pd.DataFrame(
        {
            "bandwidth": search.cv_results_["param_bandwidth"].astype(float),
            "mean_test_log_likelihood": search.cv_results_["mean_test_score"],
            "std_test_log_likelihood": search.cv_results_["std_test_score"],
        }
    )
    print(f"Selected KDE bandwidth={bandwidth_used:.6g} by 5-fold CV.")
    return kde, bandwidth_used, cv_results


def _training_ad_ratio(save_obj: dict, X_train_std: np.ndarray) -> np.ndarray | None:
    """Recover existing AD ratios for training rows from the oracle metadata."""

    ad_reference = save_obj.get("ad_reference")
    if not isinstance(ad_reference, dict):
        return None

    tau_ad = float(ad_reference.get("tau_AD", np.nan))
    if not np.isfinite(tau_ad) or tau_ad <= 0:
        return None

    distance_keys = ("D_AD_train", "train_D_AD_mean", "train_D_AD", "D_AD")
    for key in distance_keys:
        if key in ad_reference:
            distances = np.asarray(ad_reference[key], dtype=float).reshape(-1)
            if distances.size == len(X_train_std):
                return distances / tau_ad

    if "train_x_std" in ad_reference:
        ref = np.asarray(ad_reference["train_x_std"], dtype=float)
        if ref.shape == X_train_std.shape:
            k = int(
                ad_reference.get(
                    "k",
                    ad_reference.get("ad_k_neighbors", DEFAULT_AD_K_NEIGHBORS),
                )
            )
            k_eff = max(1, min(k, len(ref) - 1))
            nn = NearestNeighbors(n_neighbors=k_eff + 1)
            nn.fit(ref)
            dists, _ = nn.kneighbors(X_train_std)
            return np.mean(dists[:, 1 : k_eff + 1], axis=1) / tau_ad

    return None


def _diagnostics_dataframe(
    train_scores: dict[str, np.ndarray],
    ad_ratio: np.ndarray | None,
    ocsvm_cv: pd.DataFrame,
    kde_cv: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """Build a compact long-form diagnostics table for printing and CSV output."""

    combined = np.asarray(train_scores["combined_score"], dtype=float)
    rows: list[dict[str, Any]] = []

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        rows.append(
            {
                "section": "threshold_search",
                "metric": "fraction_training_below_threshold",
                "threshold": threshold,
                "value": float(np.mean(combined < threshold)),
            }
        )

    coverage = float(np.mean(combined >= 0.5))
    rows.extend(
        [
            {
                "section": "training_coverage",
                "metric": "combined_score_mean",
                "threshold": np.nan,
                "value": float(np.mean(combined)),
            },
            {
                "section": "training_coverage",
                "metric": "combined_score_median",
                "threshold": np.nan,
                "value": float(np.median(combined)),
            },
            {
                "section": "training_coverage",
                "metric": "combined_score_p5",
                "threshold": np.nan,
                "value": float(np.percentile(combined, 5)),
            },
            {
                "section": "training_coverage",
                "metric": "combined_score_p95",
                "threshold": np.nan,
                "value": float(np.percentile(combined, 95)),
            },
            {
                "section": "self_consistency",
                "metric": "fraction_training_combined_score_ge_0.5",
                "threshold": 0.5,
                "value": coverage,
            },
        ]
    )

    spearman_corr = np.nan
    if ad_ratio is not None:
        ad_ratio = np.asarray(ad_ratio, dtype=float)
        mask = np.isfinite(ad_ratio) & np.isfinite(combined)
        if int(mask.sum()) >= 3:
            spearman_corr = float(spearmanr(1.0 - combined[mask], ad_ratio[mask]).statistic)

    rows.append(
        {
            "section": "existing_ad_comparison",
            "metric": "spearman_corr_1_minus_combined_score_vs_AD_ratio",
            "threshold": np.nan,
            "value": spearman_corr,
        }
    )

    for _, row in ocsvm_cv.iterrows():
        rows.append(
            {
                "section": "ocsvm_nu_cv",
                "metric": "mean_f1",
                "threshold": float(row["nu"]),
                "value": float(row["mean_f1"]),
            }
        )

    for _, row in kde_cv.iterrows():
        rows.append(
            {
                "section": "kde_bandwidth_cv",
                "metric": "mean_test_log_likelihood",
                "threshold": float(row["bandwidth"]),
                "value": float(row["mean_test_log_likelihood"]),
            }
        )

    return pd.DataFrame(rows), spearman_corr


def _print_diagnostics(
    train_scores: dict[str, np.ndarray],
    spearman_corr: float,
) -> None:
    """Print human-readable diagnostics for threshold selection."""

    combined = np.asarray(train_scores["combined_score"], dtype=float)
    print("\nTraining diagnostics")
    print("--------------------")
    print("Threshold search: fraction of training points below T")
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        frac = float(np.mean(combined < threshold))
        print(f"  T={threshold:.1f}: {frac:.4f}")

    print(
        "Combined score: "
        f"mean={np.mean(combined):.4f}, "
        f"median={np.median(combined):.4f}, "
        f"p5={np.percentile(combined, 5):.4f}, "
        f"p95={np.percentile(combined, 95):.4f}"
    )
    coverage = float(np.mean(combined >= 0.5))
    print(f"Self-consistency: fraction training score >= 0.5 is {coverage:.4f}")
    if np.isfinite(spearman_corr):
        print(
            "Existing AD comparison: "
            f"Spearman corr(1 - combined_score, AD_ratio) = {spearman_corr:.4f}"
        )
    else:
        print("Existing AD comparison: unavailable in this oracle.")


def train_filter(
    oracle_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    diagnostics_path: str | os.PathLike[str],
    nu_arg: str,
    seed: int,
) -> dict:
    """Run the full train, calibrate, diagnose, and serialise workflow."""

    print(f"Loading oracle: {oracle_path}")
    save_obj = _load_oracle(oracle_path)
    _validate_oracle_metadata(save_obj)
    X_train_std = _extract_train_x_std(save_obj)
    print(f"Recovered X_train_std with shape {X_train_std.shape}.")

    print("Training OCSVM...")
    ocsvm, nu_used, ocsvm_cv = _fit_ocsvm(X_train_std, nu_arg, seed)
    ocsvm_train_raw = ocsvm.decision_function(X_train_std).reshape(-1)
    ocsvm_scale = float(np.std(ocsvm_train_raw))
    if not np.isfinite(ocsvm_scale) or ocsvm_scale <= EPS:
        ocsvm_scale = 1.0
    print(f"Calibrated OCSVM scale={ocsvm_scale:.6g}.")

    print("Training KDE...")
    kde, bandwidth_used, kde_cv = _fit_kde(X_train_std)
    kde_train_log = kde.score_samples(X_train_std).reshape(-1)
    kde_log_max = float(np.max(kde_train_log))
    print(f"Calibrated KDE log-density max={kde_log_max:.6g}.")

    artefact = {
        "ocsvm": ocsvm,
        "kde": kde,
        "ocsvm_scale": float(ocsvm_scale),
        "kde_log_max": float(kde_log_max),
        "x_cont_mean": _as_numpy_1d(save_obj["x_cont_mean"], "x_cont_mean", 9),
        "x_cont_std": _as_numpy_1d(save_obj["x_cont_std"], "x_cont_std", 9),
        "features": FEATURES,
        "cont_idxs": CONT_IDXS,
        "transform_recipe": TRANSFORM_RECIPE,
        "nu_used": float(nu_used),
        "bandwidth_used": float(bandwidth_used),
        "n_train": int(len(X_train_std)),
        "script_version": "2.0_no_head",
    }

    train_scores = score_feasibility(X_train_std, artefact)
    ad_ratio = _training_ad_ratio(save_obj, X_train_std)
    diagnostics, spearman_corr = _diagnostics_dataframe(
        train_scores, ad_ratio, ocsvm_cv, kde_cv
    )
    _print_diagnostics(train_scores, spearman_corr)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefact, output_path)
    print(f"\nSaved artefact: {output_path}")

    diagnostics_path = Path(diagnostics_path)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(diagnostics_path, index=False)
    print(f"Saved diagnostics: {diagnostics_path}")

    return artefact


def load_feasibility_filter(artefact_path: str) -> dict:
    """Load and return the feasibility-filter artefact dict from a .pkl file.

    Raises
    ------
    FileNotFoundError
        If the artefact does not exist yet. Run this script once from the
        oracle directory to train and save ``ocsvm_kde_feasibility.pkl``.
    """

    path = Path(artefact_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Feasibility-filter artefact not found: {path}. "
            "Run `python ocsvm_kde_feasibility.py` from the oracle directory "
            "first to train and save it."
        )
    artefact = joblib.load(path)
    if not isinstance(artefact, dict):
        raise TypeError(f"Expected {path} to contain an artefact dict.")
    return artefact


def prepare_features_for_filter(df: pd.DataFrame, artefact: dict) -> np.ndarray:
    """Transform and standardise candidate mixes for the feasibility filter.

    The input DataFrame must contain the columns listed in
    ``artefact['features']``. All columns are numeric and are transformed with
    the same log/log1p/raw recipe used by the LCMGP oracle, then z-scored using
    the oracle's saved ``x_cont_mean`` and ``x_cont_std``.

    Rows that become NaN or non-finite after transformation are preserved in
    the returned float array so the caller can decide how to handle them.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    features = list(artefact.get("features", FEATURES))
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise KeyError(f"Candidate DataFrame is missing required columns: {missing}")

    cont_idxs = list(artefact.get("cont_idxs", CONT_IDXS))
    recipe = artefact.get("transform_recipe", TRANSFORM_RECIPE)

    X = np.empty((len(df), len(features)), dtype=float)

    for idx, col in enumerate(features):
        X[:, idx] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

    feature_to_idx = {name: idx for idx, name in enumerate(features)}
    with np.errstate(divide="ignore", invalid="ignore"):
        for col in recipe.get("log_cols", []):
            X[:, feature_to_idx[col]] = np.log(X[:, feature_to_idx[col]])
        for col in recipe.get("log1p_cols", []):
            X[:, feature_to_idx[col]] = np.log1p(X[:, feature_to_idx[col]])

    x_cont_mean = _as_numpy_1d(artefact["x_cont_mean"], "x_cont_mean", len(cont_idxs))
    x_cont_std = _as_numpy_1d(artefact["x_cont_std"], "x_cont_std", len(cont_idxs))
    x_cont_std = np.where(x_cont_std == 0.0, 1.0, x_cont_std)
    X[:, cont_idxs] = (X[:, cont_idxs] - x_cont_mean) / x_cont_std
    return X.astype(float, copy=False)


def score_feasibility(Xs: np.ndarray, artefact: dict) -> dict:
    """Score a batch of standardised candidates.

    Parameters
    ----------
    Xs
        Standardised candidate matrix with shape ``(N, len(features))``.
    artefact
        Dict produced by this script and loaded with
        ``load_feasibility_filter``.

    Returns
    -------
    dict
        Keys are ``combined_score``, ``ocsvm_score``, ``kde_score``,
        ``ocsvm_decision``, and ``kde_log_density``. Higher
        ``combined_score`` means the candidate is more like the oracle's
        training data and therefore more feasible from a density perspective.

    Notes
    -----
    Scoring is batched internally so large composition grids do not need to be
    materialised inside sklearn in one call. Rows with non-finite inputs are
    returned as NaN scores instead of raising a sklearn error.
    """

    Xs = np.asarray(Xs, dtype=float)
    if Xs.ndim == 1:
        Xs = Xs.reshape(1, -1)
    if Xs.ndim != 2:
        raise ValueError(f"Xs must be a 2-D matrix, got shape {Xs.shape}.")
    expected_features = len(artefact.get("features", FEATURES))
    if Xs.shape[1] != expected_features:
        raise ValueError(
            f"Xs must have {expected_features} columns, got {Xs.shape[1]}."
        )

    n_rows = Xs.shape[0]
    ocsvm_decision = np.full(n_rows, np.nan, dtype=float)
    kde_log_density = np.full(n_rows, np.nan, dtype=float)

    finite_mask = np.isfinite(Xs).all(axis=1)
    finite_indices = np.flatnonzero(finite_mask)
    batch_size = int(artefact.get("score_batch_size", DEFAULT_SCORE_BATCH_SIZE))
    batch_size = max(1, batch_size)

    ocsvm = artefact["ocsvm"]
    kde = artefact["kde"]

    for start in range(0, len(finite_indices), batch_size):
        batch_idx = finite_indices[start : start + batch_size]
        X_batch = Xs[batch_idx]
        ocsvm_decision[batch_idx] = ocsvm.decision_function(X_batch).reshape(-1)
        kde_log_density[batch_idx] = kde.score_samples(X_batch).reshape(-1)

    ocsvm_scale = float(artefact.get("ocsvm_scale", 1.0))
    if not np.isfinite(ocsvm_scale) or ocsvm_scale <= EPS:
        ocsvm_scale = 1.0

    kde_log_max = float(artefact["kde_log_max"])

    ocsvm_score = np.full(n_rows, np.nan, dtype=float)
    kde_score = np.full(n_rows, np.nan, dtype=float)
    combined_score = np.full(n_rows, np.nan, dtype=float)

    if finite_indices.size:
        ocsvm_score[finite_indices] = _sigmoid_stable(
            ocsvm_decision[finite_indices] / ocsvm_scale
        )
        log_ratio = kde_log_density[finite_indices] - kde_log_max
        log_ratio = np.minimum(log_ratio, 0.0)
        log_ratio = np.maximum(log_ratio, -745.0)
        kde_score[finite_indices] = np.clip(np.exp(log_ratio), 0.0, 1.0)
        combined_score[finite_indices] = np.clip(
            0.5 * ocsvm_score[finite_indices] + 0.5 * kde_score[finite_indices],
            0.0,
            1.0,
        )

    return {
        "combined_score": combined_score,
        "ocsvm_score": ocsvm_score,
        "kde_score": kde_score,
        "ocsvm_decision": ocsvm_decision,
        "kde_log_density": kde_log_density,
    }


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one-shot training."""

    parser = argparse.ArgumentParser(
        description="Train an OCSVM + KDE soft feasibility filter from an LCMGP oracle."
    )
    parser.add_argument(
        "--oracle",
        default=DEFAULT_ORACLE,
        help=f"Path to oracle .pt file (default: {DEFAULT_ORACLE}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output joblib artefact path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--nu",
        default="auto",
        help="OCSVM nu: 'auto' for 5-fold CV, or a float in (0, 1].",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for CV fold shuffling (default: 42).",
    )
    parser.add_argument(
        "--diagnostics",
        default=DEFAULT_DIAGNOSTICS,
        help=f"Diagnostics CSV path (default: {DEFAULT_DIAGNOSTICS}).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = _parse_args()
    train_filter(
        oracle_path=args.oracle,
        output_path=args.output,
        diagnostics_path=args.diagnostics,
        nu_arg=args.nu,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
