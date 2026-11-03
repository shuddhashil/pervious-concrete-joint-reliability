"""Data-informed porosity evidence and post-run robustness analyses.

This module is deliberately separate from the search solvers.  It can be
called after a launcher run, or imported by a launcher integration, without
changing the frozen model or the Phase A--D implementations.  The only
porosity data used for the evidence analysis are measured hardened porosity
and repeated/near-repeated rows in ``training_data_reconciled.csv``.

The sensitivity and window routines use common random numbers: batching and
paired residual draws are held fixed while the porosity SD, bias, or nominal
porosity is changed.  ``r_lcb95`` is a Wilson lower bound when
``n_bootstrap=0`` and the required minimum of Wilson and study-cluster lower
bound when a positive bootstrap count is requested.  The method is recorded
in every output row so a reduced smoke analysis cannot be mistaken for final
verification.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent
DEFAULT_DATA = TEMPLATE / "tables" / "training_data_reconciled.csv"
DEFAULT_BANK = TEMPLATE / "tables" / "paired_residual_bank.csv"
DEFAULT_POROSITY_CONTRACT = TEMPLATE / "POROSITY_CONTRACT.json"

RECIPE_COLUMNS = [
    "cem (kg/m3)", "water (kg/m3)", "CA (kg/m3)", "sand (kg/m3)", "SP (kg/m3)",
    "R25-19mm", "R19-12.5mm", "R12.5-9.5mm", "R9.5-6.3mm",
    "R6.3-4.75mm", "R4.75-2.36mm",
]
GRADATION_COLUMNS = RECIPE_COLUMNS[5:]


def load_training_data(path: os.PathLike | str = DEFAULT_DATA) -> pd.DataFrame:
    """Read the reconciled CSV and make the study label explicit per row."""
    df = pd.read_csv(path).copy()
    if "study_id" not in df:
        source = "Unnamed: 0"
        if source not in df:
            raise ValueError("training data must contain Unnamed: 0 study labels")
        df["study_id"] = df[source].ffill().astype(str)
    else:
        df["study_id"] = df["study_id"].ffill().astype(str)
    missing = [c for c in RECIPE_COLUMNS + ["porosity"] if c not in df]
    if missing:
        raise ValueError(f"training data missing required columns: {missing}")
    return df


def _quantized(value: pd.Series, tolerance: float) -> pd.Series:
    if tolerance <= 0:
        return value.round(12)
    return np.rint(value.astype(float) / tolerance).astype(np.int64)


def identify_replicate_groups(
    df: pd.DataFrame,
    mass_tolerance: float = 1e-3,
    gradation_tolerance: float = 1e-3,
) -> pd.DataFrame:
    """Return same-study near-recipe groups with at least two rows.

    The default tolerances are absolute: 0.001 kg/m3 for masses and 0.001
    percentage points for each reported gradation fraction.  These tolerances
    are stated in the generated provenance report and are not fitted to
    porosity spread.
    """
    work = df.copy()
    keys = []
    for col in RECIPE_COLUMNS:
        tol = mass_tolerance if col not in GRADATION_COLUMNS else gradation_tolerance
        keys.append(_quantized(work[col].fillna(0.0), tol).astype(str).rename(col))
    key_frame = pd.concat([work["study_id"].astype(str).rename("study_id")] + keys, axis=1)
    group_key = key_frame.astype(str).agg("|".join, axis=1)
    work["replicate_group"] = group_key
    counts = work.groupby(["study_id", "replicate_group"], sort=True).size().rename("n_rows")
    counts = counts[counts >= 2].reset_index()
    if counts.empty:
        return pd.DataFrame(columns=["study_id", "replicate_group", "n_rows", "porosity_sd_pp", "porosity_median_pp", "porosity_mad_sd_pp"])
    rows = []
    for rec in counts.to_dict("records"):
        mask = (work["study_id"] == rec["study_id"]) & (work["replicate_group"] == rec["replicate_group"])
        values = work.loc[mask, "porosity"].astype(float).to_numpy()
        median = float(np.median(values))
        mad_sd = float(1.4826 * np.median(np.abs(values - median))) if len(values) else float("nan")
        rows.append({**rec, "porosity_sd_pp": float(np.std(values, ddof=1)),
                     "porosity_median_pp": median, "porosity_mad_sd_pp": mad_sd,
                     "porosity_min_pp": float(np.min(values)), "porosity_max_pp": float(np.max(values))})
    return pd.DataFrame(rows)


def _pooled_sd(groups: pd.DataFrame) -> float:
    valid = groups[groups["n_rows"] > 1]
    dof = np.maximum(valid["n_rows"].to_numpy(float) - 1.0, 0.0)
    den = float(dof.sum())
    return float(np.sqrt(np.sum(dof * valid["porosity_sd_pp"].to_numpy(float) ** 2) / den)) if den else float("nan")


def replicate_scatter_evidence(
    data_path: os.PathLike | str = DEFAULT_DATA,
    seed: int = 20260920,
    n_bootstrap: int = 5000,
    mass_tolerance: float = 1e-3,
    gradation_tolerance: float = 1e-3,
) -> tuple[pd.DataFrame, dict]:
    """Compute group-level replicate scatter and study-cluster bootstrap CIs."""
    groups = identify_replicate_groups(load_training_data(data_path), mass_tolerance, gradation_tolerance)
    if groups.empty:
        return groups, {"n_usable_groups": 0, "n_studies": 0, "pooled_sd_pp": float("nan")}
    rng = np.random.default_rng(seed)
    group_rows = groups.to_dict("records")
    boot_sd, boot_mad = [], []
    for _ in range(int(n_bootstrap)):
        sample = groups.iloc[rng.integers(0, len(groups), len(groups))]
        boot_sd.append(_pooled_sd(sample))
        boot_mad.append(float(np.nanmedian(sample["porosity_mad_sd_pp"])))
    q = lambda a, p: float(np.percentile(np.asarray(a, float), p)) if len(a) else float("nan")
    summary = {
        "n_training_rows": int(len(load_training_data(data_path))),
        "n_usable_groups": int(len(groups)),
        "n_studies_with_usable_groups": int(groups["study_id"].nunique()),
        "n_rows_in_usable_groups": int(groups["n_rows"].sum()),
        "mass_tolerance_kg_m3": float(mass_tolerance),
        "gradation_tolerance_fraction_pp": float(gradation_tolerance),
        "pooled_within_group_sd_pp": _pooled_sd(groups),
        "median_group_sd_pp": float(groups["porosity_sd_pp"].median()),
        "median_group_mad_sd_pp": float(groups["porosity_mad_sd_pp"].median()),
        "pooled_sd_bootstrap_ci95_low": q(boot_sd, 2.5),
        "pooled_sd_bootstrap_ci95_high": q(boot_sd, 97.5),
        "median_mad_bootstrap_ci95_low": q(boot_mad, 2.5),
        "median_mad_bootstrap_ci95_high": q(boot_mad, 97.5),
        "bootstrap_reps": int(n_bootstrap),
        "bootstrap_unit": "near-identical recipe groups (not rows)",
        "interpretation": "supporting evidence only; not a project-specific batch-to-batch SD",
    }
    return groups, summary


def _candidate_value(row: pd.Series, *names, default=None):
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default


def candidate_from_row(row: pd.Series | dict) -> dict:
    """Normalize common launcher CSV schemas to the evaluator design dict."""
    row = pd.Series(row)
    g = [_candidate_value(row, f"g{i}", f"g_{i}") for i in range(1, 7)]
    if any(v is None for v in g):
        raise ValueError("candidate row must contain g1..g6")
    g = np.asarray(g, float)
    if not np.isfinite(g).all() or g.sum() <= 0:
        raise ValueError("candidate gradation fractions must be finite and have positive sum")
    g /= g.sum()
    return {
        "C": float(_candidate_value(row, "C", "cement_level_kg_m3")),
        "r": float(_candidate_value(row, "r", "wc", "w/c")),
        "A": float(_candidate_value(row, "A", "CA", "CA_kg_m3")),
        "P": float(_candidate_value(row, "P", "SP", "SP_kg_m3")),
        "S": float(_candidate_value(row, "S", "sand", "sand_kg_m3")),
        "phi": float(_candidate_value(row, "phi", "porosity_nominal_pct", "porosity")),
        "g": g,
    }


def load_candidate(path: os.PathLike | str, prefer_selected: bool = True) -> dict:
    df = pd.read_csv(path)
    if prefer_selected and "selected" in df:
        chosen = df[df["selected"].astype(str).str.lower().isin(["true", "1", "yes"])]
        if not chosen.empty:
            df = chosen
    if df.empty:
        raise ValueError(f"no candidate rows in {path}")
    return candidate_from_row(df.iloc[0])


def load_porosity_contract(path: os.PathLike | str = DEFAULT_POROSITY_CONTRACT) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def porosity_contract_value(name: str, default=None):
    """Read a post-run porosity setting without duplicating it in analysis."""
    try:
        return load_porosity_contract().get(name, default)
    except FileNotFoundError:
        return default


def _truncated_normal_from_uniform(u, mean, sd, lo, hi):
    """Exact inverse-CDF truncated normal transform used for CRN draws."""
    from scipy.special import ndtr, ndtri
    if sd <= 0:
        return np.full_like(np.asarray(u, float), float(mean))
    a = float(ndtr((lo - mean) / sd))
    b = float(ndtr((hi - mean) / sd))
    return mean + sd * ndtri(a + np.asarray(u, float) * (b - a))


def build_common_random_banks(
    specs: list[tuple[float, float]],
    n_scenarios: int,
    seed: int,
    contract_path: os.PathLike | str | None = None,
):
    """Return one manufacturing bank per (porosity_sd_pp, bias_pp) pair.

    All non-porosity draws and residual rows are shared.  Porosity draws use
    shared uniforms and an exact truncated-normal inverse CDF, avoiding the
    rejection-sampler stream drift that would otherwise weaken CRN.

    The contract must be the UNCERTAINTY contract (it carries the
    manufacturing magnitudes consumed by UncertaintySpec); the previous
    default pointed at the porosity-semantics contract, which has no
    ``parameters`` block and raised KeyError.
    """
    sys.path.insert(0, str(HERE))
    import uncertainty as U
    contract = str(contract_path) if contract_path else str(TEMPLATE / "UNCERTAINTY_CONTRACT.json")
    base_spec = U.UncertaintySpec(contract, porosity_sd_override_pp=float(specs[0][0]), porosity_bias_override_pp=float(specs[0][1]))
    base = U.build_manufacturing_bank(base_spec, int(n_scenarios), int(seed), common_random=True)
    rng = np.random.default_rng(int(seed) + 7919)
    uniform = rng.random(int(n_scenarios))
    out = {}
    for sd, bias in specs:
        spec = U.UncertaintySpec(contract, porosity_sd_override_pp=float(sd), porosity_bias_override_pp=float(bias))
        lo, hi = spec.porosity_bounds_pp
        eps = _truncated_normal_from_uniform(uniform, float(bias), float(sd), lo, hi)
        out[(float(sd), float(bias))] = replace(base, eps_phi_pp=eps, spec_hash=spec.spec_hash(), stress=False)
    return out


def _scenario_arrays(nominal, mfg_bank):
    """Obtain effective means/SDs/support for one manufacturing bank."""
    sys.path.insert(0, str(HERE))
    import robust_evaluator as RE
    import fast_engine as FE
    import stage1_evaluator as E
    import uncertainty as U
    E.get_bundle()
    mfg = U.apply_manufacturing_scenarios(nominal, mfg_bank)
    eng = FE.get_engine(E)
    if eng is not None:
        mu_s, mu_q, s_s, s_q, support = eng.scenario_arrays(mfg)
    else:
        frame = RE._candidates_frame_from_scenarios(mfg)
        pred = E.predict_fast(E.ORACLE_PT, frame)
        mu_s = pred["strength_pred_mpa"].to_numpy(float)
        mu_q = pred["logperm_pred"].to_numpy(float)
        s_s = pred["strength_std_effective_mpa"].to_numpy(float)
        s_q = pred["logperm_std_effective"].to_numpy(float)
        support = RE._support_scores_for_frame(frame)
    return mfg, mu_s, mu_q, s_s, s_q, support


def evaluate_design(
    nominal: dict,
    mfg_bank,
    residual_bank=None,
    residual_bank_path: os.PathLike | str = DEFAULT_BANK,
    *,
    s_star: float,
    k_star: float,
    target: float,
    n_bootstrap: int = 0,
    seed: int = 20260920,
) -> dict:
    """Evaluate joint reliability and decomposition for one nominal design.

    The strength target, permeability target and required reliability are
    mandatory keyword arguments: there are deliberately no built-in values, so an
    analysis can never silently use a target other than the run's own.
    """
    sys.path.insert(0, str(HERE))
    import cluster_bootstrap as CB
    import robust_evaluator as RE
    import stage1_evaluator as E
    import uncertainty as U
    from dataclasses import asdict

    if residual_bank is None:
        residual_bank = U.build_residual_bank(str(residual_bank_path), mfg_bank.n, int(seed) + 1)
    mfg, mu_s, mu_q, s_s, s_q, support = _scenario_arrays(nominal, mfg_bank)
    logk_star = math.log(float(k_star))
    support_threshold = float(porosity_contract_value("support_model_threshold", E.SUPPORT_THRESHOLD))
    in_support = support >= support_threshold
    s_real = mu_s + s_s * residual_bank.e_strength
    q_real = mu_q + s_q * residual_bank.e_logk
    strength_ok = s_real >= float(s_star)
    perm_ok = q_real >= logk_star
    joint = in_support & strength_ok & perm_ok
    n = len(joint)
    k = int(joint.sum())
    r_hat = k / n
    wilson = RE.wilson_lower_bound(k, n)
    cluster_lb = float("nan")
    method = "wilson_only"
    if int(n_bootstrap) > 0:
        df = pd.read_csv(residual_bank_path)
        cb = CB.run_cluster_bootstrap(mu_s, mu_q, s_s, s_q, support, support_threshold, df,
                                      float(s_star), logk_star, int(n_bootstrap), int(seed) + 2)
        cluster_lb = float(cb["R_LCB95_cluster"])
        method = "min_wilson_and_study_cluster"
    r_lcb = float(min(wilson, cluster_lb)) if np.isfinite(cluster_lb) else float(wilson)
    return {
        "cement_kg_m3": float(nominal["C"]), "phi_nominal_pp": float(nominal["phi"]),
        "n_scenarios": int(n), "n_bootstrap": int(n_bootstrap), "target_joint_reliability": float(target),
        "R_hat_joint": float(r_hat), "R_LCB95_joint": r_lcb, "wilson_LCB95": float(wilson),
        "cluster_LCB95": cluster_lb, "R_LCB95_method": method,
        "strength_ok_probability": float((in_support & strength_ok).mean()),
        "permeability_ok_probability": float((in_support & perm_ok).mean()),
        "product_of_marginals": float((in_support & strength_ok).mean() * (in_support & perm_ok).mean()),
        "joint_probability": float(r_hat),
        "strength_only_failure": float((in_support & ~strength_ok & perm_ok).mean()),
        "permeability_only_failure": float((in_support & strength_ok & ~perm_ok).mean()),
        "both_response_failure": float((in_support & ~strength_ok & ~perm_ok).mean()),
        "support_failure": float((~in_support).mean()),
        "phi_actual_p05": float(np.percentile(mfg["phi_actual"], 5)),
        "phi_actual_p50": float(np.percentile(mfg["phi_actual"], 50)),
        "phi_actual_p95": float(np.percentile(mfg["phi_actual"], 95)),
        "nominal_design": nominal,
        "mfg_bank_hash": mfg_bank.bank_hash(), "residual_bank_hash": residual_bank.bank_hash(),
    }


def porosity_sensitivity(
    designs: dict[str, dict],
    sd_values=(0.5, 1.0, 1.5, 2.0),
    bias_values=(-1.0, 0.0, 1.0),
    n_scenarios: int = 20000,
    n_bootstrap: int = 0,
    seed: int = 20260920,
    *,
    s_star: float,
    k_star: float,
    target: float,
    output_csv: os.PathLike | str | None = None,
) -> pd.DataFrame:
    """Evaluate SD/bias sensitivity with shared manufacturing/residual draws."""
    specs = [(float(sd), float(bias)) for sd in sd_values for bias in bias_values]
    banks = build_common_random_banks(specs, int(n_scenarios), int(seed))
    rows = []
    for design_id, design in designs.items():
        residual = None
        # One residual bank per design is shared across every SD/bias cell.
        sys.path.insert(0, str(HERE))
        import uncertainty as U
        residual = U.build_residual_bank(str(DEFAULT_BANK), int(n_scenarios), int(seed) + 101)
        for sd, bias in specs:
            rec = evaluate_design(design, banks[(sd, bias)], residual, s_star=s_star, k_star=k_star,
                                  target=target, n_bootstrap=n_bootstrap, seed=seed)
            rows.append({"design_id": design_id, "porosity_sd_pp": sd, "porosity_bias_pp": bias,
                         "target_strength_mpa": float(s_star), "target_permeability_mm_s": float(k_star),
                         "target_joint_reliability": float(target),
                         **{k: v for k, v in rec.items() if k not in ("nominal_design",)}})
    result = pd.DataFrame(rows)
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
    return result


def _largest_contiguous_window(grid, records, *, target, design_phi=None):
    good = np.array([float(r["R_LCB95_joint"]) >= float(target) for r in records], bool)
    if not good.any():
        return {"phi_low": float("nan"), "phi_high": float("nan"), "window_width_pp": 0.0,
                "binding_low": "permeability", "binding_high": "strength", "n_grid_good": 0}
    idx0 = int(np.argmin(np.abs(np.asarray(grid) - float(design_phi)))) if design_phi is not None else int(np.flatnonzero(good)[0])
    if not good[idx0]:
        # Use the nearest good point only as a diagnostic; no feasible window
        # around an infeasible design is claimed.
        return {"phi_low": float("nan"), "phi_high": float("nan"), "window_width_pp": 0.0,
                "binding_low": "permeability", "binding_high": "strength", "n_grid_good": int(good.sum())}
    lo = hi = idx0
    while lo > 0 and good[lo - 1]: lo -= 1
    while hi + 1 < len(good) and good[hi + 1]: hi += 1
    low = records[lo]; high = records[hi]
    bind_low = "permeability" if low["permeability_only_failure"] >= low["strength_only_failure"] else "strength"
    bind_high = "strength" if high["strength_only_failure"] >= high["permeability_only_failure"] else "permeability"
    return {"phi_low": float(grid[lo]), "phi_high": float(grid[hi]),
            "window_width_pp": float(grid[hi] - grid[lo]), "binding_low": bind_low,
            "binding_high": bind_high, "n_grid_good": int(good.sum())}


def porosity_window_sweep(
    design: dict,
    phi_grid=None,
    n_scenarios: int = 20000,
    n_bootstrap: int = 0,
    seed: int = 20260920,
    *,
    s_star: float,
    k_star: float,
    target: float,
    output_csv: os.PathLike | str | None = None,
) -> pd.DataFrame:
    """Sweep nominal porosity while holding all other recipe inputs fixed."""
    if phi_grid is None:
        lo, hi = porosity_contract_value("deployment_domain_pp", [20.0, 30.0])
        step = float(porosity_contract_value("porosity_sweep_step_pp", 0.25))
        phi_grid = np.arange(float(lo), float(hi) + step * 0.1, step)
    phi_grid = np.asarray(phi_grid, float)
    banks = build_common_random_banks([(1.0, 0.0)], int(n_scenarios), int(seed))
    bank = banks[(1.0, 0.0)]
    sys.path.insert(0, str(HERE))
    import uncertainty as U
    residual = U.build_residual_bank(str(DEFAULT_BANK), int(n_scenarios), int(seed) + 101)
    rows = []
    for phi in phi_grid:
        cand = copy.deepcopy(design); cand["phi"] = float(phi)
        rec = evaluate_design(cand, bank, residual, s_star=s_star, k_star=k_star, target=target,
                              n_bootstrap=n_bootstrap, seed=seed)
        rec.pop("nominal_design", None)
        rec["phi_nominal_pp"] = float(phi)
        rec["target_joint_reliability"] = float(target)
        rows.append(rec)
    result = pd.DataFrame(rows)
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)
    return result


def evaluate_robust_tolerance(
    design: dict,
    delta_pp: float,
    n_scenarios: int = 20000,
    n_bootstrap: int = 0,
    seed: int = 20260920,
    *,
    s_star: float,
    k_star: float,
    target: float,
) -> dict:
    """Option-B evaluator: require reliability at phi +/- delta and centre."""
    d = abs(float(delta_pp))
    specs = [(1.0, 0.0)]
    bank = build_common_random_banks(specs, int(n_scenarios), int(seed))[(1.0, 0.0)]
    sys.path.insert(0, str(HERE))
    import uncertainty as U
    residual = U.build_residual_bank(str(DEFAULT_BANK), int(n_scenarios), int(seed) + 101)
    offsets = [-d, 0.0, d] if d else [0.0]
    evaluations = []
    for offset in offsets:
        cand = copy.deepcopy(design); cand["phi"] = float(design["phi"] + offset)
        evaluations.append(evaluate_design(cand, bank, residual, s_star=s_star, k_star=k_star, target=target,
                                           n_bootstrap=n_bootstrap, seed=seed))
    worst = min(float(e["R_LCB95_joint"]) for e in evaluations)
    out = {"delta_pp": d, "minimum_cement_kg_m3": float(design["C"]),
           "phi_nominal_pp": float(design["phi"]), "worst_R_LCB95": worst,
           "target_joint_reliability": float(target), "robust_pass": bool(worst >= target),
           "offset_evaluations": evaluations, "r_lcb95_method": evaluations[0]["R_LCB95_method"]}
    for off, ev in zip(offsets, evaluations):
        tag = "center" if off == 0 else ("low" if off < 0 else "high")
        out[f"R_LCB95_{tag}"] = float(ev["R_LCB95_joint"])
        out[f"R_hat_{tag}"] = float(ev["R_hat_joint"])
    return out


def robust_tolerance_constraint(design: dict, delta_pp: float, **kwargs) -> float:
    """NOMAD-compatible signed constraint: non-negative means feasible."""
    return float(evaluate_robust_tolerance(design, delta_pp, **kwargs)["worst_R_LCB95"] - kwargs["target"])


def _figure_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 10, "figure.dpi": 120})
    return plt


def save_sensitivity_heatmap(df: pd.DataFrame, path_stem: os.PathLike | str):
    plt = _figure_style(); import matplotlib.pyplot as _plt
    pivot = df[df["design_id"] == df["design_id"].iloc[0]].pivot(index="porosity_sd_pp", columns="porosity_bias_pp", values="R_LCB95_joint")
    fig, ax = plt.subplots(figsize=(5.2, 3.8)); im = ax.imshow(pivot.to_numpy(float), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), [f"{x:g}" for x in pivot.columns]); ax.set_yticks(range(len(pivot.index)), [f"{x:g}" for x in pivot.index])
    ax.set_xlabel("Porosity bias (percentage points)"); ax.set_ylabel("Porosity scatter SD (percentage points)"); ax.set_title("Joint reliability lower bound")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]): ax.text(j, i, f"{pivot.iloc[i,j]:.3f}", ha="center", va="center", color="white" if pivot.iloc[i,j] < .55 else "black")
    fig.colorbar(im, ax=ax, label="R_LCB95"); fig.tight_layout(); _save_figure(fig, path_stem); return fig


def save_tradeoff_figures(tradeoff: pd.DataFrame, sweep_by_delta: dict, path_stem_a, path_stem_b, *, target: float):
    plt = _figure_style(); import matplotlib.pyplot as _plt
    fig, ax = plt.subplots(figsize=(5.2, 3.8)); ax.plot(tradeoff["delta_pp"], tradeoff["minimum_cement_kg_m3"], "o-", color="#0072B2")
    ax.set_xlabel("Guaranteed porosity tolerance, delta (percentage points)"); ax.set_ylabel("Minimum cement (kg/m3)"); ax.set_title("Cost of porosity robustness"); ax.grid(alpha=.25); fig.tight_layout(); _save_figure(fig, path_stem_a)
    fig2, ax2 = plt.subplots(figsize=(6.0, 4.0)); colors = ["#0072B2", "#D55E00", "#009E73"]
    for (delta, sweep), color in zip(sorted(sweep_by_delta.items()), colors):
        ax2.plot(sweep["phi_nominal_pp"], sweep["R_LCB95_joint"], label=f"delta={delta:g} pp", color=color)
        good = sweep["R_LCB95_joint"] >= sweep["target_joint_reliability"].iloc[0]
        if bool(good.any()): ax2.fill_between(sweep["phi_nominal_pp"], 0, 1, where=good, color=color, alpha=.08)
    ax2.axhline(float(target), color="black", ls="--", lw=1, label=f"target {float(target):.2f}"); ax2.set_ylim(0, 1.02); ax2.set_xlabel("Nominal porosity (percent)"); ax2.set_ylabel("Joint reliability lower bound"); ax2.set_title("Porosity window under fixed recipe inputs"); ax2.legend(frameon=False); ax2.grid(alpha=.25); fig2.tight_layout(); _save_figure(fig2, path_stem_b); return fig, fig2


def _save_figure(fig, path_stem):
    stem = Path(path_stem); stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight"); fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight"); fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replicates", action="store_true")
    ap.add_argument("--data", default=str(DEFAULT_DATA)); ap.add_argument("--out-dir", default=str(TEMPLATE / "tables"))
    ap.add_argument("--bootstrap", type=int, default=5000)
    args = ap.parse_args(argv)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    groups, summary = replicate_scatter_evidence(args.data, n_bootstrap=args.bootstrap)
    groups.to_csv(out / "POROSITY_REPLICATE_GROUPS.csv", index=False)
    pd.DataFrame([summary]).to_csv(out / "POROSITY_REPLICATE_SCATTER.csv", index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
