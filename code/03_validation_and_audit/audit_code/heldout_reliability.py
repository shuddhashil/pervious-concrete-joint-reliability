"""Held-out calibration audit of the joint-pass probability used for acceptance.

What this checks
----------------
The inverse design accepts a recipe when the estimated probability that strength
AND permeability both reach their targets is high. That probability is obtained
by combining the candidate's predicted mean and effective SD with a bank of
paired, standardised out-of-sample residuals. This script tests, on held-out
rows, whether probabilities computed *the same way* are calibrated:

  * cross-fitting: for each of the five stored folds the residual pairs come only
    from the other folds, with every copy of the held-out physical rows removed;
  * for each held-out row and each (strength, permeability) target, the predicted
    joint-pass probability P is compared with what the row actually did (O);
  * results are reported as a reliability table, Brier score, logistic calibration
    slope, and the false-accept rate in the accept region (P >= 0.90), with
    confidence intervals that resample whole source studies.

It also (a) reproduces the stored cross-fitted coverage of the max-score screen
as an implementation check, (b) varies the hand-set uncertainty constants one at a
time, and (c) tests directly whether the applicability-domain distance of a
held-out row includes the row itself.

Design targets are never hardcoded. The target grid comes from quantiles of the
observed data; if --run-dir is given, that run's own TARGET_CONTRACT.json target
is added as an extra target.

Nothing in the frozen model, residual bank or search code is modified.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
# A3-1 audit copy: reads the deployed A3-1 base (631 records, 627-entry bank); nothing there is modified
MODELS = PKG / "GuidedEngine_20261003" / "base" / "core" / "models"

FEATURES = ["porosity", "w_per_c", "cem_kg_per_m3", "CA_kg_per_m3", "SP_kg_per_m3", "sand_kg_per_m3", "q10", "q25", "q50", "q75", "q90"]
XCOLS = [f"xstd_{f}" for f in FEATURES]
LEVELS = (0.80, 0.90, 0.95)
ACCEPT_P = 0.90


@dataclass(frozen=True)
class Config:
    """Uncertainty constants under test. Defaults equal the frozen production values."""
    s0: float = 0.75          # applicability-domain inflation starts above this ratio
    eta: float = 1.0          # inflation slope
    cap: float = 3.0          # inflation ceiling
    k_local: int = 10         # neighbours for the local residual SD
    rule: str = "max"         # how GP SD and local SD are combined
    exclude_self_in_ad: bool = False  # False reproduces the stored production scales


@dataclass
class Data:
    split: np.ndarray
    orig: np.ndarray
    study: np.ndarray
    X: np.ndarray
    yS: np.ndarray
    muS: np.ndarray
    rawS: np.ndarray
    yQ: np.ndarray
    muQ: np.ndarray
    rawQ: np.ndarray
    train_x: np.ndarray
    tau_ad: float
    k_ad: int
    prod: pd.DataFrame


def load_data(models: Path = MODELS) -> Data:
    import torch  # bundled runtime provides it; only needed to read the frozen oracle

    ref = pd.read_csv(models / "oos_residual_reference_M1global.csv")
    prod = pd.read_csv(models / "oos_production_scales_M1global.csv")
    key = ref[["split", "original_row_index"]].merge(
        prod.rename(columns={"row": "original_row_index"}), on=["split", "original_row_index"], how="left")
    if key["study"].isna().any():
        raise RuntimeError("some residual rows have no study label")
    oracle = torch.load(models / "lcmgp_ca_oracle_full_data.pt", map_location="cpu", weights_only=False)
    adr = oracle["ad_reference"]
    return Data(
        split=ref["split"].to_numpy(int), orig=ref["original_row_index"].to_numpy(int),
        study=key["study"].to_numpy(), X=ref[XCOLS].to_numpy(float),
        yS=ref["strength_true_mpa"].to_numpy(float), muS=ref["strength_pred_mpa"].to_numpy(float),
        rawS=ref["strength_std_oos_raw_mpa"].to_numpy(float),
        yQ=ref["logperm_true"].to_numpy(float), muQ=ref["logperm_pred"].to_numpy(float),
        rawQ=ref["logperm_std_oos_raw"].to_numpy(float),
        train_x=oracle["train_x"].detach().cpu().numpy().astype(float),
        tau_ad=float(adr["tau_AD"]), k_ad=int(adr["ad_k_neighbors"]), prod=key)


def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))


def ad_self_distance_test(d: Data) -> dict:
    """Does a held-out row see itself (distance 0) in the AD reference set?"""
    dist = _dist(d.X, d.train_x)
    self_d = dist[np.arange(len(d.X)), d.orig]
    return {
        "n_rows": int(len(d.X)),
        "fraction_rows_with_zero_distance_to_own_training_copy": float(np.mean(self_d < 1e-6)),
        "max_distance_to_own_training_copy": float(self_d.max()),
        "k_ad": d.k_ad,
        "tau_ad": d.tau_ad,
        "interpretation": "tau_AD is computed leave-one-out (diagonal excluded); if held-out rows include "
                          "themselves at distance 0, their D_AD is biased low relative to a new candidate.",
    }


def ad_inflation(d: Data, cfg: Config) -> np.ndarray:
    dist = _dist(d.X, d.train_x)
    if cfg.exclude_self_in_ad:
        dist[np.arange(len(d.X)), d.orig] = np.inf
    D = np.sort(dist, axis=1)[:, :d.k_ad].mean(1)
    ratio = D / d.tau_ad
    raw = 1.0 + cfg.eta * (ratio - cfg.s0) / (1.0 - cfg.s0)
    return np.where(ratio > cfg.s0, np.clip(raw, 1.0, cfg.cap), 1.0)


def local_sd(d: Data, ref_mask: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray]:
    """SD of the q nearest reference residuals, the same physical row excluded."""
    rs = (d.yS - d.muS)[ref_mask]
    rq = (d.yQ - d.muQ)[ref_mask]
    dist = _dist(d.X, d.X[ref_mask])
    same = d.orig[:, None] == d.orig[ref_mask][None, :]
    dist[same] = np.inf
    nn = np.argpartition(dist, q - 1, axis=1)[:, :q]
    return np.std(rs[nn], axis=1, ddof=1), np.std(rq[nn], axis=1, ddof=1)


def combine(gp: np.ndarray, loc: np.ndarray, rule: str) -> np.ndarray:
    gp = np.maximum(gp, 1e-6)
    return {"gp_only": gp, "local_only": np.maximum(loc, 1e-12), "max": np.maximum(gp, loc),
            "mean": 0.5 * (gp + loc), "sqrt": np.sqrt(gp ** 2 + loc ** 2)}[rule]


def fold_masks(d: Data, h: int) -> tuple[np.ndarray, np.ndarray]:
    te = d.split == h
    cal = (~te) & (~np.isin(d.orig, np.unique(d.orig[te])))
    return te, cal


def scales(d: Data, cfg: Config, cal: np.ndarray, infl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    locS, locQ = local_sd(d, cal, cfg.k_local)
    return combine(d.rawS, locS, cfg.rule) * infl, combine(d.rawQ, locQ, cfg.rule) * infl


def order_stat_q(scores: np.ndarray, p: float) -> float:
    sc = np.sort(scores)
    k = int(math.ceil((len(sc) + 1) * p))
    return float(sc[k - 1]) if 1 <= k <= len(sc) else float("inf")


def wilson(k: float, n: float, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def target_grid(d: Data, run_dir: Path | None) -> list[tuple[str, float, float]]:
    """(id, strength MPa, permeability mm/s). Grid from data quantiles; run target added if given."""
    qs = (0.30, 0.45, 0.60, 0.75)
    ss = np.quantile(d.yS, qs)
    ks = np.quantile(np.exp(d.yQ), qs)
    grid = [(f"grid_S{a:.2f}_K{b:.2f}", float(s), float(k)) for a, s in zip(qs, ss) for b, k in zip(qs, ks)]
    if run_dir is not None:
        tc = json.loads((Path(run_dir) / "TARGET_CONTRACT.json").read_text(encoding="utf-8"))
        grid.append(("run_target", float(tc["strength_target_mpa"]), float(tc["permeability_target_mm_s"])))
    return grid


def crossfit_pairs(d: Data, cfg: Config, targets) -> tuple[pd.DataFrame, dict]:
    """Predicted joint-pass probability P vs observed O for every held-out row and target,
    plus the cross-fitted max-score coverage."""
    infl = ad_inflation(d, cfg)
    rows, cover = [], {p: [] for p in LEVELS}
    width = {p: [] for p in LEVELS}
    for h in sorted(np.unique(d.split)):
        te, cal = fold_masks(d, h)
        sS, sQ = scales(d, cfg, cal, infl)
        T = np.maximum((d.muS - d.yS) / sS, (d.muQ - d.yQ) / sQ)
        for p in LEVELS:
            qv = order_stat_q(T[cal], p)
            cover[p].append(float(np.mean(T[te] <= qv)))
            width[p].append(float(np.mean(qv * sS[te])))
        eS = ((d.yS - d.muS) / sS)[cal]
        eQ = ((d.yQ - d.muQ) / sQ)[cal]
        ti = np.flatnonzero(te)
        Sreal = d.muS[ti, None] + sS[ti, None] * eS[None, :]
        Qreal = d.muQ[ti, None] + sQ[ti, None] * eQ[None, :]
        for tid, s_star, k_star in targets:
            P = ((Sreal >= s_star) & (Qreal >= math.log(k_star))).mean(1)
            O = ((d.yS[ti] >= s_star) & (d.yQ[ti] >= math.log(k_star))).astype(float)
            rows.append(pd.DataFrame({"fold": h, "row": ti, "study": d.study[ti], "target_id": tid,
                                      "S_star": s_star, "K_star": k_star, "P": P, "O": O}))
    summary = {f"coverage_{p:.2f}": float(np.mean(cover[p])) for p in LEVELS}
    summary.update({f"mean_interval_halfwidth_strength_{p:.2f}": float(np.mean(width[p])) for p in LEVELS})
    return pd.concat(rows, ignore_index=True), summary


def _logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def calibration_slope(P: np.ndarray, O: np.ndarray) -> tuple[float, float]:
    """Logistic regression of O on logit(P): returns (intercept, slope).

    Fitted by BFGS on the numerically stable negative log-likelihood, so it stays
    well behaved when the forecaster is badly overconfident.
    """
    from scipy.optimize import minimize

    x = _logit(P)
    A = np.column_stack([np.ones_like(x), x])

    def nll(b):
        z = A @ b
        return float(np.sum(np.logaddexp(0.0, z) - O * z))

    def grad(b):
        z = A @ b
        return A.T @ (1.0 / (1.0 + np.exp(-z)) - O)

    res = minimize(nll, np.array([0.0, 1.0]), jac=grad, method="BFGS")
    return float(res.x[0]), float(res.x[1])


def metrics(P: np.ndarray, O: np.ndarray) -> dict:
    acc = P >= ACCEPT_P
    icpt, slope = calibration_slope(P, O)
    return {
        "n_pairs": int(len(P)), "brier": float(np.mean((P - O) ** 2)),
        "cal_intercept": icpt, "cal_slope": slope,
        "n_accept": int(acc.sum()),
        "accept_observed_rate": float(O[acc].mean()) if acc.any() else float("nan"),
        "accept_mean_predicted": float(P[acc].mean()) if acc.any() else float("nan"),
    }


def cluster_bootstrap(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Resample whole source studies; report percentile intervals."""
    rng = np.random.default_rng(seed)
    studies = df["study"].to_numpy()
    uniq = np.unique(studies)
    idx = {s: np.flatnonzero(studies == s) for s in uniq}
    P, O = df["P"].to_numpy(), df["O"].to_numpy()
    out = {"slope": [], "accept_rate": [], "accept_gap": []}
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx[s] for s in draw])
        m = metrics(P[sel], O[sel])
        out["slope"].append(m["cal_slope"])
        out["accept_rate"].append(m["accept_observed_rate"])
        out["accept_gap"].append(m["accept_mean_predicted"] - m["accept_observed_rate"])
    res = {}
    for k, v in out.items():
        a = np.asarray(v, float)
        a = a[np.isfinite(a)]
        res[k] = {"lo95": float(np.percentile(a, 2.5)), "hi95": float(np.percentile(a, 97.5)),
                  "lower5": float(np.percentile(a, 5.0))}
    return res


DECISION_BANDS = ((0.80, 0.90), (0.90, 0.95), (0.95, 1.0000001))


def band_stats(df: pd.DataFrame, lo: float, hi: float, n_boot: int, seed: int) -> dict:
    """Observed vs predicted pass rate for pairs with lo <= P < hi, study-clustered interval."""
    sub = df[(df["P"] >= lo) & (df["P"] < hi)]
    if sub.empty:
        return {"lo": lo, "hi": hi, "n_pairs": 0}
    rng = np.random.default_rng(seed)
    studies = sub["study"].to_numpy()
    uniq = np.unique(studies)
    idx = {s: np.flatnonzero(studies == s) for s in uniq}
    P, O = sub["P"].to_numpy(), sub["O"].to_numpy()
    obs = []
    for _ in range(n_boot):
        sel = np.concatenate([idx[s] for s in rng.choice(uniq, size=len(uniq), replace=True)])
        obs.append(O[sel].mean())
    return {"lo": lo, "hi": min(hi, 1.0), "n_pairs": int(len(sub)), "n_studies": int(len(uniq)),
            "mean_predicted": float(P.mean()), "observed_rate": float(O.mean()),
            "gap_predicted_minus_observed": float(P.mean() - O.mean()),
            "observed_cluster_lo95": float(np.percentile(obs, 2.5)),
            "observed_cluster_hi95": float(np.percentile(obs, 97.5))}


BINS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0000001]


def reliability_table(df: pd.DataFrame) -> pd.DataFrame:
    cut = pd.cut(df["P"], BINS, right=False)
    rows = []
    for iv, g in df.groupby(cut, observed=True):
        lo, hi = wilson(g["O"].sum(), len(g))
        rows.append({"bin": str(iv), "n": len(g), "mean_predicted": g["P"].mean(),
                     "observed_rate": g["O"].mean(), "wilson_lo": lo, "wilson_hi": hi})
    return pd.DataFrame(rows)


def evaluate_criteria(m: dict, ci: dict, coverage: dict, stored_cov: dict | None) -> dict:
    c1a = ci["slope"]["lo95"] <= 1.2 and ci["slope"]["hi95"] >= 0.8
    c1b = ci["accept_rate"]["lower5"] >= 0.85
    c1c = (m["accept_mean_predicted"] - m["accept_observed_rate"]) <= 0.05
    out = {"C1a_slope_interval_overlaps_0.8_1.2": bool(c1a),
           "C1b_accept_region_observed_lower95_ge_0.85": bool(c1b),
           "C1c_accept_region_bias_le_0.05": bool(c1c)}
    if stored_cov is not None:
        out["C2_reproduces_stored_coverage_within_0.002"] = bool(
            all(abs(coverage[f"coverage_{p:.2f}"] - stored_cov[p]) <= 0.002 for p in LEVELS))
    return out


def stored_coverage() -> dict | None:
    path = PKG / "validation" / "joint_fold_metrics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return {p: float(df[np.isclose(df["p"], p)]["joint_cov"].mean()) for p in LEVELS}


def sensitivity_configs() -> list[tuple[str, Config]]:
    base = Config()
    out = [("base_production", base), ("ad_self_excluded", replace(base, exclude_self_in_ad=True))]
    out += [(f"s0={v}", replace(base, s0=v)) for v in (0.5, 0.9)]
    out += [(f"eta={v}", replace(base, eta=v)) for v in (0.5, 2.0)]
    out += [(f"cap={v}", replace(base, cap=v)) for v in (2.0, 5.0)]
    out += [(f"k_local={v}", replace(base, k_local=v)) for v in (5, 20)]
    out += [(f"rule={v}", replace(base, rule=v)) for v in ("gp_only", "local_only", "mean", "sqrt")]
    return out


def plot_reliability(tab: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    err = np.vstack([tab["observed_rate"] - tab["wilson_lo"], tab["wilson_hi"] - tab["observed_rate"]])
    ax.errorbar(tab["mean_predicted"], tab["observed_rate"], yerr=err, fmt="o-", capsize=3, label="held-out rows")
    ax.axvline(ACCEPT_P, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Predicted joint-pass probability (cross-fitted)")
    ax.set_ylabel("Observed joint-pass rate (held-out)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=None, help="run folder whose TARGET_CONTRACT.json adds a target")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results" / "heldout")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260921)
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    d = load_data()
    targets = target_grid(d, a.run_dir)
    (a.out_dir / "ad_self_distance_test.json").write_text(json.dumps(ad_self_distance_test(d), indent=2))

    stored = stored_coverage()
    sens_rows, main_pairs, main_summary = [], None, None
    for name, cfg in sensitivity_configs():
        pairs, summ = crossfit_pairs(d, cfg, targets)
        m = metrics(pairs["P"].to_numpy(), pairs["O"].to_numpy())
        sens_rows.append({"config": name, **{k: v for k, v in asdict(cfg).items()}, **summ, **m})
        if name == "base_production":
            main_pairs, main_summary = pairs, summ
        if name == "ad_self_excluded":
            fixed_pairs, fixed_summary = pairs, summ
        print(f"[{name}] cov90={summ['coverage_0.90']:.3f} brier={m['brier']:.4f} slope={m['cal_slope']:.3f} "
              f"accept n={m['n_accept']} obs={m['accept_observed_rate']:.3f}", flush=True)
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(a.out_dir / "sensitivity_constants.csv", index=False)

    result = {"targets_used": [{"id": t, "strength_mpa": s, "permeability_mm_s": k} for t, s, k in targets],
              "stored_coverage": stored}
    for label, pairs, summ in (("base_production", main_pairs, main_summary),
                               ("ad_self_excluded", fixed_pairs, fixed_summary)):
        m = metrics(pairs["P"].to_numpy(), pairs["O"].to_numpy())
        ci = cluster_bootstrap(pairs, a.boot, a.seed)
        rt = pairs[pairs["target_id"] == "run_target"]
        entry = {"metrics": m, "cluster_bootstrap": ci, "coverage": summ,
                 "criteria": evaluate_criteria(m, ci, summ, stored)}
        if len(rt):
            entry["run_target_only"] = metrics(rt["P"].to_numpy(), rt["O"].to_numpy())
        entry["decision_bands"] = [band_stats(pairs, lo, hi, a.boot, a.seed) for lo, hi in DECISION_BANDS]
        result[label] = entry
        pairs.to_csv(a.out_dir / f"heldout_pass_probability_pairs_{label}.csv", index=False)
        tab = reliability_table(pairs)
        tab.to_csv(a.out_dir / f"reliability_bins_{label}.csv", index=False)
        plot_reliability(tab, a.out_dir / f"reliability_diagram_{label}")

    base = sens.set_index("config").loc["base_production"]
    influential = []
    for _, r in sens.iterrows():
        if r["config"] == "base_production":
            continue
        d_acc = abs(r["accept_observed_rate"] - base["accept_observed_rate"])
        d_cov = abs(r["coverage_0.90"] - base["coverage_0.90"])
        if d_acc > 0.03 or d_cov > 0.02:
            influential.append({"config": r["config"], "accept_rate_change": float(d_acc), "coverage90_change": float(d_cov)})
    result["C3_influential_constants"] = influential
    (a.out_dir / "calibration_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result["base_production"][k] for k in ("criteria",)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
