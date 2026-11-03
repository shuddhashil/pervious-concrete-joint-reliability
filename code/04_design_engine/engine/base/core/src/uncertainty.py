"""Manufacturing/process uncertainty scenario sampler and paired predictive
residual sampler for the Robust Stage-One search.

Implements UncertaintySpec / ManufacturingScenarioBank / ResidualScenarioBank
per UNCERTAINTY_CONTRACT.json. No hidden global RNG: every bank is built from
an explicit seed and is fully serialisable (metadata + arrays).

Transformation order (exact, matches UNCERTAINTY_MODEL.md section 4):
  perturb C -> perturb added W -> [moisture correction if available: NONE
  in this run] -> derive w/c -> perturb aggregate-bin masses -> derive
  realised gradation fractions -> perturb SP if >0 -> perturb sand if >0
  (run_04 repair -- see RUN04_PROBLEM_AUDIT.md Defect A) -> draw achieved
  porosity -> (feature-vector rebuild + support check + model evaluation
  happen in robust_evaluator.py, which consumes this module's scenario
  arrays).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sha256_text(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# --------------------------------------------------------------------------
# UncertaintySpec
# --------------------------------------------------------------------------

@dataclass
class UncertaintySpec:
    """One resolved uncertainty parameter set (baseline or a named stress
    scenario), loaded from UNCERTAINTY_CONTRACT.json.

    Porosity truncation is DYNAMIC (+/-3*SD, per contract v2.0.0 Section E1)
    rather than a fixed absolute bound, and porosity_sd_pp/porosity_bias_pp
    can be overridden directly (porosity_sd_override_pp /
    porosity_bias_override_pp) to run the required sensitivity grid
    (0.5/1.0/1.5/2.0 pp SD; -1/0/+1 pp bias) without needing a second named
    "sensitivity" block for this parameter.
    """

    contract_path: str
    stress: bool = False
    contract: dict = field(default_factory=dict)
    porosity_sd_override_pp: float = None
    porosity_bias_override_pp: float = None

    def __post_init__(self):
        if not self.contract:
            with open(self.contract_path) as f:
                self.contract = json.load(f)

    def _get(self, name, key, stress_key=None):
        p = self.contract["parameters"][name]
        if self.stress:
            return p["sensitivity"][stress_key or key]
        return p["baseline"][key]

    # porosity is in percentage points; everything else is a fraction (0.0033 = 0.33%)
    @property
    def porosity_sd_pp(self):
        if self.porosity_sd_override_pp is not None:
            return self.porosity_sd_override_pp
        return self.contract["parameters"]["achieved_porosity"]["baseline"]["sd_pp"]

    @property
    def porosity_bias_pp(self):
        if self.porosity_bias_override_pp is not None:
            return self.porosity_bias_override_pp
        return self.contract["parameters"]["achieved_porosity"]["baseline"]["mean_pp"]

    @property
    def porosity_bounds_pp(self):
        """+/-3*SD truncation around the bias, per contract Section E1."""
        sd = self.porosity_sd_pp
        bias = self.porosity_bias_pp
        return (bias - 3.0 * sd, bias + 3.0 * sd)

    @property
    def cement_sd(self):
        return self._get("cement_mass", "sd")

    @property
    def cement_bounds(self):
        p = self.contract["parameters"]["cement_mass"]
        return tuple((p["sensitivity"] if self.stress else p["baseline"])["bounds"])

    @property
    def water_sd(self):
        return self._get("added_water", "sd")

    @property
    def water_bounds(self):
        p = self.contract["parameters"]["added_water"]
        return tuple((p["sensitivity"] if self.stress else p["baseline"])["bounds"])

    @property
    def bin_sd(self):
        return self._get("aggregate_bin_masses", "sd")

    @property
    def bin_bounds(self):
        p = self.contract["parameters"]["aggregate_bin_masses"]
        return tuple((p["sensitivity"] if self.stress else p["baseline"])["bounds"])

    @property
    def sp_sd(self):
        return self._get("superplasticizer", "sd")

    @property
    def sp_bounds(self):
        p = self.contract["parameters"]["superplasticizer"]
        return tuple((p["sensitivity"] if self.stress else p["baseline"])["bounds"])

    @property
    def sand_sd(self):
        return self._get("sand_mass", "sd")

    @property
    def sand_bounds(self):
        p = self.contract["parameters"]["sand_mass"]
        return tuple((p["sensitivity"] if self.stress else p["baseline"])["bounds"])

    @property
    def reliability_mode(self):
        return self.contract.get("reliability_mode")

    @property
    def support_threshold_note(self):
        return "loaded from the frozen support model at evaluation time, never hard-coded here"

    def spec_hash(self):
        return _sha256_text({"contract": self.contract, "stress": self.stress,
                              "porosity_sd_override_pp": self.porosity_sd_override_pp,
                              "porosity_bias_override_pp": self.porosity_bias_override_pp})


# --------------------------------------------------------------------------
# Truncated-normal rejection sampler (never "clip until valid" silently)
# --------------------------------------------------------------------------

def sample_truncated_normal(rng: np.random.Generator, mean, sd, lo, hi, n, max_rounds=200):
    """Rejection-sample n draws from N(mean, sd^2) truncated to [lo, hi].

    Returns (samples, n_rejected, n_rounds). Raises if truncation cannot be
    satisfied within max_rounds (rather than silently falling back to clip).
    """
    out = np.empty(n, dtype=float)
    filled = 0
    n_rejected = 0
    rounds = 0
    while filled < n and rounds < max_rounds:
        need = n - filled
        draw = rng.normal(mean, sd, size=int(need * 1.5) + 8)
        ok = draw[(draw >= lo) & (draw <= hi)]
        take = min(len(ok), need)
        out[filled:filled + take] = ok[:take]
        n_rejected += len(draw) - len(ok)
        filled += take
        rounds += 1
    if filled < n:
        raise RuntimeError(
            f"truncated-normal rejection sampling failed to fill {n} draws "
            f"(mean={mean}, sd={sd}, bounds=[{lo},{hi}]) within {max_rounds} rounds"
        )
    return out, n_rejected, rounds


# --------------------------------------------------------------------------
# ManufacturingScenarioBank
# --------------------------------------------------------------------------

@dataclass
class ManufacturingScenarioBank:
    n: int
    seed: int
    stress: bool
    delta_C: np.ndarray
    delta_W: np.ndarray
    delta_A: np.ndarray  # (n, 6)
    delta_SP: np.ndarray
    delta_S: np.ndarray
    eps_phi_pp: np.ndarray
    truncation_stats: dict
    spec_hash: str
    common_random: bool

    def bank_hash(self) -> str:
        arrs = [self.delta_C, self.delta_W, self.delta_A, self.delta_SP, self.delta_S, self.eps_phi_pp]
        h = hashlib.sha256()
        for a in arrs:
            h.update(np.ascontiguousarray(a).tobytes())
        h.update(_sha256_text({"n": self.n, "seed": self.seed, "stress": self.stress,
                                "spec_hash": self.spec_hash, "common_random": self.common_random}).encode())
        return h.hexdigest()

    def metadata(self) -> dict:
        return {
            "n": self.n, "seed": self.seed, "stress": self.stress,
            "spec_hash": self.spec_hash, "common_random": self.common_random,
            "truncation_stats": self.truncation_stats, "bank_hash": self.bank_hash(),
        }


def build_manufacturing_bank(spec: UncertaintySpec, n: int, seed: int, common_random: bool = True
                              ) -> ManufacturingScenarioBank:
    """Build N candidate-independent manufacturing perturbation draws.

    Candidate-independent because every parameter here is defined as a
    RELATIVE (fractional or percentage-point) perturbation, not an absolute
    one -- so the same draws can be applied to any nominal design, which is
    exactly what a common-random scenario bank requires for fair candidate
    comparison during the coarse search. `common_random=True` documents that
    intent in metadata; for final independent verification, build a FRESH
    bank with a different seed and (optionally) common_random=False.
    """
    rng = np.random.default_rng(seed)
    stats = {}

    lo, hi = spec.cement_bounds
    delta_C, rej, rounds = sample_truncated_normal(rng, 0.0, spec.cement_sd, lo, hi, n)
    stats["cement"] = {"n_rejected": rej, "rounds": rounds, "rejection_rate": rej / max(1, rej + n)}

    lo, hi = spec.water_bounds
    delta_W, rej, rounds = sample_truncated_normal(rng, 0.0, spec.water_sd, lo, hi, n)
    stats["water"] = {"n_rejected": rej, "rounds": rounds, "rejection_rate": rej / max(1, rej + n)}

    lo, hi = spec.bin_bounds
    delta_A = np.empty((n, 6))
    bin_rej_total, bin_rounds_max = 0, 0
    for j in range(6):
        col, rej, rounds = sample_truncated_normal(rng, 0.0, spec.bin_sd, lo, hi, n)
        delta_A[:, j] = col
        bin_rej_total += rej
        bin_rounds_max = max(bin_rounds_max, rounds)
    stats["aggregate_bins"] = {"n_rejected": bin_rej_total, "rounds_max": bin_rounds_max,
                                "rejection_rate": bin_rej_total / max(1, bin_rej_total + n * 6)}

    lo, hi = spec.sp_bounds
    delta_SP, rej, rounds = sample_truncated_normal(rng, 0.0, spec.sp_sd, lo, hi, n)
    stats["sp"] = {"n_rejected": rej, "rounds": rounds, "rejection_rate": rej / max(1, rej + n)}

    lo, hi = spec.sand_bounds
    delta_S, rej, rounds = sample_truncated_normal(rng, 0.0, spec.sand_sd, lo, hi, n)
    stats["sand"] = {"n_rejected": rej, "rounds": rounds, "rejection_rate": rej / max(1, rej + n)}

    lo, hi = spec.porosity_bounds_pp
    eps_phi, rej, rounds = sample_truncated_normal(rng, spec.porosity_bias_pp, spec.porosity_sd_pp, lo, hi, n)
    stats["porosity"] = {"n_rejected": rej, "rounds": rounds, "rejection_rate": rej / max(1, rej + n)}

    return ManufacturingScenarioBank(
        n=n, seed=seed, stress=spec.stress, delta_C=delta_C, delta_W=delta_W, delta_A=delta_A,
        delta_SP=delta_SP, delta_S=delta_S, eps_phi_pp=eps_phi, truncation_stats=stats,
        spec_hash=spec.spec_hash(), common_random=common_random,
    )


def apply_manufacturing_scenarios(nominal: dict, bank: ManufacturingScenarioBank) -> dict:
    """Apply the exact transformation order to a nominal design dict
    {C, r, A, P, S, phi, g(6)} and return per-scenario physical arrays.

    Returns dict of (n,)/(n,6) arrays: C_actual, W_actual, wc_actual,
    A_bin_actual (n,6), A_actual (n,), SP_actual, S_actual, phi_actual,
    g_actual (n,6).
    """
    n = bank.n
    C_nom = float(nominal["C"])
    wc_nom = float(nominal["r"])
    W_nom = wc_nom * C_nom
    A_nom = float(nominal["A"])
    SP_nom = float(nominal["P"])
    S_nom = float(nominal["S"])
    phi_nom = float(nominal["phi"])
    g_nom = np.asarray(nominal["g"], dtype=float)

    C_actual = C_nom * (1.0 + bank.delta_C)
    W_actual = W_nom * (1.0 + bank.delta_W)
    # no aggregate-moisture correction applied in this run (none discovered; see UNCERTAINTY_MODEL.md)
    wc_actual = W_actual / C_actual

    bin_mass_nom = A_nom * g_nom  # (6,)
    A_bin_actual = bin_mass_nom[None, :] * (1.0 + bank.delta_A)  # (n,6)
    A_bin_actual = np.maximum(A_bin_actual, 0.0)  # physical: mass cannot go negative
    A_actual = A_bin_actual.sum(axis=1)  # (n,)
    g_actual = A_bin_actual / A_actual[:, None]

    SP_actual = np.where(SP_nom > 0, SP_nom * (1.0 + bank.delta_SP), 0.0)
    # sand (S): perturbed exactly as declared in UNCERTAINTY_CONTRACT.json's
    # sand_mass parameter (multiplicative truncated-normal) when nominal
    # sand > 0; held EXACTLY zero when nominal sand == 0 (no noise-created
    # material), mirroring the SP conditional above.
    S_actual = np.where(S_nom > 0, S_nom * (1.0 + bank.delta_S), 0.0)

    phi_actual = phi_nom + bank.eps_phi_pp

    return dict(C_actual=C_actual, W_actual=W_actual, wc_actual=wc_actual,
                A_bin_actual=A_bin_actual, A_actual=A_actual, SP_actual=SP_actual,
                S_actual=S_actual, phi_actual=phi_actual, g_actual=g_actual)


# --------------------------------------------------------------------------
# ResidualScenarioBank (paired, signed, study-labelled)
# --------------------------------------------------------------------------

@dataclass
class ResidualScenarioBank:
    n: int
    seed: int
    row_indices: np.ndarray
    study_id: np.ndarray
    e_strength: np.ndarray
    e_logk: np.ndarray
    bank_csv_sha256: str
    resample_mode: str  # "iid_bootstrap" or "cluster_bootstrap_study"

    def bank_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.row_indices.tobytes())
        h.update(_sha256_text({"n": self.n, "seed": self.seed, "resample_mode": self.resample_mode,
                                "bank_csv_sha256": self.bank_csv_sha256}).encode())
        return h.hexdigest()


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_residual_bank(bank_csv_path: str, n: int, seed: int,
                         resample_mode: str = "iid_bootstrap") -> ResidualScenarioBank:
    """Draw n paired (signed_standardised_strength_residual,
    signed_standardised_logk_residual) rows from the frozen residual bank.

    resample_mode:
      "iid_bootstrap": sample rows with replacement uniformly (used for
        common-random coarse-search draws and generic Monte Carlo).
      "cluster_bootstrap_study": first sample studies with replacement, then
        one row within each sampled study (used for the final calibration-
        uncertainty bootstrap, so resampling respects between-study grouping
        rather than treating literature rows as independent).
    The two residuals in a drawn row are NEVER separated.
    """
    df = pd.read_csv(bank_csv_path)
    rng = np.random.default_rng(seed)
    if resample_mode == "iid_bootstrap":
        idx = rng.integers(0, len(df), size=n)
    elif resample_mode == "cluster_bootstrap_study":
        studies = df["study_id"].unique()
        chosen_studies = rng.choice(studies, size=n, replace=True)
        idx = np.empty(n, dtype=int)
        by_study = {s: df.index[df["study_id"] == s].to_numpy() for s in studies}
        for i, s in enumerate(chosen_studies):
            rows = by_study[s]
            idx[i] = rows[rng.integers(0, len(rows))]
    else:
        raise ValueError(f"unknown resample_mode {resample_mode}")

    sub = df.iloc[idx]
    return ResidualScenarioBank(
        n=n, seed=seed, row_indices=idx, study_id=sub["study_id"].to_numpy(),
        e_strength=sub["signed_standardised_strength_residual"].to_numpy(dtype=float),
        e_logk=sub["signed_standardised_logk_residual"].to_numpy(dtype=float),
        bank_csv_sha256=_sha256_file(bank_csv_path), resample_mode=resample_mode,
    )
