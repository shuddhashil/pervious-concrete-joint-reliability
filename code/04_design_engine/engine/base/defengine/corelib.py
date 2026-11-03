"""Frozen-science core loader + batched evaluator.

The code that DEFINES the reliability number lives in ../core (byte-identical to the A3 engine copy; hashes in ../CORE_MANIFEST.json).
This module only (1) imports it, (2) verifies the hashes, (3) offers a batched evaluator that scores many designs in one pass.
Run with the engine's bundled python (CPU torch). Decision E-01 / E-02 in DECISIONS_LOG.md."""
import os, sys, json, math, hashlib, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "core"
os.environ.setdefault("POROSITY_CONDITIONAL_CALIBRATION", "1")      # the reported method (D-019)
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "1")
sys.path.insert(0, str(CORE / "src"))
import stage1_evaluator as E            # noqa: E402
import fast_descriptors as FD           # noqa: E402
FD.install_patch(E)
import fast_engine as FE                # noqa: E402
import uncertainty as U                 # noqa: E402
import continuous_score as CS           # noqa: E402
import solvers_common as SC             # noqa: E402
import stage17_state_support as S17     # noqa: E402
import gradation_support as GS          # noqa: E402
import efficient_verification as EV     # noqa: E402
import runtime_targets as RT            # noqa: E402

CONTRACT = str(CORE / "UNCERTAINTY_CONTRACT.json")
BANK_CSV = str(CORE / "tables" / "paired_residual_bank.csv")
TRAIN_CSV = str(CORE / "tables" / "training_data_reconciled.csv")
S_STAR, K_STAR, REL_TARGET = RT.STRENGTH, RT.PERMEABILITY, RT.RELIABILITY
LOGK_STAR = math.log(K_STAR)
SUPPORT_THR = E.SUPPORT_THRESHOLD
_BANKDF = {}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def model_hash():
    """sha256 of the loaded A3 oracle file (decision E-02: never a hard-coded string)."""
    return sha256_file(E.ORACLE_PT)


def verify_core(manifest_path=None):
    """Return the list of core files whose hash differs from the frozen manifest (empty list = intact)."""
    man = json.load(open(manifest_path or (ROOT / "CORE_MANIFEST.json")))
    bad = []
    for rel, h in man.items():
        p = CORE / rel
        if not p.exists():
            if not rel.endswith(".pyc"): bad.append(f"MISSING {rel}")
        elif sha256_file(p) != h:
            bad.append(f"CHANGED {rel}")
    return bad


def spec(porosity_sd_pp=None, porosity_bias_pp=None, stress=False):
    return U.UncertaintySpec(CONTRACT, stress=stress, porosity_sd_override_pp=porosity_sd_pp, porosity_bias_override_pp=porosity_bias_pp)


def mfg_bank(n, seed, **spec_kw):
    return U.build_manufacturing_bank(spec(**spec_kw), n=n, seed=seed)


def bank_df():
    if "df" not in _BANKDF:
        _BANKDF["df"] = pd.read_csv(BANK_CSV)
    return _BANKDF["df"]


def res_bank_iid(n, seed):
    """Paired residual draws (iid bootstrap), as used by the frozen search banks."""
    df = bank_df(); rng = np.random.default_rng(seed); idx = rng.integers(0, len(df), size=n)
    return (df["signed_standardised_strength_residual"].to_numpy(float)[idx], df["signed_standardised_logk_residual"].to_numpy(float)[idx], idx)


def engine():
    E.get_bundle()
    eng = FE.get_engine(E)
    if eng is None:
        raise RuntimeError("fast engine unavailable (self-check failed); refusing to run on an unverified path")
    return eng


def batch_scenario_arrays(nominals, bank):
    """Apply the SAME manufacturing bank to every nominal design and run ONE forward pass. Returns (P, N) arrays."""
    P, N = len(nominals), bank.n
    mf = [U.apply_manufacturing_scenarios(n, bank) for n in nominals]
    cat = {k: np.concatenate([m[k] for m in mf]) for k in ("g_actual", "phi_actual", "wc_actual", "C_actual", "A_actual", "SP_actual", "S_actual")}
    muS, muQ, sS, sQ, sup = engine().scenario_arrays(cat)
    r = lambda a: np.asarray(a).reshape(P, N)
    return dict(muS=r(muS), muQ=r(muQ), sS=r(sS), sQ=r(sQ), support=r(sup), phi=r(cat["phi_actual"]))


_GPU = {"eng": None}


def use_gpu(on=True):
    """Route search-time scenario evaluation to the GPU engine (needs the CUDA python). Acceptance (verify.py) stays on the frozen CPU path."""
    if on and _GPU["eng"] is None:
        from . import gpu
        _GPU["eng"] = gpu.GPUEngine()
    if not on: _GPU["eng"] = None
    return _GPU["eng"]


def batch_search_stats(nominals, bank, e_strength, e_logk, tail_frac=0.10):
    if _GPU["eng"] is not None:
        return _GPU["eng"].search_stats(nominals, bank, e_strength, e_logk, tail_frac)
    return batch_search_stats_cpu(nominals, bank, e_strength, e_logk, tail_frac)


def batch_search_stats_cpu(nominals, bank, e_strength, e_logk, tail_frac=0.10):
    """Search-time statistics for a whole population with common random numbers.
    Returns arrays (P,): rhat (literal joint pass fraction), cvar (smooth CVaR-of-margin score, -inf if any scenario unsupported),
    support_fraction. Same formulas as the frozen robust_evaluator / continuous_score, vectorised."""
    A = batch_scenario_arrays(nominals, bank)
    S_real = A["muS"] + A["sS"] * e_strength[None, :]
    Q_real = A["muQ"] + A["sQ"] * e_logk[None, :]
    insup = A["support"] >= SUPPORT_THR
    ok = insup & (S_real >= S_STAR) & (Q_real >= LOGK_STAR)
    s_scale, k_scale = CS.resolve_scales()
    jm = np.minimum((S_real - S_STAR) / s_scale, (Q_real - LOGK_STAR) / k_scale)
    k = max(1, int(math.ceil(tail_frac * jm.shape[1])))
    cvar = np.partition(jm, k - 1, axis=1)[:, :k].mean(axis=1)
    all_sup = insup.all(axis=1)
    cvar = np.where(all_sup, cvar, -np.inf)
    return dict(rhat=ok.mean(axis=1), cvar=cvar, support_fraction=insup.mean(axis=1), strength_p05=np.percentile(S_real, 5, axis=1), logk_p05=np.percentile(Q_real, 5, axis=1))


def decode_unit(Z, cement=None):
    """Unit-cube vectors (n,12) -> nominal design dicts, cement optionally fixed (kg/m3)."""
    Z = np.array(Z, float, copy=True)
    if cement is not None:
        Z[:, 0] = (cement - SC.BND["C"][0]) / (SC.BND["C"][1] - SC.BND["C"][0])
    return SC.decode_batch(Z)
