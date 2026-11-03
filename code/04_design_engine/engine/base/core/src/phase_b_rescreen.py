"""Phase B - deduplicate DE candidates and rescore leading candidates on a
FRESH, larger, common-random bank disjoint from the Phase A search bank
(SEARCH_CONTRACT.json phase_B_rescreen)."""
from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(RUN)))
sys.path.insert(0, HERE)
import codec11 as K  # noqa: E402
import stage1_evaluator as E  # noqa: E402
import uncertainty as U  # noqa: E402
import robust_evaluator as RE  # noqa: E402
import fast_descriptors as FD  # noqa: E402
import fast_local_residual as FLR  # noqa: E402
import stage17_state_support as S17  # noqa: E402  (Stage-18 nominal state-support gate)
import gradation_support as GS  # noqa: E402
try:
    import runtime_targets as RT  # generated per run by run_stage1.py
except ImportError:  # template-level unit tests before run generation; a REAL import error (bad syntax, corrupted generated file) now propagates instead of silently falling back
    class RT:  # noqa: N801 - module-compatible fallback
        STRENGTH = 18.0
        PERMEABILITY = 13.0
        RELIABILITY = 0.90
try:  # Blind-repair: multi-fidelity plausible retention (no premature cut).
    import multifidelity as MF  # noqa: E402
except Exception:  # pragma: no cover
    MF = None

# Vectorised, numerically-verified replacement for cached_descriptors --
# see WORKLOG.md / tests/test_fast_descriptors_equivalence.py. Process-local
# monkey-patch only; the frozen file on disk is never touched.
if os.environ.get("DISABLE_FAST_DESCRIPTORS") != "1":
    FD.install_patch(E)
    print("[perf] fast_descriptors patch installed", flush=True)
if os.environ.get("DISABLE_FAST_LOCAL_RESIDUAL") != "1":
    FLR.install_patch(E.L)
    print("[perf] fast_local_residual patch installed", flush=True)

CONTRACT = os.path.join(RUN, "UNCERTAINTY_CONTRACT.json")
BANK_CSV = os.path.join(RUN, "tables", "paired_residual_bank.csv")
ORDER = "coarse_first"
RESCREEN_N = 5000
RESCREEN_SEED = 205609201
DEDUP_L2_THRESHOLD = 0.1
# Chosen empirically on this run's actual DE shortlists (918 raw rows across
# 4 seeds): 0.02 (the originally-planned value) barely merges anything
# (555 "distinct" survivors -- consecutive DE generations along the same
# converging trajectory differ by more than 0.02 in the 11-D unit-cube
# encoding even though they represent nearly the same physical design,
# since a single scalar coordinate like cement carries ~500 kg/m3 of
# physical range per unit of normalized distance). 0.1 collapses this to
# 16 genuinely distinct candidates spanning the same cement range
# [247.8, 340.0] kg/m3 -- see WORKLOG.md for the full threshold sweep
# (0.02->555, 0.05->114, 0.08->32, 0.1->16, 0.15->6, 0.2->4, 0.3->1).


def load_all_de_shortlist():
    rows = []
    for path in glob.glob(os.path.join(RUN, "traces", "de_shortlist_seed*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df is None or len(df) == 0 or "C" not in df.columns:
            continue
        rows.append(df)
    if not rows:
        raise FileNotFoundError("no usable de_shortlist_seed*.csv rows found -- run Phase A first")
    return pd.concat(rows, ignore_index=True)


def load_prior_basins():
    # Compatibility hook: production starts come only from this run's DE
    # archive.  Kept empty so callers cannot accidentally inject coordinates.
    return {}


def s17_record(cand):
    """Stage-18 nominal state-support record for one candidate.

    Bypassed (all-pass) only when DISABLE_STAGE17_SUPPORT=1, which production
    never sets (integration-test replay only).
    """
    if os.environ.get("DISABLE_STAGE17_SUPPORT") == "1":
        return {"s17_recipe_pass": True, "s17_recipe_pct": float("nan"),
                "s17_recipe_excess": 0.0, "s17_phi_in_conf90": True,
                "s17_phi_violation": 0.0, "s17_conf90_width": float("nan"),
                "s17_conf90_nmodes": 0, "s17_deploy_ok": True,
                "s17_empirical_pass": True, "s17_state_pass": True,
                "gradation_w1_min": float("nan"),
                "gradation_w1_epsilon_95": float("nan"),
                "gradation_w1_pass": True}
    ss = S17.state_support(cand)
    gc = GS.load_contract()
    w1_violation, w1_min = GS.violation(np.asarray(cand["g"], dtype=float), gc["epsilon_95"])
    return {"s17_recipe_pass": bool(ss["recipe_pass"]),
            "s17_recipe_pct": float(ss["recipe_percentile"]),
            "s17_recipe_excess": float(ss["recipe_excess"]),
            "s17_phi_in_conf90": bool(ss["phi_in_conf90"]),
            "s17_phi_violation": 0.0 if ss["phi_in_conf90"] else float(ss["phi_violation"]),
            "s17_conf90_width": float(ss["conf90_measure"]),
            "s17_conf90_nmodes": int(ss["conf90_n_components"]),
            "s17_deploy_ok": bool(ss["deployment_ok"]),
            "s17_empirical_pass": bool(ss["empirical_pass"]),
            "s17_state_pass": bool(ss["state_support_pass"]),
            "gradation_w1_min": float(w1_min),
            "gradation_w1_epsilon_95": float(gc["epsilon_95"]),
            "gradation_w1_pass": bool(w1_violation <= 0.0)}


def failure_reason(s17rec, support_fraction, rhat, rel_target=None):
    """Prescribed Phase-B failure taxonomy (first failing gate in order)."""
    rel_target = float(RT.RELIABILITY if rel_target is None else rel_target)
    if not s17rec["s17_recipe_pass"]:
        return "recipe-space support"
    if not s17rec["s17_phi_in_conf90"]:
        return "porosity-state support"
    if not s17rec["s17_deploy_ok"]:
        return "deployment phi domain"
    if not s17rec.get("gradation_w1_pass", False):
        return "gradation support"
    if support_fraction < 1.0:
        return "old forward-model AD/support"
    if rhat < rel_target:
        return "joint reliability below target"
    return "none (pass)"


def _cand_from_row(row):
    g = [float(row[f"g{i}"]) for i in range(1, 7)]
    return dict(C=float(row["C"]), r=float(row["r"]), A=float(row["A"]), P=float(row["P"]),
                S=float(row["S"]), phi=float(row["phi"]), g=g)


def dedup(df):
    """Original local full-design dedup, retaining boundary + feasible seed.

    The original greedy full-11D local representatives are deliberately kept
    to preserve the observed 16-basin diversity.  Each discarded record is
    assigned to its closest such representative; its maximum-Rhat member is
    retained in addition to the minimum-cement representative.  This avoids
    losing a feasible local start merely because it uses more cement.
    """
    df = df.sort_values(["C", "Rhat_search"], ascending=[True, False]).reset_index(drop=True)
    xs = np.array([K.encode11(_cand_from_row(row), ORDER) for _, row in df.iterrows()])
    representative_rows, representative_xs, members = [], [], []
    for i, x in enumerate(xs):
        if not representative_xs:
            representative_rows.append(i)
            representative_xs.append(x)
            members.append([i])
            continue
        distances = np.array([np.linalg.norm(x - rx) for rx in representative_xs])
        nearest = int(np.argmin(distances))
        if distances[nearest] < DEDUP_L2_THRESHOLD:
            members[nearest].append(i)
        else:
            representative_rows.append(i)
            representative_xs.append(x)
            members.append([i])
    kept, labels = [], {}
    for cluster_id, (low, cluster_members) in enumerate(zip(representative_rows, members)):
        best = df.iloc[cluster_members].sort_values(["Rhat_search", "C"], ascending=[False, True]).index[0]
        kept.append(low)
        labels[low] = f"cluster_{cluster_id}:min_cement"
        if best != low:
            kept.append(best)
            labels[best] = f"cluster_{cluster_id}:max_search_rhat"
    out = df.loc[sorted(set(kept))].copy()
    out["dedup_role"] = [labels[i] for i in out.index]
    return out.sort_values(["C", "Rhat_search"], ascending=[True, False]).reset_index(drop=True)


def main():
    t0 = time.time()
    E.get_bundle()
    spec = U.UncertaintySpec(CONTRACT)
    mfg_bank = U.build_manufacturing_bank(spec, n=RESCREEN_N, seed=RESCREEN_SEED)
    res_bank = U.build_residual_bank(BANK_CSV, n=RESCREEN_N, seed=RESCREEN_SEED)
    print(f"Rescreen bank: N={RESCREEN_N} mfg_hash={mfg_bank.bank_hash()[:12]} "
          f"res_hash={res_bank.bank_hash()[:12]}", flush=True)

    de_shortlist = load_all_de_shortlist()
    print(f"Loaded {len(de_shortlist)} DE shortlist rows (Rhat_search>=0.85) across all seeds", flush=True)
    deduped = dedup(de_shortlist)
    print(f"After local full-design clustering (L2<{DEDUP_L2_THRESHOLD}): "
          f"{len(deduped)} retained boundary/seed candidates", flush=True)

    rows = []
    for _, row in deduped.iterrows():
        cand = _cand_from_row(row)
        r = RE.evaluate_robust(
            cand, mfg_bank, res_bank, backend="fast",
            s_star=float(RT.STRENGTH), k_star=float(RT.PERMEABILITY))
        s17 = s17_record(cand)
        rhat = float(r["joint_reliability_hat"])
        if MF is not None:
            plausible = MF.plausible(rhat, RESCREEN_N, float(RT.RELIABILITY))
        else:
            plausible = bool(rhat >= float(RT.RELIABILITY))
        rows.append({"source": "DE_dedup", "tag": None, **cand_flat(cand),
                      "Rhat_rescreen": rhat, "support_fraction": r["support_fraction"],
                      "in_bounds": True,
                      "support_score_pass": bool(r["support_fraction"] >= 1.0 - 1e-12),
                      "reliability_gate": bool(rhat >= float(RT.RELIABILITY)),
                      "statistically_plausible": bool(plausible),
                      "verified": False,
                      "Rhat_search_original": row["Rhat_search"], "seed": row["seed"], "gen": row["gen"],
                     **s17,
                      "failure_reason": failure_reason(s17, r["support_fraction"], rhat)})
    if MF is not None:
        rep = MF.retention_report(len(rows), [r.get("statistically_plausible", False) for r in rows],
                                  "phase_b_rescreen", RESCREEN_N,
                                  "retain rhat+z*SE>=target-margin (no premature elimination)")
        print(f"[Phase B] retention: {rep['n_kept']}/{rep['n_in']} kept | {rep['survival_rule']}", flush=True)

    out = pd.DataFrame(rows).sort_values(["Rhat_rescreen", "C"], ascending=[False, True])
    out_path = os.path.join(RUN, "tables", "CONTINUOUS_FRONTIER_SHORTLIST.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} rows). Elapsed {time.time()-t0:.1f}s", flush=True)
    print(out[["source", "tag", "C", "Rhat_rescreen", "support_fraction"]].head(20).to_string())


def cand_flat(cand):
    return {"C": cand["C"], "r": cand["r"], "A": cand["A"], "P": cand["P"], "S": cand["S"], "phi": cand["phi"],
            **{f"g{i+1}": cand["g"][i] for i in range(6)}}


if __name__ == "__main__":
    main()
