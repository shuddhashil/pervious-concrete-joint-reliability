"""Held-out reliability audit (porosity-conditional, as in the paper) for each variant bank, with PAIRED residuals (the paper's sampler) and with
INDEPENDENTLY sampled residuals (ln k residuals permuted within each fold x porosity-tercile pool, which keeps both marginals and removes the
pairing; 5 permutation seeds). CalibrationAudit_A3-1_20261005 code is imported unchanged; only the residual pairing is changed in the copy below.
One fixed target grid for every variant (the paper bank's grid, 16 (S*, K*) pairs) so all variants are scored on the same targets.
usage: python audit_mtind.py <audit_code_dir> <variants_root> <out_dir> <variant> [<variant> ...]"""
import sys, os, json, math
from pathlib import Path
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
import heldout_porosity_conditional as P
H = P.H; VR, OUT = Path(sys.argv[2]), Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
N_PERM, NB = 5, 2000


def load(name):
    M = VR / name / "core" / "models"; H.MODELS = M; P._PHI.clear(); return H.load_data(M)


def pairs_perm(d, cfg, targets, seed):
    """P.crossfit_pairs with the ln k residual column permuted inside each calibration pool (fold x porosity tercile)."""
    b = P._porosity(d); infl = H.ad_inflation(d, cfg); rows = []; rng = np.random.default_rng(seed)
    for h in sorted(np.unique(d.split)):
        te, cal = H.fold_masks(d, h); sS, sQ = H.scales(d, cfg, cal, infl)
        eS_all = (d.yS - d.muS) / sS; eQ_all = (d.yQ - d.muQ) / sQ
        for k in range(P.N_BINS):
            ti = np.flatnonzero(te & (b == k)); pool = cal & (b == k)
            if len(ti) == 0: continue
            eS, eQ = eS_all[pool], eQ_all[pool][rng.permutation(int(pool.sum()))]
            Sreal = d.muS[ti, None] + sS[ti, None] * eS[None, :]; Qreal = d.muQ[ti, None] + sQ[ti, None] * eQ[None, :]
            for tid, s_star, k_star in targets:
                Pp = ((Sreal >= s_star) & (Qreal >= math.log(k_star))).mean(1); O = ((d.yS[ti] >= s_star) & (d.yQ[ti] >= math.log(k_star))).astype(float)
                rows.append(pd.DataFrame({"fold": h, "row": d.orig[ti], "study": d.study[ti], "target_id": tid, "P": Pp, "O": O}))
    return pd.concat(rows, ignore_index=True)


def summarise(df):
    m = H.metrics(df["P"].to_numpy(), df["O"].to_numpy()); ci = H.cluster_bootstrap(df, NB, 20260921)
    acc = H.band_stats(df, 0.90, 1.0000001, NB, 20260921)
    return dict(metrics=m, slope_ci=ci["slope"], accept_rate_ci=ci["accept_rate"], accept_gap_ci=ci["accept_gap"], accept_band=acc,
                criteria=dict(C1a=bool(ci["slope"]["lo95"] <= 1.2 and ci["slope"]["hi95"] >= 0.8), C1b=bool(ci["accept_rate"]["lower5"] >= 0.85),
                              C1c=bool(abs(m["accept_mean_predicted"] - m["accept_observed_rate"]) <= 0.05)))


d0 = load("P_check"); TARGETS = H.target_grid(d0, None); json.dump(TARGETS, open(OUT / "targets.json", "w"))
res = {}
for name in sys.argv[4:]:
    d = load(name); cfg = H.Config()
    pp, _ = P.crossfit_pairs(d, cfg, TARGETS); pp = pp.assign(row=d.orig[pp["row"].to_numpy()])[["fold", "row", "study", "target_id", "P", "O"]]
    pp.to_csv(OUT / f"pairs_{name}_paired.csv", index=False); res[name] = dict(n_records=int(len(d.orig)), n_studies=int(len(set(d.study))), paired=summarise(pp), indep=[])
    for s in range(N_PERM):
        pi = pairs_perm(d, cfg, TARGETS, 777 + s); pi.to_csv(OUT / f"pairs_{name}_indep{s}.csv", index=False); res[name]["indep"].append(summarise(pi))
    m = res[name]["paired"]["metrics"]; mi = [x["metrics"] for x in res[name]["indep"]]
    print(f"[{name}] n {len(d.orig)} | paired slope {m['cal_slope']:.3f} brier {m['brier']:.4f} accept n {m['n_accept']} obs {m['accept_observed_rate']:.3f} pred {m['accept_mean_predicted']:.3f}"
          f" | indep slope {np.mean([x['cal_slope'] for x in mi]):.3f} brier {np.mean([x['brier'] for x in mi]):.4f} accept obs {np.mean([x['accept_observed_rate'] for x in mi]):.3f}", flush=True)
    json.dump(res, open(OUT / "audit_summary.json", "w"), indent=1, default=float)
print("AUDIT DONE", flush=True)
