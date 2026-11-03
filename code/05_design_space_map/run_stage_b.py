"""Stage B + monotonic check + stability check. GPU, one process. Reads data/stageA.npz; writes data/stageB.pkl, logs/stageB_timing.json, pool_coverage.json, stability.json, monotonic.json."""
import sys, os, pickle; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify import *

DATA = os.path.join(SM, "data"); LOGS = os.path.join(SM, "logs")


def design_nom_fn(Z, Cv):
    return lambda ids: (Z[ids], Cv[ids])


def edge_cells(best, grid, margin=M_WALK):
    """per K column and level: highest stage-B-supported S index (-1 = none) -> list of target ids to force into the full evaluation: that cell, the 2 below, the 2 above."""
    nS, nK = grid.nS, grid.nK; L = best.reshape(nS, nK); out = set()
    for R in RS:
        sup = np.nan_to_num(L, nan=-1.0) >= R + margin
        for b in range(nK):
            ix = np.flatnonzero(sup[:, b]); a0 = int(ix.max()) if len(ix) else -1
            for a in range(a0 - 2, a0 + 3):
                if 0 <= a < nS: out.add(grid.tid(a, b))
    return sorted(out)


if __name__ == "__main__":
    T0 = time.time(); tm = {}; cfg, H, ctx = get_ctx(); tg = TempGuard(); grid = Grid(S_GRID, K_GRID); A = np.load(os.path.join(DATA, "stageA.npz")); Z, Cv, fam = A["Z"], A["C"], A["fam"]
    D = len(Cv); ok_nom = A["supf"] >= 1 - 1e-12; elig_base = ok_nom & A["gate11"] & A["fam11"]; wilA = wilson(torch.as_tensor(A["counts"], dtype=torch.float64, device=DEV), 4096.0)
    floors = paste_floors(ctx, grid); rank = rankings(wilA, elig_base, A["hmin11"], floors); tm["rank_s"] = round(time.time() - T0, 1)
    log = lambda *a: (print(*a, flush=True), open(os.path.join(LOGS, "stageB.log"), "a").write(" ".join(str(x) for x in a) + "\n"))
    sb = StageB(ctx, grid, tg=tg); dn = design_nom_fn(Z, Cv)
    t1 = time.time(); R1 = classify(sb, rank, dn, log=log, tag=" main"); tm["main_s"] = round(time.time() - t1, 1)
    ec = edge_cells(R1["best_LB"], grid); t1 = time.time(); R2 = classify(sb, rank, dn, log=log, tag=" edge", only=ec, noresolve=True); tm["edge_full_s"] = round(time.time() - t1, 1)
    best = R1["best_LB"].copy(); bdes = R1["best_design"].copy(); cands = dict(R1["candidates"])
    for t in ec: cands[t] = R2["candidates"][t]; best[t] = R2["best_LB"][t]; bdes[t] = R2["best_design"][t]
    ne = R1["n_eval"].copy()
    for t in ec: ne[t] = R2["n_eval"][t]
    L = best.reshape(grid.nS, grid.nK)
    # candidate pairs worth keeping for the certificates: top-3 per target
    ds = sorted({d for t in range(grid.nT) for lb, d in cands[t][:3]}); nds = dict(zip(ds, nom_dicts(decode(T(Z[ds]), T(Cv[ds]))))) if ds else {}
    keep = {t: [(lb, d, sb.cache[(d, t)].tolist(), nds[d]) for lb, d in cands[t][:3]] for t in range(grid.nT)}
    out = dict(best_LB=best, best_design=bdes, n_eval=ne, floors=floors, cands=keep, edge_forced=ec, n_pairs=len(sb.cache), n_designs_evaluated=len({d for d, t in sb.cache}), row_evals=sb.n_row_evals, stageB_seconds=sb.seconds)
    pickle.dump(out, open(os.path.join(DATA, "stageB.pkl"), "wb")); tm["stageB_gpu_s"] = round(sb.seconds, 1)
    for R in RS: tm[f"supported_{R}"] = int((np.nan_to_num(best, nan=-1) >= R + M_WALK).sum())
    tm.update(n_pairs=len(sb.cache), n_designs_evaluated=out["n_designs_evaluated"], row_evals=sb.n_row_evals); json.dump(tm, open(os.path.join(LOGS, "stageB_timing.json"), "w"), indent=1); log(json.dumps(tm))
    # ---- stability: random 25 / 50 / 75 % subsets of the pool (3 draws each), same rule, shared cache
    t1 = time.time(); full = {R: np.nan_to_num(best, nan=-1) >= R + M_WALK for R in RS}; stab = dict(rule="rebuild the 0.90 region (stage-B acceptance at 0.902) from a random subset of the pool; compare with the full pool; targets that change class", n_targets=int(grid.nT), draws=[])
    for frac in (0.25, 0.50, 0.75):
        for k in range(3):
            rng = np.random.default_rng(20261004 + int(frac * 100) * 10 + k); sub = np.zeros(D, bool); sub[rng.choice(D, int(round(frac * D)), replace=False)] = True
            rk = rankings(wilA, elig_base, A["hmin11"], floors, subset=sub); Rs = classify(sb, rk, dn, log=log, tag=f" sub{frac}/{k}"); f2 = np.nan_to_num(Rs["best_LB"], nan=-1) >= 0.90 + M_WALK; ch = f2 != full[0.90]
            stab["draws"].append(dict(fraction=frac, draw=k, n_designs=int(sub.sum()), n_supported=int(f2.sum()), n_supported_full=int(full[0.90].sum()), n_changed=int(ch.sum()), lost=int((full[0.90] & ~f2).sum()), gained=int((~full[0.90] & f2).sum()),
                                      changed_targets=[(float(grid.S[t // grid.nK]), float(grid.K[t % grid.nK])) for t in np.flatnonzero(ch)][:60]))
            log(f"[stability] {frac} draw {k}: changed {int(ch.sum())} of {grid.nT}")
    stab["seconds"] = round(time.time() - t1, 1); tm["stability_s"] = stab["seconds"]; json.dump(stab, open(os.path.join(SM, "stability.json"), "w"), indent=1)
    tm.update(gpu_peak=tg.peak, pauses=tg.pauses, total_s=round(time.time() - T0, 1), n_pairs_after_stability=len(sb.cache)); json.dump(tm, open(os.path.join(LOGS, "stageB_timing.json"), "w"), indent=1)
    pickle.dump(dict(cache=sb.cache, supall=sb.supall), open(os.path.join(DATA, "stageB_cache.pkl"), "wb")); log("STAGE B DONE", json.dumps(tm))
