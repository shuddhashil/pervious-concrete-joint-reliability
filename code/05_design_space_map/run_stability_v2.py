"""Stability (v2): rebuild the 0.90 region (stage-B acceptance at 0.902, v1 `classify`, cache shared) from random 25 / 50 / 75 % subsets of the FINAL pool, 3 draws each (same rng rule as v1), compare with the full pool. GPU. Writes stability_v2.json."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_core import *
T0 = time.time(); cfg, H, ctx = get_ctx(); log = lambda *a: (print(*a, flush=True), open(os.path.join(L2, "stability.log"), "a").write(" ".join(str(x) for x in a) + "\n")); tg = Gpu(log); grid = Grid(S_GRID, K_GRID); P = load_pool(); sb = StageB(ctx, grid, tg=tg); load_cache(sb)
st = pickle.load(open(os.path.join(D2, "state_final.pkl"), "rb")); floors = st["floors"]; full = np.nan_to_num(st["best"], nan=-1.0) >= THR90; D = len(P["C"]); elig = eligible(P)
wilA = wilson(torch.as_tensor(P["counts"], dtype=torch.float64, device=DEV), 4096.0); dn = design_nom_fn(P["Z"], P["C"])
stab = dict(rule="rebuild the 0.90 region (stage-B acceptance at 0.902, GPU rule) from a random subset of the final pool; compare with the full final pool (same rule); targets that change class", n_targets=int(grid.nT), n_designs_full_pool=int(D), n_supported_full=int(full.sum()), draws=[])
for frac in (0.75, 0.50, 0.25):
    for k in range(3):
        rng = np.random.default_rng(20261004 + int(frac * 100) * 10 + k); sub = np.zeros(D, bool); sub[rng.choice(D, int(round(frac * D)), replace=False)] = True
        rk = rankings(wilA, elig, P["hmin11"], floors, subset=sub); Rs = classify(sb, rk, dn, log=log, tag=f" sub{frac}/{k}"); f2 = np.nan_to_num(Rs["best_LB"], nan=-1) >= THR90; ch = f2 != full
        stab["draws"].append(dict(fraction=frac, draw=k, n_designs=int(sub.sum()), n_refined_or_tuned_in_subset=int((sub & (P["source"] != "sobol") & (P["source"] != "anchor")).sum()), n_supported=int(f2.sum()), n_supported_full=int(full.sum()), n_changed=int(ch.sum()), lost=int((full & ~f2).sum()), gained=int((~full & f2).sum()),
                                  changed_targets=[(float(grid.S[t // grid.nK]), float(grid.K[t % grid.nK])) for t in np.flatnonzero(ch)][:80])); log(f"[stability] {frac} draw {k}: changed {int(ch.sum())} of {grid.nT}"); torch.cuda.empty_cache(); stab["seconds"] = round(time.time() - T0, 1); stab["complete"] = False; json.dump(stab, open(os.path.join(SM, "stability_v2.json"), "w"), indent=1)
stab["complete"] = True; stab["seconds"] = round(time.time() - T0, 1); stab["gpu_peak"] = tg.peak; stab["pauses"] = tg.pauses; json.dump(stab, open(os.path.join(SM, "stability_v2.json"), "w"), indent=1); save_cache(sb); log("STABILITY DONE", stab["seconds"]); tg.stop()
