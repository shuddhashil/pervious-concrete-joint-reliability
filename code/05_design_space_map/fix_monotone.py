"""Monotonic-notch fix (v2, main thread, 2026-10-04). A design that supports a target (S, K) supports every easier target (lower S and lower K) because the pass conditions are nested. The v2 classification only
evaluates, for each target, the 10 best designs of its stage-A (nominal) ranking, so a design that supports a harder cell can be missing from the shortlist of an easier cell (the notch at 35 MPa / 1.47 mm/s).
Fix = evaluate, with the SAME acceptance code (StageB: min over 11 blends of R_LCB95, N=4096, stricter tie rule), the designs that support harder cells at the easier cells that violate monotonicity, then reclassify from the cache.
Nothing is closed by hand: a cell becomes supported only if a real design passes there; violations that remain (paste floor, gates) are reported. GPU, seconds. Writes logs_v2/monotone_fix.json; updates data_v2/state_final.pkl and stageB_v2_cache.pkl."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_core import *
log = lambda *a: (print(*a, flush=True), open(os.path.join(L2, "monotone_fix.log"), "a").write(" ".join(str(x) for x in a) + "\n"))
cfg, H, ctx = get_ctx(); tg = Gpu(log); grid = Grid(S_GRID, K_GRID); P = load_pool(); floors = paste_floors(ctx, grid); sb = StageB(ctx, grid, tg=tg); load_cache(sb)
sf = os.path.join(D2, "state_final.pkl"); st = pickle.load(open(sf, "rb")); nS, nK = grid.nS, grid.nK
best, bd, cands = best_from_cache(sb, grid.nT); elig = eligible(P); dn = design_nom_fn(P["Z"], P["C"]); out = dict(iterations=[])


def violations(best):
    F = np.nan_to_num(best, nan=-1.0).reshape(nS, nK) >= THR90; G = F.copy()
    for a in range(nS - 2, -1, -1): G[a] |= G[a + 1]
    for b in range(nK - 2, -1, -1): G[:, b] |= G[:, b + 1]
    return G & ~F, F


V, F = violations(best); out["supported_before"] = int(F.sum()); out["violations_before"] = [(float(S_GRID[a]), float(K_GRID[b])) for a, b in np.argwhere(V)]; log("supported before", int(F.sum()), "violations", out["violations_before"])
for it in range(6):
    V, F = violations(best); cells = np.argwhere(V)
    if not len(cells): log("no violation left"); break
    pairs = {}
    for a, b in cells:
        t = grid.tid(a, b); fl = floors[t]; ds = {int(bd[grid.tid(a2, b2)]) for a2 in range(a, nS) for b2 in range(b, nK) if F[a2, b2]}
        for d in ds:
            if d < 0 or not elig[d] or (d, t) in sb.cache: continue
            if not (np.isnan(fl) or P["hmin11"][d] >= fl - 1e-12): continue          # paste floor of THIS target (looser at easier targets; if it blocks, the violation stays and is reported)
            pairs.setdefault(d, []).append(t)
    n_pairs = sum(len(v) for v in pairs.values()); log(f"iteration {it + 1}: {len(cells)} violating cells, {len(pairs)} designs, {n_pairs} new pairs")
    if not pairs: out["iterations"].append(dict(it=it + 1, violating=len(cells), new_pairs=0)); break
    t0 = time.time(); sb.eval_pairs(pairs, dn); best, bd, cands = best_from_cache(sb, grid.nT); V2, F2 = violations(best)
    out["iterations"].append(dict(it=it + 1, violating=len(cells), designs=len(pairs), new_pairs=n_pairs, seconds=round(time.time() - t0, 1), supported_after=int(F2.sum()), violations_after=int(V2.sum())))
    log(f"  evaluated in {time.time() - t0:.1f}s: supported {int(F.sum())} -> {int(F2.sum())}, violations {int(V.sum())} -> {int(V2.sum())}")
V, F = violations(best); out.update(supported_after=int(F.sum()), violations_after=[(float(S_GRID[a]), float(K_GRID[b])) for a, b in np.argwhere(V)], gpu_peak=tg.peak, pauses=tg.pauses)
st["best"], st["bd"] = best, bd; pickle.dump(st, open(sf, "wb")); save_cache(sb); json.dump(out, open(os.path.join(L2, "monotone_fix.json"), "w"), indent=1); log("FIX DONE supported", int(F.sum()), "remaining violations", len(out["violations_after"]), "GPU peak", tg.peak); tg.stop()
