"""Shortlist + acceptance classification. For every target the designs are ranked by the stage-A Wilson bound among designs that pass the cheap rules
(gates and family rule at all 11 blends, nominal support, paste floor of THAT target at all 11 blends if the target has one, stage-A bound >= SCREEN). They are then evaluated in rank order with the full acceptance (StageB: min over 11 blends of R_LCB95 at N=4096)
until 10 valid designs have been evaluated (a design whose support fails at some blend is replaced by the next one) or one design reaches 0.952 (supported at all three levels: nothing better is needed, same classification as evaluating all ten).
A target is supported at R* when the best evaluated valid design has min-over-blends R_LCB95 >= R* + 0.002 (the walk's margin)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multitarget import *

SCREEN = 0.70; TOPN = 40; RESOLVE = 0.95 + M_WALK


def paste_floors(ctx, grid):
    """h_min per target (NaN = no recorded unit meets the target: paste check skipped, as walk.Ctx)."""
    f = np.full(grid.nT, np.nan)
    for a, S in enumerate(grid.S):
        for b, K in enumerate(grid.K):
            h = PS.paste_floor(ctx.data, float(S), float(K))["h_min"]
            if h is not None: f[grid.tid(a, b)] = h
    return f


def rankings(wilA, elig_base, hmin11, floors, subset=None):
    """top-TOPN eligible designs per target (indices (TOPN,nT), -1 = none). wilA (D,nT) float tensor on GPU. Deterministic: ties broken by the smaller design index."""
    D, nT = wilA.shape; idx = torch.arange(D, device=wilA.device, dtype=torch.float64)[:, None]; eb = torch.as_tensor(elig_base if subset is None else (elig_base & subset), device=wilA.device)
    hm = torch.as_tensor(hmin11, device=wilA.device)[:, None]; fl = torch.as_tensor(floors, device=wilA.device)[None, :]
    el = eb[:, None] & (torch.isnan(fl) | (hm >= fl - 1e-12)) & (wilA >= SCREEN); sc = torch.where(el, wilA - 1e-12 * idx / D, torch.full_like(wilA, -1.0)); v, i = sc.topk(min(TOPN, D), dim=0)
    return torch.where(v > 0, i, torch.full_like(i, -1)).cpu().numpy()


def classify(sb, rank, design_nom, K=TOPK, log=print, tag="", only=None, noresolve=False):
    """Adaptive evaluation described in the module doc. sb = StageB (its cache is shared between calls). Returns dict with arrays over targets:
    best_LB (nan if no valid evaluated design), best_design (-1), n_eval, resolved, and the list of evaluated designs per target."""
    if only is not None:                                                   # restrict to some targets (others get an empty ranking)
        rank = rank.copy(); keep = np.zeros(rank.shape[1], bool); keep[list(only)] = True; rank[:, ~keep] = -1
    nT = rank.shape[1]; ptr = np.zeros(nT, int); taken = [[] for _ in range(nT)]; resolved = np.zeros(nT, bool); rnd = 0
    invalid = lambda d: (sb.supall.get(d) is not None) and sb.supall[d] < 1 - 1e-12
    valid = lambda d: sb.supall.get(d) is not None and sb.supall[d] >= 1 - 1e-12
    while True:
        pairs = {}; n_new = 0; progress = False; took = []
        for t in range(nT):
            if resolved[t]: continue
            nv = sum(1 for d in taken[t] if not invalid(d))
            while nv < K and ptr[t] < rank.shape[0] and rank[ptr[t], t] >= 0:
                d = int(rank[ptr[t], t]); ptr[t] += 1
                if invalid(d): continue
                taken[t].append(d); nv += 1; progress = True; took.append((d, t))
                if (d, t) not in sb.cache: pairs.setdefault(d, []).append(t); n_new += 1
                break                                                          # one new design per target per round (early exit once resolved)
        if not progress: break
        t0 = time.time()
        if pairs: sb.eval_pairs(pairs, design_nom)
        rnd += 1
        for d, t in took:
            if (not noresolve) and valid(d) and np.nanmin(sb.cache[(d, t)]) >= RESOLVE: resolved[t] = True
        log(f"[classify{tag}] round {rnd}: {n_new} new pairs over {len(pairs)} designs, resolved {int(resolved.sum())}/{nT}, {time.time() - t0:.1f}s")
    best = np.full(nT, np.nan); bd = np.full(nT, -1); ne = np.zeros(nT, int); cands = {}
    for t in range(nT):
        c = [(float(np.nanmin(sb.cache[(d, t)])), d) for d in taken[t] if valid(d) and (d, t) in sb.cache]; cands[t] = sorted(c, reverse=True); ne[t] = len([d for d in taken[t] if not invalid(d)])
        if c: best[t], bd[t] = max(c)
    return dict(best_LB=best, best_design=bd, n_eval=ne, resolved=resolved, candidates=cands, rounds=rnd)


def flags(best, margin=M_WALK):
    return {R: np.nan_to_num(best, nan=-1.0) >= R + margin for R in RS}
