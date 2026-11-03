"""v2 final prep (GPU, seconds): from the final pool and the stage-B cache take, for every cell within 6 below and 3 above the top supported cell of each K* column, the 3 best valid designs (value, design id, 11-blend values, nominal design dict) for the strict certificates.
Output data_v2/cands_final.pkl."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_core import *
grid = Grid(S_GRID, K_GRID); P = load_pool(); c = pickle.load(open(os.path.join(D2, "stageB_v2_cache.pkl"), "rb"))
class _S: pass
sb = _S(); sb.cache, sb.supall = c["cache"], c["supall"]; best, bd, cands = best_from_cache(sb, grid.nT); top = top_cells(best, grid); want = set()
for b in range(grid.nK):
    for a in range(max(0, top[b] - 6), min(grid.nS, top[b] + 4) if top[b] >= 0 else 3): want.add(grid.tid(a, b))
ds = sorted({d for t in want for lb, d in cands[t][:3]}); nds = dict(zip(ds, nom_dicts(decode(T(P["Z"][ds]), T(P["C"][ds]))))) if ds else {}
out = {t: [(lb, d, sb.cache[(d, t)].tolist(), nds[d]) for lb, d in cands[t][:3]] for t in sorted(want)}
pickle.dump(dict(cands=out, best=best, bd=bd), open(os.path.join(D2, "cands_final.pkl"), "wb")); print("cells", len(out), "designs", len(ds))
