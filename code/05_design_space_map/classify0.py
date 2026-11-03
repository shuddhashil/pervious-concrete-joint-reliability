"""Step 1c (v3): first classification of the whole pool at R* = 0.90, replacing run_step2's classification (2026-10-22).
ctx + grid + load_pool(); paste floors; StageB with an EMPTY cache (never the old engine's cache); classify_pool (rank by stage-A Wilson bound, adaptive stage-B acceptance, forced full edge evaluation).
Writes data_v2/state_step2.pkl (best, bd, floors, n_v1) + data_v2/stageB_v2_cache.pkl and logs_v3/classify0.json (supported count at 0.90, edge summary). GPU, one process."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_core import *

T0 = time.time(); L3 = os.path.join(SM, "logs_v3"); os.makedirs(L3, exist_ok=True); LOGF = open(os.path.join(L3, "classify0.log"), "a")


def log(*a):
    s = " ".join(str(v) for v in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()


cfg, H, ctx = get_ctx(); tg = Gpu(log); grid = Grid(S_GRID, K_GRID); P = load_pool(); floors = paste_floors(ctx, grid)
sb = StageB(ctx, grid, tg=tg)                                                             # empty cache by construction
t0 = time.time(); R = classify_pool(ctx, grid, P, sb, floors, tg, log=log, tag=" v3"); sec = time.time() - t0
best = R["best"]; n_sup = int((np.nan_to_num(best, nan=-1.0) >= THR90).sum())
pickle.dump(dict(best=best, bd=R["bd"], floors=floors, n_v1=len(P["C"])), open(os.path.join(D2, "state_step2.pkl"), "wb")); save_cache(sb)
js = dict(n_pool=int(len(P["C"])), supported_090=n_sup, edge=edge_summary(best, grid), n_pairs=len(sb.cache), n_designs_evaluated=len({d for d, t in sb.cache}), row_evals=sb.n_row_evals,
          seconds=dict(classify=round(sec, 1), total=round(time.time() - T0, 1)), gpu_peak=tg.peak, pauses=tg.pauses)
json.dump(js, open(os.path.join(L3, "classify0.json"), "w"), indent=1); log(json.dumps(js, indent=1)); tg.stop()
