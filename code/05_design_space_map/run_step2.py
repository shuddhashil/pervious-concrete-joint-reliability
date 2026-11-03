"""Step 2 (v2): add the tuned designs (harvest_tuned.py) to the pool with source="tuned", same screening, stage A on the survivors, re-classification at all 920 targets at R* = 0.90 (v1 cache reused; same banks and seeds).
Writes data_v2/stageA_v2.npz, data_v2/stageB_v2_cache.pkl, data_v2/state_step2.pkl, logs_v2/step2.json. GPU, one process."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_core import *
T0 = time.time(); cfg, H, ctx = get_ctx(); log = lambda *a: (print(*a, flush=True), open(os.path.join(L2, "step2.log"), "a").write(" ".join(str(x) for x in a) + "\n")); tg = Gpu(log); grid = Grid(S_GRID, K_GRID)
P = load_pool(); n_v1 = len(P["C"]); tun = np.load(os.path.join(D2, "tuned_raw.npz")); res = dict(n_v1_pool=int(n_v1), n_tuned_harvested=int(len(tun["C"])))
t0 = time.time(); n_add, cnt, ids, keep = append_designs(ctx, grid, P, tun["Z"], tun["C"], tun["fam"], "tuned", 1, tg=tg, log=log); res.update(screen=cnt, n_added=int(n_add), screen_and_stageA_s=round(time.time() - t0, 1))
P["round"][n_v1:] = 0; save_pool(P); log("added", n_add, json.dumps(cnt))
# which harvested designs survived, by kind
kinds = tun["kinds"][keep]; res["survivors_by_kind"] = {}
for kd in sorted({k for x in tun["kinds"] for k in x.split("|")}):
    tot = int(sum(kd in x.split("|") for x in tun["kinds"])); sv = int(sum(kd in x.split("|") for x in kinds)); res["survivors_by_kind"][kd] = dict(harvested=tot, survived=sv)
floors = paste_floors(ctx, grid); sb = StageB(ctx, grid, tg=tg); load_cache(sb, os.path.join(SM, "data", "stageB_cache.pkl")); n_pairs0 = len(sb.cache)
B1 = pickle.load(open(os.path.join(SM, "data", "stageB.pkl"), "rb")); best1 = np.nan_to_num(B1["best_LB"], nan=-1.0)
t0 = time.time(); R = classify_pool(ctx, grid, P, sb, floors, tg, log=log, tag=" step2"); res["classify_s"] = round(time.time() - t0, 1); best2 = np.nan_to_num(R["best"], nan=-1.0)
f1 = best1 >= THR90; f2 = best2 >= THR90; res.update(v1_supported_090=int(f1.sum()), after_step2_supported_090=int(f2.sum()), gained=int((f2 & ~f1).sum()), lost_vs_v1=int((f1 & ~f2).sum()), new_pairs_evaluated=len(sb.cache) - n_pairs0,
    edge_v1_GPU_rule=edge_summary(np.where(f1, best1, -1), grid), edge_after_step2=edge_summary(R["best"], grid), best_design_is_tuned_cells=int(((R["bd"] >= n_v1) & f2).sum()), best_design_is_tuned_among_gained=int(((R["bd"] >= n_v1) & f2 & ~f1).sum()))
res["gpu_peak"] = tg.peak; res["pauses"] = tg.pauses; res["total_s"] = round(time.time() - T0, 1)
pickle.dump(dict(best=R["best"], bd=R["bd"], floors=floors, n_v1=n_v1), open(os.path.join(D2, "state_step2.pkl"), "wb")); save_cache(sb)
json.dump(res, open(os.path.join(L2, "step2.json"), "w"), indent=1); log(json.dumps(res, indent=1)); tg.stop()
