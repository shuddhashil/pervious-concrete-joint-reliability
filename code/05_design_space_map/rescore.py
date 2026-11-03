"""Step 1b (v3): one pool for the map, replacing the old pool build + run_stage_a + run_step2 (2026-10-22).
Pool = inputs/old_pool_designs.npz (Z, C, fam, source; source renamed "old_"+source) + data_v2/tuned_raw.npz (source "six"). Drops rows with C < 230 kg/m3 (the engine's cement floor, config walk floor),
de-duplicates (family, cement to 1e-6, Z to 1e-9), screens with the pool's own screen (v2_core.screen), then stage A (nominal blend, N=4096, bank 20261004+3) and the 11-blend cheap rules on the survivors.
MAP_SUBSAMPLE=n keeps a random n of the non-"six" rows (default_rng(0)) plus every "six" row (smoke tests only).
Writes data_v2/stageA_v2.npz (every v2_core.ARR key + source + round) and logs_v3/rescore.json. GPU, one process; v2_core.Gpu is the heat guard."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_core import *

T0 = time.time(); L3 = os.path.join(SM, "logs_v3"); os.makedirs(L3, exist_ok=True); LOGF = open(os.path.join(L3, "rescore.log"), "a")


def log(*a):
    s = " ".join(str(v) for v in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()


cfg, H, ctx = get_ctx(); tg = Gpu(log); grid = Grid(S_GRID, K_GRID)
A = np.load(os.path.join(SM, "inputs", "old_pool_designs.npz"), allow_pickle=False); Zo, Co, Fo = A["Z"], A["C"], A["fam"].astype(int); So = np.array(["old_" + str(s) for s in A["source"]], dtype="U12")
B = np.load(os.path.join(D2, "tuned_raw.npz"), allow_pickle=False); Zs, Cs, Fs = B["Z"], B["C"], B["fam"].astype(int); Ss = np.full(len(Cs), "six", dtype="U12")
Z = np.concatenate([Zo, Zs]).astype(float); C = np.concatenate([Co, Cs]).astype(float); fam = np.concatenate([Fo, Fs]); src = np.concatenate([So, Ss])
res = dict(n_old=int(len(Co)), n_six=int(len(Cs)), n_in=int(len(C)))
floor = 230.0; ok = C >= floor; res["dropped_below_floor"] = int((~ok).sum()); res["below_floor_by_source"] = {s: int(((src == s) & ~ok).sum()) for s in sorted(set(src.tolist()))}
Z, C, fam, src = Z[ok], C[ok], fam[ok], src[ok]

seen = {}; idx = []
for i in range(len(C)):
    k = (int(fam[i]), round(float(C[i]), 6), tuple(round(float(z), 9) for z in Z[i]))
    if k in seen: continue
    seen[k] = i; idx.append(i)
res["duplicates"] = int(len(C) - len(idx)); Z, C, fam, src = Z[idx], C[idx], fam[idx], src[idx]
res["n_pool"] = int(len(C)); res["by_source_pool"] = {s: int((src == s).sum()) for s in sorted(set(src.tolist()))}

sub = os.environ.get("MAP_SUBSAMPLE")
if sub:
    n = int(sub); rng = np.random.default_rng(0); six = np.flatnonzero(src == "six"); rest = np.setdiff1d(np.arange(len(C)), six)
    pick = rest if len(rest) <= n else rng.choice(rest, n, replace=False); m = np.zeros(len(C), bool); m[six] = True; m[pick] = True
    res["subsample"] = dict(requested=n, kept_six=int(len(six)), kept_other=int(len(pick))); Z, C, fam, src = Z[m], C[m], fam[m], src[m]; res["n_pool_after_subsample"] = int(len(C))

keep, cnt = screen(ctx, Z, C, fam, tg); res["screen"] = cnt; res["n_after_screen"] = int(keep.sum()); res["kept"] = int(keep.sum())
Zk, Ck, Fk, Sk = Z[keep], C[keep], fam[keep], src[keep]
t0 = time.time(); counts, supf, csm, km = stage_a(ctx, Zk, Ck, grid, 4096, 20261004 + 3, tg=tg); res["stage_a_s"] = round(time.time() - t0, 1)
t0 = time.time(); gate11, fam11, hmin11 = rules_11(ctx, Zk, Ck, Fk); res["rules11_s"] = round(time.time() - t0, 1)
res["by_source_final"] = {s: int((Sk == s).sum()) for s in sorted(set(Sk.tolist()))}
res["n_nominal_support_ok"] = int((supf >= 1 - 1e-12).sum()); res["n_all_rules"] = int(((supf >= 1 - 1e-12) & gate11 & fam11).sum())

out = dict(counts=counts, supf=supf, csm=csm, km=km, gate11=gate11, fam11=fam11, hmin11=hmin11, Z=Zk, C=Ck, fam=Fk, stratum=np.zeros(len(Ck), np.int64), typ=np.zeros(len(Ck), np.int64), source=Sk.astype("U12"), round=np.zeros(len(Ck), np.int16))
np.savez_compressed(os.path.join(D2, "stageA_v2.npz"), **out)
res.update(gpu_peak=tg.peak, pauses=tg.pauses, total_s=round(time.time() - T0, 1)); json.dump(res, open(os.path.join(L3, "rescore.json"), "w"), indent=1); log(json.dumps(res, indent=1)); tg.stop()
