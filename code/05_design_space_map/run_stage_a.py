"""Stage A: nominal blend, N = 4096 batches per design, pass counts for all 920 targets at once; cheap 11-blend rules; pool diagnostics."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multitarget import *
t0 = time.time(); cfg, H, ctx = get_ctx(); tg = TempGuard(); t_load = time.time() - t0
raw = np.load(os.path.join(SM, "data", "pool_raw.npz")); Z, Cv, fam, stratum = raw["Z"], raw["C"], raw["fam"], raw["stratum"]; D = len(Cv); grid = Grid(S_GRID, K_GRID)
t1 = time.time(); counts, supf, csm, km = stage_a(ctx, Z, Cv, grid, 4096, 20261004 + 3, tg=tg); t_a = time.time() - t1
t1 = time.time(); gate11, fam11, hmin11 = rules_11(ctx, Z, Cv, fam); t_r = time.time() - t1
ok_nom = supf >= 1 - 1e-12
np.savez_compressed(os.path.join(SM, "data", "stageA.npz"), counts=counts, supf=supf, csm=csm, km=km, gate11=gate11, fam11=fam11, hmin11=hmin11, Z=Z, C=Cv, fam=fam, stratum=stratum, typ=raw["typ"])
tm = dict(load_s=round(t_load, 1), stage_a_s=round(t_a, 1), rules11_s=round(t_r, 1), D_raw=int(D), n_nominal_support_ok_N4096=int(ok_nom.sum()), n_gate11_family11=int((gate11 & fam11).sum()), n_all=int((ok_nom & gate11 & fam11).sum()), gpu_peak=tg.peak, pauses=tg.pauses)
json.dump(tm, open(os.path.join(SM, "logs", "stageA_timing.json"), "w"), indent=1); print(json.dumps(tm, indent=1))
