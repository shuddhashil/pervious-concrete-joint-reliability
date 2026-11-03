"""Stage A for the anchored layer only; appended to the stage-A arrays of layer 1 (data/sobol_only/stageA.npz) -> data/stageA.npz (all designs)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multitarget import *
t0 = time.time(); cfg, H, ctx = get_ctx(); tg = TempGuard(); grid = Grid(S_GRID, K_GRID); old = np.load(os.path.join(SM, "data", "sobol_only", "stageA.npz")); new = np.load(os.path.join(SM, "data", "pool_anchor_raw.npz")); Z, Cv, fam = new["Z"], new["C"], new["fam"]
t1 = time.time(); counts, supf, csm, km = stage_a(ctx, Z, Cv, grid, 4096, 20261004 + 3, tg=tg); t_a = time.time() - t1; t1 = time.time(); gate11, fam11, hmin11 = rules_11(ctx, Z, Cv, fam); t_r = time.time() - t1
cat = lambda k, v: np.concatenate([old[k], v]); n0 = len(old["C"])
np.savez_compressed(os.path.join(SM, "data", "stageA.npz"), counts=cat("counts", counts), supf=cat("supf", supf), csm=cat("csm", csm), km=cat("km", km), gate11=cat("gate11", gate11), fam11=cat("fam11", fam11), hmin11=cat("hmin11", hmin11), Z=cat("Z", Z), C=cat("C", Cv), fam=cat("fam", fam), stratum=cat("stratum", new["stratum"]), typ=cat("typ", new["typ"]))
tm = dict(n_layer1=int(n0), n_layer2=int(len(Cv)), stage_a_s=round(t_a, 1), rules11_s=round(t_r, 1), n_nominal_support_ok_layer2=int((supf >= 1 - 1e-12).sum()), n_pool_eligible_layer2=int(((supf >= 1 - 1e-12) & gate11 & fam11).sum()), gpu_peak=tg.peak, pauses=tg.pauses)
json.dump(tm, open(os.path.join(SM, "logs", "stageA_extra_timing.json"), "w"), indent=1); print(json.dumps(tm))
