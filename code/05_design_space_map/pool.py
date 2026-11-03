"""Step 1: target-independent pool of candidate designs. Space-filling (scrambled Sobol, fixed seeds) over the engine's decision variables and bounds,
three families (the frozen blend projection), three strata (whole box, high-strength corner, high-permeability corner) so the pool reaches the edges, screened at the nominal blend by the engine's own gate,
family rule and support check (GPU code of the stack). Output data/pool.npz."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sm_common import *
from scipy.stats import qmc

LOGN = int(os.environ.get("POOL_LOG2", "15"))
# strata: unit-cube ranges for [C, r, A, P, S, phi] (cement uses its own bounds in kg/m3; phi stays inside the 20-30 % window); None = whole range
CEM = (200.0, float(BND["C"][1]))
zphi = lambda v: (v - BND["phi"][0]) / (BND["phi"][1] - BND["phi"][0])
STRATA = {
    "box": dict(C=CEM, r=(0, 1), A=(0, 1), P=(0, 1), S=(0, 1), phi=(zphi(PHI_WIN[0]), zphi(PHI_WIN[1]))),
    "strength": dict(C=(400.0, CEM[1]), r=(0, 0.45), A=(0, 1), P=(0, 1), S=(0, 1), phi=(zphi(PHI_WIN[0]), zphi(22.5))),
    "perm": dict(C=(200.0, 400.0), r=(0, 1), A=(0.5, 1), P=(0, 1), S=(0, 0.3), phi=(zphi(26.0), zphi(PHI_WIN[1]))),
}


def sample(ctx, fam, stratum, typ, n):
    st = STRATA[stratum]; ty = {"dir": 0, "mix": 1}[typ]; seed = 1000 * fam + 100 * list(STRATA).index(stratum) + 10 * ty + 20261004
    d = 12 if typ == "dir" else 9; U = qmc.Sobol(d, scramble=True, seed=seed).random(n)
    lo = lambda k: st[k][0]; sc = lambda u, k: st[k][0] + u * (st[k][1] - st[k][0])
    Cv = sc(U[:, 0], "C"); Z = np.zeros((n, 11)); Z[:, 0] = sc(U[:, 1], "r"); Z[:, 1] = sc(U[:, 2], "A"); Z[:, 2] = sc(U[:, 3], "P"); Z[:, 3] = sc(U[:, 4], "S"); Z[:, 4] = sc(U[:, 5], "phi")
    if typ == "dir":
        e = -np.log(np.clip(U[:, 6:12], 1e-9, 1)); g = e / e.sum(1, keepdims=True)
    else:
        bl = np.flatnonzero(np.array(ctx.cfg["families"]["labels"]) == fam); Bl = ctx.data.Bl; a = bl[np.minimum((U[:, 7] * len(bl)).astype(int), len(bl) - 1)]; b = bl[np.minimum((U[:, 8] * len(bl)).astype(int), len(bl) - 1)]
        w = U[:, 6:7]; g = w * Bl[a] + (1 - w) * Bl[b]
    Z[:, 5:] = g; return Z, Cv


def build(ctx, tg, log=print):
    ctx.eng.set_lowprec(False); bank, u = ctx.bank(256, 20261004); out = dict(Z=[], C=[], fam=[], stratum=[], typ=[]); stats = []
    for fam in (0, 1, 2):
        for stratum in STRATA:
            for typ in ("dir", "mix"):
                Z, Cv = sample(ctx, fam, stratum, typ, 2 ** LOGN); Zt = T(Z)
                with torch.no_grad():
                    g2 = ctx.fam.project(decode(Zt, T(Cv))["g"], fam); Zt = torch.cat([Zt[:, :5], g2], 1); nom = decode(Zt, T(Cv)); pen = torch.nan_to_num(ctx.gate.penalties(nom)["penalty"], posinf=1e6); fok, fd, _ = ctx.fam.check(nom["g"], fam)
                    ok = (pen <= 0) & fok; idx = ok.nonzero()[:, 0]; kept = []
                    for a in range(0, len(idx), 1500):
                        tg(); ii = idx[a:a + 1500]; st = ctx.eng.stats_t({k: v[ii] for k, v in nom.items()}, bank, None, None, u=u); kept.append(ii[st["support_fraction"] >= 1 - 1e-12])
                    kept = torch.cat(kept).cpu().numpy() if kept else np.zeros(0, int)
                out["Z"].append(Zt.cpu().numpy()[kept]); out["C"].append(Cv[kept]); out["fam"].append(np.full(len(kept), fam)); out["stratum"].append(np.full(len(kept), list(STRATA).index(stratum))); out["typ"].append(np.full(len(kept), 0 if typ == "dir" else 1))
                stats.append(dict(fam=fam, stratum=stratum, typ=typ, raw=2 ** LOGN, gate_family_ok=int(ok.sum()), kept=int(len(kept)))); log(stats[-1])
    return {k: np.concatenate(v) for k, v in out.items()}, stats


if __name__ == "__main__":
    t0 = time.time(); cfg, H, ctx = get_ctx(); tg = TempGuard()
    pool, stats = build(ctx, tg); np.savez(os.path.join(SM, "data", "pool_raw.npz"), **pool); json.dump(dict(stats=stats, seconds=time.time() - t0, gpu_peak=tg.peak, pauses=tg.pauses), open(os.path.join(SM, "logs", "pool_build.json"), "w"), indent=1)
    print("pool size", len(pool["C"]), "seconds", round(time.time() - t0), "peak", tg.peak)
