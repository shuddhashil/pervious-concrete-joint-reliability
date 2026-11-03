"""Fast level solver (decision E-07). Same search as gpu_solver.solve_level_gpu, same output format, but it stops when it stops improving
and uses few noise scenarios while the search is still exploring: phase 1 on a small bank, phase 2 (re-scored) on a larger bank, each ending
when the best score has not improved for `patience` generations. The final ranking is on a FRESH bank, exactly as before. Search only: acceptance never uses this.
Run with the GPU python."""
import time
import numpy as np, torch
from . import corelib as C, gpu_solver as GS


def solve_level_fast(cement, seed, pool=None, P=100, max_gens=100, patience=25, min_gens=15, n_lo=128, n_hi=512, n_exact=12, rerank_n=1024, tol=1e-4):
    st = GS._state(); dev = st["eng"].dev; t0 = time.time(); D = 11; gen = torch.Generator(device=dev).manual_seed(int(seed))
    tt = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=torch.float64, device=dev)
    rnd = lambda *s: torch.rand(*s, generator=gen, device=dev, dtype=torch.float64)
    nrm = lambda *s: torch.randn(*s, generator=gen, device=dev, dtype=torch.float64)

    def bank(n, s):
        es, eq, _ = C.res_bank_iid(n, seed=s + 2)
        return C.mfg_bank(n, seed=s + 1), tt(es), tt(eq), tt(np.random.default_rng(s + 3).random(n))
    obj = lambda Z, bk: GS.objective(Z, cement, *bk)
    lo, hi = bank(n_lo, seed), bank(n_hi, seed + 101)
    parts = []
    if pool:
        Zp = tt(pool); top = Zp[obj(Zp, hi)[0].argsort(descending=True)[:n_exact]]; parts.append(top)
        nw = P // 4; parts.append(torch.clamp(top[torch.randint(0, min(len(top), 4), (nw,), generator=gen, device=dev)] + 0.02 * nrm(nw, D), 0, 1))
    n_anchor = int((0.2 if pool else 0.3) * P); n_lhs = max(P - sum(len(p) for p in parts) - n_anchor, 4)
    lhs = (torch.stack([torch.randperm(n_lhs, generator=gen, device=dev) for _ in range(D)], 1).to(torch.float64) + rnd(n_lhs, D)) / n_lhs
    a = st["anchor"][torch.randint(0, len(st["anchor"]), (n_anchor,), generator=gen, device=dev)]
    pop = torch.cat([lhs, torch.clamp(a + 0.02 * nrm(n_anchor, D), 0, 1)] + parts, 0); Pn = pop.shape[0]; evals = 0

    def de(pop, bk, gens):
        nonlocal evals
        J, pen, rhat, cvar, sf = obj(pop, bk); evals += Pn; F = torch.full((Pn,), 0.5, dtype=torch.float64, device=dev); CR = torch.full((Pn,), 0.9, dtype=torch.float64, device=dev)
        best, stall, used = float(J.max()), 0, 0
        for g in range(gens):
            Fi = torch.where(rnd(Pn) < 0.1, 0.1 + 0.9 * rnd(Pn), F); CRi = torch.where(rnd(Pn) < 0.1, rnd(Pn), CR)
            r3 = rnd(Pn, Pn); r3.fill_diagonal_(-1.0); abc = r3.topk(3, dim=1).indices
            mutant = pop[abc[:, 0]] + Fi[:, None] * (pop[abc[:, 1]] - pop[abc[:, 2]])
            cross = rnd(Pn, D) < CRi[:, None]; cross[torch.arange(Pn, device=dev), torch.randint(0, D, (Pn,), generator=gen, device=dev)] = True
            v = torch.where(cross, mutant, pop); v = torch.where(v < 0, -v, v); v = torch.where(v > 1, 2 - v, v); trial = torch.clamp(v, 0, 1)
            Jt, pent, rt, ct, sft = obj(trial, bk); evals += Pn; better = Jt >= J
            pop = torch.where(better[:, None], trial, pop); J = torch.where(better, Jt, J); pen = torch.where(better, pent, pen); rhat = torch.where(better, rt, rhat)
            cvar = torch.where(better, ct, cvar); sf = torch.where(better, sft, sf); F = torch.where(better, Fi, F); CR = torch.where(better, CRi, CR)
            bj = float(J.max()); used += 1
            if bj > best + tol: best, stall = bj, 0
            else: stall += 1
            if used >= min_gens and stall >= patience: break
        return pop, J, pen, rhat, cvar, sf, used
    pop, J, pen, rhat, cvar, sf, g1 = de(pop, lo, max_gens // 2)
    pop, J, pen, rhat, cvar, sf, g2 = de(pop, hi, max(max_gens - g1, 20))
    order = J.argsort(descending=True).cpu().numpy(); popn = pop.cpu().numpy(); Jn, penn, rn, cn = J.cpu().numpy(), pen.cpu().numpy(), rhat.cpu().numpy(), cvar.cpu().numpy()
    out, seen = [], []
    for i in order:
        if any(np.abs(popn[i] - s_).max() < 0.02 for s_ in seen): continue
        seen.append(popn[i]); out.append(dict(z=popn[i].tolist(), J=float(Jn[i]), penalty=float(penn[i]), rhat_search=float(rn[i]), cvar_search=float(cn[i])))
        if len(out) >= 6: break
    Zb = tt(np.array([o["z"] for o in out])); Jf, penf, rf, cf, sff = obj(Zb, bank(rerank_n, seed + 777)); nom = GS.decode(Zb, cement)          # re-rank on a FRESH bank
    for k, o in enumerate(out):
        o["rhat"], o["cvar"], o["support_fraction"], o["penalty"] = float(rf[k]), float(cf[k]), float(sff[k]), float(penf[k])
        o["nominal"] = dict(C=float(cement), r=float(nom["r"][k]), A=float(nom["A"][k]), P=float(nom["P"][k]), S=float(nom["S"][k]), phi=float(nom["phi"][k]), g=nom["g"][k].cpu().numpy().tolist())
    out.sort(key=lambda o: (o["penalty"] > 0, -o["rhat"], -o["cvar"]))
    return dict(cement=cement, seed=seed, P=Pn, generations=[g1, g2], best=out, evals=evals, seconds=time.time() - t0)
