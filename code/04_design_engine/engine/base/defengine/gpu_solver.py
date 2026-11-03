"""GPU-resident level solver: decoding, differential evolution, gate, scenarios, scoring and selection all stay on the GPU.
Same algorithm, tiers, banks and output format as level_solver.solve_level (drop-in), so the ladder is unchanged.
Run with the GPU python."""
import math, time
import numpy as np, torch
from . import corelib as C, gpu as GP, gpu_gate as GG, level_solver as LS

_S = {}


def _state():
    if "eng" not in _S:
        C.engine(); _S["eng"] = GP.GPUEngine(); _S["gate"] = GG.GPUGate(); dev = _S["eng"].dev
        B = C.SC.BND; _S["bnd"] = {k: (float(B[k][0]), float(B[k][1])) for k in B}; _S["lna"] = (C.SC.LNA_LO, C.SC.LNA_HI)
        _S["anchor"] = torch.as_tensor(LS._training_anchor_vectors(), dtype=torch.float64, device=dev)
    return _S


def project_simplex(m):
    """Euclidean projection of each row of m (n,6) onto the simplex (Duchi et al.), identical to stage1_evaluator.project_to_simplex."""
    u, _ = torch.sort(m, dim=1, descending=True); css = torch.cumsum(u, 1); k = torch.arange(1, m.shape[1] + 1, device=m.device, dtype=m.dtype)
    cond = (u * k) > (css - 1.0); rho = (cond.long() * torch.arange(1, m.shape[1] + 1, device=m.device)[None, :]).argmax(1)       # last True index
    rho = torch.where(cond.any(1), rho, torch.zeros_like(rho)); theta = (torch.gather(css, 1, rho[:, None])[:, 0] - 1.0) / (rho + 1).to(m.dtype)
    return torch.clamp(m - theta[:, None], min=0.0)


def decode(Z, cement):
    """Z (n,11) unit-cube free variables [r, A, P, S, phi, m1..m6] -> nominal tensors (cement fixed)."""
    st = _state(); b = st["bnd"]; lo, hi = st["lna"]; Z = torch.clamp(Z, 0.0, 1.0); n = Z.shape[0]
    lin = lambda u, k: b[k][0] + u * (b[k][1] - b[k][0])
    return dict(C=torch.full((n,), float(cement), dtype=torch.float64, device=Z.device), r=lin(Z[:, 0], "r"), A=torch.exp(lo + Z[:, 1] * (hi - lo)),
                P=lin(Z[:, 2], "P"), S=lin(Z[:, 3], "S"), phi=lin(Z[:, 4], "phi"), g=project_simplex(Z[:, 5:11]))


def objective(Z, cement, bank, eS, eQ, u=None):
    """Same tiers as level_solver.objective: gate-violating < unsupported < supported (CVaR margin). Returns J, penalty, rhat, cvar, support_fraction (tensors)."""
    st = _state(); nom = decode(Z, cement); pen = st["gate"].penalties(nom)["penalty"]; ok = pen <= 0.0
    J = -1.0e4 - 1.0e3 * torch.nan_to_num(pen, posinf=1e6); rhat = torch.zeros_like(J); cvar = torch.full_like(J, -math.inf); sf = torch.zeros_like(J)
    if bool(ok.any()):
        idx = ok.nonzero()[:, 0]; s = st["eng"].stats_t({k: v[idx] for k, v in nom.items()}, bank, eS, eQ, u=u)
        rhat[idx], cvar[idx], sf[idx] = s["rhat"], s["cvar"], s["support_fraction"]
        J[idx] = torch.where(torch.isfinite(s["cvar"]), s["cvar"], -1.0e3 + 100.0 * s["support_fraction"])
    return J, pen, rhat, cvar, sf


def solve_level_gpu(cement, seed, P=80, generations=30, n_scen=256, warm=None, pool=None, n_exact=8, bank_seed=None, verbose=False, gate=None, rerank_n=1024, conditional=True):
    st = _state(); dev = st["eng"].dev; t0 = time.time(); gen = torch.Generator(device=dev).manual_seed(int(seed)); D = 11
    tt = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=torch.float64, device=dev)
    bank = C.mfg_bank(n_scen, seed=(bank_seed if bank_seed is not None else seed + 1)); es, eq, _ = C.res_bank_iid(n_scen, seed=seed + 2); eS, eQ = tt(es), tt(eq)
    u = tt(np.random.default_rng(seed + 3).random(n_scen)) if conditional else None      # common random numbers for the porosity-band residual draw
    rnd = lambda *s: torch.rand(*s, generator=gen, device=dev, dtype=torch.float64)
    exact = torch.zeros(0, D, dtype=torch.float64, device=dev)
    if pool:
        Zp = tt(pool); Jp = objective(Zp, cement, bank, eS, eQ, u)[0]; exact = Zp[Jp.argsort(descending=True)[:n_exact]]
    n_ex = len(exact); n_warm = min(len(warm or []), P // 5); n_anchor = int(0.3 * P); n_lhs = max(P - n_ex - n_warm - n_anchor, 4)
    lhs = (torch.stack([torch.randperm(n_lhs, generator=gen, device=dev) for _ in range(D)], 1).to(torch.float64) + rnd(n_lhs, D)) / n_lhs
    parts = [lhs]
    if n_anchor:
        a = st["anchor"][torch.randint(0, len(st["anchor"]), (n_anchor,), generator=gen, device=dev)]; parts.append(torch.clamp(a + 0.02 * torch.randn(n_anchor, D, generator=gen, device=dev, dtype=torch.float64), 0, 1))
    if n_warm: w = tt(warm[:n_warm]); parts.append(torch.clamp(w + 0.01 * torch.randn(n_warm, D, generator=gen, device=dev, dtype=torch.float64), 0, 1))
    if n_ex: parts.append(exact)
    pop = torch.cat(parts, 0)[:max(P, n_ex)]; P = pop.shape[0]
    J, pen, rhat, cvar, sf = objective(pop, cement, bank, eS, eQ, u); evals = P
    F = torch.full((P,), 0.5, dtype=torch.float64, device=dev); CR = torch.full((P,), 0.9, dtype=torch.float64, device=dev); hist = [float(J.max())]
    for g in range(generations):
        Fi = torch.where(rnd(P) < 0.1, 0.1 + 0.9 * rnd(P), F); CRi = torch.where(rnd(P) < 0.1, rnd(P), CR)
        r3 = rnd(P, P); r3.fill_diagonal_(-1.0); abc = r3.topk(3, dim=1).indices                                   # three distinct random members, none equal to i
        mutant = pop[abc[:, 0]] + Fi[:, None] * (pop[abc[:, 1]] - pop[abc[:, 2]])
        cross = rnd(P, D) < CRi[:, None]; cross[torch.arange(P, device=dev), torch.randint(0, D, (P,), generator=gen, device=dev)] = True
        v = torch.where(cross, mutant, pop); v = torch.where(v < 0, -v, v); v = torch.where(v > 1, 2 - v, v); trial = torch.clamp(v, 0, 1)
        Jt, pent, rt, ct, sft = objective(trial, cement, bank, eS, eQ, u); evals += P
        better = Jt >= J
        pop = torch.where(better[:, None], trial, pop); J = torch.where(better, Jt, J); pen = torch.where(better, pent, pen); rhat = torch.where(better, rt, rhat)
        cvar = torch.where(better, ct, cvar); sf = torch.where(better, sft, sf); F = torch.where(better, Fi, F); CR = torch.where(better, CRi, CR)
        if verbose and g % 5 == 0: print(f"  C={cement:5.0f} gen {g+1:3d} best J {float(J.max()):7.3f}", flush=True)
        hist.append(float(J.max()))
    order = J.argsort(descending=True).cpu().numpy(); popn = pop.cpu().numpy(); Jn, penn, rn, cn = J.cpu().numpy(), pen.cpu().numpy(), rhat.cpu().numpy(), cvar.cpu().numpy()
    out, seen = [], []
    for i in order:
        if any(np.abs(popn[i] - s_).max() < 0.02 for s_ in seen): continue
        seen.append(popn[i]); out.append(dict(z=popn[i].tolist(), J=float(Jn[i]), penalty=float(penn[i]), rhat_search=float(rn[i]), cvar_search=float(cn[i])))
        if len(out) >= 6: break
    Zb = tt(np.array([o["z"] for o in out])); fb = C.mfg_bank(rerank_n, seed=seed + 777); fs, fq, _ = C.res_bank_iid(rerank_n, seed=seed + 778)       # re-rank on a FRESH bank
    uf = tt(np.random.default_rng(seed + 779).random(rerank_n)) if conditional else None
    Jf, penf, rf, cf, sff = objective(Zb, cement, fb, tt(fs), tt(fq), uf); nom = decode(Zb, cement)
    for k, o in enumerate(out):
        o["rhat"], o["cvar"], o["support_fraction"], o["penalty"] = float(rf[k]), float(cf[k]), float(sff[k]), float(penf[k])
        o["nominal"] = dict(C=float(cement), r=float(nom["r"][k]), A=float(nom["A"][k]), P=float(nom["P"][k]), S=float(nom["S"][k]), phi=float(nom["phi"][k]), g=nom["g"][k].cpu().numpy().tolist())
    out.sort(key=lambda o: (o["penalty"] > 0, -o["rhat"], -o["cvar"]))
    return dict(cement=cement, seed=seed, n_scen=n_scen, P=P, generations=generations, best=out, history=hist, evals=evals, seconds=time.time() - t0)
