"""Global optimiser for the Guided engine: L-SHADE + batched gradient polish.
PORTED (copied, not imported) from C:\\Users\\Shuddhashil\\Desktop\\Mix Design Paper Rebuild\\research\\DefensibleEngineV2_20261003\\v2engine\\optim.py: `Run` (reduced), `_reflect`, `lshade`, `distinct_top`, `grad_polish`
(the V2 bake-off, BAKEOFF.md, picked LSHADE+GRAD). Changes against the V2 text, nothing else:
  (1) the objective is not V2's RobustEval: `Run` takes two callables, evalfn(Z, bank) -> J and smooth(Z, bank, temp) -> smooth pass probability (both supplied by gengine.global_search from the walk's own functions);
  (2) `repair(Z)` hook: after every move (L-SHADE trial, Adam step) the design is put back inside the free-variable box, the porosity window (20-30 %) and the family (blend projection of the walk), written back into Z;
  (3) the V2 family archive (MAP-Elites elites) is dropped: the family is fixed here; jDE and CMA-ES are not ported;
  (4) lshade(time_driven=False) = generation-driven schedule (deterministic), see its docstring.
Search only: acceptance is the walk's own acceptance (walk.Ctx.accept), applied by the caller."""
import math, time
import numpy as np, torch

D = 11


class Run:
    """One level-solve: objective callables, banks, random generator, best-so-far trace (V2 `Run` without the elite archive)."""
    def __init__(self, evalfn, smooth, repair, seed, dev, bank_lo, bank_hi):
        self.evalfn, self.smooth, self.repair, self.seed, self.dev = evalfn, smooth, repair, int(seed), dev
        self.gen = torch.Generator(device=dev).manual_seed(self.seed); self.lo, self.hi = bank_lo, bank_hi; self.bk = self.lo
        self.t0 = time.time(); self.trace = []; self.evals = 0

    def rnd(self, *s): return torch.rand(*s, generator=self.gen, device=self.dev, dtype=torch.float64)
    def nrm(self, *s): return torch.randn(*s, generator=self.gen, device=self.dev, dtype=torch.float64)
    def elapsed(self): return time.time() - self.t0
    def J(self, Z, bk=None): self.evals += Z.shape[0]; return self.evalfn(Z, bk if bk is not None else self.bk)
    def switch_bank(self, bk, pop): self.bk = bk; return self.J(pop, bk)
    def note(self, J): self.trace.append((self.elapsed(), float(J.max())))


def _reflect(v, x):
    """L-SHADE boundary repair on the unit cube: halfway between the parent and the violated bound."""
    return torch.where(v < 0, x / 2, torch.where(v > 1, (1 + x) / 2, v))


def lshade(run, pop, deadline, patience=None, min_gens=15, tol=1e-4, p_min=20, H=6, switch=0.5, max_gens=None, time_driven=True):
    """L-SHADE: current-to-pbest/1/bin, success-history memory of F and CR (weighted Lehmer means), external archive of replaced parents, linear population-size reduction
    (driven by elapsed fraction of the time budget or generation budget), small noise bank first then large bank (population re-scored at the switch).
    time_driven=False (change (4) against V2): with max_gens the schedule follows the generation count only, the deadline is just a safety cap, so the result does not depend on machine speed (same seed = same result)."""
    dev = run.dev; P0 = pop.shape[0]; t_start = run.elapsed(); J = run.J(pop); MF = torch.full((H,), 0.5, dtype=torch.float64, device=dev); MCR = torch.full((H,), 0.5, dtype=torch.float64, device=dev)
    hk = 0; arch = torch.zeros(0, D, dtype=torch.float64, device=dev); best, stall, g, switched = float(J.max()), 0, 0, False
    while run.elapsed() < deadline and (max_gens is None or g < max_gens):
        prog = max(((run.elapsed() - t_start) / max(deadline - t_start, 1e-9)) if (time_driven or not max_gens) else 0.0, (g / max_gens) if max_gens else 0.0)
        if not switched and prog >= switch: J = run.switch_bank(run.hi, pop); switched = True; best, stall = float(J.max()), 0
        P = pop.shape[0]; tgt = max(p_min, int(round(P0 - (P0 - p_min) * min(prog, 1.0))))
        if tgt < P:
            keep = J.argsort(descending=True)[:tgt]; pop, J = pop[keep], J[keep]; P = tgt; arch = arch[-P0:]
        ri = torch.randint(0, H, (P,), generator=run.gen, device=dev)
        CRi = torch.clamp(MCR[ri] + 0.1 * run.nrm(P), 0.0, 1.0)
        Fi = MF[ri] + 0.1 * torch.tan(math.pi * (run.rnd(P) - 0.5))
        for _ in range(4): Fi = torch.where(Fi <= 0, MF[ri] + 0.1 * torch.tan(math.pi * (run.rnd(P) - 0.5)), Fi)
        Fi = torch.clamp(torch.where(Fi <= 0, torch.full_like(Fi, 0.1), Fi), max=1.0)
        order = J.argsort(descending=True); pn = max(2, int(round(max(2.0 / P, 0.11) * P))); pbest = pop[order[torch.randint(0, pn, (P,), generator=run.gen, device=dev)]]
        ar = torch.arange(P, device=dev); r1 = (ar + 1 + torch.randint(0, max(P - 1, 1), (P,), generator=run.gen, device=dev)) % P
        union = torch.cat([pop, arch], 0); U = union.shape[0]; r2 = torch.randint(0, U, (P,), generator=run.gen, device=dev)
        for _ in range(3): bad = (r2 == ar) | (r2 == r1); r2 = torch.where(bad, torch.randint(0, U, (P,), generator=run.gen, device=dev), r2)
        mutant = pop + Fi[:, None] * (pbest - pop) + Fi[:, None] * (pop[r1] - union[r2])
        cross = run.rnd(P, D) < CRi[:, None]; cross[ar, torch.randint(0, D, (P,), generator=run.gen, device=dev)] = True
        trial = run.repair(torch.clamp(_reflect(torch.where(cross, mutant, pop), pop), 0, 1))         # (2) repair hook
        Jt = run.J(trial); better = Jt >= J; improved = Jt > J
        if bool(improved.any()):
            w = torch.clamp(Jt - J, max=1.0)[improved]; w = w / w.sum(); f, cr = Fi[improved], CRi[improved]
            MF[hk] = (w * f * f).sum() / (w * f).sum(); MCR[hk] = (w * cr * cr).sum() / max(float((w * cr).sum()), 1e-12); hk = (hk + 1) % H
            arch = torch.cat([arch, pop[improved]], 0)[-P0:]
        pop = torch.where(better[:, None], trial, pop); J = torch.where(better, Jt, J); run.note(J); g += 1; bj = float(J.max())
        if bj > best + tol: best, stall = bj, 0
        else: stall += 1
        if patience and g >= min_gens and stall >= patience:
            if switched: break
            J = run.switch_bank(run.hi, pop); switched = True; best, stall = float(J.max()), 0           # stalled on the small bank: re-score on the large bank and continue (stop only after the large-bank phase stalls)
    return pop, J


def distinct_top(pop, J, k, min_dist=0.02):
    order = J.argsort(descending=True).cpu().numpy(); Zn = pop.cpu().numpy(); out = []
    for i in order:
        if all(np.abs(Zn[i] - Zn[j]).max() >= min_dist for j in out): out.append(i)
        if len(out) >= k: break
    return pop[torch.as_tensor(out, device=pop.device, dtype=torch.long)]


def grad_polish(run, starts, deadline, lr=0.02, check_every=15, steps_max=100000, temp=0.35, chunk=4):
    """Batched multi-start gradient ascent (manual Adam, per-start step size) on the smooth worst-neighbour pass probability; the EXACT objective re-scores every `check_every`
    steps, the best exact design per start is kept, and a start whose exact score fell is reset to its best with a halved step (this is the constraint / feasibility rule).
    Differences from V2: smooth probability comes from run.smooth (the walk's DF.smooth_reliability), the repair hook acts after every Adam step, `chunk` = designs per autograd chunk."""
    K_ = starts.shape[0]; Zc = starts.clone(); run.bk = run.hi
    Zb, Jb = starts.clone(), run.J(starts, run.hi); m = torch.zeros_like(Zc); v = torch.zeros_like(Zc); lrs = torch.full((K_,), lr, dtype=torch.float64, device=run.dev); t = 0
    while run.elapsed() < deadline and t < steps_max:
        Zp = Zc.clone().requires_grad_(True)
        for a in range(0, K_, chunk):
            R = run.smooth(Zp[a:a + chunk], run.hi, temp); (-R.sum()).backward()
        gr = torch.nan_to_num(Zp.grad); t += 1; m = 0.9 * m + 0.1 * gr; v = 0.999 * v + 0.001 * gr * gr
        step = (m / (1 - 0.9 ** t)) / (torch.sqrt(v / (1 - 0.999 ** t)) + 1e-8); Zc = run.repair(torch.clamp(Zc - lrs[:, None] * step, 0, 1))
        if t % check_every == 0:
            Jc = run.J(Zc, run.hi); imp = Jc > Jb; Zb = torch.where(imp[:, None], Zc, Zb); Jb = torch.where(imp, Jc, Jb)
            drop = Jc < Jb - 0.02; Zc = torch.where(drop[:, None], Zb, Zc); lrs = torch.where(drop, lrs * 0.5, lrs); m = torch.where(drop[:, None], torch.zeros_like(m), m); v = torch.where(drop[:, None], torch.zeros_like(v), v)
            run.note(Jb)
    return Zb, Jb
