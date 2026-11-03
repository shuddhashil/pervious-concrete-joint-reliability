"""Edge refinement search. Subclass of gengine.global_search.GlobalSearch (file untouched): the SAME search (L-SHADE + gradient polish, family-restricted, cement fixed per run, walk.Ctx.accept as the exact objective: R* + 0.002 at the nominal
blend and all valid shifted blends, porosity SD 1.5, gates, family rule, support, paste floor of THAT target if recorded passers exist, none otherwise). Differences, all additive:
  * set_target(S, K): assigns the module constants that the GPU engine reads at call time (corelib.S_STAR / K_STAR / LOGK_STAR) and the paste floor of the cell; R* stays 0.90 (ctx.R_ACC = 0.902);
  * start designs: `starts` (best pool design of the cell, best design of its best supported neighbour) replace the last members of the L-SHADE start population (mode "lshade"), or are the only starts plus 4 noisy copies (mode "polish": no L-SHADE, only the gradient polish);
  * returns ALL exactly scored candidates (not only the best) so that each can be added to the pool."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sm_common import *
from gengine.global_search import GlobalSearch
from gengine import optim as O


class GS2(GlobalSearch):
    hook = staticmethod(lambda: None)                       # GPU heat check, set by the driver (v2_core.Gpu.cheap): called on every objective evaluation

    def fast_J(self, Z, c, fam, bk):
        self.hook(); return super().fast_J(Z, c, fam, bk)

    def smooth(self, Z, c, bk, temp):
        self.hook(); return super().smooth(Z, c, bk, temp)

    def set_target(self, S, K):
        C.S_STAR, C.K_STAR, C.LOGK_STAR = float(S), float(K), math.log(float(K)); self.ctx.h_min = PS.paste_floor(self.ctx.data, float(S), float(K))["h_min"]; self.ctx.R_ACC = 0.90 + M_WALK

    def search(self, S, K, fam, c, seed, starts, mode="lshade"):
        self.set_target(S, K); p = self.p; t0 = time.time(); ctx = self.ctx; run_seed = int(seed) * 100003 + int(round(c * 4)) * 7 + fam * 1000003
        bank_lo = ctx.bank(p["n_lo"], run_seed + 1); bank_hi = ctx.bank(p["n_hi"], run_seed + 2)
        run = O.Run(lambda Z, bk: self.fast_J(Z, c, fam, bk), lambda Z, bk, temp: self.smooth(Z, c, bk, temp), lambda Z: self.repair(Z, fam), run_seed, DEV, bank_lo, bank_hi)
        St = self.repair(T(np.asarray(starts, float).reshape(-1, 11)), fam); k = St.shape[0]
        if mode == "lshade":
            pop = self.init_pop(run, fam, c, p["P"]); pop = torch.cat([pop[:-k], St], 0); pop, J = O.lshade(run, pop, run.elapsed() + p["t_lshade"], patience=p["patience"], max_gens=p["max_gens"], time_driven=False)
        else:
            noisy = torch.clamp(St[torch.randint(0, k, (4,), generator=run.gen, device=DEV)] + 0.01 * run.nrm(4, 11), 0, 1); pop = self.repair(torch.cat([St, noisy], 0), fam); J = run.J(pop, run.hi)
        t_ls = run.elapsed(); st2 = O.distinct_top(pop, J, p["n_polish"]); self.eng.set_lowprec(True); Zb, Jb = O.grad_polish(run, st2, run.elapsed() + p["t_polish"], lr=p["lr"], chunk=4, steps_max=p["steps_max"]); t_gp = run.elapsed()
        cand = torch.cat([O.distinct_top(pop, J, p["n_exact"]), Zb], 0); Jc = run.J(cand, run.hi); keep = [int(i) for i in Jc.argsort(descending=True).cpu().numpy()]; Zn = cand.cpu().numpy(); sel = []
        for i in keep:
            if all(np.abs(Zn[i] - Zn[j]).max() > 0.003 for j in sel): sel.append(i)
            if len(sel) >= p["n_exact"]: break
        rnd = f"ref{c:g}s{seed}"; ex = self.exact(cand[sel], c, fam, rnd)
        order = sorted(range(len(sel)), key=lambda q: (not ex[q]["accepted"], len(ex[q]["reasons"]), -ex[q]["LB_min_all"])); b = order[0]
        return dict(S=float(S), K=float(K), fam=int(fam), C=float(c), seed=int(seed), mode=mode, runtime_s=round(time.time() - t0, 2), t_lshade_s=round(t_ls, 2), n_starts=int(k), passed=bool(ex[b]["accepted"]), LB_min_all=float(ex[b]["LB_min_all"]),
                    reasons=ex[b]["reasons"], n_accepted_candidates=int(sum(e["accepted"] for e in ex)), cand_Z=[Zn[i].tolist() for i in sel], cand_LB=[float(e["LB_min_all"]) for e in ex], cand_ok=[bool(e["accepted"]) for e in ex], gpu_temp=gpu_temp())
