"""Shared pieces of the v2 design-space map. Reuses the v1 code unchanged (multitarget.py, classify.py, run_stage_b.edge_cells); only new files.
 * pool state: data_v2/stageA_v2.npz (v1 arrays + appended designs, with a `source` text array), data_v2/stageB_v2_cache.pkl (stage-B pair cache, same bank/seed as v1)
 * screen(): the pool's own screen (gate, family rule, support at the nominal blend on the 256-batch bank of pool.py) for designs that are appended later (tuned, refined)
 * best_from_cache(): v2 classification = best valid stage-B value over EVERY (design, target) pair evaluated so far (v1 pairs included), so a bigger pool can never lose a target
 * flags90 / edge helpers; GPU monitor thread (peak temperature, pause at 76 C until 70 C)."""
import sys, os, pickle, threading, subprocess; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify import *
from run_stage_b import edge_cells, design_nom_fn

D2 = os.path.join(SM, "data_v2"); L2 = os.path.join(SM, "logs_v2"); os.makedirs(D2, exist_ok=True); os.makedirs(L2, exist_ok=True)
R90 = 0.90; THR90 = R90 + M_WALK
ARR = ("counts", "supf", "csm", "km", "gate11", "fam11", "hmin11", "Z", "C", "fam", "stratum", "typ")


class Gpu:
    """temperature guard. A thread samples nvidia-smi every 2 s (peak + cached value). __call__() (between units of work) pauses at >= 75 C until <= 70 C; cheap() is the same check on the cached value and is called inside the search (every objective call),
    so a run cannot heat the card far past the pause point (rule: pause at 76 C, back at 70 C, never above 78 C)."""
    LIM = int(os.environ.get("GPU_LIM", 75))
    def __init__(self, log=print):
        self.log = log; self.peak = gpu_temp(); self.cur = self.peak; self.pauses = []; self.samples = 0; self._stop = False; self.lock = threading.Lock(); self.paused = False
        threading.Thread(target=self._mon, daemon=True).start()
    def _mon(self):
        while not self._stop:
            t = gpu_temp(); self.samples += 1
            with self.lock: self.cur = t; self.peak = max(self.peak, t)
            time.sleep(2)
    def _pause(self, t):
        t0 = time.time(); self.log(f"GPU {t} C: pause"); self.paused = True
        try:
            while True:
                time.sleep(3); x = gpu_temp()
                with self.lock: self.cur = x
                if x <= 70: break
        finally: self.paused = False
        self.pauses.append(dict(at=time.strftime("%H:%M:%S"), temp=t, seconds=round(time.time() - t0)))
    def __call__(self):
        t = gpu_temp()
        with self.lock: self.cur = t; self.peak = max(self.peak, t)
        if t >= self.LIM: self._pause(t)
    def cheap(self):
        if self.cur >= self.LIM and not self.paused: self._pause(self.cur)
    def stop(self): self._stop = True


def load_pool():
    p = os.path.join(D2, "stageA_v2.npz")
    if not os.path.exists(p): raise FileNotFoundError(f"{p}: run rescore.py first (v3 never reads a v1 pool)")
    A = np.load(p, allow_pickle=False); return {k: A[k] for k in A.files}


def save_pool(P): np.savez(os.path.join(D2, "stageA_v2.npz"), **P)


def screen(ctx, Z, Cv, fam, tg=lambda: None):
    """pool.py's screen for appended designs: gate (nominal) and family rule (nominal), support at the nominal blend on the 256-batch bank (bank(256, 20261004)). No projection (a design that fails the family rule is dropped, not moved).
    Returns keep mask (n,) and a dict of counts (n_in, gate_fail, family_fail, support_fail, kept)."""
    ctx.eng.set_lowprec(False); bank, u = ctx.bank(256, 20261004); n = len(Cv); keep = np.zeros(n, bool); gate_ok = np.zeros(n, bool); fam_ok = np.zeros(n, bool)
    with torch.no_grad():
        for f in (0, 1, 2):
            ix = np.flatnonzero(fam == f)
            if not len(ix): continue
            nom = decode(T(Z[ix]), T(Cv[ix])); pen = torch.nan_to_num(ctx.gate.penalties(nom)["penalty"], posinf=1e6); fok, fd, _ = ctx.fam.check(nom["g"], f); gate_ok[ix] = (pen <= 0).cpu().numpy(); fam_ok[ix] = fok.cpu().numpy()
            ok = np.flatnonzero(gate_ok[ix] & fam_ok[ix])
            for a in range(0, len(ok), 1500):
                tg(); ii = torch.as_tensor(ok[a:a + 1500], device=DEV); st = ctx.eng.stats_t({k: v[ii] for k, v in nom.items()}, bank, None, None, u=u); keep[ix[ok[a:a + 1500]]] = (st["support_fraction"] >= 1 - 1e-12).cpu().numpy()
    return keep, dict(n_in=int(n), gate_fail=int((~gate_ok).sum()), family_fail_among_gate_ok=int((gate_ok & ~fam_ok).sum()), support_fail_among_gate_family_ok=int((gate_ok & fam_ok & ~keep).sum()), kept=int(keep.sum()))


def append_designs(ctx, grid, P, Z, Cv, fam, source, rnd, tg=lambda: None, log=print, fixed_keep=None):
    """screen, then stage A (nominal blend, N=4096, bank seed as v1) and the 11-blend cheap rules for the survivors; appended to the pool dict P (in place). Returns (n_added, screen counts, new ids)."""
    keep, cnt = screen(ctx, Z, Cv, fam, tg); Zk, Ck, Fk = Z[keep], Cv[keep], fam[keep]; n0 = len(P["C"])
    if keep.sum() == 0: return 0, cnt, np.zeros(0, int), keep
    counts, supf, csm, km = stage_a(ctx, Zk, Ck, grid, 4096, 20261004 + 3, tg=tg); gate11, fam11, hmin11 = rules_11(ctx, Zk, Ck, Fk)
    new = dict(counts=counts, supf=supf, csm=csm, km=km, gate11=gate11, fam11=fam11, hmin11=hmin11, Z=Zk, C=Ck, fam=Fk, stratum=np.full(len(Ck), 4), typ=np.full(len(Ck), 3))
    for k in ARR: P[k] = np.concatenate([P[k], new[k].astype(P[k].dtype)])
    P["source"] = np.concatenate([P["source"], np.full(len(Ck), source, dtype="U12")]); P["round"] = np.concatenate([P["round"], np.full(len(Ck), rnd, np.int16)])
    cnt.update(stageA_nominal_support_ok=int((supf >= 1 - 1e-12).sum()), eligible_all11=int(((supf >= 1 - 1e-12) & gate11 & fam11).sum()))
    return len(Ck), cnt, np.arange(n0, n0 + len(Ck)), keep


def eligible(P): return (P["supf"] >= 1 - 1e-12) & P["gate11"] & P["fam11"]


def best_from_cache(sb, nT, designs_ok=None):
    """best valid stage-B value per target over every cached (design, target) pair. valid = the design kept full support at all 11 blends. Returns best_LB (nan = none), best_design (-1), and per-target sorted candidate lists."""
    cands = [[] for _ in range(nT)]
    for (d, t), v in sb.cache.items():
        s = sb.supall.get(d)
        if s is not None and s >= 1 - 1e-12 and (designs_ok is None or designs_ok[d]): cands[t].append((float(np.nanmin(v)), d))
    best = np.full(nT, np.nan); bd = np.full(nT, -1)
    for t in range(nT):
        cands[t].sort(reverse=True)
        if cands[t]: best[t], bd[t] = cands[t][0]
    return best, bd, cands


def edge_cells90(best, grid, below=2, above=2):
    nS, nK = grid.nS, grid.nK; L = np.nan_to_num(best, nan=-1.0).reshape(nS, nK) >= THR90; out = set()
    for b in range(nK):
        ix = np.flatnonzero(L[:, b]); a0 = int(ix.max()) if len(ix) else -1
        for a in range(a0 - below, a0 + above + 1):
            if 0 <= a < nS: out.add(grid.tid(a, b))
    return sorted(out)


def top_cells(best, grid):
    """per K column: highest supported S index (-1 = none) at 0.90 under the acceptance rule."""
    L = np.nan_to_num(best, nan=-1.0).reshape(grid.nS, grid.nK) >= THR90
    return np.array([int(np.flatnonzero(L[:, b]).max()) if L[:, b].any() else -1 for b in range(grid.nK)])


def classify_pool(ctx, grid, P, sb, floors, tg, log=print, tag=""):
    """v2 classification on the current pool: ranking by stage-A Wilson bound (all designs, same rules as v1), adaptive stage-B evaluation (v1 `classify`, cache shared with v1), forced full evaluation (10 designs) of the 0.90 edge cells,
    then best = best valid value over everything cached. Returns dict(best, bd, cands, ranking...)."""
    elig = eligible(P); wilA = wilson(torch.as_tensor(P["counts"], dtype=torch.float64, device=DEV), 4096.0); rank = rankings(wilA, elig, P["hmin11"], floors); del wilA; torch.cuda.empty_cache()
    dn = design_nom_fn(P["Z"], P["C"]); t0 = time.time(); R1 = classify(sb, rank, dn, log=log, tag=tag + " main"); t1 = time.time() - t0
    best, bd, cands = best_from_cache(sb, grid.nT); ec = edge_cells90(best, grid); t0 = time.time(); R2 = classify(sb, rank, dn, log=log, tag=tag + " edge", only=ec, noresolve=True); t2 = time.time() - t0
    best, bd, cands = best_from_cache(sb, grid.nT)
    return dict(best=best, bd=bd, cands=cands, rank=rank, seconds_main=t1, seconds_edge=t2, edge_forced=ec)


def save_cache(sb): pickle.dump(dict(cache=sb.cache, supall=sb.supall), open(os.path.join(D2, "stageB_v2_cache.pkl"), "wb"))


def load_cache(sb, path=None):
    """v3: only data_v2/stageB_v2_cache.pkl; missing -> leave the (new) StageB cache empty. `path` is accepted and ignored so no old-engine cache can ever be loaded."""
    p = os.path.join(D2, "stageB_v2_cache.pkl")
    if not os.path.exists(p): return False
    c = pickle.load(open(p, "rb")); sb.cache, sb.supall = c["cache"], c["supall"]; return True


def edge_summary(best, grid):
    tc = top_cells(best, grid); return {f"{grid.K[b]:.4f}": (float(grid.S[tc[b]]) if tc[b] >= 0 else None) for b in range(grid.nK)}
