"""Multi-target scoring on the Guided stack. Reuses the engine's scenario sampler (GPUEngine.arrays_t, manufacturing bank), its porosity-conditional residual draw (conditional_draws),
the residual-bank bootstrap tables of gpu_lb.LB and the tie-bracket trick; no frozen file is edited. The pass condition is rebuilt from the realised strength / log-permeability
   pass(S*, K*)  =  in support  AND  S_real >= S*  AND  Q_real >= ln K*          (identical to engine.stats_t / gpu_lb.lcb95 for one target),
and counted for ALL targets at once with a 2-D histogram of (strength bin, permeability bin) followed by a reversed cumulative sum (exact integer counts; see tests M1)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sm_common import *

Z95 = gpu_lb.Z95


class Grid:
    def __init__(self, S, K):
        self.S = np.asarray(S, float); self.K = np.asarray(K, float); self.nS, self.nK = len(self.S), len(self.K); self.nT = self.nS * self.nK
        self.Sg = T(self.S); self.Qg = T([math.log(k) for k in self.K])        # same ln as C.LOGK_STAR = math.log(K*)

    def tid(self, a, b): return a * self.nK + b                                 # flat target index, strength-major


def tail_counts(Sr, Qr, ok, grid):
    """Sr, Qr, ok (P,N) -> (P, nS*nK) int64: number of scenarios with ok & Sr >= S*_a & Qr >= lnK*_b."""
    P = Sr.shape[0]; iS = torch.bucketize(Sr.contiguous(), grid.Sg, right=True); iK = torch.bucketize(Qr.contiguous(), grid.Qg, right=True)
    w = grid.nK + 1; H = torch.zeros(P, (grid.nS + 1) * w, dtype=torch.long, device=Sr.device); H.scatter_add_(1, iS * w + iK, ok.long()); H = H.view(P, grid.nS + 1, w)
    H = H.flip(1).cumsum(1).flip(1).flip(2).cumsum(2).flip(2); return H[:, 1:, 1:].reshape(P, grid.nT)


def wilson(k, n, z=Z95):
    """vectorised copy of gpu_lb.wilson_lower_bound (k tensor float64)."""
    p = k / n; den = 1 + z * z / n; ctr = p + z * z / (2 * n); mar = z * torch.sqrt((p * (1 - p) + z * z / (4 * n)) / n); return torch.clamp((ctr - mar) / den, min=0.0)


def realise(eng, nom, bank, u):
    """nom: dict of (P,) / (P,6) GPU tensors -> model arrays, realised strength, log-permeability, support flag, achieved porosity (P,N) each."""
    muS, muQ, sS, sQ, sup, phi = eng.arrays_t(nom, bank, with_phi=True); eS, eQ = eng.conditional_draws(phi, u)
    return muS, muQ, sS, sQ, muS + sS * eS, muQ + sQ * eQ, sup >= C.SUPPORT_THR, phi


# ------------------------------------------------------------------ stage A
@torch.no_grad()
def stage_a(ctx, Z, Cv, grid, N, seed, tg=lambda: None, chunk=24):
    """nominal blend only. Z (D,11) numpy, Cv (D,) -> counts (D,nT) int16, nominal support fraction, predicted CS median (MPa) and K median (mm/s)."""
    ctx.eng.set_lowprec(False); bank, u = ctx.bank(N, seed); D = len(Cv); counts = np.zeros((D, grid.nT), np.int16); supf = np.zeros(D); csm = np.zeros(D); km = np.zeros(D)
    for a in range(0, D, chunk):
        tg(); nom = decode(T(Z[a:a + chunk]), T(Cv[a:a + chunk])); muS, muQ, sS, sQ, Sr, Qr, insup, phi = realise(ctx.eng, nom, bank, u)
        counts[a:a + chunk] = tail_counts(Sr, Qr, insup, grid).cpu().numpy(); supf[a:a + chunk] = insup.double().mean(1).cpu().numpy(); csm[a:a + chunk] = Sr.median(1).values.cpu().numpy(); km[a:a + chunk] = torch.exp(Qr.median(1).values).cpu().numpy()
    return counts, supf, csm, km


@torch.no_grad()
def rules_11(ctx, Z, Cv, fam):
    """target-independent cheap rules at all valid blends (nominal + 10 shifts): gates, family rule; and min paste thickness h over those blends. Returns bool (D,), bool (D,), float (D,)."""
    D = len(Cv); gate = np.zeros(D, bool); famok = np.zeros(D, bool); hmin = np.zeros(D)
    for a in range(0, D, 1500):
        sl = slice(a, a + 1500); Zt = T(Z[sl]); nom = decode(Zt, T(Cv[sl])); nc, owner, kk = nbhd(nom, ctx.tau); pen = torch.nan_to_num(ctx.gate.penalties(nc)["penalty"], posinf=1e6); P = len(Zt)
        fk = torch.zeros(len(owner), dtype=torch.bool, device=DEV); fam_t = torch.as_tensor(fam[sl], device=DEV)
        for f in (0, 1, 2):
            m = (fam_t[owner] == f).nonzero()[:, 0]
            if len(m): ok, _, _ = ctx.fam.check(nc["g"][m], f); fk[m] = ok
        h = PS.h_mm(nc["C"], nc["r"], nc["A"], nc["g"])
        gate[sl] = (ctx.worst_over((pen <= 0).double(), owner, P, "amin", 1.0) > 0.5).cpu().numpy(); famok[sl] = (ctx.worst_over(fk.double(), owner, P, "amin", 1.0) > 0.5).cpu().numpy(); hmin[sl] = ctx.worst_over(h, owner, P, "amin", 1e9).cpu().numpy()
    return gate, famok, hmin


# ------------------------------------------------------------------ stage B
class StageB:
    """Full acceptance inputs for (design, target) pairs: minimum over the 11 blends of R_LCB95 = min(Wilson, study-cluster bootstrap 5th percentile), each blend under the stricter of the two tie rules
    (as walk.Ctx.accept), N scenarios on one shared bank. Pairs whose Wilson bound is already below `thr` are cut early (their value is then the Wilson bound, which is below thr)."""
    def __init__(self, ctx, grid, N=N_B, seed=20261004 + 7, thr=0.80 + M_WALK, tg=lambda: None):
        self.ctx, self.grid, self.N, self.thr, self.tg = ctx, grid, N, thr, tg; self.bank, self.u = ctx.bank(N, seed); self.seed = seed; self.cache = {}; self.supall = {}; self.seconds = 0.0; self.n_row_evals = 0

    def _flip(self):
        e = self.ctx.eng; e.res_X, e.res_s, e.res_p = e.res_X.flip(0), e.res_s.flip(0), e.res_p.flip(0)

    @torch.no_grad()
    def lb_rows(self, arr, mask, thr):
        """arr = realise() output for m rows; mask (m,nT) bool tensor of the targets wanted per row -> (m,nT) LB (Wilson where cut), (m,) support fraction."""
        muS, muQ, sS, sQ, Sr, Qr, insup, phi = arr; g = self.grid; N = self.N; LBm = self.ctx.LB
        cnt = tail_counts(Sr, Qr, insup, g); wil = wilson(cnt.double(), N); LB = wil.clone(); alive = mask & (wil >= thr); med = torch.quantile(phi, 0.5, dim=1); bd = torch.bucketize(med.contiguous(), LBm.edges_inner, right=False).clamp(0, 2)
        for i in alive.any(1).nonzero()[:, 0].tolist():
            B = LBm.band[int(bd[i])]; Sx = muS[i][None, :] + sS[i][None, :] * B["Es"][:, None]; Qx = muQ[i][None, :] + sQ[i][None, :] * B["Eq"][:, None]
            pr = tail_counts(Sx, Qx, insup[i][None, :].expand(Sx.shape[0], N), g).double() / N; Uu = B["units"]["study"]; psum = torch.zeros(Uu["K"], g.nT, dtype=torch.float64, device=DEV).index_add_(0, Uu["idx"], pr)
            cn = LBm.counts_for(int(bd[i]), "study"); den = cn @ Uu["nrows"]; cols = alive[i].nonzero()[:, 0]
            for c0 in range(0, len(cols), 64):
                cc = cols[c0:c0 + 64]; q = torch.quantile((cn @ psum[:, cc]) / den[:, None], 0.05, dim=0); LB[i, cc] = torch.minimum(wil[i, cc], q)
        return LB, insup.double().mean(1)

    @torch.no_grad()
    def eval_pairs(self, pairs, design_nom):
        """pairs: dict design_id -> list of target ids not yet evaluated. design_nom(ids) -> (Z, C) numpy rows. Fills self.cache[(d,t)] = LB11 (11,) with NaN for invalid shifts,
        self.supall[d] = min support fraction over the 11 blends (None if the design was cut before all blends were computed)."""
        t0 = time.time(); ids = sorted(pairs); eng = self.ctx.eng; g = self.grid; tau = self.ctx.tau; eng.set_lowprec(False)
        for a in range(0, len(ids), 12):
            self.tg(); ch = ids[a:a + 12]; Z, Cv = design_nom(ch); nom = decode(T(Z), T(Cv)); nc, owner, kk = nbhd(nom, tau); own = owner.cpu().numpy(); kkn = kk.cpu().numpy(); m = len(own)
            mask = torch.zeros(m, g.nT, dtype=torch.bool, device=DEV)
            for j, d in enumerate(ch): mask[torch.as_tensor(np.flatnonzero(own == j), device=DEV)[:, None], torch.as_tensor(pairs[d], device=DEV)[None, :]] = True
            LBr = torch.full((m, g.nT), float("nan"), dtype=torch.float64, device=DEV); supr = np.full(m, np.nan)
            sub = lambda rows: {k: v[rows] for k, v in nc.items()}
            r0 = torch.as_tensor(np.flatnonzero(kkn == 0), device=DEV)                      # pass 1: nominal row of every design (tie rule 1)
            arr = realise(eng, sub(r0), self.bank, self.u); L, sf = self.lb_rows(arr, mask[r0], self.thr); LBr[r0] = L; supr[r0.cpu().numpy()] = sf.cpu().numpy(); self.n_row_evals += len(r0); del arr
            alive_d = {j: (mask[r0[j]] & (LBr[r0[j]] >= self.thr)) for j in range(len(ch))}
            todo = [j for j in range(len(ch)) if bool(alive_d[j].any())]; rows2 = np.flatnonzero((kkn > 0) & np.isin(own, todo))   # pass 2: shifted rows of designs with an alive target (tie rule 1)
            if len(rows2):
                r2 = torch.as_tensor(rows2, device=DEV); mk = torch.stack([alive_d[int(o)] for o in own[rows2]]); arr = realise(eng, sub(r2), self.bank, self.u); L, sf = self.lb_rows(arr, mk, self.thr)
                LBr[r2] = L; supr[rows2] = sf.cpu().numpy(); self.n_row_evals += len(rows2); del arr
            alive2 = {}
            for j in todo:
                rj = torch.as_tensor(np.flatnonzero(own == j), device=DEV); alive2[j] = alive_d[j] & (LBr[rj].nan_to_num(nan=1.0).min(0).values >= self.thr)
            todo2 = [j for j in todo if bool(alive2[j].any())]
            if todo2:                                                                       # pass 3: all rows of the survivors under tie rule 2 (reversed residual order)
                rows3 = np.flatnonzero(np.isin(own, todo2)); r3 = torch.as_tensor(rows3, device=DEV); mk = torch.stack([alive2[int(o)] for o in own[rows3]])
                self._flip()
                try: arr = realise(eng, sub(r3), self.bank, self.u); L, _ = self.lb_rows(arr, mk, self.thr)
                finally: self._flip()
                LBr[r3] = torch.minimum(LBr[r3], torch.where(mk, L, torch.full_like(L, float("inf")))); self.n_row_evals += len(rows3); del arr
            LBn = LBr.cpu().numpy()
            for j, d in enumerate(ch):
                rj = np.flatnonzero(own == j); full = np.full((11, g.nT), np.nan); full[kkn[rj]] = LBn[rj]
                for t in pairs[d]: self.cache[(d, t)] = full[:, t].copy()
                if j in todo: self.supall[d] = float(np.nanmin(supr[rj]))
                else: self.supall.setdefault(d, None)
        self.seconds += time.time() - t0
