"""GPU port of the scenario evaluator.

Reproduces fast_engine.arrays_from_features (GP mean/variance, local-residual spread, domain inflation, support score) and the scenario
application + passing-size quantiles on the GPU, for a whole population of designs x a manufacturing-noise bank in one pass.
The frozen core is only READ (its arrays are copied to the GPU); equivalence with the frozen CPU engine is tested in tests/test_gpu_engine.py.
Precision: noise algebra + quantiles in float64; kernel / large products in float32 (a GTX 1650 does ~930 GFLOPS float32 vs ~97 float64);
predictive variance is recombined in float64. Run with the GPU-enabled python (system python), not the bundled CPU runtime."""
import math, copy
import numpy as np, torch
from . import corelib as C

ASC = np.log2(np.array([2.36, 4.75, 6.3, 9.5, 12.5, 19.0, 25.0]))
SIGMA_FLOOR = 1e-6


class GPUEngine:
    def __init__(self, device="cuda", chunk_rows=24576):
        self.dev = torch.device(device); self.chunk = chunk_rows
        eng = C.engine(); self.eng = eng; g = eng.gp
        f32 = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=torch.float32, device=self.dev)
        f64 = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=torch.float64, device=self.dev)
        # --- GP (closed form of the frozen ICM model) ---
        self.dk = copy.deepcopy(g.dk).to(device=self.dev, dtype=torch.float32).eval()
        self.Xtr = g.Xtr.to(device=self.dev, dtype=torch.float32)
        self.QK, self.alpha = f32(g.QK), f32(g.alpha)
        self.g_, self.W2, self.Bd, self.B, self.d, self.m0 = f64(g.g), f64(g.W2), f64(g.Bdiag), f64(g.B), f64(g.d), f64(g.m0)
        self.g32 = f32(g.g)
        with torch.no_grad():
            kd = self.dk.forward(self.Xtr[:8], self.Xtr[:8], diag=True)
            kd = kd.to_dense() if hasattr(kd, "to_dense") else kd
        assert float((kd.max() - kd.min()).abs()) < 1e-5, "k(x,x) is not constant: closed-form shortcut invalid"
        self.kd = float(kd[0])
        # --- feature transform, outputs, uncertainty chain ---
        self.log_idx, self.log1p_idx = list(eng.log_idx), list(eng.log1p_idx)
        self.xm, self.xs = f64(eng.x_mean), f64(eng.x_std); self.ym, self.ys = f64(eng.y_mean), f64(eng.y_std)
        self.res_X, self.res_s, self.res_p = f64(eng.res_X), f64(eng.res_s), f64(eng.res_p); self.local_q = eng.local_q
        self.ad_X, self.ad_k, self.ad_tau = f64(eng.ad_X), eng.ad_k, float(eng.ad_tau); self.ad_eta = eng.ad_eta
        assert eng.combine_rule == "max"
        # --- support model ---
        o = eng.ocsvm; self.sv = f64(o.support_vectors_); self.dual = f64(o.dual_coef_[0]); self.gamma = float(o._gamma); self.icpt = float(o.intercept_[0])
        self.scale = float(eng.support.get("ocsvm_scale", 1.0)); self.kde_X = f64(eng.kde_X); self.kde_h2 = 2.0 * eng.kde_h ** 2; self.kde_norm = float(eng.kde_log_norm)
        self.kde_max = float(eng.support["kde_log_max"]); self.asc = f64(ASC)
        self._banks = {}; self.lowprec = False; self._hp = {k: getattr(self, k) for k in self._HP}

    _HP = ("res_X", "res_s", "res_p", "ad_X", "sv", "dual", "kde_X")

    def set_lowprec(self, on=True):
        """Run the neighbour / density / support maths in single precision (decision E-08). Search only; equivalence is tested in tests/test_lowprec.py."""
        self.lowprec = on
        for k in self._HP: setattr(self, k, self._hp[k].to(torch.float32 if on else torch.float64))

    # ------------------------------------------------------------------ pieces
    def quantiles(self, G):
        """G (n,6) float64 coarse->fine -> (n,5) log2-mm sizes at 10/25/50/75/90 % passing (all five levels in one pass; same rule as the training inputs)."""
        n = G.shape[0]; G = G / G.sum(1, keepdim=True); p = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], dtype=G.dtype, device=G.device)
        F = torch.cat([torch.zeros(n, 1, dtype=G.dtype, device=G.device), torch.cumsum(torch.flip(G, [1]), 1)], 1); F[:, -1] = 1.0
        k = (F[:, None, :] < p[None, :, None]).sum(-1).clamp(1, 6)
        for _ in range(6):
            nxt = torch.gather(F, 1, k); prv = torch.gather(F, 1, k - 1)
            k = k + ((k < 6) & ((nxt - prv) <= 0)).long()
        lo = torch.gather(F, 1, k - 1); hi = torch.gather(F, 1, k)
        return self.asc[k - 1] + (p[None, :] - lo) / torch.clamp(hi - lo, min=1e-12) * (self.asc[k] - self.asc[k - 1])

    def bank_tensors(self, bank):
        key = id(bank)
        if key not in self._banks:
            t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=torch.float64, device=self.dev)
            self._banks[key] = dict(dC=t(bank.delta_C), dW=t(bank.delta_W), dA=t(bank.delta_A), dSP=t(bank.delta_SP), dS=t(bank.delta_S), eps=t(bank.eps_phi_pp), n=bank.n, ref=bank)
        return self._banks[key]

    def scenarios(self, nom, bt):
        """nom: dict of float64 tensors C,r,A,P,S,phi (p,), g (p,6). Applies the manufacturing bank to every design -> flat (p*N) features (float64)."""
        C_, r_, A_, P_, S_, phi_, g_ = (nom[k] for k in ("C", "r", "A", "P", "S", "phi", "g")); p = C_.shape[0]; N = bt["n"]
        Ca = C_[:, None] * (1 + bt["dC"][None, :]); Wa = (r_ * C_)[:, None] * (1 + bt["dW"][None, :]); wc = Wa / Ca
        Ab = torch.clamp((A_[:, None] * g_)[:, None, :] * (1 + bt["dA"][None, :, :]), min=0.0); Aa = Ab.sum(2); ga = Ab / Aa[..., None]
        SPa = torch.where(P_[:, None] > 0, P_[:, None] * (1 + bt["dSP"][None, :]), torch.zeros_like(Ca)); Sa = torch.where(S_[:, None] > 0, S_[:, None] * (1 + bt["dS"][None, :]), torch.zeros_like(Ca))
        phi = phi_[:, None] + bt["eps"][None, :]
        fl = lambda x: x.reshape(p * N); ga = ga.reshape(p * N, 6)
        # the frozen path rounds realised gradations to 12 decimals (its cache key); mirror it so quantiles see identical inputs
        q = self.quantiles(torch.round(ga * 1e12) / 1e12)
        return torch.stack([fl(phi), fl(wc), fl(Ca), fl(Aa), fl(SPa), fl(Sa)], 1), q

    def standardise(self, X6, q):
        X = torch.cat([X6, q], 1)
        for j in self.log_idx: X[:, j] = torch.log(X[:, j])
        for j in self.log1p_idx: X[:, j] = torch.log1p(X[:, j])
        Xs = (X - self.xm) / self.xs
        return Xs, Xs.to(torch.float32)           # float64 for neighbours / density / SVM (as the CPU engine), float32 for the GP kernel

    def model(self, Xs64, Xs):
        """Xs64 (n,11) float64 and Xs float32 standardised -> muS, muQ, sS, sQ, support (float64 tensors)."""
        with torch.no_grad():
            Ck = self.dk.forward(Xs, self.Xtr); Ck = Ck.to_dense() if hasattr(Ck, "to_dense") else Ck
            mean = self.m0[None, :] + ((Ck @ self.alpha).double()) @ self.B.T
            U = Ck @ self.QK; s = ((U * U) @ self.g32).double()
            var = self.kd * self.Bd[None, :] - s @ self.W2 + self.d[None, :]; sig = torch.sqrt(torch.clamp(var, min=1e-12))
            mu = mean * self.ys + self.ym; sr = torch.clamp(sig * self.ys, min=SIGMA_FLOOR)
            # local-residual spread (10 nearest out-of-sample residuals) and applicability-domain inflation
            Xq = Xs64.float() if self.lowprec else Xs64
            dl = torch.cdist(Xq, self.res_X); il = dl.topk(self.local_q, dim=1, largest=False).indices
            ls = self.res_s[il].std(dim=1, unbiased=True).double(); lp = self.res_p[il].std(dim=1, unbiased=True).double()
            da = torch.cdist(Xq, self.ad_X).topk(self.ad_k, dim=1, largest=False).values.mean(dim=1).double()
            ratio = da / max(self.ad_tau, 1e-12); infl = torch.where(ratio > 0.75, 1.0 + self.ad_eta * (ratio - 0.75) / 0.25, torch.ones_like(ratio)).clamp(1.0, 3.0)
            sS = torch.maximum(torch.clamp(sr[:, 0], min=1e-12), torch.clamp(ls, min=1e-12)) * infl
            sQ = torch.maximum(torch.clamp(sr[:, 1], min=1e-12), torch.clamp(lp, min=1e-12)) * infl
            # support score
            dv = torch.cdist(Xq, self.sv) ** 2; dec = ((torch.exp(-self.gamma * dv) @ self.dual) + self.icpt).double()
            oc = torch.sigmoid(torch.clamp(dec / self.scale, -60.0, 60.0))
            kld = (torch.logsumexp(-(torch.cdist(Xq, self.kde_X) ** 2) / self.kde_h2, dim=1) + self.kde_norm).double()
            kd_ = torch.clamp(torch.exp(torch.clamp(torch.clamp(kld - self.kde_max, max=0.0), min=-745.0)), 0.0, 1.0)
            sup = torch.clamp(0.5 * oc + 0.5 * kd_, 0.0, 1.0)
        return mu[:, 0], mu[:, 1], sS, sQ, sup

    # ------------------------------------------------------------------ public API
    def _nom_tensors(self, noms):
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=torch.float64, device=self.dev)
        return dict(C=t([n["C"] for n in noms]), r=t([n["r"] for n in noms]), A=t([n["A"] for n in noms]), P=t([n["P"] for n in noms]), S=t([n["S"] for n in noms]),
                    phi=t([n["phi"] for n in noms]), g=t(np.array([np.asarray(n["g"], float) for n in noms])))

    def arrays_t(self, nom, bank, with_phi=False):
        """nom: dict of float64 GPU tensors (C,r,A,P,S,phi (p,), g (p,6)). Returns (P,N) torch tensors muS, muQ, sS, sQ, support (+ achieved porosity if with_phi)."""
        bt = self.bank_tensors(bank); N = bt["n"]; P = nom["C"].shape[0]; per = max(1, self.chunk // N); outs = [[] for _ in range(6 if with_phi else 5)]
        for a in range(0, P, per):
            X6, q = self.scenarios({k: v[a:a + per] for k, v in nom.items()}, bt); res = self.model(*self.standardise(X6, q))
            for k in range(5): outs[k].append(res[k].reshape(-1, N))
            if with_phi: outs[5].append(X6[:, 0].reshape(-1, N))
        return [torch.cat(o, 0) for o in outs]

    def resid_tables(self):
        """Porosity-band residual tables for the porosity-conditional sampler used by acceptance (3 bands cut at the bank's own porosity terciles)."""
        if getattr(self, "_rt", None) is None:
            import porosity_conditional_residual as PCR
            merged, edges, _ = PCR._load_binned_bank(C.BANK_CSV, 3)
            ns = [int((merged["_bin"] == b).sum()) for b in range(3)]; M = max(ns); Es = np.zeros((3, M)); Eq = np.zeros((3, M))
            for b in range(3):
                sub = merged[merged["_bin"] == b]; Es[b, :ns[b]] = sub["signed_standardised_strength_residual"].to_numpy(float); Eq[b, :ns[b]] = sub["signed_standardised_logk_residual"].to_numpy(float)
            t = lambda a, dt=torch.float64: torch.as_tensor(a, dtype=dt, device=self.dev)
            self._rt = dict(Es=t(Es), Eq=t(Eq), n=t(ns), edges=t(np.asarray(edges, float)[1:-1]))
        return self._rt

    def conditional_draws(self, phi, u):
        """Per scenario: pick a paired residual from the bank rows in THAT scenario's porosity band (u: (N,) uniform common random numbers)."""
        rt = self.resid_tables(); b = torch.bucketize(phi.contiguous(), rt["edges"], right=False).clamp(0, 2)          # == np.digitize(phi, edges[1:-1], right=True)
        n = rt["n"][b]; idx = torch.minimum((u[None, :] * n).long(), (n - 1).long())
        return rt["Es"][b, idx], rt["Eq"][b, idx]

    def stats_t(self, nom, bank, eS, eQ, tail_frac=0.10, u=None):
        """All-GPU search statistics (torch tensors (P,)): rhat, cvar (-inf if any scenario unsupported), support_fraction.
        Pooled residuals eS/eQ ((N,) tensors) by default; if u ((N,) uniforms) is given, residuals are drawn from each scenario's own porosity band (the acceptance sampler)."""
        if u is None:
            muS, muQ, sS, sQ, sup = self.arrays_t(nom, bank); S_real = muS + sS * eS[None, :]; Q_real = muQ + sQ * eQ[None, :]
        else:
            muS, muQ, sS, sQ, sup, phi = self.arrays_t(nom, bank, with_phi=True); es, eq = self.conditional_draws(phi, u); S_real = muS + sS * es; Q_real = muQ + sQ * eq
        insup = sup >= C.SUPPORT_THR
        ok = insup & (S_real >= C.S_STAR) & (Q_real >= C.LOGK_STAR)
        s_scale, k_scale = C.CS.resolve_scales()
        jm = torch.minimum((S_real - C.S_STAR) / s_scale, (Q_real - C.LOGK_STAR) / k_scale)
        k = max(1, int(math.ceil(tail_frac * jm.shape[1])))
        cvar = jm.topk(k, dim=1, largest=False).values.mean(dim=1); cvar = torch.where(insup.all(dim=1), cvar, torch.full_like(cvar, -math.inf))
        return dict(rhat=ok.double().mean(1), cvar=cvar, support_fraction=insup.double().mean(1))

    def arrays(self, noms, bank):
        """(P,N) torch tensors muS, muQ, sS, sQ, support for every design (dict list) under the SAME bank."""
        return self.arrays_t(self._nom_tensors(noms), bank)

    def search_stats(self, noms, bank, e_strength, e_logk, tail_frac=0.10):
        """Same statistics as corelib.batch_search_stats, on the GPU. Returns numpy arrays (P,)."""
        t = lambda a: torch.as_tensor(a, dtype=torch.float64, device=self.dev)
        st = self.stats_t(self._nom_tensors(noms), bank, t(e_strength), t(e_logk), tail_frac)
        return {k: v.cpu().numpy() for k, v in st.items()}

    def scenario_numpy(self, nominal, bank):
        """Drop-in for efficient_verification.compute_scenario_predictions: (muS, muQ, sS, sQ, support, phi_actual) as numpy for ONE design."""
        muS, muQ, sS, sQ, sup = self.arrays([nominal], bank); phi = float(nominal["phi"]) + bank.eps_phi_pp
        return tuple(x[0].cpu().numpy() for x in (muS, muQ, sS, sQ, sup)) + (np.asarray(phi),)
