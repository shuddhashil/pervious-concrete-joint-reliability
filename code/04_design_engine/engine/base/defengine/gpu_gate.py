"""GPU port of the SEARCH gate (decision E-04): recipe gate (10th-NN distance), gradation W1 gate, deployment window, and the porosity-support rule with
its density stack B2 + B5 + B6 + B7-fast (B3/B4 have ~zero weight; B7's per-query mixture is replaced by one Gaussian on the same 30 neighbours).
Interval logic reproduces hdr.mask_to_intervals / covers / _dist_to_intervals exactly (runs of consecutive above-threshold grid points).
All float64 on the GPU (matrices are small). Tested against the CPU SearchGate and the frozen verdicts in tests/test_gpu_gate.py."""
import sys, math
import numpy as np, torch
from . import corelib as C

S17, GS = C.S17, C.GS


class GPUGate:
    def __init__(self, device="cuda"):
        self.dev = torch.device(device); B = S17._bundle(); CD = sys.modules["conditional_density"]; self.FLOOR = CD.FLOOR
        T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=torch.float64, device=self.dev)
        self.sm, self.ssd = T(B["scaler_mean"]), T(B["scaler_sd"]); self.grid = T(B["phi_grid"]); self.thr = float(B["conformal_thresholds"][0.1])
        self.gate_X, self.gate_k, self.gate_thr = T(B["gate_X_fit"]), int(B["gate_k"]), float(B["gate_threshold"])
        m = B["fitted_members"]; w = {n: float(x) for n, x in zip(B["member_names"], B["weights"])}; self.w = w
        b2, b5, b6, b7 = m["B2"], m["B5"], m["B6"], m["B7"]
        self.b2_S, self.b2_phi, self.b2_k, self.b2_h = T(b2.S_tr), T(b2.phi_tr), int(b2.k), float(b2.h)
        self.b5_X, self.b5_phi, self.b5_h, self.b5_hphi = T(b5.Xs), T(b5.phi_tr), float(b5.h), float(b5.h_phi)
        self.b5_K = self._gauss(self.grid[None, :], self.b5_phi[:, None], self.b5_hphi)                       # (561, G)
        self.b6 = dict(pi=T(b6.pi), mux=T(b6.mux), mup=T(b6.mup), Sxx_inv=T(b6.Sxx_inv), Sxp=T(b6.Sxp), Spp=T(b6.Spp), mu=float(b6.mu), sd=float(b6.sd), d=int(b6.d))
        self.b6["logdet"] = -torch.linalg.slogdet(self.b6["Sxx_inv"])[1]                                       # log det of Sxx = -log det of its inverse
        self.b7_X, self.b7_phi, self.b7_k, self.b7_min = T(b7.Xs), T(b7.phi_tr), int(b7.k), int(b7.min_n)
        Tg, _ = GS.load_bundle(); self.W1_T = torch.cumsum(T(Tg / Tg.sum(1, keepdims=True))[:, :5], 1); self.W_LOG = T(GS.W_LOG); self.w1_eps = float(GS.load_contract().get("epsilon_95"))

    @staticmethod
    def _gauss(x, mu, h):
        h = max(float(h), 1e-9); z = (x - mu) / h
        return torch.exp(-0.5 * z * z) / (h * math.sqrt(2.0 * math.pi))

    def density(self, Xs):
        """Stacked conditional porosity density on the grid for standardised recipes Xs (n,10): (n,G)."""
        n = Xs.shape[0]; G = self.grid; F = self.FLOOR; w = self.w
        D = torch.zeros(n, len(G), dtype=torch.float64, device=self.dev)
        d2 = torch.cdist(Xs, self.b2_S) ** 2                                                                       # B2: k nearest, Gaussian kernel mean
        i2 = d2.topk(self.b2_k, dim=1, largest=False).indices
        D += w["B2"] * (self._gauss(G[None, None, :], self.b2_phi[i2][:, :, None], self.b2_h).mean(1) + F)
        W = torch.exp(-0.5 * (torch.cdist(Xs, self.b5_X) ** 2) / self.b5_h ** 2); sw = W.sum(1)                   # B5: joint kernel density
        bad = sw < 1e-8                                                                                             # far from all data: unconditional density (as the CPU member)
        d5 = (W @ self.b5_K) / torch.where(bad, torch.ones_like(sw), sw)[:, None] + F
        d5 = torch.where(bad[:, None], (self.b5_K.mean(0) + F)[None, :], d5)
        D += w["B5"] * d5
        b = self.b6; K = b["pi"].shape[0]; d = b["d"]; dx = Xs[:, None, :] - b["mux"][None, :, :]                  # B6: joint Gaussian mixture, analytic conditional
        A = torch.einsum("nkd,kde->nke", dx, b["Sxx_inv"]); maha = (A * dx).sum(2)
        logw = torch.log(b["pi"])[None, :] - 0.5 * (d * math.log(2 * math.pi) + b["logdet"][None, :] + maha); logw = logw - torch.logsumexp(logw, 1, keepdim=True); Wc = torch.exp(logw)
        mc = b["mup"][None, :] + (A * b["Sxp"][None, :, :]).sum(2); vc = torch.clamp(b["Spp"] - torch.einsum("kd,kde,ke->k", b["Sxp"], b["Sxx_inv"], b["Sxp"]), min=1e-6)
        mc = mc * b["sd"] + b["mu"]; vc = vc * b["sd"] ** 2; sd = torch.sqrt(vc)[None, :, None]
        z = (G[None, None, :] - mc[:, :, None]) / sd; D += w["B6"] * ((Wc[:, :, None] * torch.exp(-0.5 * z * z) / (sd * math.sqrt(2 * math.pi))).sum(1) + F)
        i7 = (torch.cdist(Xs, self.b7_X) ** 2).topk(self.b7_k, dim=1, largest=False).indices                        # B7-fast: one Gaussian on the 30 nearest porosities
        v = self.b7_phi[i7]; mean = v.mean(1, keepdim=True); sd7 = torch.clamp(v.std(1, unbiased=False, keepdim=True), min=0.3)
        D += w["B7"] * (torch.exp(-0.5 * ((G[None, :] - mean) / sd7) ** 2) / (sd7 * math.sqrt(2 * math.pi)) + F)
        return D

    def penalties(self, nom):
        """nom: dict of float64 GPU tensors C,r,A,P,S,phi (p,), g (p,6). Returns dict of (p,) tensors: penalty (+ components)."""
        C_, r_, A_, P_, S_, phi, g = (nom[k] for k in ("C", "r", "A", "P", "S", "phi", "g")); p = C_.shape[0]
        Xraw = torch.cat([r_[:, None], C_[:, None], P_[:, None], S_[:, None], A_[:, None] * g], 1)
        Xt = torch.cat([Xraw[:, :1], torch.log(Xraw[:, 1:2]), torch.log1p(Xraw[:, 2:])], 1); Xs = (Xt - self.sm) / self.ssd
        dk = torch.kthvalue(torch.cdist(Xs, self.gate_X), self.gate_k, dim=1).values                                # recipe gate: 10th-NN distance
        rpass = dk <= self.gate_thr; rex = torch.clamp((dk - self.gate_thr) / self.gate_thr, min=0.0)
        Fg = torch.cumsum(g[:, :5], 1); w1 = ((Fg[:, None, :] - self.W1_T[None, :, :]).abs() * self.W_LOG).sum(-1).min(1).values; vw = torch.clamp((w1 - self.w1_eps) / self.w1_eps, min=0.0)
        vd = torch.where((phi >= S17.DEPLOY_LO) & (phi <= S17.DEPLOY_HI), torch.zeros_like(phi), torch.minimum((phi - S17.DEPLOY_LO).abs(), (phi - S17.DEPLOY_HI).abs()) / 10.0)
        D = self.density(Xs); mask = D >= self.thr; G = self.grid; n = len(G)
        j = torch.clamp(torch.searchsorted(G, phi.contiguous(), right=True) - 1, 0, n - 1)                            # last grid point <= phi
        mj = torch.gather(mask, 1, j[:, None])[:, 0]; mj1 = torch.gather(mask, 1, torch.clamp(j + 1, max=n - 1)[:, None])[:, 0]
        gj = G[j]; inside = (phi >= G[0]) & (phi <= G[-1]) & mj & ((phi == gj) | ((j < n - 1) & mj1))
        dist = torch.where(mask, (G[None, :] - phi[:, None]).abs(), torch.full_like(D, math.inf)).min(1).values
        vphi = torch.where(rpass & ~inside, dist, torch.zeros_like(dist))                                            # no HDR is fabricated outside the recipe domain
        return dict(penalty=rex + vphi / 10.0 + vd + vw, recipe_pass=rpass, phi_inside=inside & rpass, w1_pass=vw <= 0, deploy_ok=vd <= 0, v_phi=vphi)
