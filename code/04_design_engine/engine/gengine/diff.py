"""Differentiable A3-1 scenario evaluator for the gradient polish. _scenarios/_standardise/_model are COPIES of DefensibleEngineV2_20261003/v2engine/diff.py (forward evaluation only:
only the DESIGN variables get gradients, model weights are never touched). smooth_reliability differs from the V2 function only in taking a per-design cement vector and calling nbhd directly.
The exact estimator re-scores every polished design, so this surrogate only steers; support COUNTS are never differentiated."""
import torch
from gengine.common import C, nbhd, decode


def _scenarios(eng, nom, bt):
    C_, r_, A_, P_, S_, phi_, g_ = (nom[k] for k in ("C", "r", "A", "P", "S", "phi", "g")); p = C_.shape[0]; N = bt["n"]
    Ca = C_[:, None] * (1 + bt["dC"][None, :]); Wa = (r_ * C_)[:, None] * (1 + bt["dW"][None, :]); wc = Wa / Ca
    Ab = torch.clamp((A_[:, None] * g_)[:, None, :] * (1 + bt["dA"][None, :, :]), min=0.0); Aa = Ab.sum(2); ga = Ab / Aa[..., None]
    SPa = torch.where(P_[:, None] > 0, P_[:, None] * (1 + bt["dSP"][None, :]), torch.zeros_like(Ca)); Sa = torch.where(S_[:, None] > 0, S_[:, None] * (1 + bt["dS"][None, :]), torch.zeros_like(Ca))
    phi = phi_[:, None] + bt["eps"][None, :]; fl = lambda x: x.reshape(p * N)
    q = eng.quantiles(ga.reshape(p * N, 6))
    return torch.stack([fl(phi), fl(wc), fl(Ca), fl(Aa), fl(SPa), fl(Sa)], 1), q


def _standardise(eng, X6, q):
    X = torch.cat([X6, q], 1); cols = []
    for j in range(X.shape[1]):
        c = X[:, j]
        if j in eng.log_idx: c = torch.log(c)
        elif j in eng.log1p_idx: c = torch.log1p(c)
        cols.append(c)
    Xs = (torch.stack(cols, 1) - eng.xm) / eng.xs
    return Xs, Xs.to(torch.float32)


def _model(eng, Xs64, Xs):
    Ck = eng.dk.forward(Xs, eng.Xtr); Ck = Ck.to_dense() if hasattr(Ck, "to_dense") else Ck
    mean = eng.m0[None, :] + ((Ck @ eng.alpha).double()) @ eng.B.T
    U = Ck @ eng.QK; s = ((U * U) @ eng.g32).double()
    var = eng.kd * eng.Bd[None, :] - s @ eng.W2 + eng.d[None, :]; sig = torch.sqrt(torch.clamp(var, min=1e-12))
    mu = mean * eng.ys + eng.ym; sr = torch.clamp(sig * eng.ys, min=1e-6)
    Xq = Xs64.float() if eng.lowprec else Xs64
    with torch.no_grad():
        il = torch.cdist(Xq, eng.res_X).topk(eng.local_q, dim=1, largest=False).indices
        ls = eng.res_s[il].std(dim=1, unbiased=True).double(); lp = eng.res_p[il].std(dim=1, unbiased=True).double()
    da = torch.cdist(Xq, eng.ad_X).topk(eng.ad_k, dim=1, largest=False).values.mean(dim=1).double()
    ratio = da / max(eng.ad_tau, 1e-12); infl = torch.where(ratio > 0.75, 1.0 + eng.ad_eta * (ratio - 0.75) / 0.25, torch.ones_like(ratio)).clamp(1.0, 3.0)
    sS = torch.maximum(torch.clamp(sr[:, 0], min=1e-12), torch.clamp(ls, min=1e-12)) * infl; sQ = torch.maximum(torch.clamp(sr[:, 1], min=1e-12), torch.clamp(lp, min=1e-12)) * infl
    dv = torch.cdist(Xq, eng.sv) ** 2; dec = ((torch.exp(-eng.gamma * dv) @ eng.dual) + eng.icpt).double(); oc = torch.sigmoid(torch.clamp(dec / eng.scale, -60.0, 60.0))
    kld = (torch.logsumexp(-(torch.cdist(Xq, eng.kde_X) ** 2) / eng.kde_h2, dim=1) + eng.kde_norm).double()
    kd_ = torch.clamp(torch.exp(torch.clamp(torch.clamp(kld - eng.kde_max, max=0.0), min=-745.0)), 0.0, 1.0)
    return mu[:, 0], mu[:, 1], sS, sQ, torch.clamp(0.5 * oc + 0.5 * kd_, 0.0, 1.0)


def prepare(eng):
    for p in eng.dk.parameters(): p.requires_grad_(False)


def smooth_reliability(eng, Z, Cv, tau, bk, temp=0.35, chunk_designs=4):
    """Z (n,11) tensor with requires_grad, Cv (n,) cement. Returns (n,) smooth worst-neighbour joint pass probability."""
    bank, u = bk; bt = eng.bank_tensors(bank); N = bt["n"]; s_scale, k_scale = C.CS.resolve_scales(); outs = []
    for a in range(0, Z.shape[0], chunk_designs):
        nom = decode(Z[a:a + chunk_designs], Cv[a:a + chunk_designs]); n = nom["C"].shape[0]; nc, owner, kk = nbhd(nom, tau)
        X6, q = _scenarios(eng, nc, bt); Xs64, Xs = _standardise(eng, X6, q); muS, muQ, sS, sQ, sup = _model(eng, Xs64, Xs)
        phi = X6[:, 0].detach().reshape(-1, N); es, eq = eng.conditional_draws(phi, u)
        S_real = muS.reshape(-1, N) + sS.reshape(-1, N) * es; Q_real = muQ.reshape(-1, N) + sQ.reshape(-1, N) * eq
        ps = torch.sigmoid((S_real - C.S_STAR) / (temp * s_scale)) * torch.sigmoid((Q_real - C.LOGK_STAR) / (temp * k_scale)) * torch.sigmoid((sup.reshape(-1, N) - C.SUPPORT_THR) / 0.02)
        row = ps.mean(1); outs.append(torch.full((n,), 2.0, dtype=row.dtype, device=row.device).scatter_reduce(0, owner, row, "amin", include_self=True))
    return torch.cat(outs)
