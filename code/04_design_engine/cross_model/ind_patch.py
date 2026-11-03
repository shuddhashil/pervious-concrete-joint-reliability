"""Makes the engine's search run on the independent-GP model (2026-10-30). Import AFTER gengine.common (which puts the base on sys.path).
Replaces, for every GPUEngine instance, (1) GPUEngine.model (exact scoring) and (2) gengine.diff._model (the differentiable copy that steers the
gradient polish) with the same functions in which only the ICM block is swapped for the two independent GPs (IND_oracle.pt, env IND_ORACLE).
Everything from the local-residual spread on is copied unchanged from defengine/gpu.py and gengine/diff.py. The GP block runs in float32 when the
engine is in low-precision search mode (as the ICM block always does) and in float64 otherwise; acceptance decisions are re-scored in float64
afterwards by design_eval.py."""
import os, numpy as np, torch
import indkern
from defengine import gpu as GPUM
from defengine.gpu import SIGMA_FLOOR
from gengine import diff as DF

_O = {}


def _load(eng):
    if _O: return
    O = torch.load(os.environ["IND_ORACLE"], map_location="cpu", weights_only=False)
    assert np.allclose(O["x_mean"], eng.xm.cpu().numpy()) and np.allclose(O["x_std"], eng.xs.cpu().numpy()), "IND scaler differs from the engine's"
    assert np.allclose(O["y_mean"], eng.ym.cpu().numpy()) and np.allclose(O["y_std"], eng.ys.cpu().numpy())
    for dt in (torch.float64, torch.float32):
        f = lambda a: torch.as_tensor(np.asarray(a, float), device=eng.dev).to(dt)
        _O[dt] = dict(X=f(O["X"]), tasks=[dict(spec=t["spec"], a=f(t["a"]), K0i=f(t["K0inv"]), m0=t["m0"], noise=t["noise"], kd=t["kdiag"]) for t in O["tasks"]])
    print("[ind_patch] independent-GP model active", flush=True)


def ind_gp(eng, Xs64):
    _load(eng); dt = torch.float32 if eng.lowprec else torch.float64; P = _O[dt]; X = Xs64.to(dt); mu, var = [], []
    for t in P["tasks"]:
        Kq = indkern.K(t["spec"], X, P["X"]); mu.append(t["m0"] + Kq @ t["a"]); var.append(t["kd"] - ((Kq @ t["K0i"]) * Kq).sum(1) + t["noise"])
    return torch.stack(mu, 1).double(), torch.stack(var, 1).double()


def model_ind(self, Xs64, Xs):
    with torch.no_grad():
        mean, var = ind_gp(self, Xs64); sig = torch.sqrt(torch.clamp(var, min=1e-12))
        mu = mean * self.ys + self.ym; sr = torch.clamp(sig * self.ys, min=SIGMA_FLOOR)
        Xq = Xs64.float() if self.lowprec else Xs64
        dl = torch.cdist(Xq, self.res_X); il = dl.topk(self.local_q, dim=1, largest=False).indices
        ls = self.res_s[il].std(dim=1, unbiased=True).double(); lp = self.res_p[il].std(dim=1, unbiased=True).double()
        da = torch.cdist(Xq, self.ad_X).topk(self.ad_k, dim=1, largest=False).values.mean(dim=1).double()
        ratio = da / max(self.ad_tau, 1e-12); infl = torch.where(ratio > 0.75, 1.0 + self.ad_eta * (ratio - 0.75) / 0.25, torch.ones_like(ratio)).clamp(1.0, 3.0)
        sS = torch.maximum(torch.clamp(sr[:, 0], min=1e-12), torch.clamp(ls, min=1e-12)) * infl
        sQ = torch.maximum(torch.clamp(sr[:, 1], min=1e-12), torch.clamp(lp, min=1e-12)) * infl
        dv = torch.cdist(Xq, self.sv) ** 2; dec = ((torch.exp(-self.gamma * dv) @ self.dual) + self.icpt).double()
        oc = torch.sigmoid(torch.clamp(dec / self.scale, -60.0, 60.0))
        kld = (torch.logsumexp(-(torch.cdist(Xq, self.kde_X) ** 2) / self.kde_h2, dim=1) + self.kde_norm).double()
        kd_ = torch.clamp(torch.exp(torch.clamp(torch.clamp(kld - self.kde_max, max=0.0), min=-745.0)), 0.0, 1.0)
        sup = torch.clamp(0.5 * oc + 0.5 * kd_, 0.0, 1.0)
    return mu[:, 0], mu[:, 1], sS, sQ, sup


def diff_model_ind(eng, Xs64, Xs):
    mean, var = ind_gp(eng, Xs64); sig = torch.sqrt(torch.clamp(var, min=1e-12))
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


GPUM.GPUEngine.model = model_ind
DF._model = diff_model_ind
