"""Shared pieces for the Guided engine phase 2. GPU python (system anaconda). Frozen core is only READ.
neighbours_t / nbhd: copied from DefensibleEngineV2_20261003/v2engine/core.py (neighbours_t, RobustEval.nbhd without the red-team 'extra'); decode = gpu_solver.decode with a per-design cement vector."""
import os, sys, json, math, hashlib, time, subprocess
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
BASE = os.environ.get("GENGINE_BASE") or os.path.join(ROOT, "base")     # folder holding core/ and defengine/; a per-target copy sets GENGINE_BASE (run_target.py)
for p in (BASE, ROOT):
    if p not in sys.path: sys.path.insert(0, p)
import numpy as np, pandas as pd, torch
from defengine import corelib as C, gpu as GP, gpu_gate as GG
from defengine.gpu_solver import project_simplex
from gengine import gpu_lb

DEV = torch.device("cuda")
T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=torch.float64, device=DEV)
BND = {k: (float(v[0]), float(v[1])) for k, v in C.SC.BND.items()}; LNA = (C.SC.LNA_LO, C.SC.LNA_HI)
W_LOG = np.asarray(C.GS.W_LOG, float)           # six-bin W1 weights |ln d_k - ln d_(k+1)| (gradation_support)
SHIFT_LABELS = ["nominal"] + [f"{s}->{d}" for j in range(5) for s, d in ((j, j + 1), (j + 1, j))]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()


def gpu_temp():
    try: return int(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip().split()[0])
    except Exception: return -1


def cool_down(limit=80, resume=70, log=print):
    """Heat rule: above `limit` C pause (sleep) until below `resume`; returns seconds paused."""
    return 0.0  # Colab kit: heat pause disabled
    t0 = time.time(); t = gpu_temp()
    if t <= limit: return 0.0
    log(f"GPU {t} C > {limit}: pausing")
    while gpu_temp() > resume: time.sleep(15)
    return time.time() - t0


# ------------------------------------------------------------------ design vector
def decode(Z, Cv):
    """Z (n,11) unit cube [r, A, P, S, phi, m1..m6], Cv (n,) cement kg/m3 -> nominal tensors."""
    lin = lambda u, k: BND[k][0] + u * (BND[k][1] - BND[k][0]); Z = torch.clamp(Z, 0.0, 1.0)
    return dict(C=Cv, r=lin(Z[:, 0], "r"), A=torch.exp(LNA[0] + Z[:, 1] * (LNA[1] - LNA[0])), P=lin(Z[:, 2], "P"), S=lin(Z[:, 3], "S"), phi=lin(Z[:, 4], "phi"), g=project_simplex(Z[:, 5:11]))


def encode(rec, g):
    """recorded design (C excluded) -> Z (11,) numpy + list of clamped coordinates. rec = dict r,A,P,S,phi."""
    z = [(rec["r"] - BND["r"][0]) / (BND["r"][1] - BND["r"][0]), (math.log(rec["A"]) - LNA[0]) / (LNA[1] - LNA[0])] + [(rec[k] - BND[k][0]) / (BND[k][1] - BND[k][0]) for k in ("P", "S", "phi")]
    z = np.array(z + list(np.asarray(g, float) / np.sum(g)), float); clamped = [i for i in range(11) if z[i] < 0 or z[i] > 1]
    return np.clip(z, 0, 1), clamped


def nom_dicts(nom):
    g = nom["g"].detach().cpu().numpy(); n = nom["C"].shape[0]
    return [dict(C=float(nom["C"][i]), r=float(nom["r"][i]), A=float(nom["A"][i]), P=float(nom["P"][i]), S=float(nom["S"][i]), phi=float(nom["phi"][i]), g=g[i].tolist()) for i in range(n)]


# ------------------------------------------------------------------ 2 percent single adjacent-bin shifts
def neighbours_t(g, tau):
    """g (n,6) -> (gk (n,K,6), valid (n,K)). K = 11 (nominal + 5 adjacent pairs x 2 directions, mass tau); a transfer that makes a bin negative is invalid (skipped)."""
    n = g.shape[0]
    if tau <= 0: return g[:, None, :].clone(), torch.ones(n, 1, dtype=torch.bool, device=g.device)
    gk = g[:, None, :].repeat(1, 11, 1); valid = torch.ones(n, 11, dtype=torch.bool, device=g.device); k = 1
    for j in range(5):
        for s, d in ((j, j + 1), (j + 1, j)):
            valid[:, k] = (g[:, s] - tau) >= 0; gk[:, k, s] = g[:, s] - tau; gk[:, k, d] = g[:, d] + tau; k += 1
    return gk, valid


def nbhd(nom, tau):
    """Flatten the valid neighbourhood of every design: (nc dict of (m,) / (m,6), owner (m,), kk (m,) slot 0..10; slot 0 = nominal)."""
    gk, valid = neighbours_t(nom["g"], tau); idx = valid.nonzero(); owner, kk = idx[:, 0], idx[:, 1]
    return {k: (v[owner] if k != "g" else gk[owner, kk]) for k, v in nom.items()}, owner, kk


# ------------------------------------------------------------------ family rule
class Fam:
    """Family cells: a design belongs to family f iff its nearest recorded blend (W1, all 83) has family label f AND that distance <= r_fam (plan section 6)."""
    def __init__(self, blends, labels, r_fam):
        self.B = np.asarray(blends, float); self.lab = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=DEV); self.r = float(r_fam)
        self.F = T(np.cumsum(self.B[:, :5], 1)); self.W = T(W_LOG)

    def dist(self, g):
        Fg = torch.cumsum(g[:, :5], 1); d = ((Fg[:, None, :] - self.F[None]).abs() * self.W).sum(-1); return d

    def check(self, g, fam):
        d = self.dist(g); dmin, im = d.min(1); return (self.lab[im] == fam) & (dmin <= self.r), dmin, im

    def project(self, g, fam, iters=12):
        """Pull blends that fail the rule toward their nearest in-family recorded blend (bisection on the mixing weight t; t=1 passes by construction)."""
        ok, _, _ = self.check(g, fam)
        if bool(ok.all()): return g
        d = self.dist(g); d = torch.where((self.lab == fam)[None, :], d, torch.full_like(d, math.inf)); tgt = T(self.B)[d.argmin(1)]
        lo = torch.zeros(g.shape[0], dtype=torch.float64, device=DEV); hi = torch.ones_like(lo)
        for _ in range(iters):
            mid = (lo + hi) / 2; okm, _, _ = self.check((1 - mid[:, None]) * g + mid[:, None] * tgt, fam); hi = torch.where(okm, mid, hi); lo = torch.where(okm, lo, mid)
        out = (1 - hi[:, None]) * g + hi[:, None] * tgt
        return torch.where(ok[:, None], g, out)
