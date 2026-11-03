"""Shared set-up for the design-space map. Runs INSIDE a process that points at the 18/13 per-target core copy (read only);
the map code never reads the target from the core: pass conditions are rebuilt from realised strength / log-permeability. Frozen files are only imported."""
import os, sys, json, math, time
SM = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ROOT = os.path.dirname(SM)
TGT = os.path.join(ROOT, "targets", "S18_K13_R0.90")
os.environ.setdefault("GENGINE_BASE", TGT); os.environ.setdefault("GENGINE_CONFIG", os.path.join(TGT, "CONFIG.json")); os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True; sys.path.insert(0, ROOT)
import numpy as np, torch
from gengine import config, walk, gpu_lb, paste as PS
from gengine.common import C, T, DEV, BND, decode, nbhd, nom_dicts, gpu_temp, cool_down, SHIFT_LABELS, neighbours_t

S_GRID = np.arange(5.0, 51.0, 1.0)                         # 46 strength targets, 5..50 MPa step 1
K_GRID = np.geomspace(0.5, 30.0, 20)                       # 20 log-spaced permeability targets, 0.5..30 mm/s
RS = (0.80, 0.90, 0.95); M_WALK = 0.002; TOPK = 10; N_B = 4096
PHI_WIN = (float(C.S17.DEPLOY_LO), float(C.S17.DEPLOY_HI))


def get_ctx():
    cfg, H = config.load(); ctx = walk.Ctx(cfg, H); return cfg, H, ctx


class TempGuard:
    """GPU heat rule: above 76 C (rule: 78) pause until 70 C; records the peak and every pause."""
    def __init__(self, log=print): self.peak = gpu_temp(); self.pauses = []; self.log = log
    def __call__(self):
        t = gpu_temp(); self.peak = max(self.peak, t)
        if t > 76:
            s = cool_down(76, 70, log=self.log); self.pauses.append(dict(at=time.strftime("%H:%M:%S"), temp=t, seconds=round(s))); self.peak = max(self.peak, gpu_temp())
