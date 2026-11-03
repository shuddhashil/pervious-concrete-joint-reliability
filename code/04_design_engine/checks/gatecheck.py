# recompute the 20/4 368 kg/m3 design's gate penalty components on CPU
import sys, torch
sys.path.insert(0, "."); from defengine import corelib as C; C.engine() if hasattr(C, "engine") else None
from defengine.gpu_gate import GPUGate
d = dict(C=368.0, r=0.2656354234473272, A=1588.7651466556208, P=1.8839001808017795, S=0.0, phi=20.402895364345884)
g = [1.850371707708594e-17, 1.850371707708594e-17, 0.036675386940159496, 0.5870923051609132, 0.35762170909845875, 0.0186105988004685]
nom = {k: torch.tensor([v], dtype=torch.float64) for k, v in d.items()}; nom["g"] = torch.tensor([g], dtype=torch.float64)
G = GPUGate("cpu"); out = G.penalties(nom)
print({k: float(v[0]) for k, v in out.items()}, "recipe thr", G.gate_thr)
# where porosity is plausible for this recipe
Xraw = torch.cat([nom["r"][:, None], nom["C"][:, None], nom["P"][:, None], nom["S"][:, None], nom["A"][:, None] * nom["g"]], 1)
Xt = torch.cat([Xraw[:, :1], torch.log(Xraw[:, 1:2]), torch.log1p(Xraw[:, 2:])], 1); Xs = (Xt - G.sm) / G.ssd
m = (G.density(Xs)[0] >= G.thr); gr = G.grid[m]; print("plausible porosity grid points:", round(float(gr.min()), 2) if len(gr) else None, "to", round(float(gr.max()), 2) if len(gr) else None)
