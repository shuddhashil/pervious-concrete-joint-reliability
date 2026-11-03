"""Paste film thickness guard . Packing models are NOT used.
h [mm] = V_paste / (V_agg * A_ag), the same formula and constants as PackingPhysics_20261002/phase4.py column `h_ca_mm` (median 0.51 mm, range 0.15-1.83 mm over the 650 eligible rows)
and PackingPhysics_Muse_20261002/packing_models.paste_film (h = Vpaste / (6 * Vagg * sum(y/d))):
  V_paste = cement/3150 + water/1000   (m3/m3; water = w/c * cement; NO SP, NO sand -- the existing code's h_ca_mm; the SP variant h_ca_sp_mm differs by < 1 %)
  V_agg   = CA/2700                    (coarse aggregate only; sand ignored as in the existing code)
  A_ag    = 6 * sum_k g_k / d_k        (mm^-1; spheres, d_k = geometric-mean diameter of each of the six sieve bins = frozen core DIAM, mm)
Floor (G-3, generic): h_min(S*, K*) = smallest h among recorded de-duplicated units meeting the FULL target (measured mean CS >= S* AND k >= K*); the G-2 floor (CS >= S* only) is kept as h_min_cs_only for reporting. The walk requires h >= h_min at the nominal blend and at all 10 shifted blends."""
import numpy as np, torch
from gengine.common import C, T

RHO_C, RHO_W, RHO_AG = 3150.0, 1000.0, 2700.0
DIAM_MM = np.asarray(C.E.DIAM, float)


def h_mm(Cc, r, A, g):
    """Cc, r, A (n,) and g (n,6) torch float64 tensors (or numpy) -> h in mm, same shape as Cc."""
    D = torch.as_tensor(DIAM_MM, dtype=torch.float64, device=g.device) if torch.is_tensor(g) else DIAM_MM
    vp = Cc / RHO_C + r * Cc / RHO_W; vagg = A / RHO_AG; aag = 6.0 * (g / D).sum(-1)
    return vp / (vagg * aag)


def unit_h(data):
    """h (mm) of every recorded de-duplicated unit (numpy); recipe = Data.Ru [C, w/c, CA, SP, sand, phi], blend = Data.Gu."""
    return h_mm(data.Ru[:, 0], data.Ru[:, 1], data.Ru[:, 2], data.Gu)


def paste_floor(data, S, K=None):
    """Floor for target (S, K): h_min = smallest h among units meeting the FULL target (CS >= S AND k >= K). K=None -> CS-only (returned as h_min_cs_only in every case).
    Option 1 (author-locked, generic engine): if NO recorded unit meets the full target the floor is NOT computed and NOT an error: h_min = None, status "none", reason text; the walk then skips the paste check and every result is labelled
    "model-suggested, no paste floor". If it rests on fewer than 3 units, `warning` says so. Returns dict(h_min, status, reason, warning, h_min_cs_only, p5, p10 (of the CS-only set), argmin units, counts...)."""
    h = unit_h(data); cs = np.asarray(data.U.CS); m0 = cs >= S; m = m0 & (np.asarray(data.U.K) >= K) if K is not None else m0
    ids0 = np.flatnonzero(m0); ids = np.flatnonzero(m); n_full = int(m.sum()); hs0 = h[m0]; srt = np.sort(h[m])
    out = dict(S=float(S), K=None if K is None else float(K), n_units_cs_ge_S=int(m0.sum()), n_units_full_target=n_full, n_units_total=int(len(h)), h_all_units=dict(min=float(h.min()), median=float(np.median(h)), max=float(h.max())),
               h_min=None, status="none", reason=None, warning=None, h_min_cs_only=None, argmin_unit_cs_only=None, h_p5=None, h_p10=None, h_median=None, h_max=None, argmin_unit=None, argmin_rec=None, second_lowest_h=None, third_lowest_h=None)
    if len(ids0):
        out.update(h_min_cs_only=float(hs0.min()), argmin_unit_cs_only=int(ids0[np.argmin(hs0)]), h_p5=float(np.percentile(hs0, 5)), h_p10=float(np.percentile(hs0, 10)), h_median=float(np.median(hs0)), h_max=float(hs0.max()))
    if n_full == 0:
        out["reason"] = (f"no recorded de-duplicated unit has CS >= {S:g} MPa" + (f" AND k >= {K:g} mm/s" if K is not None else "") + f" (units with CS >= {S:g}: {int(m0.sum())}); paste floor not computed, paste check skipped"); return out
    j = int(ids[np.argmin(h[m])])
    out.update(h_min=float(h[m].min()), status="computed", argmin_unit=j, second_lowest_h=float(srt[1]) if len(srt) > 1 else None, third_lowest_h=float(srt[2]) if len(srt) > 2 else None,
               argmin_rec=dict(C=float(data.Ru[j, 0]), r=float(data.Ru[j, 1]), A=float(data.Ru[j, 2]), P=float(data.Ru[j, 3]), S=float(data.Ru[j, 4]), phi=float(data.Ru[j, 5]), g=data.Gu[j].tolist(), CS=float(data.U.CS[j]), K=float(data.U.K[j]), nrows=int(data.U.nrows[j])))
    if n_full < 3: out["warning"] = f"WARNING: the paste floor rests on only {n_full} recorded unit(s) (< 3); treat it as weak"
    return out
