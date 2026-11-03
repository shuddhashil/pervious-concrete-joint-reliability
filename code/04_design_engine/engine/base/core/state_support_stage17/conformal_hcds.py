"""Conformalized highest conditional density sets (CHCDS).

Second layer over the base density estimators B2..B8: the density SHAPE is
kept (multimodal/nonconvex preserved), only the density-level threshold is
recalibrated on held-out CAL density values. Base HDR and conformalized HDR
results are always reported separately.

Finite-sample validity is MARGINAL over x only; exact pointwise conditional
coverage is not claimed. Local/subgroup calibration diagnostics mandatory.
Conformal calibration never proves physical attainability.
"""

import numpy as np

from hdr import mask_to_intervals


def calibrate_levels(cal_values, alphas=(0.2, 0.1, 0.05)):
    """Density thresholds t_alpha from CAL conformity scores.

    scores = f(phi_cal | x_cal) (higher = more conforming).
    t_alpha = quantile(scores, ceil((n+1)*alpha)/n) -- standard split-
    conformal finite-sample correction. Set = {phi : f(phi|x) >= t}.
    """
    cal_values = np.asarray(cal_values, float)
    cal_values = cal_values[np.isfinite(cal_values)]
    n = len(cal_values)
    assert n > 0
    out = {}
    for a in alphas:
        q = min(1.0, float(np.ceil((n + 1) * a)) / n)
        out[float(a)] = float(np.quantile(cal_values, q))
    return out


def apply_levels(dens_grid, grid, thresholds):
    """Threshold a (nq, G) density matrix at each level -> interval unions.

    Returns {alpha: {'intervals', 'total_measure', 'n_modes', 'empty'}}.
    Gaps are preserved (no convexification); empty sets allowed and reported.
    """
    dens_grid = np.asarray(dens_grid, float)
    res = {}
    for a, t in thresholds.items():
        masks = dens_grid >= float(t)
        ivs, tot, nmode, nempty = [], [], [], 0
        for m in masks:
            intervals, total = mask_to_intervals(m, grid)
            ivs.append(intervals)
            tot.append(total)
            nmode.append(len(intervals))
            nempty += int(len(intervals) == 0)
        res[float(a)] = {'intervals': ivs, 'total_measure': np.array(tot),
                         'n_modes': np.array(nmode), 'n_empty': nempty}
    return res
