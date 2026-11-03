"""HDR utilities: density -> highest-density region, gaps never filled.

An HDR at mass level 1-alpha is {phi : f(phi|x) >= c} with c chosen so the
set carries 1-alpha mass under the estimated conditional density. Multimodal
densities yield disconnected interval unions; min/max convexification is
FORBIDDEN here.
"""

import numpy as np


def normalize_density(dens, step):
    """Normalize a grid density to integrate to 1 (trapezoid)."""
    dens = np.asarray(dens, float)
    tot = float(np.trapezoid(dens, dx=step))
    if not np.isfinite(tot) or tot <= 0:
        raise ValueError('non-normalizable density')
    return dens / tot, tot


def density_to_hdr(dens, grid, level):
    """HDR at mass `level` (e.g. 0.9). Returns dict with disjoint intervals.

    intervals: list of (lo, hi) in grid units, sorted, non-adjacent.
    total_measure: sum of component widths (percentage points).
    n_modes: number of connected components.
    threshold: density level c used.
    """
    dens = np.asarray(dens, float)
    grid = np.asarray(grid, float)
    step = float(grid[1] - grid[0])
    dens, _ = normalize_density(dens, step)
    order = np.argsort(-dens, kind='stable')
    cum = np.cumsum(dens[order]) * step
    k = int(np.searchsorted(cum, level, side='left'))
    k = min(k, len(dens) - 1)
    thr = float(dens[order[k]])
    mask = dens >= thr
    intervals = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            intervals.append((float(grid[i]), float(grid[j])))
            i = j + 1
        else:
            i += 1
    total = float(mask.sum() * step)
    return {'intervals': intervals, 'total_measure': total,
            'n_modes': len(intervals), 'threshold': thr,
            'empty': len(intervals) == 0}


def mask_to_intervals(mask, grid):
    """Contiguous runs of a boolean grid mask -> (lo, hi) intervals."""
    mask = np.asarray(mask, bool)
    grid = np.asarray(grid, float)
    intervals = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            intervals.append((float(grid[i]), float(grid[j])))
            i = j + 1
        else:
            i += 1
    step = float(grid[1] - grid[0])
    return intervals, float(np.asarray(mask).sum() * step)


def covers(intervals, value):
    """Membership of a scalar in an interval union."""
    return any(lo <= value <= hi for lo, hi in intervals)


def conditional_cdf(dens, grid):
    """CDF of a grid density (trapezoid-normalized)."""
    dens = np.asarray(dens, float)
    step = float(grid[1] - grid[0])
    dens, _ = normalize_density(dens, step)
    cdf = np.cumsum(dens) * step
    cdf = np.clip(cdf, 0.0, 1.0)
    return cdf


def crps_from_density(dens, grid, obs):
    """CRPS of an estimated conditional distribution at one observation."""
    dens = np.asarray(dens, float)
    step = float(grid[1] - grid[0])
    cdf = conditional_cdf(dens, grid)
    ind = (np.asarray(grid, float) >= obs).astype(float)
    return float(np.sum((cdf - ind) ** 2) * step)


def cde_ise_terms(dens, grid, obs):
    """Per-query terms of the L2 CDE-loss proxy: int f^2 and f(obs).

    loss contribution = int_f2 - 2 * f_obs (lower is better).
    """
    dens = np.asarray(dens, float)
    step = float(grid[1] - grid[0])
    dens_n, _ = normalize_density(dens, step)
    int_f2 = float(np.sum(dens_n ** 2) * step)
    f_obs = float(np.interp(obs, grid, dens_n))
    return int_f2, f_obs
