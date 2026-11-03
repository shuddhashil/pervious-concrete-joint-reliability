"""Continuous search-only guidance score (blind-discovery repair).

Literal Monte Carlo pass/fail reliability (Rhat) is coarse and plateaued at
small scenario counts, so the global search needs a smooth signal.  This
module provides that signal and NOTHING else:

- scenario margins are normalised by fixed documentary scales that come from
  the frozen model's unit conventions (MPa for strength, log units for
  permeability), never fitted to any historical candidate;
- the summary is a lower-tail statistic (CVaR-like mean over the worst
  ``tail_frac`` of joint margins), which moves smoothly as a basin improves;
- support failure is a hard failure (score is -inf, feasible flag False);
- literal Rhat must still be computed and logged by the caller;
- final selection and decisive verification use literal targets and the
  existing one-sided 95% LCB rule only.

This module contains NO acceptance logic: there is no function that returns
PASS/FAIL, and ``search_score`` can never mark a candidate as final PASS.
"""
from __future__ import annotations

import math
import os

import numpy as np

# Documentary unit normalisers.  They only put MPa margins and log-permeability
# margins on a comparable scale for ranking; rank order is insensitive to
# their exact values, and they were NOT fitted to any candidate coordinates.
# Strength spreads of a few MPa and logK spreads of a few tenths are the
# documented operating regime of the frozen LCM-GP oracle (see
# UNCERTAINTY_EQUATIONS.md); these constants merely encode those units.
DOCUMENTED_STRENGTH_SCALE_MPA = 2.0
DOCUMENTED_LOGK_SCALE = 0.35
DEFAULT_TAIL_FRAC = 0.10


def resolve_scales(strength_scale=None, logk_scale=None):
    """Return (strength_scale, logk_scale), allowing env override for tests."""
    s = float(os.environ.get("BLIND_STRENGTH_SCALE",
                             DOCUMENTED_STRENGTH_SCALE_MPA if strength_scale is None else strength_scale))
    k = float(os.environ.get("BLIND_LOGK_SCALE",
                             DOCUMENTED_LOGK_SCALE if logk_scale is None else logk_scale))
    if not (s > 0 and k > 0 and math.isfinite(s) and math.isfinite(k)):
        raise ValueError("continuous-score scales must be positive finite numbers")
    return s, k


def scenario_margins(s_real, logk_real, s_star, k_star,
                     strength_scale=None, logk_scale=None):
    """Per-scenario normalised margins (search-only, not acceptance)."""
    s_scale, k_scale = resolve_scales(strength_scale, logk_scale)
    s_real = np.asarray(s_real, dtype=float)
    logk_real = np.asarray(logk_real, dtype=float)
    logk_star = math.log(float(k_star))
    strength_margin = (s_real - float(s_star)) / s_scale
    permeability_margin = (logk_real - logk_star) / k_scale
    joint_margin = np.minimum(strength_margin, permeability_margin)
    return strength_margin, permeability_margin, joint_margin


def tail_score(joint_margin, tail_frac=DEFAULT_TAIL_FRAC):
    """CVaR-like search score: mean over the worst ``tail_frac`` scenarios."""
    jm = np.asarray(joint_margin, dtype=float)
    if jm.size == 0:
        raise ValueError("joint_margin must be non-empty")
    if not (0.0 < tail_frac <= 1.0):
        raise ValueError("tail_frac must be in (0, 1]")
    k = max(1, int(math.ceil(tail_frac * jm.size)))
    worst = np.partition(jm, k - 1)[:k]
    return float(worst.mean()), float(np.quantile(jm, tail_frac))


def search_score(s_real, logk_real, s_star, k_star, in_support,
                 tail_frac=DEFAULT_TAIL_FRAC,
                 strength_scale=None, logk_scale=None):
    """Smooth search-only quality for one candidate.

    Returns a dict with ``cvar`` (higher is better), ``quantile``,
    ``mean_margin``, ``feasible_for_search`` (False when ANY scenario is
    out of support -- fail closed), and the literal ``rhat`` for logging.
    The dict deliberately has no PASS/FAIL field.
    """
    s_real = np.asarray(s_real, dtype=float)
    logk_real = np.asarray(logk_real, dtype=float)
    in_support = np.asarray(in_support, dtype=bool)
    if not (s_real.shape == logk_real.shape == in_support.shape):
        raise ValueError("s_real, logk_real and in_support must share a shape")
    logk_star = math.log(float(k_star))
    strength_ok = s_real >= float(s_star)
    perm_ok = logk_real >= logk_star
    rhat = float(np.mean(in_support & strength_ok & perm_ok))
    if not bool(np.all(in_support)):
        # Support failure remains a hard failure: no smooth score can rescue
        # an unsupported candidate, and it sorts below every supported one.
        return {"cvar": float("-inf"), "quantile": float("-inf"),
                "mean_margin": float("-inf"), "rhat": rhat,
                "feasible_for_search": False,
                "support_fraction": float(np.mean(in_support))}
    _, _, joint_margin = scenario_margins(
        s_real, logk_real, s_star, k_star, strength_scale, logk_scale)
    cvar, quant = tail_score(joint_margin, tail_frac)
    return {"cvar": cvar, "quantile": quant,
            "mean_margin": float(np.mean(joint_margin)), "rhat": rhat,
            "feasible_for_search": True,
            "support_fraction": 1.0}


def feasibility_first_key(cvar_a, cement_a, feasible_a, cvar_b, cement_b, feasible_b):
    """Ordering key: a substantially more feasible candidate always wins.

    A low-cement but hopeless candidate must never displace a substantially
    more feasible one.  ``tolerance`` keeps tiny noise from flipping ranks.
    """
    tolerance = 0.05
    if feasible_a and not feasible_b:
        return -1
    if feasible_b and not feasible_a:
        return 1
    if abs(cvar_a - cvar_b) > tolerance:
        return -1 if cvar_a > cvar_b else 1
    if cement_a != cement_b:
        return -1 if cement_a < cement_b else 1
    return 0
