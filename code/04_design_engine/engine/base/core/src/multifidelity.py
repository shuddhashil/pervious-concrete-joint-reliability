"""Multi-fidelity evaluation schedule (blind-discovery repair).

Staged scenario budgets, configurable through environment variables:

- STAGE1_FID_GLOBAL   (default 32)   -- cheap global discovery;
- STAGE1_FID_MID      (default 128)  -- intermediate screen;
- STAGE1_FID_REFINED  (default 512)  -- refined search bank;
- STAGE1_FID_CONFIRM  (default 5000) -- confirmation (never decisive).

Rules:

- common random numbers (same bank seed) within each comparison stage;
- fresh/disjoint bank seeds between major confirmation stages;
- retain statistically plausible candidates, not only candidates whose noisy
  point estimate already meets the target: keep when
  ``rhat + z * SE >= target - margin`` (default z=1.28, margin=0.05);
- selection uses continuous margins plus reliability uncertainty and
  diversity (caller supplies the continuous score);
- every stage records candidate counts and survival rules.
"""
from __future__ import annotations

import math
import os


def _env_int(name, default, minimum=1):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return max(int(minimum), int(raw))
    except ValueError:
        return int(default)


def fidelity_schedule():
    """Return (global, mid, refined, confirm) scenario budgets."""
    return (_env_int("STAGE1_FID_GLOBAL", 32),
            _env_int("STAGE1_FID_MID", 128),
            _env_int("STAGE1_FID_REFINED", 512),
            _env_int("STAGE1_FID_CONFIRM", 5000))


def stage_seed(base_seed, stage_index):
    """Disjoint bank seeds between major stages (CRN within a stage)."""
    return int(base_seed) + 100003 * int(stage_index)


def standard_error(rhat, n):
    n = max(int(n), 1)
    return math.sqrt(max(float(rhat) * (1.0 - float(rhat)), 0.0) / n)


def plausible(rhat, n, target, z=1.28, margin=0.05):
    """Retain statistically plausible candidates (no premature elimination)."""
    return bool(float(rhat) + float(z) * standard_error(rhat, n) >= float(target) - float(margin))


def retention_report(n_in, kept_mask, stage, n_scenarios, rule):
    kept = [bool(x) for x in kept_mask]
    return {"stage": str(stage), "n_in": int(n_in),
            "n_kept": int(sum(kept)), "n_dropped": int(len(kept) - sum(kept)),
            "n_scenarios": int(n_scenarios), "survival_rule": str(rule)}
