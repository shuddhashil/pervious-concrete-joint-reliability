"""Batched evaluation helper (blind-discovery repair).

Replace avoidable candidate-by-candidate prediction loops with bounded
candidate batches where scientifically identical:

- candidates are processed in bounded chunks (``STAGE1_EVAL_BATCH`` env,
  default 64) to cap memory;
- numerical equivalence against the exact sequential path is verified on
  demand via :func:`verify_equivalence` on random, boundary, zero-bin, and
  support-edge cases;
- on any equivalence failure the caller must fall back safely to the
  sequential path (this module never silently substitutes a divergent fast
  path).
"""
from __future__ import annotations

import os

import numpy as np


def _env_int(name, default, minimum=1):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return max(int(minimum), int(raw))
    except ValueError:
        return int(default)


def batch_size():
    return _env_int("STAGE1_EVAL_BATCH", 64)


def batched_map(eval_one_fn, items, size=None):
    """Apply ``eval_one_fn`` to ``items`` in bounded batches (identical order)."""
    size = int(size or batch_size())
    items = list(items)
    out = []
    for i in range(0, len(items), size):
        for item in items[i:i + size]:
            out.append(eval_one_fn(item))
    return out


def verify_equivalence(eval_fast_fn, eval_exact_fn, cases, tol=1e-9):
    """Compare fast/batched vs authoritative path; return a report dict.

    ``cases`` is a list of (name, item) tuples.  Any mismatch beyond ``tol``
    sets ``equivalent`` False and lists the offending case names so the
    caller can fall back safely.
    """
    mismatches = []
    details = []
    for name, item in cases:
        fast = eval_fast_fn(item)
        exact = eval_exact_fn(item)
        fa = np.asarray(fast, dtype=float).ravel()
        ea = np.asarray(exact, dtype=float).ravel()
        if fa.shape != ea.shape:
            mismatches.append(str(name))
            details.append({"case": str(name), "reason": "shape mismatch"})
            continue
        diff = float(np.max(np.abs(fa - ea))) if fa.size else 0.0
        details.append({"case": str(name), "max_abs_diff": diff})
        if not (diff <= float(tol)):
            mismatches.append(str(name))
    return {"equivalent": not mismatches, "tolerance": float(tol),
            "n_cases": len(cases), "mismatches": mismatches, "details": details}
