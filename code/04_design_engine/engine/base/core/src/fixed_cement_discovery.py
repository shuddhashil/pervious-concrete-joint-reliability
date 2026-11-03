"""Systematic fixed-cement discovery across the full domain (blind repair).

Do not wait for continuous Phase C to find feasibility before looking across
cement.  This stage is GLOBAL discovery, not the old unsupported upward scan
from one failed basin:

- predeclared coarse cement schedule covering the full allowed interval
  (default 200 to 700 kg/m3, spacing 25 kg/m3, all configurable via env);
- at each coarse level, the remaining variables are searched with
  global/support-aware proposals and multi-fidelity scoring (caller-supplied
  ``search_level_fn`` so this module stays model-free and testable);
- no monotonicity assumption: every coarse level is evaluated, and multiple
  promising regions at different cement levels are preserved;
- every plausible region is refined from coarse spacing to ~5 kg/m3 and then
  1 kg/m3 around candidate crossings/basins;
- expensive decisive verification is reserved for plausible boundary
  candidates (caller decides via ``plausible_fn``); plainly implausible
  candidates are never sent to verification.
"""
from __future__ import annotations

import os


def _env_float(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def coarse_schedule(lo=None, hi=None, step=None):
    lo = _env_float("STAGE1_FIXED_CEMENT_LO", 200.0 if lo is None else lo)
    hi = _env_float("STAGE1_FIXED_CEMENT_HI", 700.0 if hi is None else hi)
    step = _env_float("STAGE1_FIXED_CEMENT_STEP", 25.0 if step is None else step)
    if not (hi > lo and step > 0):
        raise ValueError("fixed-cement schedule requires hi > lo and step > 0")
    levels = []
    x = float(lo)
    while x <= hi + 1e-9:
        levels.append(round(x, 6))
        x += step
    if levels[-1] < hi - 1e-9:
        levels.append(float(hi))
    return levels


def refine_levels(center, coarse_step=25.0, fine_step=5.0, micro_step=1.0,
                  lo=200.0, hi=700.0):
    """Coarse -> ~5 kg/m3 -> 1 kg/m3 refinement around one plausible region."""
    center = float(center)
    out = set()

    def grid(step):
        xs = []
        k = -int(coarse_step // max(step, 1e-9)) - 2
        while True:
            v = center + k * step
            if v > hi + 1e-9:
                break
            if v >= lo - 1e-9:
                xs.append(round(min(max(v, lo), hi), 6))
            k += 1
            if k > 1000:
                break
        return xs

    for v in grid(fine_step):
        if abs(v - center) <= coarse_step + 1e-9:
            out.add(v)
    for v in grid(micro_step):
        if any(abs(v - u) <= fine_step + 1e-9 for u in list(out) + [center]):
            out.add(round(v, 6))
    out.add(round(min(max(center, lo), hi), 6))
    return sorted(out)


def discover(search_level_fn, plausible_fn=None, lo=None, hi=None, step=None):
    """Run the full coarse schedule; return per-level records and refinements.

    ``search_level_fn(level)`` -> dict with at least ``plausible`` (bool) and
    ``quality`` (float, higher is better).  ``plausible_fn`` optionally maps a
    level record to bool; when absent, the record's own ``plausible`` flag is
    used.  No monotonicity is assumed: all coarse levels run, and every
    plausible region gets its own 25 -> 5 -> 1 refinement list.
    """
    levels = coarse_schedule(lo, hi, step)
    records = []
    for lv in levels:
        rec = dict(search_level_fn(float(lv)))
        rec["cement_level"] = float(lv)
        if plausible_fn is not None:
            rec["plausible"] = bool(plausible_fn(rec))
        else:
            rec["plausible"] = bool(rec.get("plausible", False))
        records.append(rec)
    refinements = {}
    for rec in records:
        if rec["plausible"]:
            refinements[rec["cement_level"]] = refine_levels(rec["cement_level"])
    return {"coarse_levels": levels, "records": records,
            "plausible_levels": [r["cement_level"] for r in records if r["plausible"]],
            "refinements": refinements,
            "monotonicity_assumed": False}
