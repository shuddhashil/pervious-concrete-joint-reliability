"""Lexicographic Stage-One selection (Section 10 of the driving prompt).

Decision rule, applied in this exact order:
  A. reject candidates where both targets are not jointly satisfied
     (encoded here as R_LCB95 being undefined/NaN -- in practice every
     candidate we score has a defined R_LCB95).
  B. reject any candidate with R_LCB95 < 0.90.
  C. reject candidates invalid under the frozen domain/applicability rules
     (encoded as support_fraction < 1.0 at N-scale, i.e. the model ever
     places the candidate out of the applicability domain across scenarios --
     support failures are ALREADY folded into R_LCB95 itself, so this is a
     redundant/defensive check, not a second independent gate).
  D. among the remaining candidates, choose the LOWEST PRACTICAL CEMENT
     LEVEL (5 kg/m3 resolution).
  E. among ties at the same cement level, prefer larger R_LCB95 -- NEVER
     gradation shape.
  F. statistically indistinguishable ties (|R_LCB95 diff| < a small
     tolerance) are reported as equivalent alternatives, not force-ranked.
"""
from __future__ import annotations

TIE_TOLERANCE = 0.003  # R_LCB95 difference below which candidates are "statistically indistinguishable"


def pick_best_at_level(candidates_at_level: list[dict]) -> dict:
    """candidates_at_level: list of dicts each with at least 'R_LCB95_joint'.
    Returns the single best (rule E) -- gradation is never inspected."""
    return max(candidates_at_level, key=lambda c: c["R_LCB95_joint"])


def lexicographic_select(all_candidates: list[dict]) -> dict:
    """all_candidates: list of dicts with 'cement_level_kg_m3', 'R_LCB95_joint',
    'support_fraction'. Returns a result dict describing the selection, or a
    NO_FEASIBLE_CANDIDATE record."""
    # A/B/C: reject
    passing = [c for c in all_candidates
               if c.get("R_LCB95_joint") is not None
               and c["R_LCB95_joint"] >= __import__('runtime_targets').RELIABILITY
               and c.get("support_fraction", 0.0) >= 1.0 - 1e-9
               and c.get("decisive_pass", False)]
    if not passing:
        return {"status": "NO_FEASIBLE_CANDIDATE", "reason": "no tested cement level reached R_LCB95>=0.90",
                "candidates_considered": len(all_candidates)}

    # D: lowest practical cement level among passing candidates
    min_level = min(c["cement_level_kg_m3"] for c in passing)
    at_min_level = [c for c in passing if c["cement_level_kg_m3"] == min_level]

    # E: among ties at that level, prefer larger R_LCB95 (never gradation)
    best = pick_best_at_level(at_min_level)

    # F: report statistically-indistinguishable alternatives at the SAME level
    equivalents = [c for c in at_min_level
                   if c is not best and abs(c["R_LCB95_joint"] - best["R_LCB95_joint"]) < TIE_TOLERANCE]

    return {"status": "FEASIBLE_CANDIDATE_FOUND", "minimum_passing_cement_level_kg_m3": min_level,
            "selected_candidate": best, "equivalent_alternatives_at_same_level": equivalents,
            "candidates_considered": len(all_candidates)}
