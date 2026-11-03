"""Authoritative per-run targets, loaded by every target-sensitive stage."""
from __future__ import annotations

import json
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
with (RUN / "TARGET_CONTRACT.json").open(encoding="utf-8") as f:
    _C = json.load(f)

STRENGTH = float(_C["strength_target_mpa"])
PERMEABILITY = float(_C["permeability_target_mm_s"])
RELIABILITY = float(_C["required_joint_reliability"])

if STRENGTH <= 0 or PERMEABILITY <= 0:
    raise ValueError("strength and permeability targets must be positive")
if not 0.0 < RELIABILITY < 1.0:
    raise ValueError("required joint reliability must lie strictly between 0 and 1")
