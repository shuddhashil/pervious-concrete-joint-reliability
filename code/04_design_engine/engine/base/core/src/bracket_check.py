"""Require an adjacent integer FAIL/PASS cement bracket."""
from __future__ import annotations


class BracketNotEstablishedError(Exception):
    """Raised when the selected passing level lacks an adjacent lower failure."""


def verify_adjacent_bracket(selected_level_kg_m3: float, tested_levels: dict) -> None:
    """Validate PASS at the selected level and FAIL at selected level minus one."""
    if tested_levels.get(selected_level_kg_m3) != "PASS":
        raise BracketNotEstablishedError(
            f"selected level {selected_level_kg_m3} is not recorded as PASS in tested_levels"
        )
    lower = selected_level_kg_m3 - 1.0
    if lower not in tested_levels:
        raise BracketNotEstablishedError(
            f"adjacent lower integer level {lower} was never independently tested -- "
            f"cannot claim a bracket without it"
        )
    if tested_levels[lower] != "FAIL":
        raise BracketNotEstablishedError(
            f"adjacent lower integer level {lower} is not FAIL "
            f"(status={tested_levels[lower]!r}) -- bracket not established"
        )
