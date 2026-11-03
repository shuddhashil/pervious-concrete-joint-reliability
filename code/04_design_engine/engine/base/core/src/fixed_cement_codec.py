"""10-DOF fixed-cement wrapper around the frozen stick11 codec.

x10[0..4]  : normalized scalars [r, A(log), P, S, phi] (cement REMOVED)
x10[5..9]  : 5 stick-breaking gradation variables (unchanged from stick11)

Cement is pinned to an exact practical dosage (e.g. 270.0 kg/m3) for the
whole search; NOMAD never sees a cement coordinate, so it cannot drift it.
"""
import numpy as np

import codec11 as K
import solvers_common as SC

ORDER = "coarse_first"


class FixedCementCodec:
    def __init__(self, cement_level_kg_m3: float):
        self.C_level = float(cement_level_kg_m3)
        lo, hi = SC.BND["C"]
        if not (lo - 1e-9 <= self.C_level <= hi + 1e-9):
            raise ValueError(f"cement level {self.C_level} outside declared domain [{lo},{hi}]")
        self.u0 = (self.C_level - lo) / (hi - lo)

    def decode10(self, x10):
        x10 = np.asarray(x10, dtype=float)
        assert x10.shape == (10,), f"expected 10-D vector, got {x10.shape}"
        x11 = np.empty(11)
        x11[0] = self.u0
        x11[1:11] = x10
        cand = K.decode11(x11, ORDER)
        cand["C"] = self.C_level  # eliminate any float round-trip drift; cement is EXACTLY the requested level
        return cand

    def encode10(self, cand):
        """Drop the cement coordinate from a full candidate's stick11 encoding
        (used only to build NOMAD start vectors from known candidates)."""
        cand = dict(cand)
        cand["C"] = self.C_level
        x11 = K.encode11(cand, ORDER)
        return x11[1:11]
