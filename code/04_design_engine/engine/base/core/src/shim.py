"""Portable import shim: every frozen dependency is bundled in the run."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.dirname(HERE)

# Preserve the verified precedence: frozen optimizer helpers first, followed
# by the refit wrapper. The active run's src directory remains first so its
# target-parameterized production modules are authoritative.
for path in (
    os.path.join(RUN, "frozen", "refit_src"),
    os.path.join(RUN, "frozen", "optimizer_src"),
):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.append(path)

if HERE in sys.path:
    sys.path.remove(HERE)
sys.path.insert(0, HERE)
