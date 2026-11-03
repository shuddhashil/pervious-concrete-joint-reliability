"""Deterministic worker configuration for independent optimizer starts.

Each NOMAD start is mathematically independent but must never share an output
file.  These helpers cap numerical-library threads before a spawned worker
imports NumPy/Torch, preventing the OpenBLAS oversubscription that invalidated
the first Phase-A parallel attempt.
"""
from __future__ import annotations

import os


def configure_single_thread_worker() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def configure_parent_for_spawn() -> None:
    """Set inherited caps before ProcessPoolExecutor creates Windows workers."""
    configure_single_thread_worker()
