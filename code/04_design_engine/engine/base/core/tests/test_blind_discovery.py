"""Focused unit tests for target-agnostic blind-initialisation helpers."""
from pathlib import Path
import sys

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import adaptive_init as AI


def test_frontier_keeps_low_cement_near_miss_and_is_deterministic():
    # Index 0 is the most reliable, index 1 is lower cement but a near-miss.
    # Both belong ahead of the dominated high-cement near-miss (index 2).
    order_a = AI._frontier_order([0.91, 0.89, 0.89], [420.0, 270.0, 430.0], [1, 1, 1])
    order_b = AI._frontier_order([0.91, 0.89, 0.89], [420.0, 270.0, 430.0], [1, 1, 1])
    assert order_a.tolist() == order_b.tolist()
    assert set(order_a[:2]) == {0, 1}


def test_screen_refines_all_initial_points_and_is_seed_reproducible():
    X0 = np.linspace(0.0, 1.0, 48, dtype=float).reshape(16, 3)
    calls = []

    def evaluate(X, n):
        calls.append((len(X), n))
        return X[:, 0], 200.0 + 100.0 * X[:, 1], np.ones(len(X), dtype=bool)

    out_a, log_a = AI.screen_and_refine(X0, evaluate, seed=7, rounds=2,
                                         scenario_schedule=(4, 8), refinement_sizes=(12,))
    calls_b = []

    def evaluate_b(X, n):
        calls_b.append((len(X), n))
        return X[:, 0], 200.0 + 100.0 * X[:, 1], np.ones(len(X), dtype=bool)

    out_b, log_b = AI.screen_and_refine(X0, evaluate_b, seed=7, rounds=2,
                                         scenario_schedule=(4, 8), refinement_sizes=(12,))
    assert calls[0] == (len(X0), 4)
    assert calls_b == calls
    # Wall-clock timings (elapsed_s, cost_per_candidate_s) are inherently
    # non-reproducible; seed reproducibility covers decisions, not timings.
    timing = {"elapsed_s", "cost_per_candidate_s"}
    strip = lambda log: [{k: v for k, v in entry.items() if k not in timing} for entry in log]
    assert strip(log_a) == strip(log_b)
    assert np.allclose(out_a, out_b)
