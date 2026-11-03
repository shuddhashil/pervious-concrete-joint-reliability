"""5%-increment six-bin simplex lattice: C(25,5) = 53,130 states.

State n in Z+^6, sum n = 20, g = n/20. Permanent ID = lexicographic index.
"""
import itertools

import numpy as np


def enumerate_lattice(step=0.05):
    n = int(round(1.0 / step))
    out = []
    for tup in itertools.combinations_with_replacement(range(6), n):
        counts = [0] * 6
        for b in tup:
            counts[b] += 1
        out.append(np.array(counts, dtype=float) / n)
    return out


def lattice_count(step=0.05):
    from math import comb
    n = int(round(1.0 / step))
    return comb(n + 6 - 1, 6 - 1)
