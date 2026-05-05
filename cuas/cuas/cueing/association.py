"""Greedy nearest-neighbor association with a hard cost gate."""
from typing import List, Tuple

import numpy as np


def greedy_assign(cost: np.ndarray, gate: float) -> List[Tuple[int, int]]:
    """Greedy NN matching. Returns list of (row, col) pairs with cost <= gate.

    Repeatedly takes the cheapest unclaimed pair under the gate, removes its
    row and column from contention, and continues. Greedy is the right choice
    when costs are well-separated and the gate cleanly suppresses ambiguous
    matches (per Module 2 plan §4).
    """
    if cost.size == 0:
        return []
    rows, cols = cost.shape
    triples = []
    for i in range(rows):
        for j in range(cols):
            c = float(cost[i, j])
            if c <= gate:
                triples.append((c, i, j))
    triples.sort(key=lambda t: t[0])
    used_r, used_c = set(), set()
    out: List[Tuple[int, int]] = []
    for _, i, j in triples:
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        out.append((i, j))
    return out
