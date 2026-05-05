import numpy as np
import pytest

from cuas.cueing.association import greedy_assign


def test_empty_cost_returns_empty():
    assert greedy_assign(np.zeros((0, 0)), 1.0) == []
    assert greedy_assign(np.zeros((3, 0)), 1.0) == []
    assert greedy_assign(np.zeros((0, 3)), 1.0) == []


def test_diagonal_minimum_pairs():
    cost = np.array([[0.1, 5.0],
                     [5.0, 0.2]])
    pairs = greedy_assign(cost, 1.0)
    assert sorted(pairs) == [(0, 0), (1, 1)]


def test_gate_filters_pairs_above_threshold():
    cost = np.array([[0.1, 5.0],
                     [5.0, 0.2]])
    pairs = greedy_assign(cost, 0.05)
    assert pairs == []


def test_greedy_takes_lowest_cost_first():
    # Cell (0,1)=0.05 is the global minimum; greedy must take it,
    # forcing row 0 and col 1 out, so (1,0)=0.4 wins next, not (0,0)=0.1.
    cost = np.array([[0.10, 0.05],
                     [0.40, 0.50]])
    pairs = greedy_assign(cost, 1.0)
    assert sorted(pairs) == [(0, 1), (1, 0)]


def test_unbalanced_dimensions():
    # 2 rows, 3 cols -> at most 2 pairs.
    cost = np.array([[0.1, 0.5, 0.2],
                     [0.7, 0.3, 0.6]])
    pairs = greedy_assign(cost, 1.0)
    assert len(pairs) == 2
    rows = {i for i, _ in pairs}
    cols = {j for _, j in pairs}
    assert rows == {0, 1}
    assert len(cols) == 2  # no column reused


def test_no_pair_under_gate():
    cost = np.array([[5.0, 5.0],
                     [5.0, 5.0]])
    assert greedy_assign(cost, 1.0) == []


def test_exact_gate_inclusive():
    cost = np.array([[1.0]])
    assert greedy_assign(cost, 1.0) == [(0, 0)]
