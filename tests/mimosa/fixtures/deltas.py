"""Test fixtures: precomputed weight deltas for split-criterion edge cases.

Each dict maps a client ID to a flattened weight delta ``dW``. All values are
chosen so the expected behavior is derivable by hand (see the
``similarity-clustering`` spec scenarios).
"""

from __future__ import annotations

import numpy as np

IDENTICAL_TWO: dict[str, np.ndarray] = {
    "c1": np.array([1.0, 0.0, 0.0]),
    "c2": np.array([1.0, 0.0, 0.0]),
}

ORTHOGONAL_TWO: dict[str, np.ndarray] = {
    "c1": np.array([1.0, 0.0, 0.0]),
    "c2": np.array([0.0, 1.0, 0.0]),
}

OPPOSITE_TWO: dict[str, np.ndarray] = {
    "c1": np.array([1.0, 0.0]),
    "c2": np.array([-1.0, 0.0]),
}

ZERO_TWO: dict[str, np.ndarray] = {
    "c1": np.zeros(3),
    "c2": np.zeros(3),
}

SINGLE: dict[str, np.ndarray] = {
    "c1": np.array([1.0, 2.0, 3.0]),
}

EMPTY: dict[str, np.ndarray] = {}

# Non-consecutive, non-zero-based client IDs. The partitioner must return these
# original keys, never local positional indices (regression for
# felisat/clustered-federated-learning#2).
NON_CONTIGUOUS_IDS: dict[str, np.ndarray] = {
    "client_5": np.array([1.0, 0.0]),
    "client_10": np.array([0.0, 1.0]),
    "client_99": np.array([-1.0, 0.0]),
    "client_7": np.array([0.0, -1.0]),
}

# Two ground-truth clusters: {a, b} point in +x, {c, d} point in -x.
TWO_GROUPS: dict[str, np.ndarray] = {
    "a": np.array([1.0, 0.0]),
    "b": np.array([0.8, 0.2]),
    "c": np.array([-1.0, 0.0]),
    "d": np.array([-0.8, -0.2]),
}

# High signal, aligned updates: split_criterion rejects (mean/max == 1.0).
ALIGNED_HIGH_SIGNAL: dict[str, np.ndarray] = {
    "c1": np.array([2.0, 0.0]),
    "c2": np.array([2.0, 0.0]),
    "c3": np.array([2.0, 0.0]),
}
