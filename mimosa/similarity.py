"""Similarity metrics powering the CFL decision logic.

All functions are pure, deterministic, operate on NumPy arrays only, and are
independently testable without a running Flower server.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

#: Numerical-stability epsilon added to the cosine-similarity denominator.
EPSILON: float = 1e-12


def flatten_parameters(params: Sequence[Any]) -> np.ndarray:
    """Flatten a sequence of parameter arrays into a single 1D array.

    Parameters
    ----------
    params : Sequence[ndarray]
        List of parameter arrays (any shape).

    Returns
    -------
    np.ndarray
        1D float64 array concatenating ``params`` in order.
    """
    if not params:
        return np.asarray([], dtype=np.float64)
    return np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1) for p in params])


def pairwise_cosine_similarity(deltas: dict[str, np.ndarray]) -> np.ndarray:
    """Compute the N x N cosine similarity matrix over client weight deltas.

    ``S[i, j] = cos(dW_i, dW_j) = (dW_i . dW_j) / (||dW_i|| * ||dW_j|| + EPSILON)``.

    Row/column order follows the insertion order of ``deltas``, so identical
    inputs always produce identical output. The diagonal is forced to 1.0.
    Uses float64 precision for numerical stability.

    Parameters
    ----------
    deltas : Dict[str, ndarray]
        Client ID -> flattened weight delta.

    Returns
    -------
    np.ndarray
        N x N float64 similarity matrix.
    """
    keys = list(deltas.keys())
    n = len(keys)
    if n == 0:
        return np.empty((0, 0), dtype=np.float64)
    mat = np.stack(
        [np.asarray(deltas[k], dtype=np.float64).reshape(-1) for k in keys]
    )
    norms = np.linalg.norm(mat, axis=1)
    denom = np.outer(norms, norms) + EPSILON
    sim = (mat @ mat.T) / denom
    np.fill_diagonal(sim, 1.0)
    return sim


def max_update_norm(deltas: dict[str, np.ndarray]) -> float:
    """Return ``max(||dW_i||)`` across all deltas (0.0 for an empty dict)."""
    if not deltas:
        return 0.0
    return float(
        max(
            np.linalg.norm(np.asarray(d, dtype=np.float64).reshape(-1))
            for d in deltas.values()
        )
    )


def mean_update_norm(deltas: dict[str, np.ndarray]) -> float:
    """Return ``||mean(dW_i)||`` — the norm of the mean delta vector."""
    if not deltas:
        return 0.0
    mean_vec = np.mean(
        np.stack(
            [np.asarray(d, dtype=np.float64).reshape(-1) for d in deltas.values()]
        ),
        axis=0,
    )
    return float(np.linalg.norm(mean_vec))


def split_criterion(
    deltas: dict[str, np.ndarray], eps_1: float, eps_2: float
) -> bool:
    """Evaluate the CFL split criterion for a cluster.

    Matches the reference implementation
    (felisat/clustered-federated-learning) and the paper:

    Returns ``True`` only if::

        mean_norm < eps_1 AND max_norm > eps_2

    where ``mean_norm = ||mean(dW_i)||`` and ``max_norm = max(||dW_i||)``.

    Both thresholds are strict: ``mean_norm == eps_1`` or
    ``max_norm == eps_2`` do NOT trigger a split.

    Parameters
    ----------
    deltas : Dict[str, ndarray]
        Client ID -> flattened weight delta for the cluster.
    eps_1 : float
        Mean-norm stability threshold (cluster must be stable).
    eps_2 : float
        Max-norm divergence threshold (cluster must still be diverse).

    Returns
    -------
    bool
        ``True`` if the cluster should split.
    """
    if not deltas:
        return False
    mean_norm = mean_update_norm(deltas)
    if mean_norm >= eps_1:
        return False
    max_norm = max_update_norm(deltas)
    if max_norm <= eps_2:
        return False
    return True
