"""Test fixtures: small model ``Parameters`` for registry operations.

Model parameters are plain lists of NumPy arrays (Flower ``Parameters``).
"""

from __future__ import annotations

import numpy as np

from mimosa.server.registry import Parameters


def small_parameters() -> Parameters:
    """A tiny two-tensor model: a 2x2 weight and a length-3 bias."""
    return [np.zeros((2, 2), dtype=np.float64), np.zeros(3, dtype=np.float64)]


def small_parameters_ones() -> Parameters:
    """Same shapes as :func:`small_parameters` but filled with ones."""
    return [np.ones((2, 2), dtype=np.float64), np.ones(3, dtype=np.float64)]


def small_parameters_scaled(scale: float) -> Parameters:
    """``small_parameters_ones`` scaled by ``scale`` (nonzero values)."""
    return [np.full((2, 2), scale, dtype=np.float64), np.full(3, scale, dtype=np.float64)]


def small_parameters_shapes() -> list[tuple[int, ...]]:
    """The shapes produced by :func:`small_parameters`."""
    return [(2, 2), (3,)]
