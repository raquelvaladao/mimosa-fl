"""Unit tests for ``mimosa/similarity.py`` (100% coverage target)."""

from __future__ import annotations

import numpy as np
import pytest

from mimosa.similarity import (
    EPSILON,
    flatten_parameters,
    max_update_norm,
    mean_update_norm,
    pairwise_cosine_similarity,
    split_criterion,
)
from tests.mimosa.fixtures import deltas


# --------------------------------------------------------------------------- #
# flatten_parameters                                                          #
# --------------------------------------------------------------------------- #


def test_flatten_parameters_multiple_shapes() -> None:
    params = [np.zeros((2, 2)), np.ones(3), np.array([2.0, 3.0])]
    flat = flatten_parameters(params)
    assert flat.shape == (9,)
    assert np.allclose(flat[:4], 0.0)
    assert np.allclose(flat[4:7], 1.0)
    assert np.allclose(flat[7:], [2.0, 3.0])


def test_flatten_parameters_empty() -> None:
    flat = flatten_parameters([])
    assert flat.shape == (0,)


def test_flatten_parameters_is_float64() -> None:
    flat = flatten_parameters([np.arange(3, dtype=np.int64)])
    assert flat.dtype == np.float64


# --------------------------------------------------------------------------- #
# pairwise_cosine_similarity                                                  #
# --------------------------------------------------------------------------- #


def test_cosine_identical_deltas() -> None:
    sim = pairwise_cosine_similarity(deltas.IDENTICAL_TWO)
    assert sim.shape == (2, 2)
    assert np.allclose(sim, 1.0, atol=1e-9)


def test_cosine_orthogonal_deltas() -> None:
    sim = pairwise_cosine_similarity(deltas.ORTHOGONAL_TWO)
    assert np.allclose(sim[0, 1], 0.0, atol=1e-12)
    assert np.allclose(np.diag(sim), 1.0)


def test_cosine_opposite_deltas() -> None:
    sim = pairwise_cosine_similarity(deltas.OPPOSITE_TWO)
    assert sim[0, 1] < 0.0
    assert np.allclose(sim, [[1.0, -1.0], [-1.0, 1.0]], atol=1e-9)


def test_cosine_zero_deltas_epsilon_guard() -> None:
    sim = pairwise_cosine_similarity(deltas.ZERO_TWO)
    assert np.allclose(sim[0, 1], 0.0, atol=1e-12)
    assert np.allclose(np.diag(sim), 1.0)


def test_cosine_empty() -> None:
    sim = pairwise_cosine_similarity(deltas.EMPTY)
    assert sim.shape == (0, 0)


def test_cosine_is_symmetric() -> None:
    sim = pairwise_cosine_similarity(deltas.NON_CONTIGUOUS_IDS)
    assert np.allclose(sim, sim.T)


def test_cosine_deterministic() -> None:
    a = pairwise_cosine_similarity(deltas.NON_CONTIGUOUS_IDS)
    b = pairwise_cosine_similarity(deltas.NON_CONTIGUOUS_IDS)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# max_update_norm / mean_update_norm                                          #
# --------------------------------------------------------------------------- #


def test_max_update_norm_known() -> None:
    assert max_update_norm(deltas.ORTHOGONAL_TWO) == pytest.approx(1.0)
    assert max_update_norm(deltas.NON_CONTIGUOUS_IDS) == pytest.approx(1.0)


def test_max_update_norm_empty() -> None:
    assert max_update_norm(deltas.EMPTY) == 0.0


def test_mean_update_norm_orthogonal() -> None:
    # mean of [1,0,0] and [0,1,0] = [0.5,0.5,0], norm = sqrt(0.5)
    assert mean_update_norm(deltas.ORTHOGONAL_TWO) == pytest.approx(
        np.sqrt(0.5), abs=1e-12
    )


def test_mean_update_norm_opposite() -> None:
    # mean of [1,0] and [-1,0] = [0,0], norm = 0
    assert mean_update_norm(deltas.OPPOSITE_TWO) == pytest.approx(0.0, abs=1e-12)


def test_mean_update_norm_empty() -> None:
    assert mean_update_norm(deltas.EMPTY) == 0.0


# --------------------------------------------------------------------------- #
# split_criterion                                                             #
# --------------------------------------------------------------------------- #


def test_split_criterion_identical_no_split() -> None:
    assert not split_criterion(deltas.IDENTICAL_TWO, eps_1=0.5, eps_2=1.0)


def test_split_criterion_orthogonal_split() -> None:
    # Orthogonal: mean_norm = sqrt(0.5) ~ 0.707, max_norm = 1.0
    # Need eps_1 > 0.707 and eps_2 < 1.0 to trigger a split.
    assert split_criterion(deltas.ORTHOGONAL_TWO, eps_1=0.8, eps_2=0.5)


def test_split_criterion_opposite_split() -> None:
    # mean_norm = 0.0, max_norm = 1.0
    assert split_criterion(deltas.OPPOSITE_TWO, eps_1=0.5, eps_2=0.5)


def test_split_criterion_zero_no_split() -> None:
    assert not split_criterion(deltas.ZERO_TWO, eps_1=0.5, eps_2=0.5)


def test_split_criterion_single_no_split() -> None:
    assert not split_criterion(deltas.SINGLE, eps_1=0.5, eps_2=0.5)


def test_split_criterion_empty_no_split() -> None:
    assert not split_criterion(deltas.EMPTY, eps_1=0.5, eps_2=0.5)


def test_split_criterion_eps1_boundary_strict() -> None:
    # mean_norm == eps_1 must NOT split (strictly less)
    assert not split_criterion(deltas.ALIGNED_HIGH_SIGNAL, eps_1=2.0, eps_2=1.5)
    assert split_criterion(deltas.ALIGNED_HIGH_SIGNAL, eps_1=2.1, eps_2=1.5)


def test_split_criterion_eps2_boundary_strict() -> None:
    # max_norm == eps_2 must NOT split (strictly greater)
    assert not split_criterion(deltas.OPPOSITE_TWO, eps_1=0.5, eps_2=1.0)
    assert split_criterion(deltas.OPPOSITE_TWO, eps_1=0.5, eps_2=0.9999999999)


def test_epsilon_constant() -> None:
    assert EPSILON == pytest.approx(1e-12)


def test_split_criterion_zero_max_norm_guard() -> None:
    # Guard for max_norm == 0.0: mean_norm is also 0.0, so mean_norm < eps_1
    # is True for any positive eps_1, but max_norm > eps_2 is False for any
    # non-negative eps_2. Use negative eps_1 to keep mean_norm < eps_1 True.
    assert not split_criterion(deltas.ZERO_TWO, eps_1=-1.0, eps_2=0.5)
