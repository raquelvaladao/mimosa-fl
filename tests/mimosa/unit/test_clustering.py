"""Unit tests for ``mimosa/clustering.py`` (100% coverage target)."""

from __future__ import annotations

import numpy as np

from mimosa.clustering import AgglomerativeBiPartitioner
from tests.mimosa.fixtures import deltas


def test_partition_empty() -> None:
    assert AgglomerativeBiPartitioner().partition(deltas.EMPTY) == ([], [])


def test_partition_single_client() -> None:
    assert AgglomerativeBiPartitioner().partition(deltas.SINGLE) == (["c1"], [])


def test_partition_two_identical_returns_singletons() -> None:
    group1, group2 = AgglomerativeBiPartitioner().partition(deltas.IDENTICAL_TWO)
    assert group1 == ["c1"]
    assert group2 == ["c2"]


def test_partition_two_orthogonal_returns_singletons() -> None:
    group1, group2 = AgglomerativeBiPartitioner().partition(deltas.ORTHOGONAL_TWO)
    assert sorted(group1 + group2) == ["c1", "c2"]
    assert group1 and group2


def test_partition_non_contiguous_ids_returns_original_keys() -> None:
    """Regression test for felisat/clustered-federated-learning#2.

    The partitioner must return the original client IDs, never local
    positional indices 0, 1, 2, 3.
    """
    group1, group2 = AgglomerativeBiPartitioner().partition(deltas.NON_CONTIGUOUS_IDS)
    original = set(deltas.NON_CONTIGUOUS_IDS.keys())
    assert set(group1) | set(group2) == original
    assert set(group1) & set(group2) == set()
    assert group1 and group2
    for cid in group1 + group2:
        assert cid in original
        assert cid.startswith("client_")


def test_partition_two_groups_recovered() -> None:
    group1, group2 = AgglomerativeBiPartitioner().partition(deltas.TWO_GROUPS)
    expected_a = {"a", "b"}
    expected_c = {"c", "d"}
    s1, s2 = set(group1), set(group2)
    assert {frozenset(s1), frozenset(s2)} == {frozenset(expected_a), frozenset(expected_c)}


def test_partition_degenerate_zero_deltas_still_valid() -> None:
    deltas_zero = {
        "c1": np.zeros(3),
        "c2": np.zeros(3),
        "c3": np.zeros(3),
    }
    group1, group2 = AgglomerativeBiPartitioner().partition(deltas_zero)
    assert set(group1) | set(group2) == {"c1", "c2", "c3"}
    assert set(group1) & set(group2) == set()
    assert group1 and group2


def test_partition_deterministic() -> None:
    partitioner = AgglomerativeBiPartitioner()
    g1a, g2a = partitioner.partition(deltas.TWO_GROUPS)
    g1b, g2b = partitioner.partition(deltas.TWO_GROUPS)
    assert g1a == g1b
    assert g2a == g2b


def test_partition_preserves_group_balance_for_4_clients() -> None:
    group1, group2 = AgglomerativeBiPartitioner().partition(deltas.TWO_GROUPS)
    assert len(group1) == 2
    assert len(group2) == 2
