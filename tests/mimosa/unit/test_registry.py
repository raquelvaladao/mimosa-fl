"""Unit tests for ``mimosa/server/registry.py`` (>= 95% coverage target)."""

from __future__ import annotations

import threading

import numpy as np

from mimosa.server.registry import ClusterModelRegistry, ClusterNode
from tests.mimosa.fixtures.parameters import (
    small_parameters,
    small_parameters_ones,
    small_parameters_scaled,
)


def test_registry_init_seeds_root() -> None:
    registry = ClusterModelRegistry(initial_parameters=small_parameters())
    assert registry.get_active() == [0]
    assert registry.get_parameters(0) is not None


def test_registry_init_without_parameters() -> None:
    registry = ClusterModelRegistry()
    assert registry.get_active() == [0]
    assert registry.get_parameters(0) is None


def test_get_parameters_returns_copy() -> None:
    registry = ClusterModelRegistry(initial_parameters=small_parameters())
    params = registry.get_parameters(0)
    params[0][0, 0] = 123.0
    stored = registry.get_parameters(0)
    assert stored[0][0, 0] == 0.0


def test_update_parameters() -> None:
    registry = ClusterModelRegistry(initial_parameters=small_parameters())
    registry.update_parameters(0, small_parameters_ones())
    assert np.allclose(registry.get_parameters(0)[0], 1.0)
    assert np.allclose(registry.get_parameters(0)[1], 1.0)


def test_create_child_auto_id_inherits() -> None:
    registry = ClusterModelRegistry(initial_parameters=small_parameters())
    child = registry.create_child(parent_id=0, params=small_parameters_ones())
    assert child == 1
    assert registry.get_active() == [0, 1]
    assert np.allclose(registry.get_parameters(child)[0], 1.0)


def test_create_child_explicit_id_and_counter() -> None:
    registry = ClusterModelRegistry(initial_parameters=small_parameters())
    assert registry.create_child(0, small_parameters(), child_id=7) == 7
    # counter advanced past 7 so the next auto ID is 8
    assert registry.create_child(0, small_parameters()) == 8


def test_create_child_does_not_alias_input() -> None:
    registry = ClusterModelRegistry(initial_parameters=small_parameters())
    source = small_parameters_scaled(5.0)
    registry.create_child(0, source, child_id=1)
    source[0][0, 0] = -99.0
    assert registry.get_parameters(1)[0][0, 0] == 5.0


def test_deactivate_marks_inactive_keeps_model() -> None:
    registry = ClusterModelRegistry(initial_parameters=small_parameters())
    registry.create_child(0, small_parameters_ones(), child_id=1)
    registry.deactivate(0)
    assert registry.get_active() == [1]
    assert registry.get_parameters(0) is not None


def test_get_active_sorted() -> None:
    registry = ClusterModelRegistry(initial_parameters=small_parameters())
    registry.create_child(0, small_parameters(), child_id=5)
    registry.create_child(0, small_parameters(), child_id=3)
    registry.deactivate(0)
    assert registry.get_active() == [3, 5]


def test_registry_thread_safe_concurrent_updates() -> None:
    registry = ClusterModelRegistry(initial_parameters=small_parameters())
    errors: list[Exception] = []

    def worker(cluster_id: int) -> None:
        try:
            for _ in range(200):
                registry.update_parameters(cluster_id, small_parameters_scaled(cluster_id))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(registry.get_active()) == 1  # only the root was never deactivated


def test_cluster_node_defaults() -> None:
    node = ClusterNode(id=4, parent_id=0)
    assert node.children == []
    assert node.depth == 0
    assert node.active is True
