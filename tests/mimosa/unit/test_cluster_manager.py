"""Unit tests for ``mimosa/server/cluster_manager.py`` (>= 95% coverage).

Covers the split/aggregate state machine: root initialization, split decision
(eps_1 / eps_2 boundaries), aggregate path, min_cluster_size, max_tree_depth,
soft membership and client reassignment.
"""

from __future__ import annotations

import numpy as np

from mimosa.server.cluster_manager import ClusterManager
from tests.mimosa.fixtures import deltas


def _two_client_split_deltas() -> dict[str, np.ndarray]:
    # 7 elements to match the small model (2x2 + 3) used by the strategy.
    return {
        "c1": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "c2": np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    }


def _four_client_split_deltas() -> dict[str, np.ndarray]:
    """Two aligned clients + two opposite clients → splits into two groups of 2."""
    return {
        "c1": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "c2": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "c3": np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "c4": np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    }


def test_root_initialization() -> None:
    manager = ClusterManager(eps_1=0.5, eps_2=0.5)
    assert manager.get_tree_depth(0) == 0
    assert manager.get_active_clusters() == [0]
    assert manager.get_cluster_members(0) == []


def test_new_client_assigned_to_root() -> None:
    manager = ClusterManager(eps_1=0.5, eps_2=0.5)
    assert manager.get_client_cluster("client_new") == 0
    assert manager.get_client_cluster("client_new") == 0  # idempotent


def test_assign_client_explicit() -> None:
    manager = ClusterManager(eps_1=0.5, eps_2=0.5)
    manager.assign_client("c1", 3)
    assert manager.get_client_cluster("c1") == 3


def test_update_round_below_min_cluster_size_aggregates() -> None:
    manager = ClusterManager(eps_1=0.5, eps_2=0.5, min_cluster_size=2)
    actions = manager.update_round({"c1": np.array([1.0, 0.0, 0.0])})
    assert len(actions) == 1
    assert actions[0].type == "aggregate"
    assert actions[0].cluster_id == 0
    assert actions[0].members == ["c1"]
    # tree untouched
    assert manager.get_active_clusters() == [0]


def test_update_round_aligned_updates_aggregate() -> None:
    manager = ClusterManager(eps_1=0.5, eps_2=0.5)
    actions = manager.update_round(deltas.ALIGNED_HIGH_SIGNAL)
    assert [a.type for a in actions] == ["aggregate"]


def test_update_round_splits_opposite_deltas() -> None:
    manager = ClusterManager(eps_1=0.5, eps_2=0.5, min_cluster_size=3)
    actions = manager.update_round(_four_client_split_deltas())

    assert len(actions) == 1
    action = actions[0]
    assert action.type == "split"
    assert action.cluster_id == 0
    assert len(action.child_clusters) == 2
    assert set(action.member_reassignment.keys()) == {"c1", "c2", "c3", "c4"}

    # tree mutated: root inactive, two active children at depth 1
    assert manager.cluster_tree[0].active is False
    assert manager.cluster_tree[0].children == action.child_clusters
    child1, child2 = action.child_clusters
    assert manager.get_tree_depth(child1) == 1
    assert manager.get_tree_depth(child2) == 1
    assert manager.get_active_clusters() == sorted([child1, child2])

    # members reassigned to distinct children (2 each)
    g1 = {cid for cid, cl in action.member_reassignment.items() if cl == child1}
    g2 = {cid for cid, cl in action.member_reassignment.items() if cl == child2}
    assert len(g1) == 2 and len(g2) == 2


def test_update_round_deterministic_split() -> None:
    manager = ClusterManager(eps_1=0.5, eps_2=0.5, min_cluster_size=3)
    d = _four_client_split_deltas()
    manager.update_round(d)
    first = manager.dump_tree()
    manager2 = ClusterManager(eps_1=0.5, eps_2=0.5, min_cluster_size=3)
    manager2.update_round(d)
    assert first == manager2.dump_tree()


def test_max_tree_depth_prevents_split() -> None:
    # Root already at max depth -> even a strong split signal aggregates.
    manager = ClusterManager(eps_1=0.5, eps_2=0.5, max_tree_depth=1)
    manager.cluster_tree[0].depth = 1  # simulate a child at max depth
    actions = manager.update_round(_four_client_split_deltas())
    assert [a.type for a in actions] == ["aggregate"]


def test_soft_membership_retained_on_absence() -> None:
    manager = ClusterManager(eps_1=0.5, eps_2=0.5, min_cluster_size=3)
    d = _four_client_split_deltas()
    manager.update_round(d)
    c1_cluster = manager.get_client_cluster("c1")
    c3_cluster = manager.get_client_cluster("c3")
    assert c1_cluster != c3_cluster

    # Round 2: only c1 reports. Its cluster has 1 member -> aggregate, no crash.
    actions = manager.update_round({"c1": np.array([1.0, 0.0])})
    assert [a.type for a in actions] == ["aggregate"]
    assert actions[0].cluster_id == c1_cluster

    # c3 was absent but its assignment is retained (soft membership).
    assert manager.get_client_cluster("c3") == c3_cluster


def test_get_cluster_members_after_split() -> None:
    manager = ClusterManager(eps_1=0.5, eps_2=0.5, min_cluster_size=3)
    d = _four_client_split_deltas()
    manager.update_round(d)
    child1, child2 = manager.get_active_clusters()
    members1 = manager.get_cluster_members(child1)
    members2 = manager.get_cluster_members(child2)
    assert sorted(members1 + members2) == ["c1", "c2", "c3", "c4"]


def test_dump_tree_serializable() -> None:
    manager = ClusterManager(eps_1=0.5, eps_2=0.5, min_cluster_size=3)
    manager.update_round(_four_client_split_deltas())
    dump = manager.dump_tree()
    assert isinstance(dump["nodes"], list)
    assert isinstance(dump["client_membership"], dict)
    assert len(dump["nodes"]) == 3  # root + 2 children
