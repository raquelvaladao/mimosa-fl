"""End-to-end mini CFL experiment.

Runs a 3-round, 4-client experiment with two ground-truth non-IID groups and
verifies that the cluster tree matches expectations: the root splits into two
clusters whose memberships correspond to the ground-truth groups.
"""

from __future__ import annotations

import numpy as np

from mimosa.server.cluster_manager import ClusterManager
from mimosa.server.registry import ClusterModelRegistry
from mimosa.strategy.cfl import ClusteredFLStrategy
from tests.mimosa.fixtures.parameters import small_parameters
from tests.mimosa.fixtures.simulated_clients import SimClient, SimFitRes, SimpleClientManager

# Model has 4 + 3 = 7 parameters.
MODEL_PARAMS = small_parameters()


def _group_delta(value: float) -> np.ndarray:
    vec = np.zeros(7)
    vec[0] = value
    return vec


# Two ground-truth groups: {a, b} update in +x, {c, d} update in -x.
ROUND_DELTAS: dict[str, np.ndarray] = {
    "a": _group_delta(1.0),
    "b": _group_delta(0.9),
    "c": _group_delta(-1.0),
    "d": _group_delta(-0.9),
}


def _run_round(strategy: ClusteredFLStrategy, server_round: int) -> None:
    clients = [SimClient(cid, delta) for cid, delta in ROUND_DELTAS.items()]
    configs = strategy.configure_fit(server_round, MODEL_PARAMS, SimpleClientManager(clients))
    results = [
        (client, SimFitRes(client.simulate(fit_ins.parameters)))
        for client, fit_ins in configs
    ]
    strategy.aggregate_fit(server_round, results, [])


def test_end_to_end_two_clusters_after_three_rounds() -> None:
    manager = ClusterManager(eps_1=0.5, eps_2=0.5, min_cluster_size=2, max_tree_depth=3)
    registry = ClusterModelRegistry(initial_parameters=MODEL_PARAMS)
    strategy = ClusteredFLStrategy(
        MODEL_PARAMS, manager, registry, eps_1=0.5, eps_2=0.5,
        min_cluster_size=2, max_tree_depth=3,
    )

    for round_num in range(1, 4):
        _run_round(strategy, round_num)

    active = registry.get_active()
    assert len(active) == 2, f"expected 2 active clusters, got {active}"

    members_a = set(manager.get_cluster_members(active[0]))
    members_b = set(manager.get_cluster_members(active[1]))
    expected_group1 = {"a", "b"}
    expected_group2 = {"c", "d"}
    assert {frozenset(members_a), frozenset(members_b)} == {
        frozenset(expected_group1),
        frozenset(expected_group2),
    }

    # The final history reports the cluster tree across all 3 rounds.
    assert len(strategy.history) == 3
    assert strategy.history[-1]["num_clusters"] == 2


def test_end_to_end_metrics_emitted_each_round() -> None:
    manager = ClusterManager(eps_1=0.5, eps_2=0.5, min_cluster_size=2, max_tree_depth=3)
    registry = ClusterModelRegistry(initial_parameters=MODEL_PARAMS)
    strategy = ClusteredFLStrategy(
        MODEL_PARAMS, manager, registry, eps_1=0.5, eps_2=0.5,
        min_cluster_size=2, max_tree_depth=3,
    )
    _run_round(strategy, 1)

    metrics = strategy.history[0]
    assert "round" in metrics and metrics["round"] == 1
    assert "num_clusters" in metrics
    assert "split_events" in metrics
    assert "max_norm" in metrics and metrics["max_norm"] > 0.0
    assert "mean_norm" in metrics
    assert "cluster_memberships" in metrics
    assert metrics["split_events"]  # first round splits the root
