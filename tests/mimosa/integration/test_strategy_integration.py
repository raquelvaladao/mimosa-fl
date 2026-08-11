"""Integration tests for :class:`ClusteredFLStrategy` with in-process clients.

Exercises both the spec-named legacy-style round core (``configure_fit`` /
``aggregate_fit``) and the modern message-based API (``configure_train`` /
``aggregate_train`` / ``configure_evaluate`` / ``aggregate_evaluate``) against
real Flower 1.33 types — no network, no gRPC.
"""

from __future__ import annotations

import numpy as np
import pytest

from flwr.app import ArrayRecord, ConfigRecord, Error, Message, MetricRecord, RecordDict

from mimosa._compat import arrayrecord_to_ndarrays, ndarrays_to_arrayrecord
from mimosa.server.cluster_manager import ClusterManager
from mimosa.server.registry import ClusterModelRegistry
from mimosa.strategy.cfl import ClusteredFLStrategy
from tests.mimosa.fixtures.parameters import small_parameters
from tests.mimosa.fixtures.simulated_clients import (
    MockGrid,
    SimClient,
    SimFitRes,
    SimpleClientManager,
)

MODEL_PARAMS = small_parameters()
MODEL_SIZE = 4 + 3  # 2x2 + 3


def _make_strategy(
    eps_1: float = 1.5,
    eps_2: float = 0.5,
    min_cluster_size: int = 2,
    max_tree_depth: int = 3,
    sample_fraction: float = 1.0,
) -> tuple[ClusteredFLStrategy, ClusterManager, ClusterModelRegistry]:
    manager = ClusterManager(
        eps_1=eps_1, eps_2=eps_2, min_cluster_size=min_cluster_size,
        max_tree_depth=max_tree_depth,
    )
    registry = ClusterModelRegistry(initial_parameters=MODEL_PARAMS)
    strategy = ClusteredFLStrategy(
        MODEL_PARAMS,
        manager,
        registry,
        eps_1=eps_1,
        eps_2=eps_2,
        min_cluster_size=min_cluster_size,
        max_tree_depth=max_tree_depth,
        sample_fraction=sample_fraction,
    )
    return strategy, manager, registry


def _run_legacy_round(
    strategy: ClusteredFLStrategy,
    deltas: dict[str, np.ndarray],
    server_round: int,
    fail_cids: set[str] | None = None,
):
    """Run one configure_fit -> simulate -> aggregate_fit round with SimClients."""
    fail_cids = fail_cids or set()
    clients = [SimClient(cid, delta) for cid, delta in deltas.items()]
    client_manager = SimpleClientManager(clients)
    configs = strategy.configure_fit(server_round, MODEL_PARAMS, client_manager)

    results, failures = [], []
    for client, fit_ins in configs:
        if client.cid in fail_cids:
            failures.append((client, RuntimeError("boom")))
            continue
        after = client.simulate(fit_ins.parameters)
        results.append((client, SimFitRes(after)))
    return strategy.aggregate_fit(server_round, results, failures)


def _aligned_delta(value: float) -> np.ndarray:
    vec = np.zeros(MODEL_SIZE)
    vec[0] = value
    return vec


def test_configure_fit_routes_per_cluster_parameters() -> None:
    strategy, manager, registry = _make_strategy(eps_1=0.5, eps_2=0.5, min_cluster_size=3)
    d = {
        "c1": _aligned_delta(1.0),
        "c2": _aligned_delta(1.0),
        "c3": _aligned_delta(-1.0),
        "c4": _aligned_delta(-1.0),
    }
    _run_legacy_round(strategy, d, server_round=1)

    child1, child2 = manager.get_active_clusters()
    clients = [
        SimClient("c1", _aligned_delta(1.0)), SimClient("c2", _aligned_delta(1.0)),
        SimClient("c3", _aligned_delta(-1.0)), SimClient("c4", _aligned_delta(-1.0)),
    ]
    configs = strategy.configure_fit(2, MODEL_PARAMS, SimpleClientManager(clients))

    params_by_cid = {client.cid: fit_ins.parameters for client, fit_ins in configs}
    for cid, sent in params_by_cid.items():
        expected_cluster = manager.get_client_cluster(cid)
        expected = registry.get_parameters(expected_cluster)
        assert [np.allclose(a, b) for a, b in zip(sent, expected)]


def test_aggregate_fit_computes_deltas_and_updates_root() -> None:
    # Identical high-signal deltas: mean_norm == max_norm == 2.0.
    # With eps_1 = 1.5 the mean-norm threshold is too low → no split.
    strategy, _, registry = _make_strategy(eps_1=1.5, eps_2=0.5)
    deltas = {
        "c1": _aligned_delta(2.0),
        "c2": _aligned_delta(2.0),
    }
    _run_legacy_round(strategy, deltas, server_round=1)

    # mean delta = [2,0,0,...], root model updated by that amount
    updated = registry.get_parameters(0)
    assert updated[0][0, 0] == pytest.approx(2.0)
    assert updated[0][0, 1] == pytest.approx(0.0)
    # no split happened (mean_norm >= eps_1 rejects the signal)
    assert len(registry.get_active()) == 1


def test_split_inherits_parent_weights_and_assigns_members() -> None:
    strategy, manager, registry = _make_strategy(eps_1=0.5, eps_2=0.5, min_cluster_size=3)
    deltas = {
        "c1": _aligned_delta(1.0),
        "c2": _aligned_delta(1.0),
        "c3": _aligned_delta(-1.0),
        "c4": _aligned_delta(-1.0),
    }
    params, metrics = _run_legacy_round(strategy, deltas, server_round=1)

    assert metrics["num_clusters"] == 2
    assert len(metrics["split_events"]) == 1
    child1, child2 = manager.get_active_clusters()

    # children inherited the parent's pre-round weights, then FedAvg applied
    for child in (child1, child2):
        child_model = registry.get_parameters(child)
        assert child_model is not None
        assert child_model[0].shape == (2, 2)

    assert set(metrics["cluster_memberships"][str(child1)]) | set(
        metrics["cluster_memberships"][str(child2)]
    ) == {"c1", "c2", "c3", "c4"}
    assert params is not None  # root-cluster parameters returned


def test_failures_excluded_without_crashing() -> None:
    strategy, manager, registry = _make_strategy(eps_1=0.5, eps_2=0.5)
    deltas = {
        "c1": _aligned_delta(1.0),
        "c2": _aligned_delta(-1.0),
    }
    params, metrics = _run_legacy_round(
        strategy, deltas, server_round=1, fail_cids={"c2"}
    )
    # Only c1 reported -> its cluster has 1 member -> aggregate, no crash.
    assert metrics["num_clusters"] == 1
    # Soft membership for the failed client is retained.
    assert manager.get_client_cluster("c2") == 0
    assert params is not None


def test_modern_message_round_trip() -> None:
    """configure_train + aggregate_train with real Message/ArrayRecord objects."""
    strategy, manager, registry = _make_strategy(eps_1=0.5, eps_2=0.5, min_cluster_size=3)
    grid = MockGrid(node_ids=[1, 2, 3, 4])
    deltas_by_node = {
        1: _aligned_delta(1.0), 2: _aligned_delta(1.0),
        3: _aligned_delta(-1.0), 4: _aligned_delta(-1.0),
    }

    sent_messages = strategy.configure_train(
        1, ndarrays_to_arrayrecord(MODEL_PARAMS), ConfigRecord({"lr": 0.1}), grid
    )
    assert len(sent_messages) == 4

    replies = []
    for msg in sent_messages:
        node_id = msg.metadata.dst_node_id
        sent_nd = arrayrecord_to_ndarrays(msg.content["arrays"])
        after = SimClient(str(node_id), deltas_by_node[node_id]).simulate(sent_nd)
        reply = Message(
            content=RecordDict(
                {
                    "arrays": ndarrays_to_arrayrecord(after),
                    "metrics": MetricRecord(
                        {"num-examples": 32, "train_loss": 0.5}
                    ),
                }
            ),
            reply_to=msg,
        )
        replies.append(reply)

    arrays_out, metric_record = strategy.aggregate_train(1, replies)

    assert metric_record is not None
    assert metric_record["round"] == 1
    assert metric_record["num_clusters"] == 2  # split happened
    assert arrays_out is not None
    # server-side deltas were recorded keyed by node id
    assert set(strategy.round_deltas.keys()) == {"1", "2", "3", "4"}
    assert manager.get_client_cluster("1") == manager.get_client_cluster("2")
    assert manager.get_client_cluster("3") == manager.get_client_cluster("4")
    assert manager.get_client_cluster("1") != manager.get_client_cluster("3")


def test_configure_evaluate_routes_cluster_arrays() -> None:
    strategy, manager, _ = _make_strategy(eps_1=0.5, eps_2=0.5, min_cluster_size=3)
    d = {
        "c1": _aligned_delta(1.0), "c2": _aligned_delta(1.0),
        "c3": _aligned_delta(-1.0), "c4": _aligned_delta(-1.0),
    }
    _run_legacy_round(strategy, d, server_round=1)

    grid = MockGrid(node_ids=[1, 2, 3, 4])
    eval_messages = strategy.configure_evaluate(
        2, ndarrays_to_arrayrecord(MODEL_PARAMS), ConfigRecord(), grid
    )
    assert len(eval_messages) == 4
    for msg in eval_messages:
        node_id = msg.metadata.dst_node_id
        assert msg.content["config"]["cluster-id"] == manager.get_client_cluster(str(node_id))


def test_aggregate_evaluate_per_cluster_metrics() -> None:
    strategy, manager, _ = _make_strategy(eps_1=0.5, eps_2=0.5, min_cluster_size=3)
    d = {
        "c1": _aligned_delta(1.0), "c2": _aligned_delta(1.0),
        "c3": _aligned_delta(-1.0), "c4": _aligned_delta(-1.0),
    }
    _run_legacy_round(strategy, d, server_round=1)

    child1, child2 = manager.get_active_clusters()
    # Pin the evaluation nodes to the post-split clusters.
    cluster_of_c1 = manager.get_client_cluster("c1")
    cluster_of_c2 = manager.get_client_cluster("c3")
    manager.assign_client("1", cluster_of_c1)
    manager.assign_client("2", cluster_of_c1)
    manager.assign_client("3", cluster_of_c2)
    manager.assign_client("4", cluster_of_c2)
    by_cluster = {
        cluster_of_c1: (0.9, 0.1),
        cluster_of_c2: (0.6, 0.4),
    }
    grid = MockGrid(node_ids=[1, 2, 3, 4])
    sent = strategy.configure_evaluate(
        2, ndarrays_to_arrayrecord(MODEL_PARAMS), ConfigRecord(), grid
    )
    replies = []
    for msg in sent:
        node_id = msg.metadata.dst_node_id
        cluster_id = manager.get_client_cluster(str(node_id))
        acc, loss = by_cluster[cluster_id]
        replies.append(
            Message(
                content=RecordDict(
                    {"metrics": MetricRecord({"num-examples": 10, "eval_acc": acc, "eval_loss": loss})}
                ),
                reply_to=msg,
            )
        )
    metric_record = strategy.aggregate_evaluate(2, replies)
    assert metric_record is not None
    assert float(metric_record[f"per_cluster_accuracy.{cluster_of_c1}"]) == pytest.approx(0.9)
    assert float(metric_record[f"per_cluster_accuracy.{cluster_of_c2}"]) == pytest.approx(0.6)


def test_initialize_parameters_returns_initial() -> None:
    strategy, _, _ = _make_strategy()
    initial = strategy.initialize_parameters(None)
    assert initial is not None
    assert len(initial) == 2


def test_initialize_parameters_returns_registry_root_when_no_initial() -> None:
    manager = ClusterManager(eps_1=0.5, eps_2=0.5)
    registry = ClusterModelRegistry(initial_parameters=MODEL_PARAMS)
    strategy = ClusteredFLStrategy(None, manager, registry)
    params = strategy.initialize_parameters(None)
    assert params is not None
    assert np.allclose(params[0], MODEL_PARAMS[0])


def test_configure_fit_falls_back_to_global_parameters() -> None:
    manager = ClusterManager(eps_1=0.5, eps_2=0.5)
    registry = ClusterModelRegistry()  # no model anywhere yet
    strategy = ClusteredFLStrategy(None, manager, registry)
    clients = [SimClient("c1", np.zeros(MODEL_SIZE))]
    configs = strategy.configure_fit(1, MODEL_PARAMS, SimpleClientManager(clients))
    _, fit_ins = configs[0]
    assert len(fit_ins.parameters) == 2
    assert np.allclose(fit_ins.parameters[0], MODEL_PARAMS[0])


def test_summary_logs_configuration() -> None:
    strategy, _, _ = _make_strategy()
    strategy.summary()  # must not raise


def test_dump_run_state_shape() -> None:
    strategy, _, _ = _make_strategy(eps_1=0.5, eps_2=0.5, min_cluster_size=3)
    d = {
        "c1": _aligned_delta(1.0), "c2": _aligned_delta(1.0),
        "c3": _aligned_delta(-1.0), "c4": _aligned_delta(-1.0),
    }
    _run_legacy_round(strategy, d, server_round=1)
    state = strategy.dump_run_state()
    assert "hyperparameters" in state and state["hyperparameters"]["eps_1"] == 0.5
    assert "rounds" in state and len(state["rounds"]) == 1
    assert "cluster_tree" in state
    assert len(state["cluster_tree"]["nodes"]) == 3  # root + 2 children


def test_aggregate_train_excludes_error_replies() -> None:
    strategy, manager, _ = _make_strategy(eps_1=0.5, eps_2=0.5)
    grid = MockGrid(node_ids=[1, 2])
    deltas_by_node = {1: _aligned_delta(1.0), 2: _aligned_delta(-1.0)}

    sent = strategy.configure_train(
        1, ndarrays_to_arrayrecord(MODEL_PARAMS), ConfigRecord(), grid
    )
    replies = []
    for msg in sent:
        node_id = msg.metadata.dst_node_id
        if node_id == 2:
            replies.append(
                Message(error=Error(code=1, reason="boom"), reply_to=msg)
            )
            continue
        sent_nd = arrayrecord_to_ndarrays(msg.content["arrays"])
        after = SimClient(str(node_id), deltas_by_node[node_id]).simulate(sent_nd)
        replies.append(
            Message(
                content=RecordDict(
                    {
                        "arrays": ndarrays_to_arrayrecord(after),
                        "metrics": MetricRecord({"num-examples": 32}),
                    }
                ),
                reply_to=msg,
            )
        )

    arrays_out, metric_record = strategy.aggregate_train(1, replies)
    assert set(strategy.round_deltas.keys()) == {"1"}  # failing node excluded
    assert metric_record is not None and metric_record["round"] == 1
    assert arrays_out is not None


def test_start_seeds_registry_and_runs() -> None:
    manager = ClusterManager(eps_1=0.5, eps_2=0.5)
    registry = ClusterModelRegistry()  # unseeded
    strategy = ClusteredFLStrategy(None, manager, registry, eps_1=0.5, eps_2=0.5)
    grid = MockGrid(node_ids=[1, 2])

    result = strategy.start(
        grid=grid,
        initial_arrays=ndarrays_to_arrayrecord(MODEL_PARAMS),
        num_rounds=1,
        train_config=ConfigRecord({"lr": 0.1}),
    )

    assert registry.get_parameters(0) is not None  # seeded from initial_arrays
    assert result is not None
    assert result.arrays is not None
    assert strategy.initial_parameters is None
    assert len(strategy.history) == 1


def test_constructor_seeds_unseeded_registry() -> None:
    manager = ClusterManager(eps_1=0.5, eps_2=0.5)
    registry = ClusterModelRegistry()  # unseeded
    strategy = ClusteredFLStrategy(MODEL_PARAMS, manager, registry)
    assert strategy.cluster_model_registry.get_parameters(0) is not None
    assert np.allclose(strategy.cluster_model_registry.get_parameters(0)[0], 0.0)


def test_configure_train_falls_back_when_cluster_has_no_model() -> None:
    manager = ClusterManager(eps_1=0.5, eps_2=0.5)
    registry = ClusterModelRegistry()  # no model anywhere
    strategy = ClusteredFLStrategy(None, manager, registry)
    grid = MockGrid(node_ids=[1])
    messages = strategy.configure_train(
        1, ndarrays_to_arrayrecord(MODEL_PARAMS), ConfigRecord(), grid
    )
    assert len(messages) == 1
    sent = arrayrecord_to_ndarrays(messages[0].content["arrays"])
    assert np.allclose(sent[0], MODEL_PARAMS[0])


def test_aggregate_train_ignores_reply_without_arrays() -> None:
    strategy, _, _ = _make_strategy(eps_1=0.5, eps_2=0.5)
    grid = MockGrid(node_ids=[1, 2])
    sent = strategy.configure_train(
        1, ndarrays_to_arrayrecord(MODEL_PARAMS), ConfigRecord(), grid
    )
    replies = []
    for msg in sent:
        node_id = msg.metadata.dst_node_id
        if node_id == 1:
            # Reply without an "arrays" record -> skipped.
            replies.append(
                Message(
                    content=RecordDict({"metrics": MetricRecord({"num-examples": 32})}),
                    reply_to=msg,
                )
            )
        else:
            sent_nd = arrayrecord_to_ndarrays(msg.content["arrays"])
            after = SimClient("2", _aligned_delta(1.0)).simulate(sent_nd)
            replies.append(
                Message(
                    content=RecordDict({"arrays": ndarrays_to_arrayrecord(after)}),
                    reply_to=msg,
                )
            )
    arrays_out, metric_record = strategy.aggregate_train(1, replies)
    assert set(strategy.round_deltas.keys()) == {"2"}
    assert metric_record is not None


def test_aggregate_evaluate_fallback_keys_and_missing_metrics() -> None:
    strategy, manager, _ = _make_strategy(eps_1=0.5, eps_2=0.5)
    d = {"c1": _aligned_delta(1.0), "c2": _aligned_delta(-1.0)}
    _run_legacy_round(strategy, d, server_round=1)

    cluster_of_c1 = manager.get_client_cluster("c1")
    cluster_of_c2 = manager.get_client_cluster("c2")
    manager.assign_client("1", cluster_of_c1)
    manager.assign_client("2", cluster_of_c2)

    grid = MockGrid(node_ids=[1, 2])
    sent = strategy.configure_evaluate(
        2, ndarrays_to_arrayrecord(MODEL_PARAMS), ConfigRecord(), grid
    )
    replies = []
    for i, msg in enumerate(sent):
        node_id = msg.metadata.dst_node_id
        if i == 0:
            # Node 1 replies using the "loss"/"accuracy" fallback keys.
            content = RecordDict(
                {"metrics": MetricRecord({"num-examples": 10, "loss": 0.1, "accuracy": 0.8})}
            )
        else:
            # Node 2 replies without any metric record at all.
            content = RecordDict({})
        replies.append(Message(content=content, reply_to=msg))
    metric_record = strategy.aggregate_evaluate(2, replies)
    assert metric_record is not None
    assert float(metric_record[f"per_cluster_accuracy.{cluster_of_c1}"]) == pytest.approx(0.8)


def test_aggregate_fit_skips_unconfigured_client() -> None:
    strategy, _, _ = _make_strategy(eps_1=10.0, eps_2=0.5)
    # Client "ghost" never went through configure_fit -> no sent parameters.
    ghost = SimClient("ghost", _aligned_delta(1.0))
    result = strategy.aggregate_fit(
        1,
        results=[(ghost, SimFitRes(_aligned_delta(1.0)))],
        failures=[],
    )
    assert result[1]["num_clusters"] == 1
    assert strategy.round_deltas == {}


def test_aggregate_noop_when_cluster_has_no_model() -> None:
    manager = ClusterManager(eps_1=10.0, eps_2=0.5)
    registry = ClusterModelRegistry()  # no model anywhere
    strategy = ClusteredFLStrategy(None, manager, registry)
    clients = [SimClient("c1", _aligned_delta(1.0))]
    configs = strategy.configure_fit(1, MODEL_PARAMS, SimpleClientManager(clients))
    results = [
        (client, SimFitRes(client.simulate(fit_ins.parameters)))
        for client, fit_ins in configs
    ]
    params, metrics = strategy.aggregate_fit(1, results, [])
    assert metrics["num_clusters"] == 1
    assert params is None  # registry had no root model, FedAvg is a no-op
