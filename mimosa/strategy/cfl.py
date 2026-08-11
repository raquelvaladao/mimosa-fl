"""ClusteredFLStrategy: clustered federated learning as a Flower strategy.

The strategy routes each client the parameters of the cluster it belongs to,
collects weight deltas ``dW = W_after - W_sent`` server-side, delegates
split/aggregate decisions to :class:`~mimosa.server.cluster_manager.ClusterManager`,
and maintains one model per active cluster in the
:class:`~mimosa.server.registry.ClusterModelRegistry`.

It subclasses Flower's **modern** message-based ABC
(``flwr.serverapp.strategy.Strategy``, Grid/ServerApp API, Flower >= 1.28) and
implements all five abstract methods: ``configure_train``, ``aggregate_train``,
``configure_evaluate``, ``aggregate_evaluate``, ``summary``.

The spec-named methods ``configure_fit`` / ``aggregate_fit`` /
``initialize_parameters`` are also provided with the legacy-style signature
(``ClientProxy`` / ``FitIns`` / ``FitRes``) as the testable core of the round
logic; the message-based methods reuse the same delta/clustering engine.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from logging import INFO, WARNING
from typing import Any, Optional

import numpy as np

from flwr.common.logger import log

from .. import _compat
from .._compat import (
    ArrayRecord,
    ConfigRecord,
    Grid,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
    Strategy,
    arrayrecord_to_ndarrays,
    ndarrays_to_arrayrecord,
    to_ndarrays,
)
from ..server.cluster_manager import ClusterManager, RoundAction
from ..server.registry import ClusterModelRegistry, Parameters

try:  # legacy-style types still shipped for backward compatibility (1.28-1.33)
    from flwr.common import (  # type: ignore[attr-defined]
        EvaluateIns,
        EvaluateRes,
        FitIns,
        FitRes,
    )
    from flwr.server.client_proxy import ClientProxy  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - very new Flower without legacy types
    ClientProxy = None  # type: ignore[assignment,misc]
    FitIns = None  # type: ignore[assignment,misc]
    FitRes = None  # type: ignore[assignment,misc]
    EvaluateIns = None  # type: ignore[assignment,misc]
    EvaluateRes = None  # type: ignore[assignment,misc]


class ClusteredFLStrategy(Strategy):
    """Flower strategy implementing Clustered Federated Learning (CFL).

    Parameters
    ----------
    initial_parameters : Optional[Parameters]
        Initial model parameters seeded into the root cluster. May be ``None``
        when the registry is pre-seeded or when the Grid ``start()`` call
        provides ``initial_arrays``.
    cluster_manager : ClusterManager
        Owns the cluster tree and split/aggregate decisions.
    cluster_model_registry : ClusterModelRegistry
        Stores one model per active cluster.
    eps_1 : float
        Mean-norm stability threshold (default 1.0). A cluster is a split
        candidate only if ``mean||dW|| < eps_1``.
    eps_2 : float
        Max-norm divergence threshold (default 1.0). A cluster splits only
        if ``max||dW|| > eps_2``.
    sample_fraction : float
        Fraction of available nodes sampled each round (default 1.0).
    min_cluster_size : int
        Minimum members present for split eligibility (default 2).
    max_tree_depth : int
        Maximum cluster-tree depth (default 5).
    evaluate_fn : Optional[Callable]
        Optional server-side evaluation callback (legacy-compatible).
    accept_failures : bool
        If ``False``, failed clients are excluded from the round's clustering
        (their soft membership is retained); the round still proceeds.
    """

    def __init__(
        self,
        initial_parameters: Optional[Parameters],
        cluster_manager: ClusterManager,
        cluster_model_registry: ClusterModelRegistry,
        eps_1: float = 1.0,
        eps_2: float = 1.0,
        sample_fraction: float = 1.0,
        min_cluster_size: int = 3,
        max_tree_depth: int = 5,
        evaluate_fn: Optional[Any] = None,
        accept_failures: bool = False,
    ) -> None:
        super().__init__()
        self.initial_parameters = (
            None
            if initial_parameters is None
            else [np.asarray(p, copy=True) for p in initial_parameters]
        )
        self.cluster_manager = cluster_manager
        self.cluster_model_registry = cluster_model_registry
        self.eps_1 = float(eps_1)
        self.eps_2 = float(eps_2)
        self.sample_fraction = float(sample_fraction)
        self.min_cluster_size = int(min_cluster_size)
        self.max_tree_depth = int(max_tree_depth)
        self.evaluate_fn = evaluate_fn
        self.accept_failures = accept_failures

        # Per-round state (spec R6 / cfl-strategy).
        self.sent_parameters: dict[str, list[np.ndarray]] = {}
        self.round_deltas: dict[str, np.ndarray] = {}

        # Structured run history for `mimosa tree` and plotting.
        self.history: list[dict] = []
        self._split_events: list[dict] = []

        if (
            self.initial_parameters is not None
            and self.cluster_model_registry.get_parameters(0) is None
        ):
            self.cluster_model_registry.update_parameters(0, self.initial_parameters)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        evaluate_fn: Optional[Any] = None,
    ) -> Any:
        """Seed the per-cluster model state and run the modern Grid loop.

        ``initial_arrays`` seeds the root cluster model if the registry has no
        root model yet; afterwards the strategy drives all rounds through
        ``configure_train`` / ``aggregate_train`` / ``configure_evaluate`` /
        ``aggregate_evaluate`` exactly like any other modern strategy.
        """
        if self.cluster_model_registry.get_parameters(0) is None:
            nd = arrayrecord_to_ndarrays(initial_arrays)
            self.cluster_model_registry.update_parameters(0, nd)
        if self.initial_parameters is not None:
            self.initial_parameters = None
        return super().start(
            grid=grid,
            initial_arrays=initial_arrays,
            num_rounds=num_rounds,
            timeout=timeout,
            train_config=train_config,
            evaluate_config=evaluate_config,
            evaluate_fn=evaluate_fn,
        )

    def summary(self) -> None:
        """Log the strategy configuration (modern ABC requirement)."""
        log(INFO, "\t└──> CFL hyperparameters:")
        log(INFO, "\t\t├── eps_1 (signal threshold): %.4f", self.eps_1)
        log(INFO, "\t\t├── eps_2 (divergence threshold): %.4f", self.eps_2)
        log(INFO, "\t\t├── sample_fraction: %.2f", self.sample_fraction)
        log(INFO, "\t\t├── min_cluster_size: %d", self.min_cluster_size)
        log(INFO, "\t\t└── max_tree_depth: %d", self.max_tree_depth)

    # ------------------------------------------------------------------ #
    # Modern message-based API (flwr.serverapp.strategy.Strategy)        #
    # ------------------------------------------------------------------ #

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Route each sampled node its cluster's current model (TRAIN).

        Mirrors ``configure_fit`` for the message-based API. Each message
        carries the cluster's ``ArrayRecord`` plus a ``ConfigRecord`` tagged
        with ``server-round`` and ``cluster-id``. What was sent is recorded so
        ``aggregate_train`` can reconstruct ``dW``.
        """
        node_ids = list(grid.get_node_ids())
        num = max(1, int(len(node_ids) * self.sample_fraction))
        sampled = node_ids[:num]

        self.sent_parameters.clear()
        messages: list[Message] = []
        for node_id in sampled:
            cluster_id = self.cluster_manager.get_client_cluster(node_id)
            cluster_nd = self.cluster_model_registry.get_parameters(cluster_id)
            if cluster_nd is None:
                cluster_nd = arrayrecord_to_ndarrays(arrays)
            record = RecordDict(
                {
                    "arrays": ndarrays_to_arrayrecord(cluster_nd),
                    "config": self._build_config(config, server_round, cluster_id),
                }
            )
            self.sent_parameters[str(node_id)] = [
                np.asarray(p, dtype=np.float64) for p in cluster_nd
            ]
            messages.append(
                Message(
                    content=record,
                    message_type=MessageType.TRAIN,
                    dst_node_id=node_id,
                )
            )
        log(
            INFO,
            "configure_train: routed %d node(s) across %d active cluster(s)",
            len(sampled),
            len(self.cluster_model_registry.get_active()),
        )
        return messages

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Compute server-side deltas, cluster, and emit round metrics.

        Each valid reply contributes ``dW = W_after - W_sent``. Deltas are
        pushed to the ``ClusterManager``; split/aggregate actions are executed
        against the registry (cluster-wise FedAvg). Returns the root cluster's
        ``ArrayRecord`` and a ``MetricRecord`` with per-round CFL metrics.
        """
        valid_replies, error_replies = [], []
        for msg in replies:
            (valid_replies if not msg.has_error() else error_replies).append(msg)
        if error_replies:
            log(
                WARNING,
                "aggregate_train: %d failure(s) excluded from round %d",
                len(error_replies),
                server_round,
            )

        deltas: dict[str, np.ndarray] = {}
        for msg in valid_replies:
            node_id = str(msg.metadata.src_node_id)
            record = msg.content.get("arrays")
            if record is None:
                continue
            w_after = arrayrecord_to_ndarrays(record)
            w_sent = self.sent_parameters.get(node_id)
            if w_sent is None or len(w_after) != len(w_sent):
                continue
            d = [a - s for a, s in zip(w_after, w_sent)]
            deltas[node_id] = np.concatenate([x.reshape(-1) for x in d])

        self.round_deltas = deltas
        actions = self.cluster_manager.update_round(deltas, server_round) if deltas else []
        self._process_actions(actions, deltas)
        metrics = self._build_metrics(server_round)
        self.history.append(metrics)

        root_nd = self.cluster_model_registry.get_parameters(0)
        arrays_out = ndarrays_to_arrayrecord(root_nd) if root_nd is not None else None
        return arrays_out, self._metrics_to_record(metrics)

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Route each sampled node its cluster's model for EVALUATE."""
        node_ids = list(grid.get_node_ids())
        num = max(1, int(len(node_ids) * self.sample_fraction))
        sampled = node_ids[:num]

        messages: list[Message] = []
        for node_id in sampled:
            cluster_id = self.cluster_manager.get_client_cluster(node_id)
            cluster_nd = self.cluster_model_registry.get_parameters(cluster_id)
            if cluster_nd is None:
                cluster_nd = arrayrecord_to_ndarrays(arrays)
            record = RecordDict(
                {
                    "arrays": ndarrays_to_arrayrecord(cluster_nd),
                    "config": self._build_config(config, server_round, cluster_id),
                }
            )
            messages.append(
                Message(
                    content=record,
                    message_type=MessageType.EVALUATE,
                    dst_node_id=node_id,
                )
            )
        return messages

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> Optional[MetricRecord]:
        """Aggregate evaluation metrics per cluster (not globally)."""
        valid_replies = [msg for msg in replies if not msg.has_error()]
        per_cluster: dict[int, dict[str, float]] = defaultdict(
            lambda: {"loss_w": 0.0, "acc_w": 0.0, "num_examples": 0.0}
        )
        for msg in valid_replies:
            node_id = str(msg.metadata.src_node_id)
            cluster_id = self.cluster_manager.get_client_cluster(node_id)
            metric_record = msg.content.get("metrics")
            if metric_record is None:
                continue
            num_examples = float(metric_record.get("num-examples", 1) or 1)
            loss = metric_record.get("eval_loss")
            if loss is None:
                loss = metric_record.get("loss")
            acc = metric_record.get("eval_acc")
            if acc is None:
                acc = metric_record.get("accuracy")
            bucket = per_cluster[cluster_id]
            bucket["num_examples"] += num_examples
            if loss is not None:
                bucket["loss_w"] += float(loss) * num_examples
            if acc is not None:
                bucket["acc_w"] += float(acc) * num_examples

        metrics: dict[str, Any] = {
            "round": server_round,
            "per_cluster_loss": {},
            "per_cluster_accuracy": {},
        }
        for cluster_id, bucket in per_cluster.items():
            n = bucket["num_examples"] or 1.0
            if bucket["loss_w"]:
                metrics["per_cluster_loss"][str(cluster_id)] = bucket["loss_w"] / n
            if bucket["acc_w"]:
                metrics["per_cluster_accuracy"][str(cluster_id)] = bucket["acc_w"] / n
        return self._metrics_to_record(metrics)

    # ------------------------------------------------------------------ #
    # Spec-named legacy-style API (testable core)                        #
    # ------------------------------------------------------------------ #

    def initialize_parameters(self, client_manager: Any) -> Optional[Parameters]:
        """Return the initial model parameters (root cluster)."""
        if self.initial_parameters is not None:
            return [np.asarray(p, copy=True) for p in self.initial_parameters]
        return self.cluster_model_registry.get_parameters(0)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: Any
    ) -> list[tuple[Any, Any]]:
        """Route each selected client its cluster's ``Parameters`` as ``FitIns``.

        This implements the spec interface (``cfl-strategy`` R2) with the
        legacy-style signature and is the unit-testable core that the
        message-based :meth:`configure_train` mirrors. ``parameters`` is only
        a fallback when a cluster has no model yet.
        """
        if FitIns is None:
            raise NotImplementedError(  # pragma: no cover - legacy types absent
                "Flower legacy `FitIns` is unavailable in this Flower version."
            )
        num_available = client_manager.num_available()
        sample_size = max(1, int(num_available * self.sample_fraction))
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min(sample_size, max(1, num_available)),
        )

        self.sent_parameters.clear()
        configs: list[tuple[Any, Any]] = []
        for client in clients:
            cluster_id = self.cluster_manager.get_client_cluster(client.cid)
            cluster_params = self.cluster_model_registry.get_parameters(cluster_id)
            if cluster_params is None:
                cluster_params = (
                    [np.asarray(p, copy=True) for p in parameters]
                    if parameters is not None
                    else []
                )
            fit_ins = FitIns(
                cluster_params,
                {"server_round": server_round, "cluster_id": cluster_id},
            )
            self.sent_parameters[client.cid] = [
                np.asarray(p, dtype=np.float64) for p in cluster_params
            ]
            configs.append((client, fit_ins))
        return configs

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[Any, Any]],
        failures: list[Any],
    ) -> tuple[Optional[Parameters], dict[str, Any]]:
        """Compute deltas, run clustering, execute split/aggregate.

        Implements the spec interface (``cfl-strategy`` R3/R6): failed clients
        are excluded from the round's clustering while retaining soft
        membership. Returns the updated root-cluster ``Parameters`` plus metrics.
        """
        if failures:
            log(
                WARNING,
                "aggregate_fit: %d failure(s) excluded from round %d",
                len(failures),
                server_round,
            )
        deltas: dict[str, np.ndarray] = {}
        for client, fit_res in results:
            w_after = to_ndarrays(fit_res.parameters)
            w_sent = self.sent_parameters.get(client.cid)
            if w_sent is None or len(w_after) != len(w_sent):
                continue
            d = [a - s for a, s in zip(w_after, w_sent)]
            deltas[client.cid] = np.concatenate([x.reshape(-1) for x in d])

        self.round_deltas = deltas
        actions = self.cluster_manager.update_round(deltas, server_round) if deltas else []
        self._process_actions(actions, deltas)
        metrics = self._build_metrics(server_round)
        self.history.append(metrics)
        return self.cluster_model_registry.get_parameters(0), metrics

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _build_config(
        self, config: Optional[ConfigRecord], server_round: int, cluster_id: int
    ) -> ConfigRecord:
        try:
            node_config = ConfigRecord()
        except TypeError:  # pragma: no cover - very old ConfigRecord
            node_config = {}
        if config is not None:
            for k, v in config.items():
                node_config[k] = v
        node_config["server-round"] = server_round
        node_config["cluster-id"] = cluster_id
        return node_config

    def _process_actions(
        self, actions: list[RoundAction], deltas: dict[str, np.ndarray]
    ) -> None:
        self._split_events = []
        for action in actions:
            if action.type == "split":
                parent_id = action.cluster_id
                parent_params = self.cluster_model_registry.get_parameters(parent_id)
                if parent_params is None:  # pragma: no cover - parent always has a model
                    continue
                for child_id in action.child_clusters or []:
                    self.cluster_model_registry.create_child(
                        parent_id, parent_params, child_id=child_id
                    )
                self.cluster_model_registry.deactivate(parent_id)
                for child_id in action.child_clusters or []:
                    members = [
                        cid
                        for cid, cl in (action.member_reassignment or {}).items()
                        if cl == child_id
                    ]
                    params = self.cluster_model_registry.get_parameters(child_id)
                    self._apply_fedavg(child_id, params, deltas, members)
                self._split_events.append(
                    {
                        "round": action.cluster_id,
                        "parent": parent_id,
                        "children": list(action.child_clusters or []),
                        "sizes": [
                            len(
                                [
                                    c
                                    for c, cl in (action.member_reassignment or {}).items()
                                    if cl == child
                                ]
                            )
                            for child in (action.child_clusters or [])
                        ],
                    }
                )
            else:
                params = self.cluster_model_registry.get_parameters(action.cluster_id)
                self._apply_fedavg(
                    action.cluster_id, params, deltas, action.members or []
                )

    def _apply_fedavg(
        self,
        cluster_id: int,
        params: Optional[Parameters],
        deltas: dict[str, np.ndarray],
        members: list[str],
    ) -> None:
        """Apply cluster-wise FedAvg: ``W_cluster += mean(dW_i for i in members)``."""
        if params is None or not members:
            return
        mean_delta = np.mean(np.stack([deltas[c] for c in members]), axis=0)
        updated: list[np.ndarray] = []
        offset = 0
        for arr in params:
            arr_np = np.asarray(arr, dtype=np.float64)
            n = int(arr_np.size)
            piece = arr_np.reshape(-1) + mean_delta[offset : offset + n]
            updated.append(piece.reshape(arr_np.shape))
            offset += n
        self.cluster_model_registry.update_parameters(cluster_id, updated)

    def _build_metrics(self, server_round: int) -> dict[str, Any]:
        active = self.cluster_model_registry.get_active()
        memberships: dict[str, list[str]] = {
            str(c): self.cluster_manager.get_cluster_members(c) for c in active
        }
        max_norm, mean_norm, sum_norm = 0.0, 0.0, 0.0
        if self.round_deltas:
            max_norm = float(
                max(np.linalg.norm(d) for d in self.round_deltas.values())
            )
            mean_norm = float(
                np.linalg.norm(
                    np.mean(np.stack(list(self.round_deltas.values())), axis=0)
                )
            )
            # Norm of the SUM of deltas (matches the reference implementation's
            # ``||sum_i dW_i||`` metric plotted by ``display_train_stats``).
            sum_norm = float(
                np.linalg.norm(
                    np.sum(np.stack(list(self.round_deltas.values())), axis=0)
                )
            )
        return {
            "round": server_round,
            "num_clusters": len(active),
            "cluster_memberships": memberships,
            "split_events": list(self._split_events),
            "max_norm": max_norm,
            "mean_norm": mean_norm,
            "sum_norm": sum_norm,
        }

    def _metrics_to_record(self, metrics: dict[str, Any]) -> MetricRecord:
        """Flatten numeric metrics into a ``MetricRecord``.

        Flower 1.33's ``MetricRecord`` only accepts numeric scalars (``int``,
        ``float``, or numeric lists). Structured values such as
        ``cluster_memberships`` and ``split_events`` are intentionally skipped
        here — they remain available through :attr:`history` and
        :meth:`dump_run_state`.
        """
        flat: dict[str, Any] = {}

        def add(key: str, value: Any) -> None:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                flat[key] = value
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    add(f"{key}.{sub_key}", sub_value)
            elif isinstance(value, (list, tuple)) and all(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in value
            ):
                flat[key] = value

        for key, value in metrics.items():
            add(key, value)
        return MetricRecord(flat)

    def dump_run_state(self) -> dict:
        """JSON-serializable run state for ``mimosa tree`` and plotting."""
        return {
            "hyperparameters": {
                "eps_1": self.eps_1,
                "eps_2": self.eps_2,
                "sample_fraction": self.sample_fraction,
                "min_cluster_size": self.min_cluster_size,
                "max_tree_depth": self.max_tree_depth,
            },
            "rounds": self.history,
            "cluster_tree": self.cluster_manager.dump_tree(),
        }
