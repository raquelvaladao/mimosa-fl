# API Reference

mimosa-fl provides five public modules. The core math/state modules depend only
on NumPy (and scikit-learn for the default partitioner); the strategy targets
Flower's modern message-based API.

## `mimosa.similarity`

Pure, deterministic similarity metrics over client weight deltas.

| Function | Signature | Description |
|---|---|---|
| `flatten_parameters` | `(params: Sequence[ndarray]) -> ndarray` | Concatenate parameter arrays into one 1D array. |
| `pairwise_cosine_similarity` | `(deltas: Dict[str, ndarray]) -> ndarray` | N x N cosine similarity matrix (`float64`). |
| `max_update_norm` | `(deltas: Dict[str, ndarray]) -> float` | `max(||dW_i||)`. |
| `mean_update_norm` | `(deltas: Dict[str, ndarray]) -> float` | `||mean(dW_i)||`. |
| `split_criterion` | `(deltas, eps_1, eps_2) -> bool` | `max_norm > eps_1 AND mean_norm / max_norm < eps_2` (strict). |

Constant `EPSILON = 1e-12` guards the cosine-similarity denominator.

## `mimosa.clustering`

| Class | Description |
|---|---|
| `BiPartitioner` (Protocol) | Interface: `partition(deltas) -> (group1, group2)` returning **original client IDs**. |
| `AgglomerativeBiPartitioner` | Agglomerative clustering (complete linkage) on `-S` with `n_clusters=2`. |

## `mimosa.server.registry`

| Symbol | Description |
|---|---|
| `Parameters` | `list[np.ndarray]` — model parameters. |
| `ClusterNode` | Dataclass: `id`, `parent_id`, `children`, `depth`, `active`. |
| `ClusterModelRegistry` | Thread-safe per-cluster model store. |

`ClusterModelRegistry(initial_parameters=None)` — methods:

| Method | Description |
|---|---|
| `get_parameters(cluster_id) -> Parameters \| None` | Copy of a cluster's model. |
| `update_parameters(cluster_id, params)` | Store a model. |
| `create_child(parent_id, params, child_id=None) -> int` | New cluster seeded with inherited weights. |
| `deactivate(cluster_id)` | Mark a cluster inactive (on split). |
| `get_active() -> list[int]` | Active (leaf) cluster IDs. |

## `mimosa.server.cluster_manager`

| Symbol | Description |
|---|---|
| `RoundAction` | Dataclass: `type` (`"split"`/`"aggregate"`), `cluster_id`, `child_clusters`, `member_reassignment`, `members`. |
| `ClusterManager` | The cluster-tree "brain". |

`ClusterManager(eps_1, eps_2, min_cluster_size=2, max_tree_depth=5, bi_partitioner=None)` — methods:

| Method | Description |
|---|---|
| `update_round(deltas) -> list[RoundAction]` | Evaluate split criterion, mutate tree, emit actions. |
| `get_client_cluster(cid) -> int` | Cluster for a client (auto-assigns new clients). |
| `assign_client(cid, cluster_id)` | Explicit assignment. |
| `get_cluster_members(cluster_id) -> list[str]` | Sorted member IDs. |
| `get_active_clusters() -> list[int]` | Active cluster IDs. |
| `get_tree_depth(cluster_id) -> int` | Depth in the tree. |
| `dump_tree() -> dict` | JSON-serializable tree + membership. |

## `mimosa.strategy.cfl`

`ClusteredFLStrategy(initial_parameters, cluster_manager, cluster_model_registry, eps_1=1.0, eps_2=1.0, sample_fraction=1.0, min_cluster_size=2, max_tree_depth=5, evaluate_fn=None, accept_failures=False)`

Subclasses `flwr.serverapp.strategy.Strategy` (Flower >= 1.28). Implements the
five abstract methods:

| Method | Purpose |
|---|---|
| `configure_train(round, arrays, config, grid) -> Iterable[Message]` | Route each node its cluster's model (TRAIN). |
| `aggregate_train(round, replies) -> (ArrayRecord \| None, MetricRecord \| None)` | Compute `dW`, cluster, split/aggregate, emit metrics. |
| `configure_evaluate(round, arrays, config, grid) -> Iterable[Message]` | Route each node its cluster's model (EVALUATE). |
| `aggregate_evaluate(round, replies) -> MetricRecord \| None` | Per-cluster loss/accuracy. |
| `summary()` | Log hyperparameters. |

Plus spec-compatible helpers:

| Method | Purpose |
|---|---|
| `initialize_parameters(client_manager)` | Root-cluster parameters. |
| `configure_fit(round, parameters, client_manager) -> list[(ClientProxy, FitIns)]` | Legacy-style per-cluster routing (testable core). |
| `aggregate_fit(round, results, failures) -> (Parameters, metrics)` | Legacy-style round processing. |
| `start(grid, initial_arrays, num_rounds=3, ...)` | Seed registry and run the Grid loop. |
| `dump_run_state() -> dict` | Hyperparameters + per-round history + cluster tree. |

## `mimosa.cli`

- `mimosa run <app_dir> [--config <yaml>] [flags...]` — wraps `flwr run`.
- `mimosa tree <run_dir> [--format text|json|dot]` — render a cluster tree.
- `mimosa --version` — mimosa-fl + upstream Flower versions.
