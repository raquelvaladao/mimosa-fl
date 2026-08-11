# CFL Tutorial

This tutorial walks through the `quickstart-cfl-pytorch` example step by step and
explains how Clustered Federated Learning (CFL) works inside mimosa-fl.

## 1. The problem

In standard Federated Averaging (FedAvg), every client trains the *same* global
model. If clients have **non-IID** data (each client sees a different
distribution), a single global model is a compromise that fits nobody well.

CFL solves this by **grouping clients by the direction of their updates** and
training one model per group:

1. In round R the server sends each client its *cluster's* current model.
2. Clients train locally and return their updated weights.
3. The server computes each client's weight delta `dW = W_after - W_sent`.
4. If a cluster's updates are sufficiently *divergent* (and strong enough), it
   **splits** into two clusters.
5. Each cluster continues with cluster-wise FedAvg — its own model.

## 2. The example data

`examples/quickstart-cfl-pytorch/data.py` splits MNIST so that:

- clients `0 .. num_clients/2 - 1` only see digits **0–4**,
- clients `num_clients/2 .. num_clients - 1` only see digits **5–9**.

These are two *ground-truth* clusters. Their update directions diverge, so CFL
should discover them automatically.

## 3. Server wiring

`server.py` builds three objects:

```python
from mimosa.server.cluster_manager import ClusterManager
from mimosa.server.registry import ClusterModelRegistry
from mimosa.strategy.cfl import ClusteredFLStrategy

cluster_manager = ClusterManager(eps_1=0.5, eps_2=0.5, min_cluster_size=2, max_tree_depth=3)
registry = ClusterModelRegistry(initial_parameters=ndarrays)
strategy = ClusteredFLStrategy(ndarrays, cluster_manager, registry, eps_1=0.5, eps_2=0.5)
```

- **`ClusterManager`** owns the binary cluster tree and decides split vs.
  aggregate every round.
- **`ClusterModelRegistry`** stores one model per active cluster and inherits
  weights on split.
- **`ClusteredFLStrategy`** is the Flower strategy that ties it together.

Then:

```python
strategy.start(
    grid=grid,
    initial_arrays=initial_arrays,
    train_config=ConfigRecord({"lr": 0.01, "local-epochs": 1}),
    num_rounds=5,
)
```

`start()` runs the modern Grid loop (`configure_train` → clients train →
`aggregate_train` → cluster/split → evaluate) for 5 rounds.

## 4. Client wiring

`client.py` implements the Flower `train` and `evaluate` handlers. Because
mimosa exchanges **plain NumPy arrays** (model-agnostic, per the spec), the
client rebuilds its `state_dict` positionally:

```python
names = list(model.state_dict().keys())
state_dict = {name: torch.from_numpy(arr) for name, arr in zip(names, arrays.to_numpy_ndarrays())}
model.load_state_dict(state_dict)
```

## 5. Running it

```bash
pip install -e ..            # mimosa-fl
pip install -e .             # the example
mimosa run .                # or: flwr run .
```

Watch the server log: round 1 typically reports `configure_train: routed 6
node(s) across 2 active cluster(s)` — the root split into two clusters, one per
digit group.

## 6. Inspecting the result

The server writes `metrics.json` with the full run state:

```bash
mimosa tree .                # ASCII tree:
# Cluster 0
# └── Cluster 1 (active)
# └── Cluster 2 (active)
```

The two active clusters correspond to the two digit groups, and each cluster's
accuracy converges on its own distribution.
