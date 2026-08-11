# Quickstart: CFL with PyTorch

A minimal end-to-end example showing **Clustered Federated Learning (CFL)** with
mimosa-fl. Six simulated clients hold non-IID MNIST data — half only see digits
0–4, the other half only see digits 5–9. Because their weight-update directions
diverge, CFL automatically discovers the two groups and trains a dedicated model
per cluster.

## Run it

```bash
pip install -e ..            # install mimosa-fl from the repo root
pip install -e .             # install this example (from this directory)
mimosa run .                # or: flwr run .
```

Or, with an explicit config override:

```bash
mimosa run . --config config.yaml
```

## What to expect

- Round 1 usually **splits the root cluster** into two children, one per digit
  group.
- Subsequent rounds aggregate within each cluster (cluster-wise FedAvg), so each
  cluster's accuracy converges on its own data distribution.
- The server prints per-round metrics (``num_clusters``, ``max_norm``,
  ``mean_norm``, split events) and writes ``metrics.json``.

## Inspect the cluster tree

```bash
mimosa tree .                      # ASCII tree
mimosa tree . --format json        # raw structure
mimosa tree . --format dot         # Graphviz DOT
```

## Files

| File        | Purpose                                                    |
|-------------|------------------------------------------------------------|
| `model.py`  | Small CNN (28x28 -> 10 classes).                            |
| `data.py`   | Non-IID MNIST partition (two ground-truth digit groups).    |
| `client.py` | ClientApp: train / evaluate handlers.                       |
| `server.py` | ServerApp wiring `ClusteredFLStrategy` + registry + tree.   |

## Configuration

Edit `pyproject.toml` → `[tool.flwr.app.config]` (or override at run time):

| Key                 | Meaning                                   | Default |
|---------------------|-------------------------------------------|---------|
| `num-server-rounds` | Number of federated rounds                | 5       |
| `learning-rate`     | Client SGD learning rate                  | 0.01    |
| `local-epochs`      | Local epochs per round                    | 1       |
| `eps-1`             | Max-norm signal threshold                 | 0.5     |
| `eps-2`             | Orthogonality ratio threshold             | 0.5     |
| `min-cluster-size`  | Minimum members for split eligibility     | 2       |
| `max-tree-depth`    | Maximum cluster-tree depth                | 3       |

See the [CFL tutorial](../../docs/cfl/tutorial.md) for a step-by-step walkthrough.
