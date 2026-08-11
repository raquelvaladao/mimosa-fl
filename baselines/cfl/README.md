# CFL Baseline — Sattler et al. (arXiv:1910.01991)

Reproduces the paper's MNIST non-IID experiment with mimosa-fl's
`ClusteredFLStrategy` and produces a head-to-head comparison against vanilla
FedAvg — the headline result of the master's project.

## Setup

```bash
pip install -e ..              # mimosa-fl from the repo root
pip install -e .               # this baseline
```

## Run

**CFL run** (default):

```bash
mimosa run baselines/cfl
# or: flwr run baselines/cfl
```

**FedAvg comparison run** (same data split, model, seeds, rounds):

```bash
mimosa run baselines/cfl -- --server-app fedavg_baseline:app
```

**Generate the comparison figure** (after both runs):

```bash
python compare.py
```

## Outputs

```
output/
  cfl/
    metrics.json   -- per-round CFL metrics + cluster tree (feeds `mimosa tree`)
    accuracy.png   -- per-cluster accuracy vs round with split markers (Fig 1)
    norms.png      -- mean/max update norm vs round with eps_1/eps_2 lines (Fig 2)
  fedavg/
    metrics.json   -- per-round global FedAvg accuracy
  comparison.png   -- CFL (weighted) vs FedAvg accuracy (headline figure)
  results.json     -- structured metrics for both methods
```

Inspect the final cluster tree:

```bash
mimosa tree baselines/cfl/output/cfl
```

## Configuration

`config.yaml` (or `pyproject.toml` → `[tool.flwr.app.config]`) controls the
experiment:

| Key                | Default | Meaning                                  |
|--------------------|---------|------------------------------------------|
| `num-rounds`       | 50      | Federated rounds                         |
| `num-clients`      | 100     | Clients (paper setup)                    |
| `num-clusters`     | 2       | Ground-truth clusters (paper setup)      |
| `eps-1` / `eps-2`  | 0.5     | Split thresholds                         |
| `min-cluster-size` | 5       | Minimum members for split eligibility    |
| `max-tree-depth`   | 3       | Maximum cluster-tree depth               |
| `learning-rate`    | 0.1     | Client SGD learning rate                 |
| `local-epochs`     | 1       | Local epochs per round                   |
| `seed`             | 42      | Deterministic partitioning / initialization |

## Ground-truth clusters

`data_utils.partition_non_iid` assigns cluster *c* the digit classes
`{2c, 2c+1}`; every client draws from its cluster's two classes. The result is
exactly `num_clusters` ground-truth groups (constraint C3), which the learned
cluster tree should recover.

> **`num_clusters` is a data-generation parameter, not an algorithm parameter.**
> It controls how many non-IID groups are *baked into the synthetic partition*
> so the experiment has a known answer to validate against. The CFL algorithm
> (`ClusterManager` / `ClusteredFLStrategy`) never receives it — it discovers
> the number of clusters itself from `eps_1` / `eps_2` / `min_cluster_size` /
> `max_tree_depth`.

## Notes

- The partitioner uses `Dict[str, ndarray]` keyed by client ID, so no
  positional-index aliasing (felisat/clustered-federated-learning#2).
- `compare.py` refuses to plot from partial data (C7): it exits with an error if
  either `metrics.json` is missing.
