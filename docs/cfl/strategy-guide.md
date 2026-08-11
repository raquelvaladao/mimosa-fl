# Strategy Guide

How to choose `eps_1` / `eps_2`, interpret the cluster tree, and know when CFL
helps over standard FedAvg.

## The two thresholds

Every round, for each cluster with `>= min_cluster_size` members present, the
server computes:

- `max_norm = max(||dW_i||)` — the strongest update magnitude,
- `mean_norm = ||mean(dW_i)||` — the norm of the average update.

A cluster **splits** only if all of these hold:

```
max_norm > eps_1                  (updates carry enough signal)
mean_norm / max_norm < eps_2      (updates point in divergent directions)
|members| >= min_cluster_size     (enough data to split on)
depth < max_tree_depth            (tree has room)
```

### Choosing `eps_1` (signal)

`eps_1` is a **noise floor**. If `max_norm <= eps_1`, updates are too small to
cluster meaningfully — the cluster just aggregates. Set `eps_1` *above* the
typical update magnitude you expect from random noise:

- **Too low** → random fluctuations trigger spurious splits.
- **Too high** → clusters never form (CFL degrades to FedAvg).

Start around `0.5–1.0` on normalized models and inspect `max_norm` in the
norms figure. `eps_1` should sit just below the *signal* updates and above the
*noise* updates.

### Choosing `eps_2` (divergence)

`eps_2` controls how "pointing in different directions" updates must be.

- `mean_norm / max_norm == 1.0` means all updates are aligned (parallel) → no
  split.
- `mean_norm / max_norm` near `0.0` means updates cancel (orthogonal or
  opposite) → strong split signal.

A smaller `eps_2` requires more divergence to split; a larger `eps_2` splits
more eagerly. `0.5` is a reasonable default — it requires updates to average to
less than half the maximum norm.

## Interpreting the cluster tree

`mimosa tree <run_dir>` renders the learned tree:

```
Cluster 0                <- root (all clients), now inactive after splitting
├── Cluster 1 (active)   <- leaf: a homogeneous client group
└── Cluster 2 (active)
```

- **Active (leaf)** clusters are the final groups; each has its own model.
- **Inactive** nodes were split and no longer receive updates.
- Members reassignments happen exactly once, at the split round.

A healthy result: the tree recovers the *ground-truth* groups (e.g., digit
groups in the quickstart), each leaf is tight (aligned updates), and no leaf
splits further.

## When CFL helps vs. FedAvg

**CFL helps** when client data is **clustered non-IID**: distinct groups with
distinct distributions. Example signals:

- Clients fall into recognizable groups (geography, device type, label skew).
- A single global model plateaus while per-group models would do better.
- Update directions correlate with client groups.

**CFL does not help** (use FedAvg) when:

- Data is IID or mildly skewed — clustering adds overhead without benefit.
- Every client is unique (updates never align) — you would need personalized
  FL (out of scope) instead.
- The signal is too weak (`max_norm` rarely exceeds `eps_1`).

## Soft membership and dynamic rosters

Clients that miss a round keep their last cluster assignment. When they return
they receive their cluster's current model and their delta participates in that
cluster's next split decision. New clients are assigned to the root cluster
(if active) or the shallowest active cluster.
