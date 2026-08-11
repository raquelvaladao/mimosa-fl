"""ServerApp for the CFL quickstart.

Wires up ``ClusteredFLStrategy`` with a ``ClusterManager`` and a
``ClusterModelRegistry``, runs ``num-server-rounds`` rounds over the Grid, and
persists the run state (cluster tree + per-round metrics) to ``metrics.json``
for ``mimosa tree``.
"""

from __future__ import annotations

import json

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from mimosa._compat import arrayrecord_to_ndarrays
from mimosa.server.cluster_manager import ClusterManager
from mimosa.server.registry import ClusterModelRegistry
from mimosa.strategy.cfl import ClusteredFLStrategy

from client import global_evaluate
from model import Net

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config

    # Initial model -> parameters (as plain NumPy arrays).
    model = Net()
    initial_arrays = ArrayRecord(model.state_dict())
    ndarrays = arrayrecord_to_ndarrays(initial_arrays)

    # CFL server components.
    cluster_manager = ClusterManager(
        eps_1=cfg["eps-1"],
        eps_2=cfg["eps-2"],
        min_cluster_size=cfg.get("min-cluster-size", 3),
        max_tree_depth=cfg["max-tree-depth"],
        min_split_round=cfg.get("min-split-round", 1),
    )
    registry = ClusterModelRegistry(initial_parameters=ndarrays)
    strategy = ClusteredFLStrategy(
        ndarrays,
        cluster_manager,
        registry,
        eps_1=cfg["eps-1"],
        eps_2=cfg["eps-2"],
        min_cluster_size=cfg.get("min-cluster-size", 3),
        max_tree_depth=cfg["max-tree-depth"],
        sample_fraction=1.0,
    )

    # Run the modern Grid loop.
    strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord(
            {
                "lr": cfg["learning-rate"],
                "local-epochs": cfg["local-epochs"],
            }
        ),
        num_rounds=cfg["num-server-rounds"],
        evaluate_fn=global_evaluate,
    )

    # Persist run state for post-run analysis and `mimosa tree`.
    with open("metrics.json", "w", encoding="utf-8") as handle:
        json.dump(strategy.dump_run_state(), handle, indent=2)
    print("\nCluster tree saved to metrics.json — visualize with: mimosa tree .")
