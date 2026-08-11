"""Vanilla FedAvg baseline on the SAME data split, model, seeds and rounds.

Only the strategy differs from the CFL experiment — this is the head-to-head
needed for the master's headline comparison figure. Run it with::

    mimosa run baselines/cfl -- --server-app fedavg_baseline:app
"""

from __future__ import annotations

import json
import os

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from data_utils import get_central_test_loader
from experiment import _device, client_app, test
from models import CFLNet, load_parameters_from_state_dict, load_state_dict_from_parameters
from plot_utils import plot_comparison

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    output_dir = cfg["output-dir"]

    model = CFLNet()
    initial_arrays = ArrayRecord(load_parameters_from_state_dict(model))

    global_accuracies: list[dict] = []

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model_eval = CFLNet()
        load_state_dict_from_parameters(model_eval, arrays.to_numpy_ndarrays())
        _, accuracy = test(model_eval, get_central_test_loader(), _device())
        global_accuracies.append({"round": server_round, "accuracy": accuracy})
        return MetricRecord({"accuracy": accuracy})

    strategy = FedAvg(fraction_train=cfg["sample-fraction"], fraction_evaluate=1.0)
    strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord(
            {"lr": cfg["learning-rate"], "local-epochs": cfg["local-epochs"]}
        ),
        num_rounds=cfg["num-rounds"],
        evaluate_fn=global_evaluate,
    )

    fedavg_dir = os.path.join(output_dir, "fedavg")
    os.makedirs(fedavg_dir, exist_ok=True)
    with open(os.path.join(fedavg_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump({"rounds": global_accuracies}, handle, indent=2)
    print(f"\n[baseline] FedAvg results written to {fedavg_dir}/")
