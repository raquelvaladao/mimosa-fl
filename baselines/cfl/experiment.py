"""CFL baseline experiment: ServerApp + ClientApp for the paper reproduction.

Two execution modes:
- **normal**: ``mimosa run baselines/cfl`` (or ``flwr run``) — runs the CFL
  experiment over the Grid/ServerApp API and writes ``output/cfl/``.
- **notebook**: a follow-up ``experiment.ipynb`` stepping through the same
  logic cell-by-cell.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid, ServerApp

from mimosa._compat import arrayrecord_to_ndarrays
from mimosa.server.cluster_manager import ClusterManager
from mimosa.server.registry import ClusterModelRegistry
from mimosa.strategy.cfl import ClusteredFLStrategy

from data_utils import build_client_dataloaders, get_central_test_loader
from models import CFLNet, load_parameters_from_state_dict, load_state_dict_from_parameters
from plot_utils import plot_accuracy, plot_norms

# --------------------------------------------------------------------------- #
# Shared client-side helpers                                                   #
# --------------------------------------------------------------------------- #


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model: CFLNet, trainloader, epochs: int, lr: float, device) -> float:
    """Local SGD training; returns the final batch loss."""
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    loss = torch.tensor(0.0)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
    return float(loss.item())


def test(model: CFLNet, testloader, device) -> tuple[float, float]:
    """Return ``(loss, accuracy)`` over a DataLoader."""
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss_sum += float(criterion(output, labels).item())
            correct += int((output.argmax(1) == labels).sum().item())
            total += len(labels)
    return loss_sum / total, correct / total if total else 0.0


# --------------------------------------------------------------------------- #
# ClientApp                                                                   #
# --------------------------------------------------------------------------- #

client_app = ClientApp()


@client_app.train()
def client_train(msg: Message, context: Context) -> Message:
    """Train on the client's non-IID partition and reply with new weights."""
    model = CFLNet()
    load_state_dict_from_parameters(model, msg.content["arrays"].to_numpy_ndarrays())

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader = _dataloaders(
        partition_id,
        num_partitions,
        int(context.run_config["num-clusters"]),
    )[partition_id]

    loss = train(
        model,
        trainloader,
        int(context.run_config["local-epochs"]),
        float(msg.content["config"]["lr"]),
        _device(),
    )
    return Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord(load_parameters_from_state_dict(model)),
                "metrics": MetricRecord(
                    {"train_loss": loss, "num-examples": len(trainloader.dataset)}
                ),
            }
        ),
        reply_to=msg,
    )


@client_app.evaluate()
def client_evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the cluster's model on a held-out slice of the test set."""
    model = CFLNet()
    load_state_dict_from_parameters(model, msg.content["arrays"].to_numpy_ndarrays())

    testloader = get_central_test_loader()
    eval_loss, eval_acc = test(model, testloader, _device())
    return Message(
        content=RecordDict(
            {
                "metrics": MetricRecord(
                    {
                        "eval_loss": eval_loss,
                        "eval_acc": eval_acc,
                        "num-examples": len(testloader.dataset),
                    }
                )
            }
        ),
        reply_to=msg,
    )


# --------------------------------------------------------------------------- #
# Server-side helpers                                                         #
# --------------------------------------------------------------------------- #

_dataloader_cache: dict[tuple[int, int], dict[str, object]] = {}


def _dataloaders(partition_id: int, num_partitions: int, num_clusters: int):
    """Deterministically build the per-client dataloaders once per run.

    ``num_clusters`` is a **data-generation** parameter: it controls how many
    ground-truth non-IID groups are baked into the synthetic partition. It is
    NOT passed to the CFL algorithm, which discovers the number of clusters
    itself from ``eps_1`` / ``eps_2``.
    """
    key = (num_partitions, num_clusters)
    if key not in _dataloader_cache:
        _dataloader_cache[key] = build_client_dataloaders(
            num_clients=num_partitions,
            num_clusters=num_clusters,
            seed=42,
            batch_size=32,
        )
    return _dataloader_cache[key]


class RecordingCFLStrategy(ClusteredFLStrategy):
    """CFL strategy that folds per-cluster evaluation accuracy into history."""

    def aggregate_evaluate(self, server_round: int, replies: Iterable[object]):
        record = super().aggregate_evaluate(server_round, replies)
        if record is not None and self.history:
            last = self.history[-1]
            last["per_cluster_accuracy"] = {
                key.split(".")[-1]: float(value)
                for key, value in record.items()
                if key.startswith("per_cluster_accuracy.")
            }
        return record


def build_strategy(eps_1: float, eps_2: float, min_cluster_size: int, max_tree_depth: int, ndarrays: list):
    manager = ClusterManager(
        eps_1=eps_1, eps_2=eps_2, min_cluster_size=min_cluster_size, max_tree_depth=max_tree_depth
    )
    registry = ClusterModelRegistry(initial_parameters=ndarrays)
    return RecordingCFLStrategy(
        ndarrays, manager, registry,
        eps_1=eps_1, eps_2=eps_2, min_cluster_size=min_cluster_size, max_tree_depth=max_tree_depth,
        sample_fraction=1.0,
    )


# --------------------------------------------------------------------------- #
# ServerApp (CFL run)                                                         #
# --------------------------------------------------------------------------- #

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    num_clients = cfg["num-clients"]
    output_dir = cfg["output-dir"]

    model = CFLNet()
    initial_arrays = ArrayRecord(load_parameters_from_state_dict(model))
    ndarrays = arrayrecord_to_ndarrays(initial_arrays)

    strategy = build_strategy(
        cfg["eps-1"], cfg["eps-2"], cfg["min-cluster-size"], cfg["max-tree-depth"], ndarrays
    )
    strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord(
            {"lr": cfg["learning-rate"], "local-epochs": cfg["local-epochs"]}
        ),
        num_rounds=cfg["num-rounds"],
    )

    cfl_dir = os.path.join(output_dir, "cfl")
    os.makedirs(cfl_dir, exist_ok=True)
    with open(os.path.join(cfl_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(strategy.dump_run_state(), handle, indent=2)

    plot_accuracy(strategy.history, os.path.join(cfl_dir, "accuracy.png"))
    plot_norms(strategy.history, cfg["eps-1"], cfg["eps-2"], os.path.join(cfl_dir, "norms.png"))
    print(f"\n[baseline] CFL results written to {cfl_dir}/")
