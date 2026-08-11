"""ClientApp for the CFL quickstart.

Implements the Flower ``train`` / ``evaluate`` handlers. The server routes each
node its own cluster's model; the client loads it positionally (mimosa exchanges
plain NumPy arrays, so the state_dict is rebuilt by parameter-name order).
"""

from __future__ import annotations

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from data import load_data, load_centralized_dataset
from model import Net

app = ClientApp()


def _load_arrays_into_model(model: torch.nn.Module, arrays) -> None:
    """Load a list of NumPy arrays into ``model`` by state_dict key order."""
    names = list(model.state_dict().keys())
    ndarrays = arrays.to_numpy_ndarrays()
    state_dict = {
        name: torch.from_numpy(arr) for name, arr in zip(names, ndarrays)
    }
    model.load_state_dict(state_dict)


def train(
    model: torch.nn.Module,
    trainloader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> float:
    """Run local SGD training; returns the last batch's loss."""
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


def test(
    model: torch.nn.Module, testloader, device: torch.device
) -> tuple[float, float]:
    """Evaluate a model; returns ``(loss, accuracy)``."""
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
    accuracy = correct / total if total else 0.0
    return loss_sum / total, accuracy


def global_evaluate(server_round: int, arrays) -> MetricRecord:
    """Centralized evaluation used by the ServerApp's ``evaluate_fn``."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net()
    _load_arrays_into_model(model, arrays)
    loss, accuracy = test(model, load_centralized_dataset(), device)
    return MetricRecord({"accuracy": accuracy, "loss": loss})


@app.train()
def train_round(msg: Message, context: Context) -> Message:
    """Train on the client's non-IID partition and reply with updated weights."""
    model = Net()
    _load_arrays_into_model(model, msg.content["arrays"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    samples_per_client = context.run_config.get("samples-per-client")
    rotation_degrees = context.run_config.get("rotation-degrees", 0.0)
    noise_seed = context.run_config.get("noise-seed", 42)
    rotated_client_ids = context.run_config.get("rotated-client-ids")
    trainloader, _ = load_data(
        partition_id, num_partitions, batch_size, samples_per_client,
        rotation_degrees, noise_seed, rotated_client_ids,
    )

    train_loss = train(
        model,
        trainloader,
        int(context.run_config["local-epochs"]),
        float(msg.content["config"]["lr"]),
        device,
    )

    return Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord(model.state_dict()),
                "metrics": MetricRecord(
                    {
                        "train_loss": train_loss,
                        "num-examples": len(trainloader.dataset),
                    }
                ),
            }
        ),
        reply_to=msg,
    )


@app.evaluate()
def evaluate_round(msg: Message, context: Context) -> Message:
    """Evaluate the cluster's model on the client's held-out validation set."""
    model = Net()
    _load_arrays_into_model(model, msg.content["arrays"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    rotation_degrees = context.run_config.get("rotation-degrees", 0.0)
    noise_seed = context.run_config.get("noise-seed", 42)
    rotated_client_ids = context.run_config.get("rotated-client-ids")
    _, valloader = load_data(
        partition_id, num_partitions, batch_size,
        rotation_degrees=rotation_degrees, noise_seed=noise_seed,
        rotated_client_ids=rotated_client_ids,
    )

    eval_loss, eval_acc = test(model, valloader, device)
    return Message(
        content=RecordDict(
            {
                "metrics": MetricRecord(
                    {
                        "eval_loss": eval_loss,
                        "eval_acc": eval_acc,
                        "num-examples": len(valloader.dataset),
                    }
                )
            }
        ),
        reply_to=msg,
    )
