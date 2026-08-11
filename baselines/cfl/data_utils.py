"""MNIST non-IID partitioning for the CFL baseline.

Ports the reference implementation's setup (felisat/clustered-federated-learning):
clients are split into exactly ``num_clusters`` ground-truth groups, each group
sees its own digit classes, and every client draws a fixed number of samples per
class so that the partition is deterministic for a given seed.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

_DATA_ROOT = "./data"


def load_mnist(train: bool):
    """Load the (optionally downloaded) MNIST split."""
    return datasets.MNIST(
        root=_DATA_ROOT, train=train, download=True, transform=transforms.ToTensor()
    )


def partition_non_iid(
    num_clients: int,
    num_clusters: int = 2,
    samples_per_class: int = 50,
    seed: int = 42,
) -> dict[str, Subset]:
    """Return ``{cid: Subset}`` where clusters have disjoint digit classes.

    Cluster ``c`` is assigned digit classes ``{2c, 2c+1}``. Every client in a
    cluster draws ``samples_per_class`` samples from each of its cluster's two
    classes. Clients within a cluster therefore produce aligned updates, while
    different clusters point in different directions — exactly two ground-truth
    clusters for validation (C3).
    """
    rng = np.random.default_rng(seed)
    dataset = load_mnist(train=True)
    targets = np.asarray(dataset.targets)

    partitions: dict[str, Subset] = {}
    clients_per_cluster = max(1, num_clients // num_clusters)

    for cluster in range(num_clusters):
        classes = [2 * cluster, 2 * cluster + 1]
        indices_by_class = {
            c: np.where(targets == c)[0] for c in classes
        }
        for i in range(clients_per_cluster):
            cid = f"client_{cluster * clients_per_cluster + i}"
            picked = np.concatenate(
                [
                    rng.choice(indices_by_class[c], size=samples_per_class, replace=False)
                    for c in classes
                ]
            )
            rng.shuffle(picked)
            partitions[cid] = Subset(dataset, picked.tolist())

    return partitions


def build_client_dataloaders(
    num_clients: int,
    num_clusters: int = 2,
    samples_per_class: int = 50,
    seed: int = 42,
    batch_size: int = 32,
) -> dict[str, DataLoader]:
    """Return ``{cid: DataLoader}`` over the deterministic non-IID partition."""
    partitions = partition_non_iid(num_clients, num_clusters, samples_per_class, seed)
    return {
        cid: DataLoader(subset, batch_size=batch_size, shuffle=True)
        for cid, subset in partitions.items()
    }


def get_central_test_loader(batch_size: int = 64) -> DataLoader:
    """Full MNIST test set for server-side evaluation."""
    return DataLoader(load_mnist(train=False), batch_size=batch_size)


def ground_truth_clusters(num_clients: int, num_clusters: int = 2) -> dict[str, int]:
    """Map each client ID to its ground-truth cluster index."""
    clients_per_cluster = max(1, num_clients // num_clusters)
    return {
        f"client_{cluster * clients_per_cluster + i}": cluster
        for cluster in range(num_clusters)
        for i in range(clients_per_cluster)
    }
