"""Non-IID MNIST partitioning for the CFL quickstart.

Clients are divided into two ground-truth groups: the first half only sees
digits 0-4, the second half only sees digits 5-9. Their weight-update
directions therefore diverge and CFL splits them into two clusters.

The MNIST dataset and the per-client index slices are **cached in memory** so
that repeated calls (10 clients x 50 rounds) do not reload the full dataset
from disk every round.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

_DATA_ROOT = "./data"

# Cache loaded datasets and deterministic per-client index slices. This is what
# makes a 10-client / 50-round run tractable on CPU.
_DATASET_CACHE: dict[tuple[bool, str], object] = {}
_SLICE_CACHE: dict[tuple[int, int, frozenset[int], int], np.ndarray] = {}
_SEED = 42


def _get_mnist(train: bool):
    key = (train, _DATA_ROOT)
    if key not in _DATASET_CACHE:
        _DATASET_CACHE[key] = datasets.MNIST(
            root=_DATA_ROOT,
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )
    return _DATASET_CACHE[key]


def _group_and_slice(
    partition_id: int,
    num_partitions: int,
    allowed: set[int],
    samples_per_client: int | None = None,
):
    """Return a deterministic per-client slice of indices within ``allowed``."""
    key = (partition_id, num_partitions, frozenset(allowed), samples_per_client or 0)
    if key in _SLICE_CACHE:
        return _SLICE_CACHE[key]

    train = _get_mnist(train=True)
    indices = [i for i, (_, y) in enumerate(train) if int(y) in allowed]
    group_clients = max(1, num_partitions // 2)
    position = int(partition_id) % group_clients
    result = np.array_split(np.array(indices, dtype=np.int64), group_clients)[position]

    if samples_per_client and len(result) > samples_per_client:
        rng = np.random.default_rng(_SEED)
        result = rng.choice(result, size=samples_per_client, replace=False)

    _SLICE_CACHE[key] = result
    return result


def _rotate_subset(dataset, indices, degrees: float, seed: int):
    """Return a TensorDataset with each image rotated by a deterministic angle."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for idx in indices:
        x, y = dataset[int(idx)]
        angle = float(rng.uniform(-degrees, degrees))
        xs.append(transforms.functional.rotate(x, angle))
        ys.append(y)
    return TensorDataset(torch.stack(xs), torch.tensor(ys))


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int = 32,
    samples_per_client: int | None = None,
    rotation_degrees: float = 0.0,
    noise_seed: int = 42,
    rotated_client_ids: set[int] | None = None,
):
    """Return ``(trainloader, valloader)`` for a non-IID client partition.

    Parameters
    ----------
    rotation_degrees : float
        If > 0 and this client is in ``rotated_client_ids`` (or
        ``rotated_client_ids`` is ``None`` meaning all clients), every
        training image is rotated by a random angle drawn uniformly from
        ``[-rotation_degrees, +rotation_degrees]`` using a deterministic
        seed ``noise_seed + partition_id``.
    rotated_client_ids : set[int] | None
        Which clients receive rotation noise.  ``None`` = all clients.
    """
    group = 0 if partition_id < num_partitions // 2 else 1
    allowed = set(range(0, 5)) if group == 0 else set(range(5, 10))

    train = _get_mnist(train=True)
    my_train_idx = _group_and_slice(
        partition_id, num_partitions, allowed, samples_per_client
    )
    apply_rot = rotation_degrees and (
        rotated_client_ids is None or partition_id in rotated_client_ids
    )
    if apply_rot:
        train_ds = _rotate_subset(
            train, my_train_idx.tolist(), rotation_degrees, noise_seed + partition_id
        )
    else:
        train_ds = Subset(train, my_train_idx.tolist())
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test = _get_mnist(train=False)
    test_idx = [i for i, (_, y) in enumerate(test) if int(y) in allowed][:1000]
    valloader = DataLoader(Subset(test, test_idx), batch_size=batch_size)

    return trainloader, valloader


def load_centralized_dataset(batch_size: int = 64):
    """Return a DataLoader over the full MNIST test set (server-side eval)."""
    test = _get_mnist(train=False)
    return DataLoader(test, batch_size=batch_size)
