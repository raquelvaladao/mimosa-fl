"""Test fixtures: simulated in-process clients (no gRPC, no network).

A :class:`SimClient` models a client that, when given the parameters the server
sent, returns its "trained" weights as ``sent + delta`` — so the resulting
server-side weight delta is exactly the client's configured ``delta``. This
makes split/aggregate behaviour fully deterministic in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimClient:
    """A mock client identified by ``cid`` that returns a fixed weight delta."""

    cid: str
    delta: np.ndarray
    num_examples: int = 32

    def simulate(self, sent_parameters: list[np.ndarray]) -> list[np.ndarray]:
        """Return after-training weights: ``sent + delta`` (unflattened)."""
        sent = [np.asarray(p, dtype=np.float64) for p in sent_parameters]
        flat_sent = np.concatenate([p.reshape(-1) for p in sent])
        flat_after = flat_sent + np.asarray(self.delta, dtype=np.float64).reshape(-1)
        out: list[np.ndarray] = []
        offset = 0
        for p in sent:
            n = int(p.size)
            out.append(flat_after[offset : offset + n].reshape(p.shape))
            offset += n
        return out


@dataclass
class SimFitRes:
    """Minimal stand-in for a ``FitRes`` (parameters only)."""

    parameters: list[np.ndarray]
    num_examples: int = 32
    metrics: dict = field(default_factory=dict)


class SimpleClientManager:
    """Deterministic in-memory client manager used by the strategy's tests."""

    def __init__(self, clients: list[SimClient]) -> None:
        self._clients = list(clients)

    def num_available(self) -> int:
        return len(self._clients)

    def sample(self, num_clients: int, min_num_clients: int = 1) -> list[SimClient]:
        return self._clients[: max(num_clients, min_num_clients)]


class MockGrid:
    """Minimal ``Grid`` stand-in exposing node sampling and reply capture."""

    def __init__(self, node_ids: list[int]) -> None:
        self._node_ids = [int(n) for n in node_ids]
        self.sent_messages: list[object] = []
        self.replies: list[object] = []

    def get_node_ids(self) -> list[int]:
        return list(self._node_ids)

    def send_and_receive(self, messages, timeout=None):  # noqa: ANN001
        self.sent_messages = list(messages)
        return list(self.replies)
