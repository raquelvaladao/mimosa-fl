"""Cluster model registry.

Stores one set of model parameters per active cluster and handles model
inheritance when a cluster splits. Framework-agnostic: it only manipulates
plain lists of NumPy arrays (Flower ``Parameters`` = ``list[ndarray]``).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

#: Model parameters are represented as a plain list of NumPy arrays.
Parameters = list[np.ndarray]


@dataclass
class ClusterNode:
    """A node in the binary cluster tree.

    Attributes
    ----------
    id : int
        Monotonically increasing cluster ID (root = 0).
    parent_id : Optional[int]
        Parent cluster ID, or ``None`` for the root.
    children : List[int]
        Child cluster IDs (empty for active leaves).
    depth : int
        Tree depth measured from the root (root depth = 0).
    active : bool
        Whether the cluster is a leaf currently receiving updates.
    """

    id: int
    parent_id: Optional[int]
    children: list[int] = field(default_factory=list)
    depth: int = 0
    active: bool = True


class ClusterModelRegistry:
    """Thread-safe store of one ``Parameters`` per cluster.

    Parameters
    ----------
    initial_parameters : Optional[Parameters]
        Initial model parameters seeded into the root cluster (ID 0). If
        ``None`` the root is registered as active but has no model until
        :meth:`update_parameters` is called.
    """

    def __init__(self, initial_parameters: Optional[Parameters] = None) -> None:
        self.models: dict[int, Parameters] = {}
        self.active: set[int] = set()
        self._next_id = 0
        self._lock = threading.RLock()
        if initial_parameters is not None:
            self.models[0] = _copy_parameters(initial_parameters)
        # The root cluster (ID 0) always exists, mirroring the ClusterManager.
        self.active.add(0)

    def get_parameters(self, cluster_id: int) -> Optional[Parameters]:
        """Return a copy of the current parameters for ``cluster_id``."""
        with self._lock:
            params = self.models.get(cluster_id)
            return _copy_parameters(params) if params is not None else None

    def update_parameters(self, cluster_id: int, params: Parameters) -> None:
        """Store ``params`` as the model of ``cluster_id``."""
        with self._lock:
            self.models[cluster_id] = _copy_parameters(params)

    def create_child(
        self,
        parent_id: int,
        params: Parameters,
        child_id: Optional[int] = None,
    ) -> int:
        """Create a new child cluster seeded with ``params``.

        Parameters
        ----------
        parent_id : int
            Parent cluster ID (informational; used to stay in sync with the
            ClusterManager's tree).
        params : Parameters
            Parameters the child inherits (the parent's current model).
        child_id : Optional[int]
            Explicit child ID. When ``None`` a fresh ID is allocated; when
            provided it is used directly so the registry can stay in sync with
            IDs emitted by :class:`~mimosa.server.cluster_manager.ClusterManager`.

        Returns
        -------
        int
            The new child cluster ID.
        """
        with self._lock:
            if child_id is None:
                self._next_id += 1
                child_id = self._next_id
            else:
                self._next_id = max(self._next_id, child_id)
            self.models[child_id] = _copy_parameters(params)
            self.active.add(child_id)
            return child_id

    def deactivate(self, cluster_id: int) -> None:
        """Mark a cluster inactive (called when it is split)."""
        with self._lock:
            self.active.discard(cluster_id)

    def get_active(self) -> list[int]:
        """Return all active (leaf) cluster IDs, sorted ascending."""
        with self._lock:
            return sorted(self.active)


def _copy_parameters(params: Parameters) -> Parameters:
    """Deep-copy a parameter list so stored models are never aliased."""
    return [np.asarray(p, copy=True) for p in params]
