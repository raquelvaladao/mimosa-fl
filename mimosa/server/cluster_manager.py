"""ClusterManager: the "brain" of the CFL decision logic.

Maintains the binary cluster tree, tracks client-to-cluster membership (with
soft membership for dynamic rosters), evaluates the eps_1/eps_2 split criterion
each round, and emits either split or aggregate actions. It has no dependency
on the strategy or the model registry.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from ..clustering import AgglomerativeBiPartitioner, BiPartitioner
from ..similarity import split_criterion
from .registry import ClusterNode


@dataclass
class RoundAction:
    """Decision emitted by :meth:`ClusterManager.update_round` for one cluster.

    Attributes
    ----------
    type : Literal["split", "aggregate"]
        Whether the cluster should split or just aggregate this round.
    cluster_id : int
        The cluster the action applies to.
    child_clusters : Optional[List[int]]
        For a split: the two new child cluster IDs.
    member_reassignment : Optional[Dict[str, int]]
        For a split: client ID -> new child cluster ID.
    members : Optional[List[str]]
        The client IDs that contributed deltas to this cluster this round.
    aggregated_parameters : Optional[object]
        Reserved for forward compatibility; the strategy computes the actual
        cluster-wise FedAvg update and stores it in the registry.
    """

    type: Literal["split", "aggregate"]
    cluster_id: int
    child_clusters: Optional[list[int]] = None
    member_reassignment: Optional[dict[str, int]] = None
    members: Optional[list[str]] = None
    aggregated_parameters: Optional[object] = None


class ClusterManager:
    """Tracks the cluster tree and makes per-round split/aggregate decisions.

    Parameters
    ----------
    eps_1 : float
        Mean-norm stability/divergence threshold. A cluster is a split candidate
        only if ``mean||dW|| < eps_1`` (updates point in divergent directions),
        matching the reference implementation.
    eps_2 : float
        Max-norm signal threshold. A cluster splits only if ``max||dW|| > eps_2``
        (updates still carry learning signal), matching the reference implementation.
    min_cluster_size : int
        Minimum members present this round for split eligibility (default 3).
    max_tree_depth : int
        Maximum tree depth (root = depth 0); clusters at this depth never split.
    min_split_round : int
        Minimum server round before a cluster is allowed to split (default 1).
    bi_partitioner : BiPartitioner
        Pluggable partitioning strategy (defaults to
        :class:`AgglomerativeBiPartitioner`).
    """

    def __init__(
        self,
        eps_1: float,
        eps_2: float,
        min_cluster_size: int = 3,
        max_tree_depth: int = 5,
        min_split_round: int = 1,
        bi_partitioner: Optional[BiPartitioner] = None,
    ) -> None:
        self.eps_1 = float(eps_1)
        self.eps_2 = float(eps_2)
        self.min_cluster_size = int(min_cluster_size)
        self.max_tree_depth = int(max_tree_depth)
        self.min_split_round = int(min_split_round)
        self.bi_partitioner = bi_partitioner or AgglomerativeBiPartitioner()

        self.cluster_tree: dict[int, ClusterNode] = {
            0: ClusterNode(id=0, parent_id=None, children=[], depth=0, active=True)
        }
        self.client_membership: dict[str, int] = {}
        self._next_cluster_id = 1
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Membership                                                          #
    # ------------------------------------------------------------------ #

    def get_client_cluster(self, cid: object) -> int:
        """Return the active cluster ID for a client.

        New clients (no recorded membership) are auto-assigned to the root
        cluster if it is active, otherwise to the shallowest active cluster.
        """
        key = str(cid)
        with self._lock:
            if key in self.client_membership:
                return self.client_membership[key]
            cluster_id = self._default_cluster_for_new_client()
            self.client_membership[key] = cluster_id
            return cluster_id

    def assign_client(self, cid: object, cluster_id: int) -> None:
        """Explicitly assign ``cid`` to ``cluster_id`` (for new clients)."""
        with self._lock:
            self.client_membership[str(cid)] = cluster_id

    def get_cluster_members(self, cluster_id: int) -> list[str]:
        """Return all clients assigned to ``cluster_id`` (sorted)."""
        with self._lock:
            return sorted(
                c for c, cl in self.client_membership.items() if cl == cluster_id
            )

    def get_active_clusters(self) -> list[int]:
        """Return IDs of all active (leaf) clusters, sorted ascending."""
        with self._lock:
            return sorted(cid for cid, node in self.cluster_tree.items() if node.active)

    def get_tree_depth(self, cluster_id: int) -> int:
        """Return the depth of ``cluster_id`` in the tree (-1 if unknown)."""
        with self._lock:
            node = self.cluster_tree.get(cluster_id)
            return node.depth if node is not None else -1

    # ------------------------------------------------------------------ #
    # Round processing                                                    #
    # ------------------------------------------------------------------ #

    def update_round(
        self, deltas: dict[str, np.ndarray], current_round: int = 1
    ) -> list[RoundAction]:
        """Process one round's deltas and emit split/aggregate actions.

        Deltas are grouped by each client's current cluster assignment. For
        every cluster with ``>= min_cluster_size`` members present, the split
        criterion is evaluated; if met, the cluster is bi-partitioned and the
        tree is mutated (child nodes created, parent deactivated, members
        reassigned). The returned actions let the strategy update the model
        registry accordingly.

        Parameters
        ----------
        deltas : Dict[str, ndarray]
            Client ID -> flattened weight delta ``dW``.
        current_round : int
            Current server round number (used for ``min_split_round`` gating).

        Returns
        -------
        List[RoundAction]
            One action per cluster that received deltas this round.
        """
        by_cluster: dict[int, list[str]] = {}
        for cid in deltas:
            cluster_id = self.get_client_cluster(cid)
            by_cluster.setdefault(cluster_id, []).append(cid)

        actions: list[RoundAction] = []
        with self._lock:
            for cluster_id in sorted(by_cluster):
                node = self.cluster_tree.get(cluster_id)
                if node is None or not node.active:
                    continue
                members = sorted(by_cluster[cluster_id])
                cluster_deltas = {c: deltas[c] for c in members}
                action = self._decide(cluster_id, cluster_deltas, current_round)
                if action.type == "split":
                    self._finalize_split(action)
                actions.append(action)
        return actions

    def _decide(
        self, cluster_id: int, cluster_deltas: dict[str, np.ndarray], current_round: int = 1
    ) -> RoundAction:
        """Apply the state machine and produce one action for a cluster."""
        node = self.cluster_tree[cluster_id]
        members = list(cluster_deltas.keys())

        def aggregate() -> RoundAction:
            return RoundAction(
                type="aggregate", cluster_id=cluster_id, members=members
            )

        if len(members) < self.min_cluster_size:
            return aggregate()
        if node.depth >= self.max_tree_depth:
            return aggregate()
        if current_round < self.min_split_round:
            return aggregate()
        if not split_criterion(cluster_deltas, self.eps_1, self.eps_2):
            return aggregate()

        group1, group2 = self.bi_partitioner.partition(cluster_deltas)
        if not group1 or not group2:
            return aggregate()
        # Guard: neither child may be a singleton.
        if len(group1) < 2 or len(group2) < 2:
            return aggregate()

        child1 = self._allocate_cluster_id()
        child2 = self._allocate_cluster_id()
        reassignment = {c: child1 for c in group1}
        reassignment.update({c: child2 for c in group2})
        return RoundAction(
            type="split",
            cluster_id=cluster_id,
            child_clusters=[child1, child2],
            member_reassignment=reassignment,
            members=members,
        )

    # ------------------------------------------------------------------ #
    # Tree mutation                                                       #
    # ------------------------------------------------------------------ #

    def _allocate_cluster_id(self) -> int:
        cluster_id = self._next_cluster_id
        self._next_cluster_id += 1
        return cluster_id

    def _finalize_split(self, action: RoundAction) -> None:
        """Mutate the tree: add children, deactivate parent, reassign members."""
        parent = self.cluster_tree[action.cluster_id]
        for child_id in action.child_clusters or []:
            self.cluster_tree[child_id] = ClusterNode(
                id=child_id,
                parent_id=parent.id,
                children=[],
                depth=parent.depth + 1,
                active=True,
            )
        parent.active = False
        parent.children = list(action.child_clusters or [])
        for cid, child_id in (action.member_reassignment or {}).items():
            self.client_membership[cid] = child_id

    def _default_cluster_for_new_client(self) -> int:
        """Pick a cluster for a brand-new client (root if active, else shallowest)."""
        node = self.cluster_tree.get(0)
        if node is not None and node.active:
            return 0
        active = self.get_active_clusters()
        if active:
            return min(active, key=lambda c: (self.cluster_tree[c].depth, c))
        return 0

    # ------------------------------------------------------------------ #
    # Introspection                                                       #
    # ------------------------------------------------------------------ #

    def dump_tree(self) -> dict:
        """JSON-serializable representation of the cluster tree and membership."""
        with self._lock:
            nodes = [
                {
                    "id": cid,
                    "parent_id": node.parent_id,
                    "children": list(node.children),
                    "depth": node.depth,
                    "active": node.active,
                }
                for cid, node in sorted(self.cluster_tree.items())
            ]
            memberships = {cid: cl for cid, cl in sorted(self.client_membership.items())}
        return {"nodes": nodes, "client_membership": memberships}
