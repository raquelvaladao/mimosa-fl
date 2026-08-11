"""Bi-partitioning of client clusters.

Defines the pluggable :class:`BiPartitioner` interface and the default
:class:`AgglomerativeBiPartitioner` which wraps sklearn's
``AgglomerativeClustering`` (complete linkage on the negated cosine similarity
matrix) and always produces exactly two groups.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .similarity import pairwise_cosine_similarity


class BiPartitioner(Protocol):
    """Interface for strategies that split a cluster into exactly two groups.

    Implementations receive a dict keyed by **actual client IDs** and must
    return the original keys (never positional indices) — see the regression
    fix for felisat/clustered-federated-learning#2.
    """

    def partition(
        self, deltas: dict[str, np.ndarray]
    ) -> tuple[list[str], list[str]]:
        """Split ``deltas`` into two groups of client IDs.

        Returns ``(group1, group2)`` where both contain the original string keys
        of ``deltas``. Degenerate cases: zero clients -> ``([], [])``; one
        client -> ``([cid], [])``.
        """
        ...


class AgglomerativeBiPartitioner:
    """Agglomerative bi-partitioner using complete-linkage clustering.

    Computes the pairwise cosine similarity matrix ``S``, negates it to form a
    distance matrix ``D = -S``, and runs ``AgglomerativeClustering`` with
    ``metric="precomputed"`` and ``linkage="complete"`` to produce exactly two
    clusters. Output is deterministic for the same input.
    """

    def __init__(self) -> None:
        self._clustering_cls = _import_agglomerative_clustering()

    def partition(
        self, deltas: dict[str, np.ndarray]
    ) -> tuple[list[str], list[str]]:
        """Split ``deltas`` into two groups, returning the original client IDs."""
        keys = list(deltas.keys())
        n = len(keys)
        if n == 0:
            return [], []
        if n == 1:
            return keys, []
        if n == 2:
            # Exactly two points always form two singleton clusters under any
            # linkage; hard-coding keeps this deterministic without sklearn.
            return [keys[0]], [keys[1]]

        sim = pairwise_cosine_similarity(deltas)
        dist = -sim
        model = self._clustering_cls(
            n_clusters=2, metric="precomputed", linkage="complete"
        )
        labels = model.fit_predict(dist)
        group1 = [k for k, lab in zip(keys, labels) if int(lab) == 0]
        group2 = [k for k, lab in zip(keys, labels) if int(lab) == 1]

        # Guard against the degenerate all-one-label output: fall back to a
        # deterministic balanced split so both groups are always non-empty.
        if not group1 or not group2:  # pragma: no cover - sklearn always emits 2 labels
            mid = n // 2
            return keys[:mid], keys[mid:]
        return group1, group2


def _import_agglomerative_clustering():
    """Import sklearn lazily so the similarity module stays import-light."""
    from sklearn.cluster import AgglomerativeClustering  # type: ignore[import-not-found]

    return AgglomerativeClustering
