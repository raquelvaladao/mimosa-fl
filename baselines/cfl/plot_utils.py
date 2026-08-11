"""Plotting utilities for the CFL baseline.

Reproduces the reference implementation's ``display_train_stats`` figures plus
the CFL-vs-FedAvg comparison (the master's headline figure).
"""

from __future__ import annotations

from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_train_stats(
    history: list[dict],
    eps_1: float,
    eps_2: float,
    communication_rounds: Optional[int] = None,
    output_path: Optional[str] = None,
) -> None:
    """Reference-style training stats figure.

    Faithfully reproduces ``display_train_stats`` from the reference
    implementation (felisat/clustered-federated-learning):

    - **Left**: mean per-cluster accuracy over rounds with a +/- std band,
      black vertical lines at split rounds, and a ``Clusters: ...`` annotation
      with the final cluster memberships.
    - **Right**: ``||sum_i dW_i||`` and ``max_i ||dW_i||`` update norms with
      ``eps_1`` (dashed) and ``eps_2`` (dotted) threshold lines and split lines.

    ``history`` is the strategy's per-round metrics (``strategy.history``),
    optionally enriched with ``per_cluster_accuracy``.
    """
    if communication_rounds is None:
        communication_rounds = max((e["round"] for e in history), default=0)
    rounds = [e["round"] for e in history]
    split_rounds = [e["round"] for e in history if e.get("split_events")]

    fig = plt.figure(figsize=(12, 4))

    # --- left: accuracy mean/std across clusters --------------------------
    plt.subplot(1, 2, 1)
    acc_values = [list(e.get("per_cluster_accuracy", {}).values()) for e in history]
    acc_mean = np.array([float(np.mean(a)) if a else np.nan for a in acc_values])
    acc_std = np.array([float(np.std(a)) if a else 0.0 for a in acc_values])
    plt.fill_between(
        rounds, acc_mean - acc_std, acc_mean + acc_std, alpha=0.5, color="C0"
    )
    plt.plot(rounds, acc_mean, color="C0")
    for s in split_rounds:
        plt.axvline(x=s, linestyle="-", color="k", label="Split")

    if history:
        last = history[-1]
        memberships = last.get("cluster_memberships", {})
        clusters = [sorted(m) for m in memberships.values()]
        clusters.sort(key=lambda m: m[0] if m else "")
        plt.text(
            x=communication_rounds,
            y=1,
            ha="right",
            va="top",
            s=f"Clusters: {clusters}",
        )

    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy")
    plt.xlim(0, communication_rounds)
    plt.ylim(0, 1)

    # --- right: update norms ----------------------------------------------
    plt.subplot(1, 2, 2)
    plt.plot(
        rounds,
        [e.get("sum_norm", 0.0) for e in history],
        color="C1",
        label=r"$\|\sum_i\Delta W_i \|$",
    )
    plt.plot(
        rounds,
        [e.get("max_norm", 0.0) for e in history],
        color="C2",
        label=r"$\max_i\|\Delta W_i \|$",
    )
    plt.axhline(y=eps_1, linestyle="--", color="k", label=r"$\varepsilon_1$")
    plt.axhline(y=eps_2, linestyle=":", color="k", label=r"$\varepsilon_2$")
    for s in split_rounds:
        plt.axvline(x=s, linestyle="-", color="k", label="Split")

    plt.xlabel("Communication Rounds")
    plt.legend()
    plt.xlim(0, communication_rounds)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
    plt.show()
    plt.close(fig)


def plot_accuracy(
    history: list[dict],
    output_path: str,
    eps_lines: Optional[tuple[float, float]] = None,
) -> None:
    """Per-cluster accuracy vs round with split-event markers (reference Fig 1)."""
    rounds = [entry["round"] for entry in history]
    split_rounds = [
        entry["round"]
        for entry in history
        if entry.get("split_events")
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    clusters = set()
    for entry in history:
        clusters.update(entry.get("per_cluster_accuracy", {}).keys())
    for cluster in sorted(clusters, key=int):
        accs = [
            entry.get("per_cluster_accuracy", {}).get(cluster)
            for entry in history
        ]
        ax.plot(rounds, accs, marker="o", label=f"Cluster {cluster}")

    if split_rounds:
        ax.axvline(split_rounds[0], color="gray", linestyle="--", label="split")
    if eps_lines is not None:
        eps_1, eps_2 = eps_lines
        ax.axhline(eps_1, color="tab:red", linestyle=":", alpha=0.6, label="eps_1")
        ax.axhline(eps_2, color="tab:green", linestyle=":", alpha=0.6, label="eps_2")

    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    ax.set_title("CFL: per-cluster accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_norms(history: list[dict], eps_1: float, eps_2: float, output_path: str) -> None:
    """Mean/max update norm vs round with eps_1/eps_2 threshold lines (Fig 2)."""
    rounds = [entry["round"] for entry in history]
    max_norms = [entry["max_norm"] for entry in history]
    mean_norms = [entry["mean_norm"] for entry in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, max_norms, marker="o", label="max ||dW||")
    ax.plot(rounds, mean_norms, marker="s", label="mean ||dW||")
    ax.axhline(eps_1, color="tab:red", linestyle="--", label=f"eps_1 = {eps_1}")
    ax.axhline(eps_2, color="tab:green", linestyle="--", label=f"eps_2 = {eps_2}")
    ax.set_xlabel("Round")
    ax.set_ylabel("Update norm")
    ax.set_title("CFL: update norms vs round")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_comparison(
    cfl_history: list[dict],
    fedavg_history: list[dict],
    output_path: str,
) -> None:
    """FedAvg global accuracy vs CFL weighted accuracy on a shared axis."""
    fig, ax = plt.subplots(figsize=(8, 5))

    cfl_rounds = [e["round"] for e in cfl_history]
    cfl_weights = [len(e.get("cluster_memberships", {})) for e in cfl_history]
    cfl_weighted = [
        _weighted_accuracy(e)
        for e in cfl_history
    ]
    ax.plot(cfl_rounds, cfl_weighted, marker="o", label="mimosa CFL (weighted)")

    fa_rounds = [e["round"] for e in fedavg_history]
    fa_acc = [e.get("accuracy") for e in fedavg_history]
    ax.plot(fa_rounds, fa_acc, marker="s", label="FedAvg (global)")

    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    ax.set_title("CFL vs FedAvg on non-IID MNIST")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _weighted_accuracy(entry: dict) -> float:
    """Weighted per-cluster accuracy (weighted by cluster member count)."""
    acc = entry.get("per_cluster_accuracy", {})
    memberships = entry.get("cluster_memberships", {})
    total_weight = sum(len(memberships.get(c, [])) for c in acc) or 1
    return (
        sum(float(acc[c]) * len(memberships.get(c, [])) for c in acc) / total_weight
        if acc
        else 0.0
    )
