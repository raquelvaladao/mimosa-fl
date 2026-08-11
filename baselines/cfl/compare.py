"""Produce the CFL-vs-FedAvg comparison figure and results table.

Reads the two runs' ``metrics.json`` files (written by ``experiment.py`` and
``fedavg_baseline.py``) and emits ``output/comparison.png`` plus
``output/results.json``. Fails with a clear error if either run's results are
missing, so figures are never generated from partial data (C7).
"""

from __future__ import annotations

import json
import os
import sys

from plot_utils import plot_comparison


def load_metrics(path: str) -> dict:
    if not os.path.isfile(path):
        sys.exit(f"ERROR: missing run results at {path}. "
                 "Run the CFL and FedAvg experiments first (see README).")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _weighted_accuracy(entry: dict) -> float:
    acc = entry.get("per_cluster_accuracy", {})
    memberships = entry.get("cluster_memberships", {})
    if not acc:
        return 0.0
    total = sum(len(memberships.get(c, [])) for c in acc) or 1
    return sum(float(acc[c]) * len(memberships.get(c, [])) for c in acc) / total


def main(output_dir: str = "output") -> int:
    cfl = load_metrics(os.path.join(output_dir, "cfl", "metrics.json"))
    fedavg = load_metrics(os.path.join(output_dir, "fedavg", "metrics.json"))

    cfl_history = cfl["rounds"]
    fedavg_history = fedavg["rounds"]
    if not cfl_history or not fedavg_history:
        sys.exit("ERROR: one of the runs has an empty history.")

    comparison_path = os.path.join(output_dir, "comparison.png")
    plot_comparison(cfl_history, fedavg_history, comparison_path)

    final_cfl = _weighted_accuracy(cfl_history[-1])
    final_fedavg = fedavg_history[-1].get("accuracy", 0.0)
    results = {
        "cfl_final_weighted_accuracy": final_cfl,
        "fedavg_final_accuracy": final_fedavg,
        "num_clusters_final": cfl_history[-1].get("num_clusters"),
        "cfl_history": cfl_history,
        "fedavg_history": fedavg_history,
    }
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Comparison figure written to {comparison_path}")
    print(f"Final accuracy  CFL (weighted): {final_cfl:.4f}   FedAvg: {final_fedavg:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
