"""``mimosa tree`` — visualize a cluster tree from a run's metrics JSON."""

from __future__ import annotations

import json
import os
from argparse import Namespace


def _load_run_state(run_dir: str, metrics_file: str) -> dict:
    path = os.path.join(run_dir, metrics_file)
    if not os.path.isfile(path):
        raise SystemExit(
            f"metrics file not found: {path} (run the experiment first)."
        )
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _render_text(tree: dict) -> None:
    nodes = {node["id"]: node for node in tree.get("nodes", [])}

    def walk(cluster_id: int, prefix: str = "", is_last: bool = True) -> None:
        node = nodes.get(cluster_id)
        if node is None:
            return
        label = f"Cluster {cluster_id}" + (" (active)" if node["active"] else "")
        print(prefix + ("└── " if is_last else "├── ") + label)
        children = node.get("children", [])
        for index, child in enumerate(children):
            walk(child, prefix + ("    " if is_last else "│   "), index == len(children) - 1)

    roots = [node for node in tree.get("nodes", []) if node["parent_id"] is None]
    for root in roots:
        print(f"Cluster {root['id']}")
        for index, child in enumerate(root.get("children", [])):
            walk(child, "", index == len(root["children"]) - 1)


def _render_dot(tree: dict) -> str:
    lines = ["digraph cluster_tree {"]
    for node in tree.get("nodes", []):
        attrs = 'style="filled", fillcolor="lightgreen"' if node["active"] else "color=gray"
        lines.append(f'  "{node["id"]}" [{attrs}];')
    for node in tree.get("nodes", []):
        for child in node.get("children", []):
            lines.append(f'  "{node["id"]}" -> "{child}";')
    lines.append("}")
    return "\n".join(lines)


def tree_command(args: Namespace) -> int:
    """Render the cluster tree in the requested format."""
    data = _load_run_state(args.run_dir, args.metrics)
    tree = data.get("cluster_tree", data)

    if args.format == "json":
        print(json.dumps(tree, indent=2))
    elif args.format == "dot":
        print(_render_dot(tree))
    else:
        _render_text(tree)
    return 0
