"""Tests for the ``mimosa`` CLI (tree rendering, version, run command)."""

from __future__ import annotations

import json

from mimosa.cli.cli import build_parser
from mimosa.cli.tree import _render_dot, _render_text

TREE = {
    "nodes": [
        {"id": 0, "parent_id": None, "children": [1, 2], "depth": 0, "active": False},
        {"id": 1, "parent_id": 0, "children": [], "depth": 1, "active": True},
        {"id": 2, "parent_id": 0, "children": [], "depth": 1, "active": True},
    ],
    "client_membership": {"c1": 1, "c2": 2},
}


def test_parser_has_expected_subcommands() -> None:
    parser = build_parser()
    assert parser.parse_args(["run", "appdir", "--config", "cfg.yaml"]).app_dir == "appdir"
    tree_args = parser.parse_args(["tree", "rundir", "--format", "json"])
    assert tree_args.format == "json"


def test_tree_json_format(tmp_path) -> None:  # noqa: ANN001
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"cluster_tree": TREE}), encoding="utf-8")

    class Args:
        run_dir = str(tmp_path)
        metrics = "metrics.json"
        format = "json"

    from mimosa.cli.tree import tree_command

    assert tree_command(Args()) == 0


def test_tree_text_renders_active_markers() -> None:
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        _render_text(TREE)
    out = buffer.getvalue()
    assert "Cluster 0" in out
    assert "Cluster 1 (active)" in out
    assert "Cluster 2 (active)" in out


def test_tree_dot_output() -> None:
    dot = _render_dot(TREE)
    assert dot.startswith("digraph cluster_tree")
    assert '"0" -> "1";' in dot
    assert '"0" -> "2";' in dot


def test_tree_missing_metrics_file_errors(tmp_path) -> None:  # noqa: ANN001
    from mimosa.cli.tree import tree_command

    class Args:
        run_dir = str(tmp_path)
        metrics = "missing.json"
        format = "text"

    try:
        tree_command(Args())
    except SystemExit as exc:
        assert exc.code is not None
    else:  # pragma: no cover
        raise AssertionError("expected SystemExit for missing metrics file")


def test_cli_version_flag() -> None:
    from mimosa.cli.cli import main

    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    else:  # pragma: no cover
        raise AssertionError("expected SystemExit(0) from --version")


def test_cli_tree_command_through_main(tmp_path) -> None:  # noqa: ANN001
    """The full CLI entry path renders a tree without crashing (Windows-safe)."""
    import io
    from contextlib import redirect_stdout

    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"cluster_tree": TREE}), encoding="utf-8")

    from mimosa.cli.cli import main

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = main(["tree", str(tmp_path), "--format", "text"])
    assert code == 0
    assert "Cluster 1 (active)" in buffer.getvalue()
