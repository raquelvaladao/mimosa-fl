"""mimosa CLI entry point.

Usage::

    mimosa run <app_dir> [--config <config.yaml>] [flwr run flags...]
    mimosa tree <run_dir> [--format text|json|dot]
    mimosa --version
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from .. import __version__


def _flwr_version() -> str:
    try:
        import flwr

        return getattr(flwr, "__version__", "unknown")
    except Exception:  # pragma: no cover - flwr not installed
        return "not installed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mimosa",
        description="mimosa-fl: Clustered Federated Learning on Flower.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"mimosa-fl {__version__} (upstream Flower {_flwr_version()})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Run a CFL experiment (wraps `flwr run`)."
    )
    run_parser.add_argument("app_dir", help="Path to the Flower app directory.")
    run_parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Optional YAML config with mimosa-fl defaults (exposed as MIMOSA_CONFIG).",
    )
    run_parser.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Additional flags passed through to `flwr run` unchanged.",
    )

    tree_parser = subparsers.add_parser(
        "tree", help="Visualize a cluster tree from a run's metrics JSON."
    )
    tree_parser.add_argument("run_dir", help="Directory containing the run metrics.")
    tree_parser.add_argument(
        "--format",
        choices=["text", "json", "dot"],
        default="text",
        help="Output format (default: text).",
    )
    tree_parser.add_argument(
        "--metrics",
        default="metrics.json",
        help="Metrics file name relative to run_dir (default: metrics.json).",
    )
    return parser


def _ensure_utf8_stdout() -> None:
    """Best-effort UTF-8 stdout so box-drawing tree output works on Windows."""
    try:
        for stream in (sys.stdout, sys.stderr):
            if stream is not None and hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # pragma: no cover - exotic stream
        pass


def main(argv: Optional[list[str]] = None) -> int:
    _ensure_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        from .run import run_command

        return run_command(args)
    if args.command == "tree":
        from .tree import tree_command

        return tree_command(args)
    parser.error(f"unknown command: {args.command}")  # pragma: no cover
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
