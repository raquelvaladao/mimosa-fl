"""``mimosa run`` — wrap ``flwr run`` with mimosa-fl defaults."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from argparse import Namespace


def _flwr_command() -> list[str]:
    """Return the command prefix used to invoke the Flower CLI."""
    exe = shutil.which("flwr")
    if exe:
        return [exe]
    return [sys.executable, "-m", "flwr"]


def run_command(args: Namespace) -> int:
    """Execute ``flwr run <app_dir>`` passing through all extra flags.

    ``--config <yaml>`` (mimosa-specific) loads a YAML file of mimosa-fl
    defaults and exposes it to the app via the ``MIMOSA_CONFIG`` environment
    variable; every other flag is passed through to ``flwr run`` untouched.
    """
    cmd = _flwr_command() + ["run", args.app_dir]
    if args.passthrough:
        cmd.extend(args.passthrough)

    env = dict(os.environ)
    if args.config:
        env["MIMOSA_CONFIG"] = os.path.abspath(args.config)

    print(f"[mimosa] running: {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)
