#!/usr/bin/env python3
"""
Main training launcher for Jackal Crossroad environment.

This script reuses existing training scripts and provides one unified entry point.
"""

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_MAP = {
    "random": "train_random_agent.py",
    "geometric": "train_geometric_agent.py",
    "sac": "train_sac_agent.py",
}


def run_script(script_name, forwarded_args=None):
    """Run one training script as a separate process."""
    script_dir = Path(__file__).resolve().parent
    script_path = script_dir / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Training script not found: {script_path}")

    cmd = [sys.executable, str(script_path)]
    if forwarded_args:
        cmd.extend(forwarded_args)

    print(f"\n=== Running {script_name} ===")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_name} failed with exit code {result.returncode}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Unified launcher for Jackal Crossroad training scripts"
    )
    parser.add_argument(
        "--agent",
        choices=["random", "geometric", "sac", "all"],
        default="sac",
        help="Select which training script to run (default: sac)",
    )
    parser.add_argument(
        "--sac-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments forwarded to train_sac_agent.py (use after --sac-args)",
    )

    args = parser.parse_args()

    if args.agent == "all":
        run_script(SCRIPT_MAP["random"])
        run_script(SCRIPT_MAP["geometric"])
        run_script(SCRIPT_MAP["sac"], args.sac_args)
        return

    script_name = SCRIPT_MAP[args.agent]
    forwarded_args = args.sac_args if args.agent == "sac" else None
    run_script(script_name, forwarded_args)


if __name__ == "__main__":
    main()
