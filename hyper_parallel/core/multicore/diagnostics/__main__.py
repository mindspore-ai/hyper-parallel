# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Offline entry point for collected execution-monitor reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hyper_parallel.core.multicore.diagnostics.execution.reports import summarize_execution_reports


def main(argv: list[str] | None = None) -> int:
    """Summarize a collected run, returning a nonzero status for incomplete coverage.

    Args:
        argv: Optional command-line arguments; None reads process arguments.

    Returns:
        Zero for complete coverage, two when requested observations are missing or invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    summarize = commands.add_parser("summarize", help="check ranks/steps and summarize Host events")
    summarize.add_argument("root", type=Path)
    summarize.add_argument("--world-size", type=int, required=True)
    summarize.add_argument("--steps", required=True, help="comma-separated zero-based sampled steps")
    args = parser.parse_args(argv)
    try:
        steps = [int(value) for value in args.steps.split(",")]
        report = summarize_execution_reports(args.root, world_size=args.world_size, expected_steps=steps)
    except ValueError as error:
        parser.error(str(error))
    text = json.dumps(report, indent=2) + "\n"
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "summary.json").write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
