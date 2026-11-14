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
"""Command-line interface for the autoresearch loop.

Commands::

    python .agent/skills/autoresearch/scripts/autoresearch.py init \\
        --run-dir <dir> --target <file> --benchmark-cmd "<cmd>"
    python .agent/skills/autoresearch/scripts/autoresearch.py baseline --run-dir <dir>
    python .agent/skills/autoresearch/scripts/autoresearch.py iterate \\
        --run-dir <dir> --description "<what this experiment tries>"
    python .agent/skills/autoresearch/scripts/autoresearch.py status --run-dir <dir>

``iterate`` expects the experimental edit to sit uncommitted in the run's
target files; everything after that point - commit, gate, benchmark,
decision, bookkeeping, restore - is owned by the tool.
"""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CONFIG_NAME, load_run_config  # pylint: disable=wrong-import-position
from loop import (  # pylint: disable=wrong-import-position
    RESULTS_HEADER,
    iterate,
    record_baseline,
    run_status,
)

_TEMPLATES = Path(__file__).resolve().parent.parent / "templates"


def _repo_root(start: Path) -> Path:
    """Locate the enclosing git repository root."""
    done = subprocess.run(
        ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True,
    )
    return Path(done.stdout.strip())


def command_init(args: argparse.Namespace) -> int:
    """Scaffold a run directory: run.json plus the living documents."""
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    _repo_root(run_dir)  # fail fast when the run dir is outside a git repository
    config_file = run_dir / CONFIG_NAME
    if config_file.exists() and not args.force:
        print(f"refusing to overwrite existing {config_file} (use --force)")
        return 1
    config_file.write_text(json.dumps({
        "target_files": list(args.target),
        "benchmark_cmd": shlex.split(args.benchmark_cmd),
        "gate_cmd": shlex.split(args.gate_cmd) if args.gate_cmd else [],
        "metric_pattern": args.metric_pattern,
        "metric_lower_is_better": True,
        "noise_fraction": args.noise_fraction,
        "bench_must_match": list(args.bench_must_match),
        "gate_must_match": list(args.gate_must_match),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name in ("manual.md", "ideas.md", "learnings.md", "experiment_log.md"):
        destination = run_dir / name
        if not destination.exists():
            shutil.copyfile(_TEMPLATES / name, destination)
    results = run_dir / "results.tsv"
    if not results.exists():
        results.write_text(RESULTS_HEADER, encoding="utf-8")
    print(f"initialized autoresearch run at {run_dir}")
    print("next: edit manual.md [SETUP], seed ideas.md, then run `baseline`")
    return 0


def _load(args: argparse.Namespace):
    """Load the run configuration for a subcommand."""
    run_dir = Path(args.run_dir).resolve()
    return load_run_config(run_dir, _repo_root(run_dir))


def command_baseline(args: argparse.Namespace) -> int:
    """Benchmark the current commit and record the first keep row."""
    outcome = record_baseline(_load(args), args.description)
    print(f"{outcome.status}: metric={outcome.metric} ({outcome.detail})")
    return 0 if outcome.kept else 1


def command_iterate(args: argparse.Namespace) -> int:
    """Run one experiment iteration on the pending target-file edit."""
    outcome = iterate(_load(args), args.description)
    print(f"{outcome.status}: metric={outcome.metric} best_before={outcome.best_before}")
    print(f"experiment_commit={outcome.experiment_commit} ({outcome.detail})")
    return 0 if outcome.status != "crash" else 1


def command_status(args: argparse.Namespace) -> int:
    """Print a short summary of the run."""
    print(run_status(_load(args), tail=args.tail))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python .agent/skills/autoresearch/scripts/autoresearch.py",
        description="Bounded autonomous-experiment loop (benchmark + gate + bookkeeping).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="scaffold a run directory")
    init.add_argument("--run-dir", required=True)
    init.add_argument("--target", action="append", required=True,
                      help="repo-relative file the agent may modify (repeatable)")
    init.add_argument("--benchmark-cmd", required=True)
    init.add_argument("--gate-cmd", default="")
    init.add_argument("--metric-pattern", default=r"total_ms=(?P<metric>[0-9.]+)")
    init.add_argument("--noise-fraction", type=float, default=0.02)
    init.add_argument("--bench-must-match", action="append", default=[])
    init.add_argument("--gate-must-match", action="append", default=[])
    init.add_argument("--force", action="store_true")
    init.set_defaults(handler=command_init)

    baseline = commands.add_parser("baseline", help="record the starting metric")
    baseline.add_argument("--run-dir", required=True)
    baseline.add_argument("--description", default="baseline")
    baseline.set_defaults(handler=command_baseline)

    iteration = commands.add_parser("iterate", help="measure the pending edit")
    iteration.add_argument("--run-dir", required=True)
    iteration.add_argument("--description", required=True)
    iteration.set_defaults(handler=command_iterate)

    status = commands.add_parser("status", help="summarize the run")
    status.add_argument("--run-dir", required=True)
    status.add_argument("--tail", type=int, default=5)
    status.set_defaults(handler=command_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
