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
"""Drive a run with NPU memory-snapshot recording and verify what it produced.

The record and dump calls belong in the training process, which owns the
device, so this script passes the window and output path through the
environment and `mem_hook` is what reads them. Collection is judged by the
snapshot files actually being there with their counter sidecars, not by the
run's exit code alone.

Pure stdlib; the device-side snapshot API is never imported here.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

SIDECAR_SUFFIX = ".stats.json"


def _snapshot_env(args: argparse.Namespace, snapshot: Path) -> dict:
    """Environment carrying the window and output path that `mem_hook` reads."""
    env = dict(os.environ)
    env["MEM_SNAPSHOT_PATH"] = str(snapshot)
    env["MEM_SNAPSHOT_STEPS"] = args.steps
    env["MEM_SNAPSHOT_STACKS"] = args.stacks
    return env


def check_output(snapshot: Path) -> tuple[list[Path], list[str]]:
    """Find the dumps `mem_hook` wrote for this base path.

    `mem_hook.dump` tags each snapshot, so the files are siblings named
    ``<stem>.<tag><suffix>`` with a ``<name>.stats.json`` sidecar beside each.

    Args:
        snapshot: The base path handed to the run.

    Returns:
        ``(complete, problems)`` — dumps that have their counter sidecar, and a
        message per dump that is missing one.
    """
    found = sorted(snapshot.parent.glob(f"{snapshot.stem}.*{snapshot.suffix}"))
    if not found:
        return [], [f"no snapshot matching {snapshot.stem}.*{snapshot.suffix} "
                    f"in {snapshot.parent}; the entry must call mem_hook.start()/step()"]
    complete, problems = [], []
    for path in found:
        sidecar = path.with_suffix(path.suffix + SIDECAR_SUFFIX)
        if sidecar.is_file():
            complete.append(path)
        else:
            problems.append(f"{path.name}: missing {sidecar.name}; the peak and the "
                            f"allocator totals cannot be cross-checked without it")
    return complete, problems


def main(argv: list[str] | None = None) -> int:
    """Launch the run with snapshot recording; verify and report the dumps."""
    parser = argparse.ArgumentParser(
        prog="python .agent/skills/memory-analysis/scripts/collect_snapshot.py",
        description="Drive a run with NPU memory-snapshot recording.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--launch", required=True,
                        help="run command (argv string, shlex-split)")
    parser.add_argument("--steps", default="1-3",
                        help="inclusive step window to dump (read by mem_hook)")
    parser.add_argument("--stacks", default="all",
                        help="stack detail to record; 'all' keeps the Python frames "
                             "that attribution needs")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = run_dir / "memory_snapshot.pickle"
    launch = shlex.split(args.launch)

    print(f"[collect_snapshot] steps={args.steps} stacks={args.stacks} "
          f"base={snapshot}", flush=True)
    print(f"[collect_snapshot] {' '.join(launch)}", flush=True)
    done = subprocess.run(launch, cwd=str(run_dir),
                          env=_snapshot_env(args, snapshot), check=False)

    complete, problems = check_output(snapshot)
    for problem in problems:
        print(f"[collect_snapshot] incomplete: {problem}", flush=True)
    status = "ok" if done.returncode == 0 and complete else "incomplete"
    print(f"[collect_snapshot] exit={done.returncode} dumps={len(complete)} "
          f"status={status}", flush=True)
    for path in complete:
        print(f"[collect_snapshot] next: python "
              f"{Path(__file__).with_name('breakdown.py')} --snapshot {path}", flush=True)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
