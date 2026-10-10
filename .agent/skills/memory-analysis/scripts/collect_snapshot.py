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
"""Drive a run with NPU memory-snapshot recording and gather the snapshot.

Public-toolchain skeleton: it launches the provided run command with snapshot
recording requested over the first few steps and reports where the snapshot
file landed. The actual record/dump calls live in the training entry (it owns
the device); this script passes the window and output path through the
environment the entry reads. Recording specifics an internal setup may need
are marked ``[待对齐]``.

Pure stdlib; the device-side snapshot API is never imported here.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Launch the run with snapshot recording; report the snapshot path."""
    parser = argparse.ArgumentParser(
        prog="python .agent/skills/memory-analysis/scripts/collect_snapshot.py",
        description="Drive a run with NPU memory-snapshot recording.",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--launch", required=True,
                        help="run command (argv string, shlex-split)")
    parser.add_argument("--steps", default="1-3",
                        help="step window to record (the entry honors MEM_SNAPSHOT_STEPS)")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = run_dir / "memory_snapshot.pickle"
    launch = shlex.split(args.launch)

    print(f"[collect_snapshot] steps={args.steps} snapshot={snapshot}", flush=True)
    print(f"[collect_snapshot] {' '.join(launch)}", flush=True)
    done = subprocess.run(
        launch, cwd=str(run_dir),
        env=_snapshot_env(args.steps, snapshot), check=False)
    status = "ok" if done.returncode == 0 and snapshot.exists() else "incomplete"
    print(f"[collect_snapshot] exit={done.returncode} status={status}", flush=True)
    if status != "ok" and not snapshot.exists():
        print("[collect_snapshot] note: the training entry must dump the snapshot to "
              "MEM_SNAPSHOT_PATH over MEM_SNAPSHOT_STEPS; wire it if absent [待对齐]",
              flush=True)
    return 0 if status == "ok" else 1


def _snapshot_env(steps: str, snapshot: Path) -> dict:
    """Environment carrying the snapshot window and output path for the entry."""
    import os  # pylint: disable=import-outside-toplevel

    env = dict(os.environ)
    env["MEM_SNAPSHOT_STEPS"] = steps
    env["MEM_SNAPSHOT_PATH"] = str(snapshot)
    return env


if __name__ == "__main__":
    sys.exit(main())
