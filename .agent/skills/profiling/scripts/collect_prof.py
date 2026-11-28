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
"""Drive a run under the Ascend profiler and check that the output is complete.

Two collection modes:

* ``inrun`` (default) — the training script profiles itself through
  ``prof_hook.profile_steps``, which reads the ``PROF_*`` variables this script
  exports. This is the validated path: it controls warmup/active precisely and
  emits ``step_trace_time.csv``, which the breakdown needs.
* ``msprof`` — wrap the entry with MindStudio ``msprof``. ``--output`` and
  ``--application`` are its real flags (``msprof --help``, CANN 9.0.0), but the
  step schedule is then the application's own. ``[待对齐]``: aic-metrics / l2 /
  extra domains per the internal profiling setup.

Collection is judged by the profiler's own completion markers, not by the
output directory merely being non-empty: a complete set has an
``ASCEND_PROFILER_OUTPUT`` directory holding ``analyse.done`` and
``step_trace_time.csv``.

The profiler is invoked as an external tool and ``prof_hook`` is never imported
here, so this script loads and lint-checks without ``torch_npu`` installed.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

OUTPUT_SUBDIR = "ASCEND_PROFILER_OUTPUT"
# Written by the profiler only once parsing finished; `step_trace_time.csv` is
# the file the step breakdown is built from, and it needs Level1 or above.
REQUIRED = ("analyse.done", "step_trace_time.csv")


def _msprof_command(launch: list[str], out_dir: Path) -> list[str]:
    """Wrap the run command with ``msprof`` collection (argv list)."""
    return [
        "msprof", "--output", str(out_dir),
        "--application", " ".join(shlex.quote(part) for part in launch),
    ]


def _prof_env(args: argparse.Namespace, out_dir: Path) -> dict:
    """Environment carrying the schedule that ``prof_hook`` reads."""
    env = dict(os.environ)
    env["PROF_OUT"] = str(out_dir)
    env["PROF_WARMUP"] = str(args.warmup)
    env["PROF_ACTIVE"] = str(args.active)
    env["PROF_LEVEL"] = args.level
    return env


def check_output(out_dir: Path) -> tuple[list[Path], list[str]]:
    """Find complete profiler output sets under ``out_dir``.

    Args:
        out_dir: Directory the run was told to write into.

    Returns:
        ``(complete, problems)`` — the output directories holding every required
        marker, and a message per directory that is missing one.
    """
    found = sorted(path for path in out_dir.rglob(OUTPUT_SUBDIR) if path.is_dir())
    if not found:
        return [], [f"no {OUTPUT_SUBDIR} under {out_dir}"]
    complete, problems = [], []
    for path in found:
        missing = [name for name in REQUIRED if not (path / name).exists()]
        if missing:
            problems.append(f"{path}: missing {', '.join(missing)}")
        else:
            complete.append(path)
    return complete, problems


def main(argv: list[str] | None = None) -> int:
    """Launch the run under the profiler; verify and report the output."""
    parser = argparse.ArgumentParser(
        prog="python .agent/skills/profiling/scripts/collect_prof.py",
        description="Drive a run under the Ascend profiler and verify its output.",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--launch", required=True,
                        help="run command (argv string, shlex-split)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="steps to skip before profiling (read by prof_hook)")
    parser.add_argument("--active", type=int, default=3,
                        help="steady-state steps to profile (read by prof_hook)")
    parser.add_argument("--level", default="Level1",
                        help="profiler level; below Level1 there is no step_trace_time.csv")
    parser.add_argument("--mode", choices=("inrun", "msprof"), default="inrun",
                        help="inrun: the script uses prof_hook; msprof: wrap the entry")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    out_dir = run_dir / "prof"
    out_dir.mkdir(parents=True, exist_ok=True)
    launch = shlex.split(args.launch)
    command = _msprof_command(launch, out_dir) if args.mode == "msprof" else launch

    print(f"[collect_prof] mode={args.mode} warmup={args.warmup} "
          f"active={args.active} level={args.level}", flush=True)
    print(f"[collect_prof] {' '.join(command)}", flush=True)
    done = subprocess.run(command, cwd=str(run_dir), env=_prof_env(args, out_dir), check=False)

    complete, problems = check_output(out_dir)
    for problem in problems:
        print(f"[collect_prof] incomplete: {problem}", flush=True)
    status = "ok" if done.returncode == 0 and complete else "incomplete"
    print(f"[collect_prof] exit={done.returncode} out={out_dir} "
          f"complete={len(complete)} status={status}", flush=True)
    if complete:
        print(f"[collect_prof] next: python {Path(__file__).with_name('breakdown.py')} "
              f"--prof-dir {out_dir}", flush=True)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
