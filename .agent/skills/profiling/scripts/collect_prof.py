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
"""Drive a run under the Ascend profiler and gather the output set.

Public-toolchain skeleton: it launches the provided run command with the
profiler enabled (MindStudio msprof wrapping the entry, or the in-run
``torch_npu.profiler`` when the recipe already sets it up) and reports where
the profiling output landed. Warmup is skipped and a few steady-state steps
are profiled. Profiler-config specifics an internal setup may require are
marked ``[待对齐]``.

The profiler is invoked as an external tool, never imported in-process, so
this script loads and lint-checks without it installed.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def _msprof_command(launch: list[str], out_dir: Path) -> list[str]:
    """Wrap the run command with ``msprof`` collection (argv list).

    The flag surface is the public msprof one; ``--output`` sets the result
    directory and ``--application`` carries the launched command. [待对齐]:
    aic-metrics / l2 / extra domains per the internal profiling setup.
    """
    return [
        "msprof", "--output", str(out_dir),
        "--application", " ".join(shlex.quote(part) for part in launch),
    ]


def main(argv: list[str] | None = None) -> int:
    """Launch the run under the profiler; report the output directory."""
    parser = argparse.ArgumentParser(
        prog="python .agent/skills/profiling/scripts/collect_prof.py",
        description="Drive a run under the Ascend profiler and gather its output.",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--launch", required=True,
                        help="run command (argv string, shlex-split)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="steps to skip before profiling (recorded, honored by the recipe)")
    parser.add_argument("--active", type=int, default=3,
                        help="steady-state steps to profile")
    parser.add_argument("--mode", choices=("msprof", "inrun"), default="msprof",
                        help="msprof wraps the entry; inrun assumes the recipe sets torch_npu.profiler")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    out_dir = run_dir / "prof"
    out_dir.mkdir(parents=True, exist_ok=True)
    launch = shlex.split(args.launch)

    if args.mode == "msprof":
        command = _msprof_command(launch, out_dir)
    else:
        # The recipe owns torch_npu.profiler; we only pass warmup/active hints
        # through the environment it reads, and run the command as given.
        command = launch
    print(f"[collect_prof] mode={args.mode} warmup={args.warmup} active={args.active}", flush=True)
    print(f"[collect_prof] {' '.join(command)}", flush=True)
    done = subprocess.run(command, cwd=str(run_dir), env=_prof_env(args), check=False)
    produced = any(out_dir.iterdir()) if out_dir.exists() else False
    status = "ok" if done.returncode == 0 and produced else "incomplete"
    print(f"[collect_prof] exit={done.returncode} out={out_dir} status={status}", flush=True)
    return 0 if status == "ok" else 1


def _prof_env(args: argparse.Namespace) -> dict:
    """Environment carrying the warmup/active hints for an in-run profiler."""
    import os  # pylint: disable=import-outside-toplevel

    env = dict(os.environ)
    env["PROF_WARMUP"] = str(args.warmup)
    env["PROF_ACTIVE"] = str(args.active)
    return env


if __name__ == "__main__":
    sys.exit(main())
