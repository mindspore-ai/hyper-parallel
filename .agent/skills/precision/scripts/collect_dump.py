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
"""Collect a precision dump by driving a run under an msprobe dump config.

This is the public-toolchain skeleton: it writes an msprobe dump config,
launches the provided run command with that config active, and reports where
the dump landed. The exact auto-instrumentation strategy and thresholds that
an internal precision-align implementation uses are marked ``[待对齐]`` and
left for the internal logic; the standard msprobe flow works as-is.

msprobe is invoked as an external tool (it is part of the MindStudio
accuracy toolchain); it is never imported in-process, so this script loads
and lint-checks without the tool installed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# msprobe dump levels: L0 module-level, L1 API-level, "mix" for both.
_LEVELS = ("L0", "L1", "mix")
_CONFIG_NAME = "msprobe_dump_config.json"


def _write_dump_config(run_dir: Path, level: str, step: str) -> Path:
    """Write a minimal msprobe dump config into the run directory.

    The schema here is the public msprobe ``task: statistics`` shape. The
    internal align implementation may force extra module scopes or a tighter
    tolerance - that delta is ``[待对齐]`` and belongs in the config this
    function emits once the internal logic is available.
    """
    config = {
        "task": "statistics",
        "dump_path": str(run_dir / "dump"),
        "level": level,
        "step": _parse_step_range(step),
        # [待对齐] scope / tensor-vs-statistics / tolerance per internal logic.
        "scope": [],
    }
    path = run_dir / _CONFIG_NAME
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def _parse_step_range(step: str) -> list[int]:
    """Turn ``"0-2"`` or ``"3"`` into an explicit ``[start, end]`` list."""
    if "-" in step:
        start, end = step.split("-", 1)
        return [int(start), int(end)]
    value = int(step)
    return [value, value]


def _launch_under_dump(launch: list[str], config_path: Path, run_dir: Path) -> int:
    """Run the command with the dump config exported; return its exit code.

    The run command is an argv list by contract (shell constructs live in the
    launched script, never here). ``MSPROBE_CONFIG_PATH`` is the public env
    hook msprobe reads; the launched training/inference entry picks it up.
    """
    env_line = f"MSPROBE_CONFIG_PATH={config_path}"
    print(f"[collect_dump] launching under {env_line}", flush=True)
    done = subprocess.run(
        list(launch), cwd=str(run_dir),
        env=_dump_env(config_path), check=False,
    )
    return done.returncode


def _dump_env(config_path: Path) -> dict:
    """Environment for the launched run with the msprobe config exported."""
    import os  # pylint: disable=import-outside-toplevel

    env = dict(os.environ)
    env["MSPROBE_CONFIG_PATH"] = str(config_path)
    return env


def main(argv: list[str] | None = None) -> int:
    """Write the dump config, launch the run, report the dump location."""
    parser = argparse.ArgumentParser(
        prog="python .agent/skills/precision/scripts/collect_dump.py",
        description="Drive a run under an msprobe dump config and gather the dump.",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--launch", required=True,
                        help="run command (argv string, shlex-split)")
    parser.add_argument("--level", default="L1", choices=_LEVELS)
    parser.add_argument("--step", default="0-2")
    args = parser.parse_args(argv)

    import shlex  # pylint: disable=import-outside-toplevel

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = _write_dump_config(run_dir, args.level, args.step)
    code = _launch_under_dump(shlex.split(args.launch), config_path, run_dir)
    dump_dir = run_dir / "dump"
    status = "ok" if code == 0 and dump_dir.exists() else "incomplete"
    print(f"[collect_dump] run exit={code} dump={dump_dir} status={status}", flush=True)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
