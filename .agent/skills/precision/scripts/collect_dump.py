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
"""Drive one run under an msprobe dump config and verify the dump landed.

Runs on the launching host; it never imports ``msprobe`` or ``torch``. The
in-process end is ``precision_hook.py``, which the launched entry imports.

**Why a hook and not an environment variable.** msprobe reads exactly one
``MSPROBE_*`` variable (``MSPROBE_LOG_LEVEL``). There is no config-path
variable it picks up, so exporting one and launching a command instruments
nothing and produces an empty dump. The run must construct
``PrecisionDebugger`` itself; this script exports the ``PRECISION_DUMP_*``
variables that ``precision_hook`` reads, so the launch side and the run side
agree on one window.

**One dump per process.** ``PrecisionDebugger`` is a process-wide singleton,
so baseline and candidate are two invocations of this script, never two
collections inside one run.

Verification is per step, not per exit code: the configured steps must each
have produced a step directory holding ``dump.json``. A run that exits 0
having dumped nothing is reported as a failure, because a silently empty
dump is what a comparison then reads as "no divergence".
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

_LEVELS = ("L0", "L1", "mix")
_TASKS = ("statistics", "tensor")
_CONFIG_SUFFIX = ".config.json"
# msprobe rejects these outright in a dump path (FileCheckException on an
# "illegal path"), so a dump directory named after a config value can fail.
_PATH_FORBIDDEN = set("[]{}(),;:*?\"'<>|&$`!\n\t")


def parse_steps(spec: str) -> list[int]:
    """Expand ``"a-b"`` / ``"n"`` / ``"n,m"`` into an explicit step list.

    msprobe's ``step`` is a list of steps to dump, so the range has to be
    expanded here: a config carrying ``[0, 2]`` dumps steps 0 and 2 and
    skips step 1. Empty means every step.
    """
    text = (spec or "").strip()
    if not text:
        return []
    steps: list[int] = []
    for part in text.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        if "-" in chunk:
            first, _, last = chunk.partition("-")
            low, high = int(first), int(last)
            if low > high:
                raise ValueError(f"step range {chunk!r} runs backwards")
            steps.extend(range(low, high + 1))
        else:
            steps.append(int(chunk))
    return sorted(set(steps))


def check_dump_path(dump_path: Path) -> None:
    """Fail early on a path msprobe's own validator would reject."""
    bad = sorted(_PATH_FORBIDDEN.intersection(str(dump_path)))
    if bad:
        raise ValueError(
            f"dump path {dump_path} contains characters msprobe rejects: "
            f"{''.join(bad)!r}. Use a plain alphanumeric path.")


def build_config(dump_path: Path, steps: list[int], task: str, level: str,
                 ranks: list[int]) -> dict:
    """Return a config in msprobe's real schema (per-task nested options)."""
    config = {
        "task": task,
        "dump_path": str(dump_path),
        "rank": list(ranks),
        "step": list(steps),
        "level": level,
        task: {"scope": [], "list": [], "data_mode": ["all"]},
    }
    if task == "statistics":
        config[task]["summary_mode"] = "statistics"
    return config


def dump_env(config_path: Path, dump_path: Path, steps: list[int],
             task: str, level: str, ranks: list[int]) -> dict:
    """Environment for the launched run, carrying the window to the hook."""
    env = dict(os.environ)
    env["PRECISION_DUMP_CONFIG"] = str(config_path)
    env["PRECISION_DUMP_PATH"] = str(dump_path)
    env["PRECISION_DUMP_STEPS"] = ",".join(str(s) for s in steps)
    env["PRECISION_DUMP_TASK"] = task
    env["PRECISION_DUMP_LEVEL"] = level
    env["PRECISION_DUMP_RANKS"] = ",".join(str(r) for r in ranks)
    return env


def verify(dump_path: Path, steps: list[int]) -> tuple[list[str], list[str]]:
    """Return (present, missing) step directories that hold a ``dump.json``.

    With no steps configured msprobe dumps every step, so presence of at
    least one complete step directory is the whole check.
    """
    if not dump_path.exists():
        return [], [f"step{s}" for s in steps] or ["<any>"]
    landed = {p.name for p in dump_path.iterdir()
              if p.is_dir() and (p / "dump.json").is_file()}
    if not steps:
        return sorted(landed), [] if landed else ["<any>"]
    want = [f"step{s}" for s in steps]
    return ([w for w in want if w in landed],
            [w for w in want if w not in landed])


def main(argv: list[str] | None = None) -> int:
    """Write the config, launch the run, verify each configured step landed."""
    parser = argparse.ArgumentParser(
        prog="python .agent/skills/precision/scripts/collect_dump.py",
        description="Drive one run under an msprobe dump config and verify it.")
    parser.add_argument("--run-dir", required=True,
                        help="working directory for the run; dump lands under it")
    parser.add_argument("--launch", required=True,
                        help="run command (argv string, shlex-split)")
    parser.add_argument("--task", default="statistics", choices=_TASKS,
                        help="statistics (summaries) or tensor (full tensors; "
                             "the only mode whose comparison carries Cosine)")
    parser.add_argument("--level", default="L1", choices=_LEVELS)
    parser.add_argument("--step", default="0-1",
                        help='steps to dump, "a-b" / "n" / "n,m"; empty = all')
    parser.add_argument("--ranks", default="",
                        help="comma-separated ranks; empty = all")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_path = run_dir / "dump"
    check_dump_path(dump_path)
    steps = parse_steps(args.step)
    ranks = parse_steps(args.ranks)
    config = build_config(dump_path, steps, args.task, args.level, ranks)
    config_path = dump_path.parent / f"dump{_CONFIG_SUFFIX}"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[collect_dump] config={config_path} task={args.task} "
          f"level={args.level} steps={steps or 'all'}", flush=True)

    launch = shlex.split(args.launch)
    code = subprocess.run(
        launch, cwd=str(run_dir), check=False,
        env=dump_env(config_path, dump_path, steps, args.task, args.level, ranks),
    ).returncode

    present, missing = verify(dump_path, steps)
    print(f"[collect_dump] run exit={code} dump={dump_path}", flush=True)
    print(f"[collect_dump] steps present: {present or 'none'}", flush=True)
    if missing:
        print(f"[collect_dump] steps MISSING: {missing}", flush=True)
        print("[collect_dump] a run that exits 0 with no dump means the entry "
              "never constructed PrecisionDebugger - import precision_hook and "
              "call start() / capture() / step() inside it.", flush=True)
    if code != 0:
        return 1
    return 0 if present and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
