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
"""In-run msprobe hook: instrument the training process and dump chosen steps.

msprobe has **no environment-variable entry point** - the installed package
reads exactly one ``MSPROBE_*`` variable (``MSPROBE_LOG_LEVEL``) and nothing
else. Instrumentation happens only when ``PrecisionDebugger`` is constructed
*inside* the training process, so a collector that merely exports a config
path and launches a command dumps nothing at all. This module is that
in-process end, mirroring ``profiling/scripts/prof_hook.py`` and
``memory-analysis/scripts/mem_hook.py``::

    import sys
    sys.path.insert(0, ".agent/skills/precision/scripts")
    import precision_hook

    precision_hook.start()                  # reads PRECISION_DUMP_* env vars
    for step, batch in enumerate(loader):
        with precision_hook.capture():      # start()/stop() around the step
            train_step(batch)
        precision_hook.step()

``collect_dump.py`` exports ``PRECISION_DUMP_PATH`` / ``PRECISION_DUMP_STEPS``
/ ``PRECISION_DUMP_TASK`` / ``PRECISION_DUMP_LEVEL`` / ``PRECISION_DUMP_RANKS``;
this module is what reads them, so the launch side and the run side agree on
one window.

**One dump per process.** ``PrecisionDebugger`` is a process-wide singleton:
constructing a second one is silently a no-op and keeps the first config, so
a baseline and a candidate collected in the same process yield one dump and
an empty second directory. ``start()`` raises instead of letting that pass.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

DEFAULT_PATH = "precision_dump"
DEFAULT_STEPS = "0-1"
DEFAULT_TASK = "statistics"
DEFAULT_LEVEL = "L1"
TASKS = ("statistics", "tensor")

_state: dict = {"debugger": None, "step": 0, "dump": None, "config": None}


def parse_steps(spec: str) -> list[int]:
    """Expand ``"a-b"`` or ``"n"`` or ``"n,m"`` into an explicit step list.

    msprobe's ``step`` is a **list of steps to dump**, not a range: a config
    carrying ``[0, 2]`` dumps step 0 and step 2 and silently skips step 1.
    It does also accept the string form ``["0-2"]`` and expands that itself,
    but an explicit list is what the written config can be checked against,
    so this returns integers. An empty spec means every step.
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


def build_config(dump_path: Path, steps: list[int], task: str, level: str,
                 ranks: list[int] | None = None) -> dict:
    """Return a config in msprobe's real schema.

    Verified against the installed package's own ``config.json``: ``task``,
    ``dump_path``, ``rank``, ``step`` and ``level`` are top-level, and the
    per-task options (``scope`` / ``list`` / ``data_mode`` / ``summary_mode``)
    live in a block **named after the task**. A top-level ``scope`` is not
    part of the schema and is ignored.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    config = {
        "task": task,
        "dump_path": str(dump_path),
        "rank": list(ranks or []),
        "step": list(steps),
        "level": level,
        task: {"scope": [], "list": [], "data_mode": ["all"]},
    }
    if task == "statistics":
        config[task]["summary_mode"] = "statistics"
    return config


def _env(name: str, fallback: str) -> str:
    """Environment value, falling back when unset or empty."""
    return os.environ.get(name) or fallback


def start(path: str | None = None, steps: str | None = None,
          task: str | None = None, level: str | None = None,
          ranks: str | None = None) -> Any:
    """Write the dump config and construct the one debugger for this process.

    Args:
        path: Dump directory; defaults to ``$PRECISION_DUMP_PATH`` or
            ``./precision_dump``. msprobe validates this path strictly -
            brackets, commas and other special characters are rejected
            outright, so a directory named after a config value can fail here.
        steps: Steps to dump, ``"a-b"`` / ``"n"`` / ``"n,m"``; defaults to
            ``$PRECISION_DUMP_STEPS`` or ``0-1``. Empty means every step.
        task: ``statistics`` (per-tensor summaries) or ``tensor`` (full
            tensors; the only mode whose comparison carries ``Cosine``).
        level: msprobe dump level, ``L0`` module / ``L1`` API / ``mix``.
        ranks: Comma-separated ranks; empty means every rank.

    Returns:
        The ``PrecisionDebugger`` instance, so a caller can drive it directly.

    Raises:
        RuntimeError: if called twice in one process (see module docstring).
    """
    if _state["debugger"] is not None:
        raise RuntimeError(
            "precision_hook.start() was already called in this process. "
            "PrecisionDebugger is a process-wide singleton, so a second dump "
            "config is ignored and its dump directory stays empty - collect "
            "the baseline and the candidate in separate processes.")
    dump_path = Path(path or _env("PRECISION_DUMP_PATH", DEFAULT_PATH)).resolve()
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    config = build_config(
        dump_path,
        parse_steps(steps if steps is not None
                    else _env("PRECISION_DUMP_STEPS", DEFAULT_STEPS)),
        task or _env("PRECISION_DUMP_TASK", DEFAULT_TASK),
        level or _env("PRECISION_DUMP_LEVEL", DEFAULT_LEVEL),
        parse_steps(ranks if ranks is not None
                    else _env("PRECISION_DUMP_RANKS", "")),
    )
    config_path = dump_path.parent / f"{dump_path.name}.config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    from msprobe.pytorch import PrecisionDebugger  # pylint: disable=import-outside-toplevel

    _state["debugger"] = PrecisionDebugger(config_path=str(config_path))
    _state["dump"] = dump_path
    _state["config"] = config_path
    _state["step"] = 0
    print(f"[precision_hook] task={config['task']} level={config['level']} "
          f"steps={config['step'] or 'all'} dump={dump_path}", flush=True)
    return _state["debugger"]


@contextmanager
def capture() -> Iterator[Any]:
    """Enable dumping for the enclosed step body.

    msprobe records only between ``start()`` and ``stop()``; without the
    ``stop()`` the step's ``dump.json`` is never finalised.
    """
    debugger = _state["debugger"]
    if debugger is None:
        raise RuntimeError("precision_hook.start() has not been called")
    debugger.start()
    try:
        yield debugger
    finally:
        debugger.stop()


def step() -> None:
    """Advance msprobe's step counter; dumping follows the configured list."""
    debugger = _state["debugger"]
    if debugger is None:
        raise RuntimeError("precision_hook.start() has not been called")
    debugger.step()
    _state["step"] += 1


def dumped_steps() -> list[str]:
    """Step directories that actually landed, for the collector to check."""
    dump = _state["dump"]
    if dump is None or not dump.exists():
        return []
    return sorted(p.name for p in dump.iterdir() if p.is_dir())
