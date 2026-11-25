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

``collect_dump.py`` exports ``PRECISION_DUMP_CONFIG``. This hook reads that
file without rewriting it. Without a supplied file, explicit arguments and
the remaining ``PRECISION_DUMP_*`` variables configure a new private file.

**One dump per process.** ``PrecisionDebugger`` is a process-wide singleton:
constructing a second one is silently a no-op and keeps the first config, so
a baseline and a candidate collected in the same process yield one dump and
an empty second directory. ``start()`` raises instead of letting that pass.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from _precision_config import (
    DEFAULT_LEVEL, DEFAULT_PATH, DEFAULT_STEPS, DEFAULT_TASK,
    build_config, parse_steps, read_config, write_config,
)

_state: dict = {"debugger": None, "step": 0, "dump": None, "config": None}


def _check_overrides(config: dict, path: str | None, steps: str | None,
                     task: str | None, level: str | None, ranks: str | None) -> None:
    """Reject explicit arguments that disagree with a supplied config file."""
    supplied = {
        "dump_path": str(Path(path).resolve()) if path is not None else None,
        "step": parse_steps(steps) if steps is not None else None,
        "task": task, "level": level,
        "rank": parse_steps(ranks) if ranks is not None else None,
    }
    for key, value in supplied.items():
        expected = config[key]
        if key in ("step", "rank"):
            expected = sorted(set(expected))
        elif key == "dump_path":
            expected = str(Path(expected).resolve())
        if value is not None and value != expected:
            raise ValueError(f"explicit {key}={value!r} conflicts with precision config value {expected!r}")


def start(path: str | None = None, steps: str | None = None,
          task: str | None = None, level: str | None = None,
          ranks: str | None = None, *, config_path: str | None = None) -> Any:
    """Use a supplied config or generate one, then construct the debugger.

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
        config_path: Configuration file, otherwise ``$PRECISION_DUMP_CONFIG``.
            When supplied, the file is authoritative; explicit arguments must
            agree with it, and other environment settings are ignored. The file
            must be readable by every worker and use an absolute dump path.

    Returns:
        The ``PrecisionDebugger`` instance, so a caller can drive it directly.

    Raises:
        RuntimeError: if called twice in one process (see module docstring).
        ValueError: if configuration is invalid or explicit arguments conflict.
        OSError: if the supplied configuration cannot be read.
    """
    if _state["debugger"] is not None:
        raise RuntimeError(
            "precision_hook.start() was already called in this process. "
            "PrecisionDebugger is a process-wide singleton, so a second dump "
            "config is ignored and its dump directory stays empty - collect "
            "the baseline and the candidate in separate processes.")
    source = config_path if config_path is not None else os.environ.get("PRECISION_DUMP_CONFIG")
    if source is not None:
        if not source.strip():
            raise ValueError("precision config path must not be empty")
        selected_path = Path(source).resolve()
        config = read_config(selected_path)
        _check_overrides(config, path, steps, task, level, ranks)
    else:
        config = build_config(
            Path(path if path is not None else os.environ.get("PRECISION_DUMP_PATH", DEFAULT_PATH)),
            parse_steps(steps if steps is not None else os.environ.get("PRECISION_DUMP_STEPS", DEFAULT_STEPS)),
            task if task is not None else os.environ.get("PRECISION_DUMP_TASK", DEFAULT_TASK),
            level if level is not None else os.environ.get("PRECISION_DUMP_LEVEL", DEFAULT_LEVEL),
            parse_steps(ranks if ranks is not None else os.environ.get("PRECISION_DUMP_RANKS", "")),
        )
        selected_path = write_config(config)
    dump_path = Path(config["dump_path"])

    # msprobe is optional; import only when instrumentation is requested.
    from msprobe.pytorch import PrecisionDebugger  # pylint: disable=import-outside-toplevel

    _state["debugger"] = PrecisionDebugger(config_path=str(selected_path))
    _state["dump"] = dump_path
    _state["config"] = selected_path
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
