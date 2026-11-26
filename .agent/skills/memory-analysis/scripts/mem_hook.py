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
"""In-run memory-snapshot hook: record history and dump at chosen moments.

Imported by the training script, inside the training process, so ``torch_npu``
is imported at module top as elsewhere in this repository. It is never imported
by ``collect_snapshot.py``, which runs on the launching host.

``collect_snapshot.py`` exports ``MEM_SNAPSHOT_PATH`` / ``MEM_SNAPSHOT_STEPS``
/ ``MEM_SNAPSHOT_STACKS``; this module is what reads them, so the launch side
and the run side agree on one window::

    import sys
    sys.path.insert(0, ".agent/skills/memory-analysis/scripts")
    import mem_hook

    mem_hook.start()                       # enable history recording
    for step, batch in enumerate(loader):
        train_step(batch)
        mem_hook.step()                    # dumps inside the configured window

``dump(tag)`` can be called from anywhere at any time — including from an
autograd hook mid-backward — so a run that is going badly can be inspected
while it is still running instead of after it finishes.

Each dump writes two files: the snapshot pickle, and a ``.stats.json`` sidecar
holding ``memory_reserved`` / ``memory_allocated`` / ``max_memory_allocated``.
Those counters live only in the process, and ``breakdown.py`` needs them to
cross-check its totals and to say whether the snapshot is the peak — a
post-step snapshot holds no activations and is normally well below the peak.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch_npu

DEFAULT_PATH = "memory_snapshot.pickle"
DEFAULT_STEPS = "1-3"

_state: dict = {"step": 0, "window": None, "base": None, "dumps": []}


def _parse_window(spec: str) -> tuple[int, int]:
    """Parse ``"a-b"`` or ``"n"`` into an inclusive step window."""
    text = (spec or "").strip()
    if "-" in text:
        first, _, last = text.partition("-")
        return int(first), int(last)
    value = int(text)
    return value, value


def start(path: str | None = None, steps: str | None = None,
          stacks: str | None = None) -> None:
    """Begin recording allocation history.

    Args:
        path: Snapshot path; defaults to ``$MEM_SNAPSHOT_PATH`` or
            ``./memory_snapshot.pickle``. Each dump adds its tag to the stem.
        steps: Inclusive step window for automatic dumps, ``"a-b"`` or ``"n"``;
            defaults to ``$MEM_SNAPSHOT_STEPS`` or ``1-3``.
        stacks: Passed through to the recorder; defaults to
            ``$MEM_SNAPSHOT_STACKS`` or ``all``. Python frames are what makes
            attribution possible, so lowering this makes the report coarser.
    """
    base = Path(path or os.environ.get("MEM_SNAPSHOT_PATH") or DEFAULT_PATH).resolve()
    base.parent.mkdir(parents=True, exist_ok=True)
    _state["base"] = base
    _state["step"] = 0
    _state["dumps"] = []
    _state["window"] = _parse_window(steps or os.environ.get("MEM_SNAPSHOT_STEPS")
                                     or DEFAULT_STEPS)
    torch_npu.npu.memory._record_memory_history(  # pylint: disable=protected-access
        enabled="all", context="all",
        stacks=stacks or os.environ.get("MEM_SNAPSHOT_STACKS") or "all")


def stop() -> None:
    """Stop recording allocation history."""
    torch_npu.npu.memory._record_memory_history(enabled=None)  # pylint: disable=protected-access


def _counters(device: int) -> dict:
    """The allocator counters that the snapshot itself does not carry."""
    return {
        "device": device,
        "memory_reserved": int(torch.npu.memory_reserved(device)),
        "memory_allocated": int(torch.npu.memory_allocated(device)),
        "max_memory_allocated": int(torch.npu.max_memory_allocated(device)),
        "max_memory_reserved": int(torch.npu.max_memory_reserved(device)),
    }


def dump(tag: str = "snapshot", device: int | None = None) -> Path:
    """Write a snapshot and its counter sidecar, and return the snapshot path.

    Safe to call at any moment, including from an autograd hook while the graph
    is alive, which is the only way to see activations in the composition.

    Args:
        tag: Appended to the snapshot stem to keep dumps apart.
        device: Device whose counters to record; defaults to the current one.

    Returns:
        The snapshot file path.
    """
    base = _state["base"] or Path(
        os.environ.get("MEM_SNAPSHOT_PATH") or DEFAULT_PATH).resolve()
    target = base.with_name(f"{base.stem}.{tag}{base.suffix}")
    target.parent.mkdir(parents=True, exist_ok=True)
    torch_npu.npu.memory._dump_snapshot(str(target))  # pylint: disable=protected-access
    # `_dump_snapshot` creates the file write-only (mode 0200), so not even its
    # owner can read it back. Restore owner read/write; the snapshot records
    # source paths, so it stays private to the owner.
    target.chmod(0o600)
    index = torch.npu.current_device() if device is None else device
    sidecar = target.with_suffix(target.suffix + ".stats.json")
    sidecar.write_text(json.dumps({"tag": tag, **_counters(index)}, indent=2),
                       encoding="utf-8")
    _state["dumps"].append(target)
    print(f"[mem_hook] dumped {target} (+{sidecar.name})", flush=True)
    return target


def step() -> None:
    """Advance the step counter and dump when inside the configured window."""
    _state["step"] += 1
    current = _state["step"]
    window = _state["window"] or _parse_window(DEFAULT_STEPS)
    if window[0] <= current <= window[1]:
        dump(f"step{current}")


def dumps() -> list[Path]:
    """Snapshot paths written so far in this process."""
    return list(_state["dumps"])
