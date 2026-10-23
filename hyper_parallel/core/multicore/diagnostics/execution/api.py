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
"""Lightweight Host execution reports sharing the existing diagnostic session boundary."""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch

from hyper_parallel.core.multicore.diagnostics.common.call import DiagnosticCall, LaunchContext
from hyper_parallel.core.multicore.diagnostics.common.schedule import DiagnosticAction
from hyper_parallel.core.multicore.diagnostics.common.session import _activate_session, _deactivate_session
from hyper_parallel.core.multicore.diagnostics.execution.inspection import (
    _execution_modules, _loaded_native_libraries, inspect_execution,
)


class ExecutionMonitor:
    """Observe selected steps without changing native launch arguments or tensor values."""

    def __init__(
        self, model: torch.nn.Module, *, output_dir: str | Path,
        schedule: Callable[[int], DiagnosticAction] | None = None, optimizer: Any = None,
        max_events_per_step: int = 10000,
    ) -> None:
        """Configure a session without activating it or creating output files.

        Args:
            model: Model whose executor states are inspected at sampled step boundaries.
            output_dir: Root for independent per-process session directories.
            schedule: Existing zero-based diagnostic schedule, or None to record every step.
            optimizer: Optional optimizer whose resolved options enter the identity report.
            max_events_per_step: Bound on retained Host records; excess records are counted.
        """
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if schedule is not None and not callable(schedule):
            raise TypeError("schedule must be callable or None")
        if (not isinstance(max_events_per_step, int) or isinstance(max_events_per_step, bool)
                or max_events_per_step < 1):
            raise ValueError("max_events_per_step must be a positive integer")
        self._model: torch.nn.Module | None = model
        self._optimizer = optimizer
        self.output_dir = Path(output_dir).expanduser()
        self.schedule = schedule
        self.max_events_per_step = max_events_per_step
        self.step_num = 0
        self.directory: Path | None = None
        self._entered = False
        self._action = DiagnosticAction.NONE
        self._events: list[dict[str, Any]] = []
        self._dropped = 0
        self._completed_steps: list[int] = []

    def _select_action(self) -> DiagnosticAction:
        action = self.schedule(self.step_num) if self.schedule is not None else DiagnosticAction.RECORD
        if not isinstance(action, DiagnosticAction):
            raise TypeError("schedule must return DiagnosticAction")
        return action

    def __enter__(self) -> ExecutionMonitor:
        """Claim the common session and publish a read-only identity snapshot."""
        if self._entered or self.directory is not None:
            raise RuntimeError("an execution monitor can be entered only once")
        _activate_session("execution", self)
        try:
            self._action = self._select_action()
            identity = inspect_execution(self._model, optimizer=self._optimizer)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.directory = Path(tempfile.mkdtemp(
                prefix=f"rank_{identity['rank']:04d}_pid_{os.getpid()}_", dir=self.output_dir))
            self._write("identity.json", identity)
            self._entered = True
        except BaseException:
            _deactivate_session(self)
            raise
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Release observation even when the training body or report output fails."""
        try:
            self._write("session.json", {
                "completed_steps": self._completed_steps,
                "status": "failed" if exc_type is not None else "closed",
                "unfinished_step": self.step_num if self._events else None,
                "interpretation": "Only step() marks a complete training step; pending events are not accepted.",
            })
        finally:
            self._entered = False
            self._events.clear()
            self._model = None
            self._optimizer = None
            _deactivate_session(self)

    def _write(self, name: str, value: dict[str, Any]) -> None:
        if self.directory is None:
            raise RuntimeError("execution monitor has no output directory")
        path = self.directory / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def is_capture_enabled(self) -> bool:
        """Retain Host metadata only during explicitly recorded steps."""
        return self._entered and self._action in (DiagnosticAction.RECORD, DiagnosticAction.RECORD_AND_SAVE)

    def prepare_call(self, context: LaunchContext) -> DiagnosticCall:
        """Leave native arguments unchanged without requesting graph metadata.

        Args:
            context: Normal kernel arguments supplied by the common diagnostic adapter.
        """
        return DiagnosticCall(context.ordinary_arguments())

    def record_event(self, kind: str, metadata: dict[str, Any]) -> None:
        """Retain existing Host metadata without copying device tensors.

        Args:
            kind: Producer event category.
            metadata: Already available Host values describing the event.
        """
        if not self.is_capture_enabled():
            return
        if len(self._events) < self.max_events_per_step:
            self._events.append({"kind": kind, **metadata})
        else:
            self._dropped += 1

    def step(self) -> None:
        """Mark one complete training step, publish its records and advance the schedule."""
        if not self._entered:
            raise RuntimeError("step() requires an entered execution monitor")
        if self.is_capture_enabled():
            self._write(f"step_{self.step_num:06d}.json", {
                "schema_version": 1, "step": self.step_num, "events": self._events,
                "dropped_events": self._dropped,
                "execution_modules": _execution_modules(self._model),
                "native_libraries": _loaded_native_libraries(),
                "scope": "Host metadata only; build timings nest inside forward/backward and are not step latency.",
            })
            self._completed_steps.append(self.step_num)
        self._events = []
        self._dropped = 0
        self.step_num += 1
        self._action = self._select_action()


def execution_monitor(
    model: torch.nn.Module, *, output_dir: str | Path,
    schedule: Callable[[int], DiagnosticAction] | None = None, optimizer: Any = None,
    max_events_per_step: int = 10000,
) -> ExecutionMonitor:
    """Create a scoped Host observer using the shared profiler/dump schedule contract.

    Call step() after each complete optimizer step. All relevant ranks must enter
    matching training calls; this observer adds no collective or synchronization.

    Args:
        model: Model providing execution-state metadata.
        output_dir: Root directory for independent per-process reports.
        schedule: Zero-based schedule shared with the other diagnostic tools.
        optimizer: Optional optimizer for startup metadata.
        max_events_per_step: Maximum retained Host records before reporting truncation.

    Returns:
        Context manager advanced by one step() call per complete training step.
    """
    return ExecutionMonitor(model, output_dir=output_dir, schedule=schedule, optimizer=optimizer,
                            max_events_per_step=max_events_per_step)
