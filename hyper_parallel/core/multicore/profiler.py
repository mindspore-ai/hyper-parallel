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
"""Standalone Torch MegaKernel profiling API and capture lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from threading import Lock
from typing import Any

import torch

from hyper_parallel.core.multicore.profiling import _PreparedMegaKernelRuntime
from hyper_parallel.core.multicore.scheduler.config import RuntimeConfigC
from hyper_parallel.core.multicore.trace_merger import merge_chrome_traces


__all__ = ["ProfilerAction", "mega_kernel_profile", "merge_chrome_traces", "schedule"]


class ProfilerAction(Enum):
    """Action selected for one user-visible training step."""

    NONE = 0
    WARMUP = 1
    RECORD = 2
    RECORD_AND_SAVE = 3


@dataclass(frozen=True)
class _ProfilerSchedule:
    """Callable schedule with Torch-profiler-compatible window arguments."""

    skip_first: int
    wait: int
    warmup: int
    active: int
    repeat: int

    def __call__(self, step: int) -> ProfilerAction:
        """Return the capture action for one zero-based training step."""
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError(f"step must be a non-negative integer, got {step!r}")
        if step < self.skip_first:
            return ProfilerAction.NONE

        relative_step = step - self.skip_first
        cycle_length = self.wait + self.warmup + self.active
        cycle_index, cycle_step = divmod(relative_step, cycle_length)
        if self.repeat and cycle_index >= self.repeat:
            return ProfilerAction.NONE
        if cycle_step < self.wait:
            return ProfilerAction.NONE
        if cycle_step < self.wait + self.warmup:
            return ProfilerAction.WARMUP
        if cycle_step == cycle_length - 1:
            return ProfilerAction.RECORD_AND_SAVE
        return ProfilerAction.RECORD


def schedule(
    *,
    wait: int,
    warmup: int,
    active: int,
    repeat: int = 0,
    skip_first: int = 0,
) -> Callable[[int], ProfilerAction]:
    """Create a repeatable MegaKernel capture schedule.

    Args:
        wait: Unrecorded steps between capture windows.
        warmup: Recorded-but-discarded steps before each active window.
        active: Steps retained in every active window.
        repeat: Number of cycles. Zero repeats indefinitely.
        skip_first: Initial unrecorded steps before the first cycle.

    Returns:
        Callable mapping a zero-based step number to ProfilerAction.
    """
    values = {
        "wait": wait,
        "warmup": warmup,
        "active": active,
        "repeat": repeat,
        "skip_first": skip_first,
    }
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    if active <= 0:
        raise ValueError(f"active must be positive, got {active}")
    return _ProfilerSchedule(
        skip_first=skip_first,
        wait=wait,
        warmup=warmup,
        active=active,
        repeat=repeat,
    )


_ACTIVE_LOCK = Lock()
_ACTIVE_PROFILER = None


@dataclass
class _RawInvocation:
    """One completed invocation copied from Device to Host."""

    runtime: _PreparedMegaKernelRuntime
    data: bytes
    direction: str
    step: int
    invocation_id: int


class _MegaKernelCall:
    """Internal launch arguments and profiling-buffer ownership for one kernel call."""

    def __init__(
        self,
        *,
        runtime_config: Any,
        event_counters: Any,
        profile_buffer: Any,
        clear_event_counters: Any | None = None,
        profiler: TorchMegaKernelProfiler | None = None,
        runtime: _PreparedMegaKernelRuntime | None = None,
        profile_buffer_key: tuple[int, int] | None = None,
        direction: str = "",
        step: int = 0,
        invocation_id: int = 0,
    ) -> None:
        """Store launch arguments and optional active-session bookkeeping."""
        self.runtime_config = runtime_config
        self.event_counters = event_counters
        self.profile_buffer = profile_buffer
        self.clear_event_counters = event_counters if clear_event_counters is None else clear_event_counters
        self._profiler = profiler
        self._runtime = runtime
        self._profile_buffer_key = profile_buffer_key
        self._direction = direction
        self._step = step
        self._invocation_id = invocation_id
        self._finished = False

    def complete(self) -> None:
        """Mark the direct-write profile buffer as complete after a successful launch."""
        if self._finished:
            return
        if self._profiler is not None:
            self._profiler._complete_call(self)
        self._finished = True

    def cancel(self) -> None:
        """Discard an unlaunched call and release its private profile buffer."""
        if self._finished:
            return
        if self._profiler is not None:
            self._profiler._cancel_call(self)
        self._finished = True


class TorchMegaKernelProfiler:
    """Standalone schedule-driven MegaKernel internal profiler for Torch."""

    def __init__(
        self,
        *,
        schedule: Callable[[int], ProfilerAction] | None,
        on_trace_ready: Callable[[Any], None] | None,
        detailed_task_names: bool,
        max_pending_calls: int,
    ) -> None:
        """Validate options and initialize the schedule-driven capture state."""
        if schedule is not None and not callable(schedule):
            raise TypeError("schedule must be callable or None")
        if on_trace_ready is not None and not callable(on_trace_ready):
            raise TypeError("on_trace_ready must be callable or None")
        if not isinstance(detailed_task_names, bool):
            raise TypeError("detailed_task_names must be bool")
        if not isinstance(max_pending_calls, int) or isinstance(max_pending_calls, bool) or max_pending_calls <= 0:
            raise ValueError("max_pending_calls must be a positive integer, " f"got {max_pending_calls!r}")
        self._schedule = schedule
        self._on_trace_ready = on_trace_ready
        self._detailed_task_names = detailed_task_names
        self._max_pending_calls = max_pending_calls
        self._step = 0
        self._action = self._action_for_step(0)
        self._entered = False
        self._next_invocation_id = 0
        self._pending: list[_MegaKernelCall] = []
        self._profile_buffer_pool: dict[tuple[int, int], list[Any]] = {}
        self._raw_invocations: list[_RawInvocation] = []
        self._last_trace: dict[str, Any] | None = None
        self._window_index = 0

    @property
    def step_num(self) -> int:
        """Return the zero-based step currently being recorded."""
        return self._step

    @property
    def current_action(self) -> ProfilerAction:
        """Return the current schedule action."""
        return self._action

    def __enter__(self) -> TorchMegaKernelProfiler:
        """Activate this profiler as the process-local MegaKernel session."""
        # pylint: disable=global-statement
        global _ACTIVE_PROFILER
        with _ACTIVE_LOCK:
            if _ACTIVE_PROFILER is not None:
                raise RuntimeError("nested MegaKernel profiler contexts are not supported")
            _ACTIVE_PROFILER = self
        self._entered = True
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Finalize successful captures or discard pending data on failure."""
        # pylint: disable=global-statement
        global _ACTIVE_PROFILER
        try:
            if exc_type is None:
                self._finish_context()
            else:
                self._drain(keep=False)
        finally:
            for call in self._pending:
                self._release_profile_buffer(call, cache=False)
            self._pending.clear()
            self._release_profile_buffer_pool()
            with _ACTIVE_LOCK:
                if _ACTIVE_PROFILER is self:
                    _ACTIVE_PROFILER = None
            self._entered = False

    def step(self) -> None:
        """Finish the current schedule step and advance to the next one."""
        if not self._entered:
            raise RuntimeError("profiler.step() must be called inside its context")
        next_action = self._action_for_step(self._step + 1)
        if self._action is ProfilerAction.WARMUP:
            self._drain(keep=False)
        elif self._action is ProfilerAction.RECORD_AND_SAVE:
            self._drain(keep=True)
            self._finalize_window()
        elif self._action is ProfilerAction.RECORD and next_action not in (
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
        ):
            self._drain(keep=True)
            self._finalize_window()
        self._step += 1
        self._action = next_action

    def export_chrome_trace(self, path: str | Path) -> dict[str, Any]:
        """Write the most recently completed internal capture window."""
        if self._last_trace is None:
            raise RuntimeError(
                "no completed MegaKernel capture window is available; "
                "call step() at the window boundary or leave the context first"
            )
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self._last_trace, ensure_ascii=False),
            encoding="utf-8",
        )
        return self._last_trace

    def _action_for_step(self, step: int) -> ProfilerAction:
        if self._schedule is None:
            return ProfilerAction.RECORD
        action = self._schedule(step)
        if not isinstance(action, ProfilerAction):
            raise TypeError("MegaKernel schedule must return ProfilerAction, " f"got {type(action).__name__}")
        return action

    def _capture_enabled(self) -> bool:
        return self._action is not ProfilerAction.NONE

    def _acquire_call(
        self,
        runtime: _PreparedMegaKernelRuntime,
        event_counters: Any,
        direction: str,
    ) -> _MegaKernelCall:
        if len(self._pending) >= self._max_pending_calls:
            self._drain(keep=self._action is not ProfilerAction.WARMUP)
        invocation_id = self._next_invocation_id
        profile_buffer, profile_buffer_key = self._take_profile_buffer(
            runtime,
            event_counters,
        )
        self._next_invocation_id += 1
        call = _MegaKernelCall(
            runtime_config=runtime.profile_tensor,
            event_counters=event_counters,
            clear_event_counters=event_counters,
            profiler=self,
            profile_buffer=profile_buffer,
            runtime=runtime,
            direction=direction,
            step=self._step,
            profile_buffer_key=profile_buffer_key,
            invocation_id=invocation_id,
        )
        self._pending.append(call)
        return call

    def _complete_call(self, call: _MegaKernelCall) -> None:
        if call not in self._pending:
            raise RuntimeError("completed MegaKernel profiling call is not pending")

    def _take_profile_buffer(
        self,
        runtime: _PreparedMegaKernelRuntime,
        event_counters: Any,
    ) -> tuple[Any, tuple[int, int]]:
        """Take one ordinary NPU buffer that is private to the next invocation."""
        profile_buffer_key = (runtime.device_id, runtime.buffer_size)
        available_buffers = self._profile_buffer_pool.get(profile_buffer_key)
        if available_buffers:
            profile_buffer = available_buffers.pop()
        else:
            profile_buffer = torch.empty(
                (runtime.buffer_size,),
                dtype=torch.uint8,
                device=event_counters.device,
            )
        profile_buffer.zero_()
        return profile_buffer, profile_buffer_key

    def _cancel_call(self, call: _MegaKernelCall) -> None:
        if call in self._pending:
            self._pending.remove(call)
        self._release_profile_buffer(call, cache=False)

    def _release_profile_buffer(
        self,
        call: _MegaKernelCall,
        *,
        cache: bool,
    ) -> None:
        """Return a completed buffer to the session pool or release its storage."""
        profile_buffer = call.profile_buffer
        profile_buffer_key = call._profile_buffer_key
        if profile_buffer is None or profile_buffer_key is None:
            return
        if cache:
            self._profile_buffer_pool.setdefault(profile_buffer_key, []).append(profile_buffer)
        else:
            profile_buffer.untyped_storage().resize_(0)
        call.profile_buffer = None
        call._profile_buffer_key = None

    def _release_profile_buffer_pool(self) -> None:
        """Release ordinary NPU buffers retained for reuse by this session."""
        for available_buffers in self._profile_buffer_pool.values():
            for profile_buffer in available_buffers:
                profile_buffer.untyped_storage().resize_(0)
        self._profile_buffer_pool.clear()

    def _drain(self, *, keep: bool) -> None:
        if not self._pending:
            if not keep:
                self._raw_invocations.clear()
            return
        unfinished = [call for call in self._pending if not call._finished]
        if unfinished:
            raise RuntimeError("cannot copy MegaKernel profiling buffers before launches complete")

        device_ids = {call._runtime.device_id for call in self._pending if call._runtime is not None}
        for device_id in device_ids:
            torch.npu.synchronize(device_id)

        for call in self._pending:
            runtime = call._runtime
            profile_buffer = call.profile_buffer
            if keep and runtime is not None and profile_buffer is not None:
                host_data = profile_buffer.cpu().numpy().tobytes()
                self._raw_invocations.append(
                    _RawInvocation(
                        runtime=runtime,
                        data=host_data,
                        direction=call._direction,
                        step=call._step,
                        invocation_id=call._invocation_id,
                    )
                )
            self._release_profile_buffer(call, cache=True)
        self._pending.clear()
        if not keep:
            self._raw_invocations.clear()

    def _finish_context(self) -> None:
        if self._action is ProfilerAction.WARMUP:
            self._drain(keep=False)
            return
        if self._pending or self._raw_invocations:
            self._drain(keep=True)
            if self._raw_invocations:
                self._finalize_window()

    def _finalize_window(self) -> None:
        self._last_trace = self._build_trace(self._raw_invocations)
        self._raw_invocations.clear()
        self._window_index += 1
        if self._on_trace_ready is not None:
            self._on_trace_ready(self)

    def _build_trace(self, invocations: list[_RawInvocation]) -> dict[str, Any]:
        if not invocations:
            return {
                "traceEvents": [],
                "megaKernelCycleTrace": {
                    "schemaVersion": 1,
                    "windowIndex": self._window_index,
                    "invocationCount": 0,
                    "recordCount": 0,
                    "droppedRecordCount": 0,
                    "detailedTaskNames": self._detailed_task_names,
                    "invocations": [],
                    "warnings": [],
                },
            }

        parsed_invocations = [
            (
                invocation,
                invocation.runtime.parse(
                    invocation.data,
                    detailed_task_names=self._detailed_task_names,
                ),
            )
            for invocation in invocations
        ]
        anchors = [trace["megaKernelCycleTrace"]["anchorCycle"] for _, trace in parsed_invocations]
        global_anchor = min(anchors)
        trace_events: list[dict[str, Any]] = []
        metadata_keys = set()
        dropped_count = 0
        record_count = 0
        invocation_metadata = []

        for invocation, trace in parsed_invocations:
            metadata = trace["megaKernelCycleTrace"]
            frequency = metadata["cycleFrequencyMHz"]
            shift_us = (metadata["anchorCycle"] - global_anchor) / frequency
            dropped_count += metadata["droppedRecordCount"]
            record_count += metadata["recordCount"]
            invocation_metadata.append(
                {
                    "invocationId": invocation.invocation_id,
                    "step": invocation.step,
                    "direction": invocation.direction,
                    "kernelName": invocation.runtime.kernel_name,
                    "rank": metadata.get("rank", invocation.runtime.rank),
                    "deviceId": metadata.get("deviceId", invocation.runtime.device_id),
                    "anchorCycle": metadata["anchorCycle"],
                    "recordCount": metadata["recordCount"],
                    "droppedRecordCount": metadata["droppedRecordCount"],
                }
            )
            for event in trace["traceEvents"]:
                if event["ph"] == "M":
                    key = (event["name"], event["pid"], event["tid"])
                    if key in metadata_keys:
                        continue
                    metadata_keys.add(key)
                    trace_events.append(event)
                    continue
                copied_event = dict(event)
                copied_event["ts"] += shift_us
                copied_event["args"] = {
                    **event.get("args", {}),
                    "invocation_id": invocation.invocation_id,
                    "step": invocation.step,
                    "direction": invocation.direction,
                }
                trace_events.append(copied_event)

        metadata_events = [event for event in trace_events if event["ph"] == "M"]
        duration_events = sorted(
            (event for event in trace_events if event["ph"] != "M"),
            key=lambda event: (event["ts"], event["pid"], event["tid"]),
        )
        return {
            "traceEvents": metadata_events + duration_events,
            "megaKernelCycleTrace": {
                "schemaVersion": 1,
                "windowIndex": self._window_index,
                "anchorCycle": global_anchor,
                "invocationCount": len(invocations),
                "recordCount": record_count,
                "droppedRecordCount": dropped_count,
                "detailedTaskNames": self._detailed_task_names,
                "invocations": invocation_metadata,
                "warnings": ([] if dropped_count == 0 else [f"Device dropped {dropped_count} cycle trace records"]),
            },
        }


def _enable_runtime_config_tensor(disabled_tensor: Any) -> Any:
    """Clone a RuntimeConfig tensor and enable its dedicated profiling field."""
    profiled_tensor = disabled_tensor.clone()
    enable_offset = RuntimeConfigC.cycle_profiling_enabled.offset
    profiled_tensor[enable_offset] = 1
    return profiled_tensor


def _prepare_mega_kernel_call(
    runtime: _PreparedMegaKernelRuntime,
    *,
    direction: str,
    fallback_event_counters: Any,
) -> _MegaKernelCall:
    """Select fast or profiled launch arguments for one Torch MegaKernel call."""
    with _ACTIVE_LOCK:
        profiler = _ACTIVE_PROFILER
    if profiler is None or not profiler._capture_enabled():
        return _MegaKernelCall(
            runtime_config=runtime.normal_tensor,
            event_counters=fallback_event_counters,
            profile_buffer=fallback_event_counters,
            clear_event_counters=fallback_event_counters,
        )
    return profiler._acquire_call(
        runtime,
        fallback_event_counters,
        direction,
    )


# The argument name intentionally mirrors torch.profiler.profile.
def mega_kernel_profile(
    *,
    schedule: Callable[[int], ProfilerAction] | None = None,
    on_trace_ready: Callable[[Any], None] | None = None,
    detailed_task_names: bool = False,
    max_pending_calls: int = 16,
) -> TorchMegaKernelProfiler:
    # pylint: disable=redefined-outer-name
    """Create the standalone Torch MegaKernel profiler.

    The returned context manager records only MegaKernel-internal events. It
    does not start torch.profiler and its export operation does not merge
    with an external trace.

    Args:
        schedule: Optional step schedule. Without one, all calls are retained.
        on_trace_ready: Optional callback invoked after each completed window.
        detailed_task_names: Include owner and task numbers in event names.
        max_pending_calls: Maximum in-flight invocation buffers before an
            intermediate D2H drain and reuse.

    Returns:
        Torch standalone MegaKernel profiler context manager.
    """
    return TorchMegaKernelProfiler(
        schedule=schedule,
        on_trace_ready=on_trace_ready,
        detailed_task_names=detailed_task_names,
        max_pending_calls=max_pending_calls,
    )
