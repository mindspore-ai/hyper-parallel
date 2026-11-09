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
"""Unit tests for the standalone Torch MegaKernel profiler frontend."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

from hyper_parallel.core.multicore.diagnostics.common.session import _reset_active_session_for_test
from hyper_parallel.core.multicore.diagnostics import DiagnosticAction, schedule
from hyper_parallel.core.multicore.diagnostics.profiler import api as torch_profiler
from hyper_parallel.core.multicore.diagnostics.common.call import DiagnosticCall, LaunchArguments
from hyper_parallel.core.multicore.diagnostics.common.session import _get_active_session
from hyper_parallel.core.multicore.torch import ops as multicore_adapter


def _prepare_profile_call(runtime, *, direction, fallback_event_counters):
    """Test private profiler lifecycle with a synthetic prepared trace runtime."""
    session = _get_active_session("profiler")
    if session is None:
        return DiagnosticCall(LaunchArguments(
            runtime.normal_tensor, fallback_event_counters, fallback_event_counters, fallback_event_counters,
        ))
    return session.acquire_call(runtime, fallback_event_counters, direction)


class _FakeStorage:
    """Track explicit release of a fake tensor storage."""

    def __init__(self) -> None:
        """Initialize isolated fake lifecycle state."""
        self.released = False
        self.resize_calls = 0

    def resize_(self, size: int) -> None:
        """Record whether the storage was resized to zero."""
        self.resize_calls += 1
        self.released = size == 0


class _FakeArray:
    """Array-like wrapper exposing a byte conversion."""

    def __init__(self, data: bytes) -> None:
        """Store synthetic device bytes for lifecycle assertions."""
        self._data = data

    def tobytes(self) -> bytes:
        """Return the synthetic Host payload."""
        return self._data


class _FakeProfileBuffer:
    """Small tensor-like profile buffer used by profiler lifecycle tests."""

    def __init__(self, data: bytes) -> None:
        """Store synthetic device bytes for lifecycle assertions."""
        self._data = data
        self.storage = _FakeStorage()
        self.cpu_calls = 0
        self.zero_calls = 0

    def cpu(self) -> _FakeProfileBuffer:
        """Record a simulated Device-to-Host transfer."""
        self.cpu_calls += 1
        return self

    def zero_(self) -> _FakeProfileBuffer:
        """Record buffer initialization and return the tensor-like object."""
        self.zero_calls += 1
        return self

    def numpy(self) -> _FakeArray:
        """Return an array-like wrapper for the synthetic payload."""
        return _FakeArray(self._data)

    def untyped_storage(self) -> _FakeStorage:
        """Return the storage object used to verify explicit release."""
        return self.storage


class _FakeBuffer:
    """Event-counter stand-in that records requested slices."""

    def __init__(self) -> None:
        """Initialize isolated fake lifecycle state."""
        self.device = "npu:0"
        self.requested_slices = []

    def __getitem__(self, key: Any) -> tuple[str, Any]:
        """Record unexpected slicing of the event buffer."""
        self.requested_slices.append(key)
        return ("view", key)


class _FakeRuntime:
    """Prepared-runtime stand-in that returns deterministic parsed traces."""

    rank = 2
    normal_tensor = "disabled-config"
    profile_tensor = "enabled-config"
    device = "runtime-device"
    device_id = 0
    kernel_name = "FakeMegaKernel"
    buffer_size = 4

    def __init__(self) -> None:
        """Initialize isolated fake lifecycle state."""
        self.parse_calls = []

    def parse(self, data: bytes, *, detailed_task_names: bool) -> dict[str, Any]:
        """Return a deterministic trace for one fake invocation payload."""
        self.parse_calls.append((data, detailed_task_names))
        anchor = int(data.decode("ascii"))
        return {
            "traceEvents": [
                {
                    "name": "process_name",
                    "ph": "M",
                    "pid": 2,
                    "tid": 0,
                    "args": {"name": "FakeMegaKernel"},
                },
                {
                    "name": "Compute",
                    "cat": "MegaKernelInternal",
                    "ph": "X",
                    "pid": 2,
                    "tid": 1000,
                    "ts": 1.0,
                    "dur": 2.0,
                    "args": {},
                },
            ],
            "megaKernelCycleTrace": {
                "anchorCycle": anchor,
                "cycleFrequencyMHz": 50.0,
                "recordCount": 1,
                "droppedRecordCount": 0,
            },
        }


class _FakeNpu:
    """NPU runtime stand-in that records synchronization calls."""

    def __init__(self) -> None:
        """Initialize isolated fake lifecycle state."""
        self.synchronized_devices = []

    def synchronize(self, device_id: int) -> None:
        """Record a simulated NPU synchronization."""
        self.synchronized_devices.append(device_id)


class TestMegaKernelSchedule(unittest.TestCase):
    """Validate Torch-compatible schedule window semantics."""

    def test_schedule_actions(self):
        """Feature: MegaKernel diagnostics.
        Description: Map each configured schedule region to its expected action.
        Expectation: Schedule actions.
        """
        capture_schedule = schedule(
            skip_first=1,
            wait=1,
            warmup=1,
            active=2,
            repeat=1,
        )

        actions = [capture_schedule(step) for step in range(6)]

        self.assertEqual(
            actions,
            [
                DiagnosticAction.NONE,
                DiagnosticAction.NONE,
                DiagnosticAction.WARMUP,
                DiagnosticAction.RECORD,
                DiagnosticAction.RECORD_AND_SAVE,
                DiagnosticAction.NONE,
            ],
        )

    def test_schedule_rejects_invalid_values(self):
        """Feature: MegaKernel diagnostics.
        Description: Reject schedule values that cannot form a valid capture cycle.
        Expectation: Schedule rejects invalid values.
        """
        with self.assertRaisesRegex(ValueError, "active must be positive"):
            schedule(wait=0, warmup=0, active=0)
        with self.assertRaisesRegex(ValueError, "wait must be"):
            schedule(wait=-1, warmup=0, active=1)


class TestTorchMegaKernelProfiler(unittest.TestCase):
    """Validate call selection, private Device buffers, D2H boundaries, and export."""

    def tearDown(self) -> None:
        """Reset the process-local diagnostic session after each test."""
        _reset_active_session_for_test()

    def test_inactive_call_uses_disabled_runtime_without_profile_allocation(self):
        """Feature: MegaKernel diagnostics.
        Description: Keep inactive calls on the disabled RuntimeConfig and fallback buffer.
        Expectation: Inactive call uses disabled runtime without profile allocation.
        """
        runtime = _FakeRuntime()
        event_counters = _FakeBuffer()

        with patch.object(torch_profiler.torch, "empty") as mock_empty:
            call = _prepare_profile_call(
                runtime,
                direction="forward",
                fallback_event_counters=event_counters,
            )

        self.assertEqual(call.runtime_config, runtime.normal_tensor)
        self.assertEqual(call.event_counters, event_counters)
        self.assertIs(call.diagnostic_buffer, event_counters)
        self.assertEqual(call.clear_event_counters, event_counters)
        self.assertEqual(event_counters.requested_slices, [])
        mock_empty.assert_not_called()

    def test_inactive_eventless_call_does_not_allocate_profile_storage(self):
        """Feature: MegaKernel diagnostics.
        Description: Keep MegaGate free of profile allocation when capture is off.
        Expectation: Inactive eventless call does not allocate profile storage.
        """
        runtime = _FakeRuntime()

        with patch.object(torch_profiler.torch, "empty") as mock_empty:
            call = _prepare_profile_call(runtime, direction="forward", fallback_event_counters=None)

        self.assertEqual(call.runtime_config, runtime.normal_tensor)
        self.assertIsNone(call.event_counters)
        mock_empty.assert_not_called()

    def test_active_eventless_call_allocates_only_a_profile_buffer(self):
        """Feature: MegaKernel diagnostics.
        Description: Capture MegaGate without creating or clearing an event-counter tensor.
        Expectation: Active eventless call allocates only a profile buffer.
        """
        runtime = _FakeRuntime()
        profile_buffer = _FakeProfileBuffer(b"1000")
        profiler = torch_profiler.TorchMegaKernelProfiler(
            schedule=None,
            on_trace_ready=None,
            detailed_task_names=False,
            max_pending_calls=4,
        )

        with patch.object(torch_profiler.torch, "empty", return_value=profile_buffer) as mock_empty:
            with profiler:
                call = _prepare_profile_call(runtime, direction="forward", fallback_event_counters=None)
                self.assertEqual(call.runtime_config, runtime.profile_tensor)
                self.assertIsNone(call.event_counters)
                self.assertIsNone(call.clear_event_counters)
                self.assertIs(call.diagnostic_buffer, profile_buffer)
                self.assertEqual(mock_empty.call_args.kwargs["device"], runtime.device)
                call.cancel()
                call.cancel()

        self.assertEqual(profile_buffer.storage.resize_calls, 1)

    def test_active_window_exports_internal_trace_at_step_boundary(self):
        """Feature: MegaKernel diagnostics.
        Description: Export retained active-window calls when the step closes the window.
        Expectation: Active window exports internal trace at step boundary.
        """
        runtime = _FakeRuntime()
        event_counters = _FakeBuffer()
        fake_npu = _FakeNpu()
        ready = []
        profile_buffer = _FakeProfileBuffer(b"1000")
        capture_schedule = schedule(
            wait=0,
            warmup=0,
            active=1,
            repeat=1,
        )
        profiler = torch_profiler.TorchMegaKernelProfiler(
            schedule=capture_schedule,
            on_trace_ready=ready.append,
            detailed_task_names=True,
            max_pending_calls=4,
        )

        with (
            patch.object(torch_profiler.torch, "npu", fake_npu, create=True),
            patch.object(torch_profiler.torch, "empty", return_value=profile_buffer),
        ):
            with profiler:
                call = _prepare_profile_call(
                    runtime,
                    direction="forward",
                    fallback_event_counters=event_counters,
                )
                self.assertEqual(call.runtime_config, runtime.profile_tensor)
                self.assertEqual(
                    call.clear_event_counters,
                    event_counters,
                )
                self.assertIs(call.diagnostic_buffer, profile_buffer)
                self.assertEqual(profile_buffer.zero_calls, 1)
                call.complete()
                self.assertEqual(runtime.parse_calls, [])
                profiler.step()

        self.assertEqual(fake_npu.synchronized_devices, [0])
        self.assertEqual(len(runtime.parse_calls), 1)
        self.assertEqual(profile_buffer.cpu_calls, 1)
        self.assertTrue(profile_buffer.storage.released)
        self.assertEqual(ready, [profiler])
        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir) / "mega_kernel_trace.json"
            trace = profiler.export_chrome_trace(output_path)
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                trace,
            )
        event = next(item for item in trace["traceEvents"] if item["ph"] == "X")
        self.assertEqual(event["args"]["direction"], "forward")
        self.assertEqual(event["args"]["step"], 0)
        self.assertEqual(event["args"]["invocation_id"], 0)
        self.assertEqual(
            trace["megaKernelCycleTrace"]["invocations"][0],
            {
                "invocationId": 0,
                "step": 0,
                "direction": "forward",
                "kernelName": "FakeMegaKernel",
                "rank": 2,
                "deviceId": 0,
                "anchorCycle": 1000,
                "recordCount": 1,
                "droppedRecordCount": 0,
            },
        )

    def test_diagnostic_buffer_is_hidden_from_public_low_level_signatures(self):
        """Feature: MegaKernel diagnostics.
        Description: Keep the new native ABI input private to managed MegaMoe execution.
        Expectation: Diagnostic buffer is hidden from public low level signatures.
        """
        self.assertNotIn(
            "diagnostic_buffer",
            inspect.signature(multicore_adapter.mega_moe).parameters,
        )
        self.assertNotIn(
            "diagnostic_buffer",
            inspect.signature(multicore_adapter.mega_moe_grad).parameters,
        )

    def test_context_exit_discards_host_data_on_parse_failure(self) -> None:
        """Feature: MegaKernel profiler exception cleanup.

        Description: Fail parsing after multiple calls have drained to Host.
        Expectation: Propagate the parser error and release all capture resources.
        """
        runtime = _FakeRuntime()
        profile_buffer = _FakeProfileBuffer(b"1000")
        profiler = torch_profiler.mega_kernel_profile(max_pending_calls=1)
        parse_error = ValueError("invalid profile header")

        with (
            patch.object(torch_profiler.torch, "npu", _FakeNpu(), create=True),
            patch.object(torch_profiler.torch, "empty", return_value=profile_buffer),
            patch.object(runtime, "parse", side_effect=parse_error),
        ):
            with self.assertRaises(ValueError) as raised:
                with profiler:
                    for _ in range(3):
                        profiler.acquire_call(runtime, None, "forward").complete()
                    self.assertEqual(
                        len(profiler._raw_invocations), 2,
                        f"Expected 2 drained invocations, got {len(profiler._raw_invocations)}",
                    )

        self.assertIs(raised.exception, parse_error, f"Expected {parse_error!r}, got {raised.exception!r}")
        self.assertEqual(
            len(profiler._raw_invocations), 0,
            f"Expected 0 retained invocations, got {len(profiler._raw_invocations)}",
        )
        self.assertEqual(len(profiler._pending), 0, f"Expected 0 pending calls, got {len(profiler._pending)}")
        self.assertEqual(
            len(profiler._profile_buffer_pool), 0,
            f"Expected 0 pooled buffers, got {len(profiler._profile_buffer_pool)}",
        )
        self.assertTrue(
            profile_buffer.storage.released,
            f"Expected released=True, got {profile_buffer.storage.released}",
        )
        self.assertIsNone(_get_active_session())
        self.assertFalse(profiler._entered, f"Expected entered=False, got {profiler._entered}")

    def test_step_discards_host_data_immediately_on_parse_failure(self) -> None:
        """Feature: MegaKernel capture window failure.

        Description: Catch a parser exception at a scheduled step boundary.
        Expectation: Discard raw data immediately without reporting or retrying the window.
        """
        runtime = _FakeRuntime()
        profile_buffer = _FakeProfileBuffer(b"1000")
        ready = []
        profiler = torch_profiler.mega_kernel_profile(
            schedule=schedule(wait=0, warmup=0, active=1, repeat=1),
            on_trace_ready=ready.append,
        )

        with (
            patch.object(torch_profiler.torch, "npu", _FakeNpu(), create=True),
            patch.object(torch_profiler.torch, "empty", return_value=profile_buffer),
            patch.object(runtime, "parse", side_effect=ValueError("invalid profile header")) as parse,
        ):
            with profiler:
                profiler.acquire_call(runtime, None, "forward").complete()
                with self.assertRaisesRegex(ValueError, "invalid profile header"):
                    profiler.step()
                self.assertEqual(
                    len(profiler._raw_invocations), 0,
                    f"Expected 0 retained invocations, got {len(profiler._raw_invocations)}",
                )
                self.assertIsNone(profiler._last_trace, f"Expected None, got {profiler._last_trace}")
                self.assertEqual(profiler._window_index, 0, f"Expected window index 0, got {profiler._window_index}")
                self.assertEqual(len(ready), 0, f"Expected 0 callbacks, got {len(ready)}")

        parse.assert_called_once()
        self.assertTrue(
            profile_buffer.storage.released,
            f"Expected released=True, got {profile_buffer.storage.released}",
        )

    def test_context_exit_discards_host_data_when_drain_fails(self) -> None:
        """Feature: MegaKernel profiler final cleanup.

        Description: Exit with retained Host data and an unfinished pending launch.
        Expectation: Clear raw data even though drain fails before the parser runs.
        """
        runtime = _FakeRuntime()
        profile_buffer = _FakeProfileBuffer(b"1000")
        profiler = torch_profiler.mega_kernel_profile(max_pending_calls=1)

        with (
            patch.object(torch_profiler.torch, "npu", _FakeNpu(), create=True),
            patch.object(torch_profiler.torch, "empty", return_value=profile_buffer),
        ):
            with self.assertRaisesRegex(RuntimeError, "before launches complete"):
                with profiler:
                    profiler.acquire_call(runtime, None, "forward").complete()
                    profiler.acquire_call(runtime, None, "forward")
                    self.assertEqual(
                        len(profiler._raw_invocations), 1,
                        f"Expected 1 drained invocation, got {len(profiler._raw_invocations)}",
                    )

        self.assertEqual(
            len(profiler._raw_invocations), 0,
            f"Expected 0 retained invocations, got {len(profiler._raw_invocations)}",
        )
        self.assertEqual(len(profiler._pending), 0, f"Expected 0 pending calls, got {len(profiler._pending)}")
        self.assertEqual(
            len(profiler._profile_buffer_pool), 0,
            f"Expected 0 pooled buffers, got {len(profiler._profile_buffer_pool)}",
        )
        self.assertTrue(
            profile_buffer.storage.released,
            f"Expected released=True, got {profile_buffer.storage.released}",
        )
        self.assertIsNone(_get_active_session())
        self.assertFalse(profiler._entered, f"Expected entered=False, got {profiler._entered}")

    def test_pending_invocations_use_distinct_private_buffers(self):
        """Feature: MegaKernel diagnostics.
        Description: Do not overwrite an earlier layer call before the window is drained.
        Expectation: Pending invocations use distinct private buffers.
        """
        runtime = _FakeRuntime()
        event_counters = _FakeBuffer()
        fake_npu = _FakeNpu()
        profile_buffers = [
            _FakeProfileBuffer(b"1000"),
            _FakeProfileBuffer(b"1100"),
        ]
        profiler = torch_profiler.TorchMegaKernelProfiler(
            schedule=None,
            on_trace_ready=None,
            detailed_task_names=False,
            max_pending_calls=4,
        )

        with (
            patch.object(torch_profiler.torch, "npu", fake_npu, create=True),
            patch.object(
                torch_profiler.torch,
                "empty",
                side_effect=profile_buffers,
            ) as mock_empty,
        ):
            with profiler:
                first = _prepare_profile_call(
                    runtime, direction="forward", fallback_event_counters=event_counters
                )
                second = _prepare_profile_call(
                    runtime, direction="forward", fallback_event_counters=event_counters
                )
                self.assertIsNot(first.diagnostic_buffer, second.diagnostic_buffer)
                first.complete()
                second.complete()

        self.assertEqual(mock_empty.call_count, 2)
        self.assertTrue(all(buffer.storage.released for buffer in profile_buffers))

    def test_nested_contexts_are_rejected(self):
        """Reject nested profiler contexts in the same process."""
        first = torch_profiler.TorchMegaKernelProfiler(
            schedule=None,
            on_trace_ready=None,
            detailed_task_names=False,
            max_pending_calls=1,
        )
        second = torch_profiler.TorchMegaKernelProfiler(
            schedule=None,
            on_trace_ready=None,
            detailed_task_names=False,
            max_pending_calls=1,
        )

        with first:
            with self.assertRaisesRegex(RuntimeError, "nested"):
                with second:
                    pass


if __name__ == "__main__":
    unittest.main()
