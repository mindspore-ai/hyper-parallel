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

import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hyper_parallel.core.multicore.profiler import ProfilerAction, profiler as torch_profiler
from hyper_parallel.core.multicore.torch import ops as multicore_adapter


class _FakeStorage:
    """Track explicit release of a fake tensor storage."""

    def __init__(self):
        self.released = False

    def resize_(self, size):
        """Record whether the storage was resized to zero."""
        self.released = size == 0


class _FakeArray:
    """Array-like wrapper exposing a byte conversion."""

    def __init__(self, data):
        self._data = data

    def tobytes(self):
        """Return the synthetic Host payload."""
        return self._data


class _FakeProfileBuffer:
    """Small tensor-like profile buffer used by profiler lifecycle tests."""

    def __init__(self, data):
        self._data = data
        self.storage = _FakeStorage()
        self.cpu_calls = 0
        self.zero_calls = 0

    def cpu(self):
        """Record a simulated Device-to-Host transfer."""
        self.cpu_calls += 1
        return self

    def zero_(self):
        """Record buffer initialization and return the tensor-like object."""
        self.zero_calls += 1
        return self

    def numpy(self):
        """Return an array-like wrapper for the synthetic payload."""
        return _FakeArray(self._data)

    def untyped_storage(self):
        """Return the storage object used to verify explicit release."""
        return self.storage


class _FakeBuffer:
    """Event-counter stand-in that records requested slices."""

    def __init__(self):
        self.device = "npu:0"
        self.requested_slices = []

    def __getitem__(self, key):
        self.requested_slices.append(key)
        return ("view", key)


class _FakeRuntime:
    """Prepared-runtime stand-in that returns deterministic parsed traces."""

    rank = 2
    normal_tensor = "disabled-config"
    profile_tensor = "enabled-config"
    device_id = 0
    kernel_name = "FakeMegaKernel"
    buffer_size = 4

    def __init__(self):
        self.parse_calls = []

    def parse(self, data, *, detailed_task_names):
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

    def __init__(self):
        self.synchronized_devices = []

    def synchronize(self, device_id):
        """Record a simulated NPU synchronization."""
        self.synchronized_devices.append(device_id)


class TestMegaKernelSchedule(unittest.TestCase):
    """Validate Torch-compatible schedule window semantics."""

    def test_schedule_actions(self):
        """Map each configured schedule region to its expected action."""
        capture_schedule = torch_profiler.schedule(
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
                ProfilerAction.NONE,
                ProfilerAction.NONE,
                ProfilerAction.WARMUP,
                ProfilerAction.RECORD,
                ProfilerAction.RECORD_AND_SAVE,
                ProfilerAction.NONE,
            ],
        )

    def test_schedule_rejects_invalid_values(self):
        """Reject schedule values that cannot form a valid capture cycle."""
        with self.assertRaisesRegex(ValueError, "active must be positive"):
            torch_profiler.schedule(wait=0, warmup=0, active=0)
        with self.assertRaisesRegex(ValueError, "wait must be"):
            torch_profiler.schedule(wait=-1, warmup=0, active=1)


class TestTorchMegaKernelProfiler(unittest.TestCase):
    """Validate call selection, private Device buffers, D2H boundaries, and export."""

    def tearDown(self):
        setattr(torch_profiler, "_ACTIVE_PROFILER", None)

    def test_inactive_call_uses_disabled_runtime_without_profile_allocation(self):
        """Keep inactive calls on the disabled RuntimeConfig and fallback buffer."""
        runtime = _FakeRuntime()
        event_counters = _FakeBuffer()

        with patch.object(torch_profiler.torch, "empty") as mock_empty:
            call = torch_profiler.prepare_mega_kernel_call(
                runtime,
                direction="forward",
                fallback_event_counters=event_counters,
            )

        self.assertEqual(call.runtime_config, runtime.normal_tensor)
        self.assertEqual(call.event_counters, event_counters)
        self.assertIs(call.profile_buffer, event_counters)
        self.assertEqual(call.clear_event_counters, event_counters)
        self.assertEqual(event_counters.requested_slices, [])
        mock_empty.assert_not_called()

    def test_active_window_exports_internal_trace_at_step_boundary(self):
        """Export retained active-window calls when the step closes the window."""
        runtime = _FakeRuntime()
        event_counters = _FakeBuffer()
        fake_npu = _FakeNpu()
        ready = []
        profile_buffer = _FakeProfileBuffer(b"1000")
        capture_schedule = torch_profiler.schedule(
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
            patch.object(torch_profiler.torch, "npu", fake_npu),
            patch.object(torch_profiler.torch, "empty", return_value=profile_buffer),
        ):
            with profiler:
                call = torch_profiler.prepare_mega_kernel_call(
                    runtime,
                    direction="forward",
                    fallback_event_counters=event_counters,
                )
                self.assertEqual(call.runtime_config, runtime.profile_tensor)
                self.assertEqual(
                    call.clear_event_counters,
                    event_counters,
                )
                self.assertIs(call.profile_buffer, profile_buffer)
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

    def test_profile_buffer_is_hidden_from_public_low_level_signatures(self):
        """Keep the new native ABI input private to managed MegaMoe execution."""
        self.assertNotIn(
            "profile_buffer",
            inspect.signature(multicore_adapter.mega_moe).parameters,
        )
        self.assertNotIn(
            "profile_buffer",
            inspect.signature(multicore_adapter.mega_moe_grad).parameters,
        )

    def test_pending_invocations_use_distinct_private_buffers(self):
        """Do not overwrite an earlier layer call before the window is drained."""
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
            patch.object(torch_profiler.torch, "npu", fake_npu),
            patch.object(
                torch_profiler.torch,
                "empty",
                side_effect=profile_buffers,
            ) as mock_empty,
        ):
            with profiler:
                first = torch_profiler.prepare_mega_kernel_call(
                    runtime, direction="forward", fallback_event_counters=event_counters
                )
                second = torch_profiler.prepare_mega_kernel_call(
                    runtime, direction="forward", fallback_event_counters=event_counters
                )
                self.assertIsNot(first.profile_buffer, second.profile_buffer)
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
