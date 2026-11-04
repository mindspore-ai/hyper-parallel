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
"""Unit tests for the process-local Trainer memory profiler."""

import os
import stat
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, call, patch

from hyper_parallel.trainer.config.resolver import resolve_component
from hyper_parallel.trainer.config.training import MemoryConfig
from hyper_parallel.trainer.runtime.memory_profiler import MemoryProfiler


MODULE = "hyper_parallel.trainer.runtime.memory_profiler"


def _write_only_snapshot(filename: str) -> None:
    """Reproduce torch-npu's owner-write-only snapshot creation."""
    Path(filename).write_bytes(b"snapshot")
    os.chmod(filename, stat.S_IWUSR)


class TestMemoryProfiler(unittest.TestCase):
    """Verify allocator history scheduling and failure cleanup."""

    @staticmethod
    def _device() -> tuple[MagicMock, MagicMock]:
        """Build a fake accelerator namespace and allocator API."""
        memory_api = MagicMock()
        device_api = MagicMock(memory=memory_api)
        device_api.max_memory_reserved.return_value = 4096
        device_api.max_memory_allocated.return_value = 2048
        return device_api, memory_api

    @staticmethod
    def _reset(profiler: MemoryProfiler, config: Optional[MemoryConfig]) -> None:
        """Start a session with representative logical ranks."""
        profiler.reset(
            config,
            global_rank=4,
            tp_rank=2,
            dp_rank=3,
        )

    def test_recovery_restores_recording_window_without_copying_history(self) -> None:
        """Feature: Reopen restored allocator history at the next training step.

        Description: Capture an active window, abort it, and load state in a freshly constructed profiler.
        Expectation: Load restores counters; step reopens history once and respects the original window end.
        """
        device, memory = self._device()
        config = MemoryConfig(enable=True, start_step=1, end_step=4, dump_ranks=[4])
        with (
            patch(f"{MODULE}.get_device_type", return_value="cuda"),
            patch(f"{MODULE}.get_torch_device", return_value=device),
        ):
            original = MemoryProfiler()
            self._reset(original, config)
            original.step()
            original.step()
            saved = original.state_dict()
            original.abort()
            replacement = MemoryProfiler()
            self._reset(replacement, config)
            count = memory._record_memory_history.call_count
            replacement.load_state_dict(saved)
            self.assertEqual(replacement.current_step, 3)
            self.assertFalse(replacement._history_started)
            self.assertEqual(memory._record_memory_history.call_count, count)
            with (
                patch(f"{MODULE}.os.makedirs"),
                patch(f"{MODULE}.os.chmod"),
                patch(f"{MODULE}.time.strftime", return_value="2026-09-15-12-30-45") as timestamp,
            ):
                replacement.step()
                self.assertTrue(replacement._history_started)
                self.assertEqual(memory._record_memory_history.call_count, count + 1)
                replacement.step()
                self.assertEqual(memory._record_memory_history.call_count, count + 1)
                replacement.step()
            memory._dump_snapshot.assert_called_once()
            timestamp.assert_called_once_with("%Y-%m-%d-%H-%M-%S")
            memory._dump_snapshot.assert_called_once_with(
                os.path.join(config.save_path, "snapshot_2026-09-15-12-30-45_4.pickle")
            )
            self.assertFalse(replacement._history_started)
            self.assertTrue(replacement.enable)
            count = memory._record_memory_history.call_count
            replacement.load_state_dict(replacement.state_dict())
            self.assertEqual(memory._record_memory_history.call_count, count)

    def test_recovery_after_recording_window_does_not_reopen_empty_history(self) -> None:
        """Feature: Completed allocator windows do not restart after recovery.

        Description: Restore after the last recorded step but before the original dump control point.
        Expectation: No backend history is reopened for an already completed window.
        """
        for statistics in (False, True):
            with self.subTest(statistics=statistics):
                profiler = MemoryProfiler()
                profiler.enable = True
                profiler.mem_info = statistics
                profiler.end_step = 4
                with (
                    patch(f"{MODULE}.get_torch_device") as device,
                    patch.object(profiler, "_dump") as dump,
                    patch.object(profiler, "_log_memory_info") as log,
                ):
                    profiler.load_state_dict({"current_step": 5})
                    self.assertEqual(profiler._session_active, statistics)
                    profiler.step()
                device.assert_not_called()
                dump.assert_not_called()
                self.assertEqual(log.call_count, int(statistics))
                self.assertTrue(profiler.enable)
                self.assertFalse(profiler._history_started)

    def test_recovery_before_window_keeps_recording_deferred(self) -> None:
        """Feature: Preserve future allocator recording boundaries.

        Description: Resume before the recording window with memory statistics still enabled.
        Expectation: Resume does not start history early; the configured boundary starts it once.
        """
        device, memory = self._device()
        profiler = MemoryProfiler()
        with (
            patch(f"{MODULE}.get_device_type", return_value="cuda"),
            patch(f"{MODULE}.get_torch_device", return_value=device),
        ):
            self._reset(profiler, MemoryConfig(enable=True, start_step=3, end_step=4, mem_info=True))
            saved = profiler.state_dict()
            profiler.abort()
            profiler.load_state_dict(saved)
            memory._record_memory_history.assert_not_called()
            self.assertTrue(profiler.mem_info)
            for _ in range(3):
                profiler.step()
            memory._record_memory_history.assert_called_once_with(stacks="all", max_entries=sys.maxsize)
            profiler.abort()

    def test_cleanup_preserves_configuration_and_restore_only_needs_progress(self) -> None:
        """Feature: Local configuration survives cleanup without replica backup.

        Description: Stop or abort history/statistics sessions, then restore only their committed counter.
        Expectation: Cleanup prevents all further activity; restore reuses configuration at the next step.
        """
        for cleanup in ("stop", "abort"):
            for history, statistics in ((True, True), (True, False), (False, True)):
                with self.subTest(cleanup=cleanup, history=history, statistics=statistics):
                    device, memory = self._device()
                    profiler = MemoryProfiler()
                    config = MemoryConfig(enable=history, mem_info=statistics, start_step=0, end_step=4)
                    with (
                        patch(f"{MODULE}.get_device_type", return_value="cuda"),
                        patch(f"{MODULE}.get_torch_device", return_value=device) as resolve,
                        patch(f"{MODULE}.os.makedirs"),
                        patch(f"{MODULE}.os.chmod"),
                    ):
                        self._reset(profiler, config)
                        saved = profiler.state_dict()
                        self.assertEqual(saved, {"current_step": int(history)})
                        getattr(profiler, cleanup)()
                        self.assertEqual(profiler.enable, history)
                        self.assertEqual(profiler.mem_info, statistics)
                        self.assertFalse(profiler._session_active)
                        if history:
                            self.assertIs(profiler._device_api, device)
                            self.assertIs(profiler._memory_api, memory)
                        calls = list(device.mock_calls)
                        profiler.step()
                        self.assertEqual(device.mock_calls, calls)
                        self.assertEqual(profiler.state_dict(), saved)
                        resolves = resolve.call_count
                        profiler.load_state_dict(saved)
                        self.assertEqual(resolve.call_count, resolves)
                        self.assertEqual(profiler.enable, history)
                        self.assertEqual(profiler.mem_info, statistics)
                        self.assertTrue(profiler._session_active)
                        starts = memory._record_memory_history.call_count
                        logs = device.reset_peak_memory_stats.call_count
                        profiler.step()
                        self.assertEqual(memory._record_memory_history.call_count, starts + int(history))
                        self.assertEqual(device.reset_peak_memory_stats.call_count, logs + int(statistics))
                        profiler.abort()
                        device_api, memory_api = profiler._device_api, profiler._memory_api
                        self._reset(profiler, None)
                        self.assertFalse(profiler.enable)
                        self.assertFalse(profiler.mem_info)
                        self.assertFalse(profiler._session_active)
                        self.assertIs(profiler._device_api, device_api)
                        self.assertIs(profiler._memory_api, memory_api)

    def test_defaults_match_mindspeed_fsdp(self) -> None:
        """Expose the MindSpeed-compatible memory configuration defaults.

        Feature: Public memory profiler configuration.
        Description: Construct MemoryConfig without overrides.
        Expectation: All fields retain the documented names and defaults.
        """
        self.assertEqual(
            vars(MemoryConfig()),
            {
                "enable": False,
                "start_step": 1,
                "end_step": 2,
                "save_path": "./memory_snapshot",
                "dump_ranks": [0],
                "stacks": "all",
                "mem_info": False,
            },
        )

    def test_disabled_session_does_not_resolve_device(self) -> None:
        """Keep the default disabled configuration free of device side effects.

        Feature: Disabled memory profiling.
        Description: Reset, step, and stop a disabled profiler.
        Expectation: No accelerator API is resolved.
        """
        profiler = MemoryProfiler()
        with patch(f"{MODULE}.get_device_type") as get_device_type_mock:
            self._reset(profiler, MemoryConfig())
            profiler.step()
            profiler.stop()

        get_device_type_mock.assert_not_called()
        self.assertEqual(profiler.current_step, 0)

    def test_closed_window_records_then_dumps(self) -> None:
        """Record the session-relative closed interval including its end step.

        Feature: Bounded allocator history.
        Description: Advance a profiling session through steps zero to four.
        Expectation: Recording starts at one and dumps after training step three.
        """
        profiler = MemoryProfiler()
        device_api, memory_api = self._device()
        config = MemoryConfig(
            enable=True,
            start_step=1,
            end_step=3,
            dump_ranks=[4],
        )
        with (
            patch(f"{MODULE}.get_device_type", return_value="cuda"),
            patch(f"{MODULE}.get_torch_device", return_value=device_api),
            patch(f"{MODULE}.os.makedirs") as makedirs_mock,
            patch(f"{MODULE}.os.chmod"),
            patch(f"{MODULE}.time.strftime", return_value="2026-09-15-12-30-45"),
        ):
            self._reset(profiler, config)
            profiler.step()
            profiler.step()
            profiler.step()
            memory_api._dump_snapshot.assert_not_called()
            self.assertTrue(profiler._history_started)
            profiler.step()

        self.assertEqual(
            memory_api._record_memory_history.call_args_list,
            [call(stacks="all", max_entries=sys.maxsize), call(enabled=None)],
        )
        memory_api._dump_snapshot.assert_called_once_with(
            "./memory_snapshot/snapshot_2026-09-15-12-30-45_4.pickle"
        )
        makedirs_mock.assert_called_once_with("./memory_snapshot", exist_ok=True)
        self.assertFalse(profiler._history_started)

    def test_equal_boundaries_record_one_complete_step(self) -> None:
        """Record a complete construction or training step with equal boundaries.

        Feature: Equal profiling boundaries.
        Description: Advance sessions with both boundaries at zero or two.
        Expectation: History stays active through the selected step and then dumps once.
        """
        for step in (0, 2):
            with self.subTest(step=step):
                profiler = MemoryProfiler()
                device_api, memory_api = self._device()
                config = MemoryConfig(enable=True, start_step=step, end_step=step, dump_ranks=[4])
                with (
                    patch(f"{MODULE}.get_device_type", return_value="npu"),
                    patch(f"{MODULE}.get_torch_device", return_value=device_api),
                    patch(f"{MODULE}.os.makedirs"),
                    patch(f"{MODULE}.os.chmod"),
                ):
                    self._reset(profiler, config)
                    for _ in range(step):
                        profiler.step()
                    memory_api._record_memory_history.assert_called_once_with(stacks="all", max_entries=sys.maxsize)
                    memory_api._dump_snapshot.assert_not_called()
                    profiler.step()

                self.assertEqual(
                    memory_api._record_memory_history.call_args_list,
                    [call(stacks="all", max_entries=sys.maxsize), call(enabled=None)],
                )
                memory_api._dump_snapshot.assert_called_once()
                self.assertTrue(profiler.enable)
                self.assertFalse(profiler._session_active)

    def test_unselected_rank_records_without_dump_or_barrier(self) -> None:
        """Record on every rank while restricting independent snapshot files.

        Feature: Distributed snapshot rank filtering.
        Description: Run a profiling session on a rank not selected for output.
        Expectation: History records and stops without filesystem or barrier work.
        """
        profiler = MemoryProfiler()
        device_api, memory_api = self._device()
        config = MemoryConfig(enable=True, start_step=0, end_step=0, dump_ranks=[0])
        with (
            patch(f"{MODULE}.get_device_type", return_value="cuda"),
            patch(f"{MODULE}.get_torch_device", return_value=device_api),
            patch(f"{MODULE}.os.makedirs") as makedirs_mock,
            patch(f"{MODULE}.os.chmod"),
        ):
            self._reset(profiler, config)
            profiler.step()

        self.assertEqual(
            memory_api._record_memory_history.call_args_list,
            [call(stacks="all", max_entries=sys.maxsize), call(enabled=None)],
        )
        memory_api._dump_snapshot.assert_not_called()
        makedirs_mock.assert_not_called()

    def test_normal_stop_dumps_partial_window_once(self) -> None:
        """Dump and close history when training ends before the configured boundary.

        Feature: Normal early completion.
        Description: Stop twice after starting a partial profiling window.
        Expectation: History dumps and stops exactly once.
        """
        profiler = MemoryProfiler()
        device_api, memory_api = self._device()
        config = MemoryConfig(
            enable=True,
            start_step=1,
            end_step=5,
            dump_ranks=[4],
        )
        with (
            patch(f"{MODULE}.get_device_type", return_value="cuda"),
            patch(f"{MODULE}.get_torch_device", return_value=device_api),
            patch(f"{MODULE}.os.makedirs"),
            patch(f"{MODULE}.os.chmod"),
        ):
            self._reset(profiler, config)
            profiler.step()
            profiler.stop()
            profiler.stop()

        self.assertEqual(
            memory_api._record_memory_history.call_args_list,
            [call(stacks="all", max_entries=sys.maxsize), call(enabled=None)],
        )
        memory_api._dump_snapshot.assert_called_once()

    def test_abort_stops_without_dumping(self) -> None:
        """Release allocator history without filesystem or collective work on errors.

        Feature: Failed-training cleanup.
        Description: Abort after allocator history has started.
        Expectation: History stops without a snapshot or directory creation.
        """
        profiler = MemoryProfiler()
        device_api, memory_api = self._device()
        config = MemoryConfig(enable=True, start_step=0, end_step=5, dump_ranks=[4])
        with (
            patch(f"{MODULE}.get_device_type", return_value="cuda"),
            patch(f"{MODULE}.get_torch_device", return_value=device_api),
            patch(f"{MODULE}.os.makedirs") as makedirs_mock,
            patch(f"{MODULE}.os.chmod"),
        ):
            self._reset(profiler, config)
            profiler.abort()

        memory_api._record_memory_history.assert_has_calls(
            [call(stacks="all", max_entries=sys.maxsize), call(enabled=None)]
        )
        memory_api._dump_snapshot.assert_not_called()
        makedirs_mock.assert_not_called()
        self.assertFalse(profiler._history_started)

    def test_dump_failure_still_stops_history(self) -> None:
        """Close allocator history even when writing a snapshot fails.

        Feature: Snapshot failure cleanup.
        Description: Raise an I/O error from the allocator snapshot API.
        Expectation: The I/O error propagates after history is disabled.
        """
        profiler = MemoryProfiler()
        device_api, memory_api = self._device()
        memory_api._dump_snapshot.side_effect = OSError("disk full")
        config = MemoryConfig(enable=True, start_step=0, end_step=0, dump_ranks=[4])
        with (
            patch(f"{MODULE}.get_device_type", return_value="cuda"),
            patch(f"{MODULE}.get_torch_device", return_value=device_api),
            patch(f"{MODULE}.os.makedirs"),
            patch(f"{MODULE}.os.chmod"),
        ):
            self._reset(profiler, config)
            with self.assertRaisesRegex(OSError, "disk full"):
                profiler.step()

        memory_api._record_memory_history.assert_has_calls(
            [call(stacks="all", max_entries=sys.maxsize), call(enabled=None)]
        )
        self.assertFalse(profiler._history_started)

    def test_record_failure_attempts_history_cleanup(self) -> None:
        """Attempt cleanup when allocator history startup reports a failure.

        Feature: Allocator startup failure cleanup.
        Description: Make the record API fail before model construction.
        Expectation: The profiler issues a disable call and becomes inactive.
        """
        profiler = MemoryProfiler()
        device_api, memory_api = self._device()
        memory_api._record_memory_history.side_effect = [
            RuntimeError("record failed"),
            None,
        ]
        config = MemoryConfig(enable=True, start_step=0, end_step=1)
        with (
            patch(f"{MODULE}.get_device_type", return_value="cuda"),
            patch(f"{MODULE}.get_torch_device", return_value=device_api),
        ):
            with self.assertRaisesRegex(RuntimeError, "record failed"):
                self._reset(profiler, config)

        self.assertEqual(
            memory_api._record_memory_history.call_args_list,
            [call(stacks="all", max_entries=sys.maxsize), call(enabled=None)],
        )
        self.assertFalse(profiler._history_started)
        self.assertFalse(profiler._session_active)

    def test_stop_failure_retries_without_masking_error(self) -> None:
        """Retry cleanup when the allocator history disable call fails once.

        Feature: Allocator shutdown failure cleanup.
        Description: Fail the first normal disable call and succeed during abort.
        Expectation: The original error propagates after history is disabled.
        """
        profiler = MemoryProfiler()
        device_api, memory_api = self._device()
        memory_api._record_memory_history.side_effect = [
            None,
            RuntimeError("stop failed"),
            None,
        ]
        config = MemoryConfig(enable=True, start_step=0, end_step=5, dump_ranks=[])
        with (
            patch(f"{MODULE}.get_device_type", return_value="cuda"),
            patch(f"{MODULE}.get_torch_device", return_value=device_api),
        ):
            self._reset(profiler, config)
            with self.assertRaisesRegex(RuntimeError, "stop failed"):
                profiler.stop()

        self.assertEqual(memory_api._record_memory_history.call_count, 3)
        self.assertFalse(profiler._history_started)
        self.assertFalse(profiler._session_active)

    def test_reset_stops_stale_history_without_dump(self) -> None:
        """Release a previous Trainer session before replacing singleton state.

        Feature: Singleton session replacement.
        Description: Reset a profiler while its previous history is active.
        Expectation: Stale history stops without creating a snapshot.
        """
        profiler = MemoryProfiler()
        device_api, memory_api = self._device()
        config = MemoryConfig(enable=True, start_step=0, end_step=5, dump_ranks=[4])
        with (
            patch(f"{MODULE}.get_device_type", return_value="cuda"),
            patch(f"{MODULE}.get_torch_device", return_value=device_api),
        ):
            self._reset(profiler, config)
            self._reset(profiler, MemoryConfig())

        memory_api._record_memory_history.assert_has_calls(
            [call(stacks="all", max_entries=sys.maxsize), call(enabled=None)]
        )
        memory_api._dump_snapshot.assert_not_called()

    def test_mem_info_is_independent_of_history(self) -> None:
        """Log and reset peak statistics when history snapshots are disabled.

        Feature: Peak memory reporting.
        Description: Enable mem_info without allocator history snapshots.
        Expectation: Logical ranks are logged and peak statistics are reset.
        """
        profiler = MemoryProfiler()
        device_api, _ = self._device()
        config = MemoryConfig(mem_info=True)
        with (
            patch(f"{MODULE}.get_device_type", return_value="npu"),
            patch(f"{MODULE}.get_torch_device", return_value=device_api),
            patch(f"{MODULE}.logger.info") as info_mock,
        ):
            self._reset(profiler, config)
            profiler.step()

        info_mock.assert_called_once()
        self.assertIn("global_rank=%s", info_mock.call_args.args[0])
        self.assertEqual(info_mock.call_args.args[2:5], (4, 2, 3))
        device_api.reset_peak_memory_stats.assert_called_once_with()

    def test_invalid_configuration_fails_before_device_access(self) -> None:
        """Reject invalid scheduling and allocator options at the reset boundary.

        Feature: Memory configuration validation.
        Description: Reset with invalid boundaries, paths, ranks, and stacks.
        Expectation: Every invalid option fails before accelerator discovery.
        """
        invalid_configs = (
            MemoryConfig(start_step=-1),
            MemoryConfig(start_step=2, end_step=1),
            MemoryConfig(save_path=""),
            MemoryConfig(dump_ranks=[-1]),
            MemoryConfig(stacks="native"),
        )
        with patch(f"{MODULE}.get_device_type") as get_device_type_mock:
            for config in invalid_configs:
                with self.subTest(config=config), self.assertRaises((TypeError, ValueError)):
                    self._reset(MemoryProfiler(), config)

        get_device_type_mock.assert_not_called()

    def test_unsupported_device_and_allocator_api_fail_clearly(self) -> None:
        """Report missing accelerator capabilities during session initialization.

        Feature: Allocator capability detection.
        Description: Start immediate or deferred history on CPU and an incomplete accelerator API.
        Expectation: Reset validates history APIs before either immediate or deferred recording.
        """
        for start in (0, 1):
            for device_type, error in (("cpu", "CUDA or NPU"), ("cuda", "_record_memory_history, _dump_snapshot")):
                with self.subTest(start=start, device=device_type):
                    profiler = MemoryProfiler()
                    config = MemoryConfig(enable=True, start_step=start)
                    device_api = MagicMock(memory=types.SimpleNamespace())
                    with (
                        patch(f"{MODULE}.get_device_type", return_value=device_type) as resolve,
                        patch(f"{MODULE}.get_torch_device", return_value=device_api),
                    ):
                        with self.assertRaisesRegex(RuntimeError, error):
                            self._reset(profiler, config)
                        resolve.assert_called_once_with()

    def test_snapshot_is_owner_readable_with_write_only_npu_backend(self) -> None:
        """Repair the owner-write-only files created by torch-npu.

        Feature: Readable allocator snapshot export.
        Description: Dump through a fake backend that writes an owner-write-only file.
        Expectation: The exported snapshot is owner-readable without granting other users access.
        """
        profiler = MemoryProfiler()
        device_api, memory_api = self._device()
        memory_api._dump_snapshot.side_effect = _write_only_snapshot
        with tempfile.TemporaryDirectory() as output_dir:
            config = MemoryConfig(enable=True, start_step=0, end_step=0, dump_ranks=[4], save_path=output_dir)
            with (
                patch(f"{MODULE}.get_device_type", return_value="npu"),
                patch(f"{MODULE}.get_torch_device", return_value=device_api),
            ):
                self._reset(profiler, config)
                profiler.step()
            snapshot_path = next(Path(output_dir).glob("*.pickle"))
            self.assertEqual(snapshot_path.read_bytes(), b"snapshot")
            self.assertEqual(stat.S_IMODE(snapshot_path.stat().st_mode), stat.S_IRUSR | stat.S_IWUSR)

    def test_nested_mapping_resolves_to_memory_config(self) -> None:
        """Resolve the public memory mapping through the typed config resolver.

        Feature: Nested YAML memory configuration.
        Description: Resolve representative memory values with strict types.
        Expectation: The resulting MemoryConfig preserves every supplied value.
        """
        config = resolve_component(
            {
                "enable": True,
                "start_step": 0,
                "end_step": 4,
                "save_path": "/tmp/snapshots",
                "dump_ranks": [1, 3],
                "stacks": "python",
                "mem_info": True,
            },
            annotation=MemoryConfig,
            path="$.memory",
        )

        self.assertEqual(config.start_step, 0)
        self.assertEqual(config.end_step, 4)
        self.assertEqual(config.dump_ranks, [1, 3])
        self.assertEqual(config.stacks, "python")
        self.assertTrue(config.mem_info)


if __name__ == "__main__":
    unittest.main()
