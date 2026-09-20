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

import sys
import types
import unittest
from unittest.mock import MagicMock, call, patch

from hyper_parallel.trainer.config.resolver import resolve_component
from hyper_parallel.trainer.config.training import MemoryConfig
from hyper_parallel.trainer.runtime.memory_profiler import MemoryProfiler
from tests.common.mark_utils import arg_mark


MODULE = "hyper_parallel.trainer.runtime.memory_profiler"


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
    def _reset(profiler: MemoryProfiler, config: MemoryConfig) -> None:
        """Start a session with representative logical ranks."""
        profiler.reset(
            config,
            global_rank=4,
            tp_rank=2,
            dp_rank=3,
        )

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
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
                "max_entries": None,
                "mem_info": False,
            },
        )

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
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

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_half_open_window_records_then_dumps(self) -> None:
        """Record the session-relative half-open interval before its end step.

        Feature: Bounded allocator history.
        Description: Advance a profiling session through steps zero to three.
        Expectation: Recording starts at one and dumps before training step three.
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
            patch(f"{MODULE}.time.strftime", return_value="2026-09-15-12-30"),
        ):
            self._reset(profiler, config)
            profiler.step()
            profiler.step()
            profiler.step()

        self.assertEqual(
            memory_api._record_memory_history.call_args_list,
            [call(stacks="all", max_entries=sys.maxsize), call(enabled=None)],
        )
        memory_api._dump_snapshot.assert_called_once_with(
            "./memory_snapshot/snapshot_2026-09-15-12-30_4.pickle"
        )
        makedirs_mock.assert_called_once_with("./memory_snapshot", exist_ok=True)
        self.assertFalse(profiler._history_started)

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_equal_boundaries_record_dump_and_stop_at_reset(self) -> None:
        """Handle an equal zero-step boundary without leaving history active.

        Feature: Equal profiling boundaries.
        Description: Configure both boundaries at reset-time step zero.
        Expectation: History records, dumps, and stops during reset.
        """
        profiler = MemoryProfiler()
        device_api, memory_api = self._device()
        config = MemoryConfig(
            enable=True,
            start_step=0,
            end_step=0,
            dump_ranks=[4],
        )
        with (
            patch(f"{MODULE}.get_device_type", return_value="npu"),
            patch(f"{MODULE}.get_torch_device", return_value=device_api),
            patch(f"{MODULE}.os.makedirs"),
        ):
            self._reset(profiler, config)

        self.assertEqual(
            memory_api._record_memory_history.call_args_list,
            [call(stacks="all", max_entries=sys.maxsize), call(enabled=None)],
        )
        memory_api._dump_snapshot.assert_called_once()
        self.assertFalse(profiler.enable)
        self.assertFalse(profiler._session_active)

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_unselected_rank_records_without_dump_or_barrier(self) -> None:
        """Record on every rank while restricting independent snapshot files.

        Feature: Distributed snapshot rank filtering.
        Description: Run a profiling session on a rank not selected for output.
        Expectation: History records and stops without filesystem or barrier work.
        """
        profiler = MemoryProfiler()
        device_api, memory_api = self._device()
        config = MemoryConfig(enable=True, start_step=0, end_step=1, dump_ranks=[0])
        with (
            patch(f"{MODULE}.get_device_type", return_value="cuda"),
            patch(f"{MODULE}.get_torch_device", return_value=device_api),
            patch(f"{MODULE}.os.makedirs") as makedirs_mock,
        ):
            self._reset(profiler, config)
            profiler.step()

        self.assertEqual(
            memory_api._record_memory_history.call_args_list,
            [call(stacks="all", max_entries=sys.maxsize), call(enabled=None)],
        )
        memory_api._dump_snapshot.assert_not_called()
        makedirs_mock.assert_not_called()

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
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
            max_entries=123,
        )
        with (
            patch(f"{MODULE}.get_device_type", return_value="cuda"),
            patch(f"{MODULE}.get_torch_device", return_value=device_api),
            patch(f"{MODULE}.os.makedirs"),
        ):
            self._reset(profiler, config)
            profiler.step()
            profiler.stop()
            profiler.stop()

        self.assertEqual(
            memory_api._record_memory_history.call_args_list,
            [call(stacks="all", max_entries=123), call(enabled=None)],
        )
        memory_api._dump_snapshot.assert_called_once()

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
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
        ):
            self._reset(profiler, config)
            profiler.abort()

        memory_api._record_memory_history.assert_has_calls(
            [call(stacks="all", max_entries=sys.maxsize), call(enabled=None)]
        )
        memory_api._dump_snapshot.assert_not_called()
        makedirs_mock.assert_not_called()
        self.assertFalse(profiler._history_started)

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_dump_failure_still_stops_history(self) -> None:
        """Close allocator history even when writing a snapshot fails.

        Feature: Snapshot failure cleanup.
        Description: Raise an I/O error from the allocator snapshot API.
        Expectation: The I/O error propagates after history is disabled.
        """
        profiler = MemoryProfiler()
        device_api, memory_api = self._device()
        memory_api._dump_snapshot.side_effect = OSError("disk full")
        config = MemoryConfig(enable=True, start_step=0, end_step=1, dump_ranks=[4])
        with (
            patch(f"{MODULE}.get_device_type", return_value="cuda"),
            patch(f"{MODULE}.get_torch_device", return_value=device_api),
            patch(f"{MODULE}.os.makedirs"),
        ):
            self._reset(profiler, config)
            with self.assertRaisesRegex(OSError, "disk full"):
                profiler.step()

        memory_api._record_memory_history.assert_has_calls(
            [call(stacks="all", max_entries=sys.maxsize), call(enabled=None)]
        )
        self.assertFalse(profiler._history_started)

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
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

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
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

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
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

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
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

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_invalid_configuration_fails_before_device_access(self) -> None:
        """Reject invalid scheduling and allocator options at the reset boundary.

        Feature: Memory configuration validation.
        Description: Reset with invalid boundaries, paths, ranks, stacks, and limits.
        Expectation: Every invalid option fails before accelerator discovery.
        """
        invalid_configs = (
            MemoryConfig(start_step=-1),
            MemoryConfig(start_step=2, end_step=1),
            MemoryConfig(save_path=""),
            MemoryConfig(dump_ranks=[-1]),
            MemoryConfig(stacks="native"),
            MemoryConfig(max_entries=0),
        )
        with patch(f"{MODULE}.get_device_type") as get_device_type_mock:
            for config in invalid_configs:
                with self.subTest(config=config), self.assertRaises((TypeError, ValueError)):
                    self._reset(MemoryProfiler(), config)

        get_device_type_mock.assert_not_called()

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_unsupported_device_and_allocator_api_fail_clearly(self) -> None:
        """Report missing accelerator capabilities before model construction.

        Feature: Allocator capability detection.
        Description: Enable history on CPU and on an incomplete accelerator API.
        Expectation: Reset raises a descriptive RuntimeError in both cases.
        """
        config = MemoryConfig(enable=True)
        with patch(f"{MODULE}.get_device_type", return_value="cpu"):
            with self.assertRaisesRegex(RuntimeError, "CUDA or NPU"):
                self._reset(MemoryProfiler(), config)

        device_api = MagicMock(memory=types.SimpleNamespace())
        with (
            patch(f"{MODULE}.get_device_type", return_value="cuda"),
            patch(f"{MODULE}.get_torch_device", return_value=device_api),
        ):
            with self.assertRaisesRegex(RuntimeError, "_record_memory_history, _dump_snapshot"):
                self._reset(MemoryProfiler(), config)

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
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
                "max_entries": 1000,
                "mem_info": True,
            },
            expected_type=MemoryConfig,
            path="$.memory",
        )

        self.assertEqual(config.start_step, 0)
        self.assertEqual(config.end_step, 4)
        self.assertEqual(config.dump_ranks, [1, 3])
        self.assertEqual(config.stacks, "python")
        self.assertEqual(config.max_entries, 1000)


if __name__ == "__main__":
    unittest.main()
