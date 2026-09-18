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
"""Unit tests for the MindFormers-derived profiler callback."""

import json
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import torch

from hyper_parallel.trainer.callbacks.profiler_callback import ProfilerCallback
from hyper_parallel.trainer.config.training import ProfilingConfig
from hyper_parallel.trainer.state import TrainerState
from tests.common.mark_utils import arg_mark


class TestProfilerCallback(unittest.TestCase):
    """Verify profiler configuration, rank selection, and lifecycle behavior."""

    @staticmethod
    def _trainer(
        profiler: ProfilingConfig,
        *,
        rank: int = 0,
        world_size: int = 1,
        pipeline_stages: int = 1,
    ) -> types.SimpleNamespace:
        """Build the minimal trainer state required by the callback."""
        accelerator = types.SimpleNamespace(
            tp_size=2,
            pp_size=pipeline_stages,
            ep_size=4,
            sequence_parallel=True,
        )
        config = types.SimpleNamespace(profiler=profiler, accelerator=accelerator)
        return types.SimpleNamespace(
            config=config,
            global_rank=rank,
            world_size=world_size,
            mesh=types.SimpleNamespace(dp_size=8),
        )

    @staticmethod
    def _cpu_profiler_patches(fake_profiler: MagicMock) -> tuple[MagicMock, MagicMock, MagicMock]:
        """Patch CPU profiler factories and return their mocks."""
        schedule = patch("hyper_parallel.trainer.callbacks.profiler_callback.torch.profiler.schedule")
        handler = patch("hyper_parallel.trainer.callbacks.profiler_callback.torch.profiler.tensorboard_trace_handler")
        profile = patch("hyper_parallel.trainer.callbacks.profiler_callback.torch.profiler.profile")
        schedule_mock = schedule.start()
        handler_mock = handler.start()
        profile_mock = profile.start()
        schedule_mock.return_value = "schedule"
        handler_mock.return_value = "handler"
        profile_mock.return_value = fake_profiler
        return schedule_mock, handler_mock, profile_mock

    def tearDown(self) -> None:
        """Stop patches created by helper methods."""
        patch.stopall()

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_disabled_profile_does_not_initialize_backend(self) -> None:
        """Keep a disabled profile from initializing a backend.

        Feature: Optional trainer profiling.
        Description: Construct the callback with profiling disabled.
        Expectation: The callback does not resolve or create a profiler backend.
        """
        trainer = self._trainer(ProfilingConfig())

        with patch("hyper_parallel.trainer.callbacks.profiler_callback.get_device_type") as get_device_type_mock:
            callback = ProfilerCallback(trainer)

        self.assertIsNone(callback.profiler)
        get_device_type_mock.assert_not_called()

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_cpu_schedule_and_lifecycle_match_mindformers(self) -> None:
        """Capture the inclusive CPU profiling window.

        Feature: MindFormers-compatible profiler scheduling.
        Description: Run a mocked CPU profile from configured steps three through four.
        Expectation: The schedule, output path, and lifecycle calls match the configured window.
        """
        fake_profiler = MagicMock()
        schedule_mock, _, profile_mock = self._cpu_profiler_patches(fake_profiler)
        config = ProfilingConfig(
            enabled=True,
            start_step=3,
            stop_step=4,
            output_path="./profiles",
            memory=False,
            with_stack=True,
        )

        with patch("hyper_parallel.trainer.callbacks.profiler_callback.get_device_type", return_value="cpu"):
            callback = ProfilerCallback(self._trainer(config))

        schedule_mock.assert_called_once_with(wait=0, warmup=0, active=2, repeat=1, skip_first=3)
        profile_mock.assert_called_once_with(
            activities=[torch.profiler.ProfilerActivity.CPU],
            profile_memory=False,
            with_stack=True,
            schedule="schedule",
            on_trace_ready="handler",
        )
        self.assertEqual(callback.output_path, Path("profiles/profile/rank_0"))

        for step in range(1, 5):
            state = TrainerState(global_step=step)
            callback.on_step_begin(state)
            callback.on_step_end(state)

        fake_profiler.start.assert_called_once_with()
        self.assertEqual(fake_profiler.step.call_count, 5)
        fake_profiler.stop.assert_called_once_with()
        self.assertIsNone(callback.profiler)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_start_on_init_starts_at_train_begin(self) -> None:
        """Start initialization collection before the first step.

        Feature: Profiling from trainer initialization.
        Description: Enable start-on-init and invoke the train-begin callback.
        Expectation: Profiling and metadata recording start once before step callbacks.
        """
        fake_profiler = MagicMock()
        schedule_mock, _, _ = self._cpu_profiler_patches(fake_profiler)
        config = ProfilingConfig(
            enabled=True,
            start_step=1,
            stop_step=3,
            start_on_init=True,
        )

        with patch("hyper_parallel.trainer.callbacks.profiler_callback.get_device_type", return_value="cpu"):
            callback = ProfilerCallback(self._trainer(config))

        schedule_mock.assert_called_once_with(wait=0, warmup=0, active=3, repeat=1, skip_first=1)
        callback.on_train_begin(TrainerState())
        fake_profiler.start.assert_called_once_with()
        fake_profiler.step.assert_called_once_with()
        fake_profiler.add_metadata_json.assert_called_once()

        callback.on_step_begin(TrainerState(global_step=1))
        fake_profiler.start.assert_called_once_with()
        fake_profiler.step.assert_called_once_with()

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_pipeline_and_explicit_rank_selection_are_combined(self) -> None:
        """Combine pipeline leaders with explicitly selected ranks.

        Feature: Distributed profiler rank selection.
        Description: Select one explicit rank and each pipeline stage leader in an eight-rank world.
        Expectation: A selected stage leader initializes profiling while an unselected rank does not.
        """
        profile_mock = MagicMock()
        _, _, backend_profile_mock = self._cpu_profiler_patches(profile_mock)
        config = ProfilingConfig(
            enabled=True,
            rank_ids=[3],
            pipeline_stage_leaders=True,
        )

        with patch("hyper_parallel.trainer.callbacks.profiler_callback.get_device_type", return_value="cpu"):
            selected = ProfilerCallback(self._trainer(config, rank=4, world_size=8, pipeline_stages=2))
            skipped = ProfilerCallback(self._trainer(config, rank=2, world_size=8, pipeline_stages=2))

        self.assertIs(selected.profiler, profile_mock)
        self.assertIsNone(skipped.profiler)
        backend_profile_mock.assert_called_once()

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_empty_rank_list_profiles_all_ranks(self) -> None:
        """Treat an empty rank list as selecting every rank.

        Feature: Distributed profiler rank selection.
        Description: Construct the callback on a nonzero rank with an empty rank list.
        Expectation: The callback creates a profiler for that rank.
        """
        fake_profiler = MagicMock()
        _, _, profile_mock = self._cpu_profiler_patches(fake_profiler)
        config = ProfilingConfig(enabled=True, rank_ids=[])

        with patch("hyper_parallel.trainer.callbacks.profiler_callback.get_device_type", return_value="cpu"):
            callback = ProfilerCallback(self._trainer(config, rank=5, world_size=8))

        self.assertIs(callback.profiler, fake_profiler)
        profile_mock.assert_called_once()

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_rank_ids_must_be_within_world_size(self) -> None:
        """Reject profile ranks outside the distributed world.

        Feature: Profiler configuration validation.
        Description: Configure rank eight for a world containing ranks zero through seven.
        Expectation: Callback construction raises a descriptive value error.
        """
        config = ProfilingConfig(enabled=True, rank_ids=[8])

        with self.assertRaisesRegex(ValueError, r"\[0, 8\)"):
            ProfilerCallback(self._trainer(config, world_size=8))

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_cuda_backend_collects_cpu_and_cuda_activities(self) -> None:
        """Collect CPU and CUDA activities for a CUDA backend.

        Feature: CUDA trainer profiling.
        Description: Resolve a mocked CUDA device while constructing the profiler callback.
        Expectation: The profiler receives both CPU and CUDA activities.
        """
        fake_profiler = MagicMock()
        _, _, profile_mock = self._cpu_profiler_patches(fake_profiler)
        config = ProfilingConfig(enabled=True)

        with patch("hyper_parallel.trainer.callbacks.profiler_callback.get_device_type", return_value="cuda"):
            ProfilerCallback(self._trainer(config))

        self.assertEqual(
            profile_mock.call_args.kwargs["activities"],
            [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        )

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_invalid_steps_reset_to_mindformers_defaults(self) -> None:
        """Reset an inverted collection window to default steps.

        Feature: Profiler schedule normalization.
        Description: Configure a start step later than the stop step.
        Expectation: The callback resets the window to steps one through ten.
        """
        schedule_mock, _, _ = self._cpu_profiler_patches(MagicMock())
        config = ProfilingConfig(enabled=True, start_step=9, stop_step=2)

        with patch("hyper_parallel.trainer.callbacks.profiler_callback.get_device_type", return_value="cpu"):
            callback = ProfilerCallback(self._trainer(config))

        self.assertEqual((callback.start_step, callback.stop_step), (1, 10))
        schedule_mock.assert_called_once_with(wait=0, warmup=0, active=10, repeat=1, skip_first=1)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_npu_options_metadata_and_mstx_are_forwarded(self) -> None:
        """Forward Ascend options, metadata, and MSTX ranges.

        Feature: Ascend NPU trainer profiling.
        Description: Run two steps against mocked torch-npu profiler and MSTX APIs.
        Expectation: NPU configuration, topology metadata, and step ranges reach torch-npu.
        """
        fake_profiler = MagicMock()
        fake_module = types.SimpleNamespace()
        fake_module.profiler = types.SimpleNamespace(
            ProfilerActivity=types.SimpleNamespace(CPU="cpu", NPU="npu"),
            ProfilerLevel=types.SimpleNamespace(Level0="level0", Level1="level1", Level2="level2"),
            ExportType=types.SimpleNamespace(Text="text"),
            schedule=MagicMock(return_value="schedule"),
            tensorboard_trace_handler=MagicMock(return_value="handler"),
            profile=MagicMock(return_value=fake_profiler),
            _ExperimentalConfig=MagicMock(return_value="experimental"),
        )
        fake_module.npu = types.SimpleNamespace(
            mstx=types.SimpleNamespace(range_start=MagicMock(return_value=7), range_end=MagicMock()),
            current_stream=MagicMock(return_value="stream"),
        )
        config = ProfilingConfig(
            enabled=True,
            stop_step=2,
            level=1,
            mstx=True,
        )

        with (
            patch.dict("sys.modules", {"torch_npu": fake_module}),
            patch("hyper_parallel.trainer.callbacks.profiler_callback.get_device_type", return_value="npu"),
        ):
            callback = ProfilerCallback(self._trainer(config, world_size=16, pipeline_stages=2))
            fake_profiler.add_metadata_json.assert_not_called()
            state = TrainerState()
            callback.on_step_begin(state)
            state.global_step += 1
            callback.on_step_end(state)
            callback.on_step_begin(state)
            state.global_step += 1
            callback.on_step_end(state)

        getattr(fake_module.profiler, "_ExperimentalConfig").assert_called_once_with(
            profiler_level="level1",
            data_simplification=False,
            mstx=True,
            export_type=["text"],
        )
        profile_kwargs = fake_module.profiler.profile.call_args.kwargs
        self.assertEqual(profile_kwargs["activities"], ["cpu", "npu"])
        self.assertEqual(profile_kwargs["experimental_config"], "experimental")
        metadata_name, metadata_json = fake_profiler.add_metadata_json.call_args.args
        self.assertEqual(metadata_name, "distributed_args")
        self.assertEqual(
            json.loads(metadata_json),
            {
                "tensor_model_parallel_size": 2,
                "pipeline_model_parallel_size": 2,
                "data_parallel_size": 8,
                "expert_model_parallel_size": 4,
                "sequence_parallel": True,
                "parallel_mode": "distributed",
                "world_size": 16,
            },
        )
        self.assertEqual(
            fake_module.npu.mstx.range_start.call_args_list,
            [call("step 1", "stream"), call("step 2", "stream")],
        )
        self.assertEqual(fake_module.npu.mstx.range_end.call_args_list, [call(7), call(7)])


class TestProfilingConfig(unittest.TestCase):
    """Verify the breaking public profiler configuration surface."""

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_defaults_use_nested_profiler_names(self) -> None:
        """Expose concise defaults inside the profiler section.

        Feature: Nested trainer profiler configuration.
        Description: Construct profiling configuration without overrides.
        Expectation: Every public field has the documented default value.
        """
        config = ProfilingConfig()

        self.assertEqual(
            vars(config),
            {
                "enabled": False,
                "start_step": 1,
                "stop_step": 10,
                "start_on_init": False,
                "memory": True,
                "rank_ids": None,
                "pipeline_stage_leaders": False,
                "output_path": None,
                "level": 1,
                "with_stack": False,
                "data_simplification": False,
                "mstx": False,
            },
        )

if __name__ == "__main__":
    unittest.main()
