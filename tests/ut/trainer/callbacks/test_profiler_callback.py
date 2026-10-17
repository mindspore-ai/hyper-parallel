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
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.trainer.callbacks.profiler_callback import ProfilerCallback
from hyper_parallel.trainer.config.training import ProfilingAnalysisConfig, ProfilingConfig
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
    def test_analysis_options_are_forwarded_only_to_npu(self) -> None:
        """Route automatic analysis controls only to the Ascend trace handler.

        Feature: Backend-specific trace analysis.
        Description: Construct callbacks for each backend and analysis option combination.
        Expectation: Only NPU receives analyse_flag and async_mode.
        """
        for device in ("cpu", "cuda", "npu"):
            for enabled, asynchronous in ((True, False), (False, False), (True, True), (False, True)):
                with self.subTest(device=device, enabled=enabled, asynchronous=asynchronous):
                    backend = MagicMock()
                    config = ProfilingConfig(
                        enable=True,
                        analysis=ProfilingAnalysisConfig(enable=enabled, async_mode=asynchronous),
                    )
                    with (
                        patch.object(ProfilerCallback, "_resolve_profiler_module", return_value=(device, backend)),
                        patch.object(ProfilerCallback, "_get_npu_experimental_config"),
                    ):
                        callback = ProfilerCallback(self._trainer(config))
                    options = {"analyse_flag": enabled, "async_mode": asynchronous} if device == "npu" else {}
                    backend.tensorboard_trace_handler.assert_called_once_with(str(callback.output_path), **options)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_analysis_rejects_non_boolean_options(self) -> None:
        """Reject ambiguous analysis settings before resolving a backend.

        Feature: Analysis configuration validation.
        Description: Supply string and integer values to boolean analysis options.
        Expectation: Invalid options fail with their full configuration path.
        """
        for name in ("enable", "async_mode"):
            for value in ("false", 1, None):
                with self.subTest(name=name, value=value):
                    config = ProfilingConfig(enable=True)
                    setattr(config.analysis, name, value)
                    with patch.object(ProfilerCallback, "_resolve_profiler_module") as backend:
                        with self.assertRaisesRegex(TypeError, f"profiler.analysis.{name}"):
                            ProfilerCallback(self._trainer(config))
                        backend.assert_not_called()

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
            enable=True,
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
    def test_collection_starts_at_first_step(self) -> None:
        """Start collection at the first training step.

        Feature: Profiler lifecycle boundaries.
        Description: Invoke the train-begin callback followed by the first step callback.
        Expectation: Profiling and metadata recording start only at the first step.
        """
        fake_profiler = MagicMock()
        schedule_mock, _, _ = self._cpu_profiler_patches(fake_profiler)
        config = ProfilingConfig(
            enable=True,
            start_step=1,
            stop_step=3,
        )

        with patch("hyper_parallel.trainer.callbacks.profiler_callback.get_device_type", return_value="cpu"):
            callback = ProfilerCallback(self._trainer(config))

        schedule_mock.assert_called_once_with(wait=0, warmup=0, active=3, repeat=1, skip_first=1)
        callback.on_train_begin(TrainerState())
        fake_profiler.start.assert_not_called()
        fake_profiler.step.assert_not_called()
        fake_profiler.add_metadata_json.assert_not_called()

        callback.on_step_begin(TrainerState(global_step=1))
        fake_profiler.start.assert_called_once_with()
        fake_profiler.step.assert_called_once_with()
        fake_profiler.add_metadata_json.assert_called_once()

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_only_explicit_ranks_are_selected(self) -> None:
        """Collect profiles only on explicitly selected ranks.

        Feature: Distributed profiler rank selection.
        Description: Select global rank three in an eight-rank world.
        Expectation: Only rank three initializes profiling.
        """
        profile_mock = MagicMock()
        _, _, backend_profile_mock = self._cpu_profiler_patches(profile_mock)
        config = ProfilingConfig(
            enable=True,
            rank_ids=[3],
        )

        with patch("hyper_parallel.trainer.callbacks.profiler_callback.get_device_type", return_value="cpu"):
            selected = ProfilerCallback(self._trainer(config, rank=3, world_size=8))
            skipped = [ProfilerCallback(self._trainer(config, rank=rank, world_size=8)) for rank in (0, 2, 4)]

        self.assertIs(selected.profiler, profile_mock)
        for callback in skipped:
            self.assertIsNone(callback.profiler)
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
        config = ProfilingConfig(enable=True, rank_ids=[])

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
        config = ProfilingConfig(enable=True, rank_ids=[8])

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
        config = ProfilingConfig(enable=True)

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
        config = ProfilingConfig(enable=True, start_step=9, stop_step=2)

        with patch("hyper_parallel.trainer.callbacks.profiler_callback.get_device_type", return_value="cpu"):
            callback = ProfilerCallback(self._trainer(config))

        self.assertEqual((callback.start_step, callback.stop_step), (1, 10))
        schedule_mock.assert_called_once_with(wait=0, warmup=0, active=10, repeat=1, skip_first=1)

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_npu_options_and_metadata_are_forwarded(self) -> None:
        """Forward Ascend options and topology metadata.

        Feature: Ascend NPU trainer profiling.
        Description: Run two steps against a mocked torch-npu profiler.
        Expectation: NPU configuration and topology metadata reach torch-npu without requiring MSTX APIs.
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
        config = ProfilingConfig(
            enable=True,
            start_step=2,
            stop_step=2,
            warmup_steps=1,
            level=1,
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
            mstx=False,
            export_type=["text"],
        )
        profile_kwargs = fake_module.profiler.profile.call_args.kwargs
        fake_module.profiler.schedule.assert_called_once_with(wait=0, warmup=1, active=1, repeat=1, skip_first=1)
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

    @arg_mark(["cpu_linux"], "level0", "onecard", "essential")
    def test_close_flushes_partial_window_once(self) -> None:
        """Flush a partial collection window when training ends early.

        Feature: Partial profiler-window cleanup.
        Description: Close collection after one of three requested steps.
        Expectation: The profiler stops once and repeated close calls are harmless.
        """
        fake_profiler = MagicMock()
        self._cpu_profiler_patches(fake_profiler)
        with patch("hyper_parallel.trainer.callbacks.profiler_callback.get_device_type", return_value="cpu"):
            callback = ProfilerCallback(self._trainer(ProfilingConfig(enable=True, stop_step=3)))
        callback.on_step_begin(TrainerState())
        callback.on_step_end(TrainerState(global_step=1))
        callback.on_train_end(TrainerState(global_step=1))
        callback.close()
        fake_profiler.stop.assert_called_once_with()
        self.assertIsNone(callback.profiler)


    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_real_cpu_trace_contains_only_selected_steps(self) -> None:
        """Check inclusive schedule boundaries in an exported CPU trace.

        Feature: Real profiler schedule and trace export.
        Description: Label CPU steps with zero, one, or two preceding warmup steps.
        Expectation: The exported trace contains only the inclusive recording window.
        """
        for start_step, warmup_steps in ((2, 0), (3, 1), (3, 2), (1, 0)):
            with self.subTest(start_step=start_step, warmup_steps=warmup_steps):
                with tempfile.TemporaryDirectory() as output_dir:
                    config = ProfilingConfig(
                        enable=True, start_step=start_step, stop_step=4, warmup_steps=warmup_steps,
                        memory=False, output_path=output_dir,
                    )
                    with patch(
                        "hyper_parallel.trainer.callbacks.profiler_callback.get_device_type", return_value="cpu",
                    ):
                        callback = ProfilerCallback(self._trainer(config))
                    callback.on_train_begin(TrainerState())
                    for step in range(1, 6):
                        state = TrainerState(global_step=step - 1)
                        callback.on_step_begin(state)
                        with torch.profiler.record_function(f"training-step-{step}"):
                            torch.ones(2).sum()
                        callback.on_step_end(state)
                    callback.on_train_end(TrainerState(global_step=5))
                    traces = list(Path(output_dir).rglob("*.pt.trace.json"))
                    self.assertEqual(len(traces), 1)
                    with traces[0].open(encoding="utf-8") as trace_file:
                        trace = json.load(trace_file)
                    labels = {event["name"] for event in trace["traceEvents"]
                              if event["name"].startswith("training-step-")}
                    self.assertEqual(labels, {f"training-step-{step}" for step in range(start_step, 5)})

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_invalid_warmup_fails_before_backend_access(self) -> None:
        """Reject invalid warmup types and steps outside the preceding window.

        Feature: Profiler warmup configuration validation.
        Description: Configure non-integer, negative, or excessive warmup steps.
        Expectation: Invalid values fail before profiler backend discovery.
        """
        for warmup_steps, error in ((True, TypeError), (1.5, TypeError), (-1, ValueError), (3, ValueError)):
            with self.subTest(warmup_steps=warmup_steps):
                config = ProfilingConfig(enable=True, start_step=3, warmup_steps=warmup_steps)
                with patch("hyper_parallel.trainer.callbacks.profiler_callback.get_device_type") as device_mock:
                    with self.assertRaisesRegex(error, "profiler.warmup_steps"):
                        ProfilerCallback(self._trainer(config))
                    device_mock.assert_not_called()
        with self.assertRaisesRegex(ValueError, "profiler.warmup_steps"):
            ProfilerCallback(self._trainer(ProfilingConfig(enable=True, warmup_steps=1)))


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
                "enable": False,
                "start_step": 1,
                "stop_step": 10,
                "warmup_steps": 0,
                "memory": True,
                "rank_ids": None,
                "output_path": None,
                "level": 1,
                "with_stack": False,
                "data_simplification": False,
                "analysis": ProfilingAnalysisConfig(),
            },
        )

if __name__ == "__main__":
    unittest.main()
