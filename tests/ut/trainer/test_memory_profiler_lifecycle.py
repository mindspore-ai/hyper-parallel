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
"""Unit tests for memory profiler integration with Trainer lifecycles."""

import types
import unittest
from unittest.mock import MagicMock, patch

from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.config.training import MemoryConfig
from hyper_parallel.trainer.text_trainer import TextTrainer
from hyper_parallel.trainer.vlm_trainer import VLMTrainer
from tests.common.mark_utils import arg_mark


class _ProbeError(RuntimeError):
    """Stop a lifecycle at the exact probe location."""


class TestMemoryProfilerLifecycle(unittest.TestCase):
    """Verify every Trainer uses the singleton at the required boundaries."""

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_reset_runs_after_setup_and_before_model_build(self) -> None:
        """Initialize profiling at the earliest point where rank metadata exists.

        Feature: Trainer memory profiler initialization.
        Description: Stop each Trainer constructor when it resets the profiler.
        Expectation: Reset follows setup, precedes model build, and errors trigger abort.
        """
        trainer_cases = (
            ("hyper_parallel.trainer.base", BaseTrainer),
            ("hyper_parallel.trainer.text_trainer", TextTrainer),
            ("hyper_parallel.trainer.vlm_trainer", VLMTrainer),
        )
        for module_name, trainer_class in trainer_cases:
            events = []

            def setup(base: BaseTrainer) -> None:
                """Provide the rank metadata produced by distributed setup."""
                events.append("setup")
                base.global_rank = 4
                base.mesh = types.SimpleNamespace(tp_rank=2, dp_rank=3)

            def reset(*args: object, **kwargs: object) -> None:
                """Stop construction when memory profiling is initialized."""
                del args, kwargs
                events.append("reset")
                raise _ProbeError

            with self.subTest(trainer=trainer_class.__name__):
                with (
                    patch.object(BaseTrainer, "_setup", autospec=True, side_effect=setup),
                    patch.object(BaseTrainer, "_build_model", autospec=True) as build_model_mock,
                    patch(f"{module_name}.memory_profiler.reset", side_effect=reset),
                    patch(f"{module_name}.memory_profiler.abort", side_effect=lambda: events.append("abort")),
                ):
                    with self.assertRaises(_ProbeError):
                        trainer_class(types.SimpleNamespace(memory=MemoryConfig()))

                    self.assertEqual(events, ["setup", "reset", "abort"])
                    build_model_mock.assert_not_called()

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_train_step_advances_profiler_before_accessing_trainer_state(self) -> None:
        """Advance the profiler before batch fetch and all other step work.

        Feature: Trainer step integration.
        Description: Make profiler.step raise on each Trainer implementation.
        Expectation: The error occurs before any incomplete Trainer state is accessed.
        """
        trainer_cases = (
            ("hyper_parallel.trainer.base", BaseTrainer),
            ("hyper_parallel.trainer.text_trainer", TextTrainer),
            ("hyper_parallel.trainer.vlm_trainer", VLMTrainer),
        )
        for module_name, trainer_class in trainer_cases:
            trainer = trainer_class.__new__(trainer_class)
            with self.subTest(trainer=trainer_class.__name__):
                with patch(
                        f"{module_name}.memory_profiler.step",
                        side_effect=_ProbeError,
                ) as step_mock:
                    with self.assertRaises(_ProbeError):
                        trainer_class.train_step(trainer, None)

                    step_mock.assert_called_once_with()

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_normal_train_stops_profiler_before_train_end_callbacks(self) -> None:
        """Flush memory history before ordinary train-end callbacks run.

        Feature: Normal Trainer completion.
        Description: Run each Trainer with zero configured iterations.
        Expectation: Memory stop precedes train-end callbacks and distributed teardown.
        """
        trainer_cases = (
            ("hyper_parallel.trainer.base", BaseTrainer),
            ("hyper_parallel.trainer.text_trainer", TextTrainer),
            ("hyper_parallel.trainer.vlm_trainer", VLMTrainer),
        )
        for module_name, trainer_class in trainer_cases:
            events = []
            trainer = self._empty_trainer(trainer_class, events)
            with self.subTest(trainer=trainer_class.__name__):
                with (
                    patch.object(trainer, "on_train_begin", side_effect=lambda: events.append("train_begin")),
                    patch.object(trainer, "on_train_end", side_effect=lambda: events.append("train_end")),
                    patch(f"{module_name}.memory_profiler.stop", side_effect=lambda: events.append("memory_stop")),
                    patch(f"{module_name}.synchronize", side_effect=lambda: events.append("synchronize")),
                    patch(f"{module_name}.HyperIter", return_value=MagicMock(), create=True),
                ):
                    trainer_class.train(trainer)

            self.assertEqual(
                events,
                ["train_begin", "memory_stop", "train_end", "synchronize", "destroy"],
            )

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_failed_train_aborts_without_normal_stop_or_train_end(self) -> None:
        """Stop allocator history without dumping when training raises.

        Feature: Exceptional Trainer completion.
        Description: Raise from the first regular training callback.
        Expectation: Every Trainer aborts memory history and skips normal finalization.
        """
        trainer_cases = (
            ("hyper_parallel.trainer.base", BaseTrainer),
            ("hyper_parallel.trainer.text_trainer", TextTrainer),
            ("hyper_parallel.trainer.vlm_trainer", VLMTrainer),
        )
        for module_name, trainer_class in trainer_cases:
            events = []
            trainer = self._empty_trainer(trainer_class, events)
            with self.subTest(trainer=trainer_class.__name__):
                with (
                    patch.object(trainer, "on_train_begin", side_effect=_ProbeError),
                    patch.object(trainer, "on_train_end") as train_end_mock,
                    patch(f"{module_name}.memory_profiler.abort", side_effect=lambda: events.append("abort")),
                    patch(f"{module_name}.memory_profiler.stop") as stop_mock,
                ):
                    with self.assertRaises(_ProbeError):
                        trainer_class.train(trainer)

                    self.assertEqual(events, ["abort"])
                    stop_mock.assert_not_called()
                    train_end_mock.assert_not_called()

    @staticmethod
    def _empty_trainer(trainer_class: type, events: list[str]):
        """Build the minimum state for a successful zero-iteration train call."""
        base = BaseTrainer.__new__(BaseTrainer)
        base.config = types.SimpleNamespace(
            dataloader=types.SimpleNamespace(use_background_prefetcher=False),
        )
        base.local_rank = 0
        base.state = types.SimpleNamespace(global_step=0, epoch=0)
        base.train_iters = 0
        base.train_epochs = 0
        base.train_dataloader = []
        base.destroy_distributed = lambda: events.append("destroy")
        if trainer_class is BaseTrainer:
            return base

        trainer = trainer_class.__new__(trainer_class)
        trainer.base = base
        return trainer


if __name__ == "__main__":
    unittest.main()
