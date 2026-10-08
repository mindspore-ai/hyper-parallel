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
# pylint: disable=wrong-import-position

import types
import unittest
from contextlib import ExitStack
from functools import partial
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("torchdata.stateful_dataloader")

from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.config.training import MemoryConfig
from hyper_parallel.trainer.text_trainer import TextTrainer
from hyper_parallel.trainer.vlm_trainer import VLMTrainer
from tests.common.mark_utils import arg_mark


class _ProbeError(RuntimeError):
    """Stop a lifecycle at the exact probe location."""


def _setup_trainer(base: BaseTrainer, *, events: list[str]) -> None:
    """Provide the rank metadata produced by distributed setup."""
    events.append("setup")
    base.global_rank = 4
    base.mesh = types.SimpleNamespace(tp_rank=2, dp_rank=3)


def _reset_profiler(*args: object, events: list[str], **kwargs: object) -> None:
    """Stop construction when memory profiling is initialized."""
    del args, kwargs
    events.append("reset")
    raise _ProbeError


def _setup_trainer_with_integration(base: BaseTrainer, *, events: list[str], integration: MagicMock) -> None:
    """Provide distributed metadata and model-integration diagnostics."""
    _setup_trainer(base, events=events)
    base.optimizer = object()
    base.model_integration = integration


class TestMemoryProfilerLifecycle(unittest.TestCase):
    """Verify every Trainer uses the singleton at the required boundaries."""

    @arg_mark(["cpu_linux", "cpu_macos"], "level0", "onecard", "essential")
    def test_construction_preserves_model_integration_hooks(self) -> None:
        """Retain data and optimizer diagnostics while adding memory recording.

        Feature: Profiler and model-integration coexistence.
        Description: Build all three trainers with mocked components and diagnostics.
        Expectation: Reset precedes model construction, and diagnostic hooks keep their order.
        """
        for trainer_class in (BaseTrainer, TextTrainer, VLMTrainer):
            events = []
            integration = MagicMock()
            integration.attach_optimizer.side_effect = lambda _, events=events: events.append("attach_optimizer")

            with self.subTest(trainer=trainer_class.__name__), ExitStack() as stack:
                stack.enter_context(patch.object(
                    BaseTrainer, "_setup", autospec=True,
                    side_effect=partial(_setup_trainer_with_integration, events=events, integration=integration),
                ))
                stack.enter_context(patch(
                    "hyper_parallel.trainer.runtime.memory_profiler.memory_profiler.reset",
                    side_effect=lambda *args, events=events, **kwargs: events.append("reset"),
                ))
                base_stages = (
                    "_build_model", "_build_loss", "_build_dataset", "_build_dataloader",
                    "_compute_train_iters", "_build_optimizer", "_build_lr_scheduler",
                    "_build_training_context", "_init_callbacks", "attach_model_integration_data_pipeline",
                )
                for stage in base_stages:
                    stack.enter_context(patch.object(
                        BaseTrainer, stage, side_effect=partial(events.append, stage),
                    ))
                for stage in ("_build_model_assets", "_build_data_transform", "_build_collate_fn"):
                    stack.enter_context(patch.object(trainer_class, stage))
                if trainer_class is not BaseTrainer:
                    stack.enter_context(patch.object(trainer_class, "_build_get_batch"))

                trainer = trainer_class(types.SimpleNamespace(memory=MemoryConfig()))
                base = trainer if trainer_class is BaseTrainer else trainer.base
                integration.attach_optimizer.assert_called_once_with(base.optimizer)

            expected = ["setup", "reset", "_build_model", "_build_loss", "_build_dataset", "_build_dataloader"]
            if trainer_class is not BaseTrainer:
                expected.append("attach_model_integration_data_pipeline")
            expected.extend([
                "_compute_train_iters", "_build_optimizer", "attach_optimizer",
                "_build_lr_scheduler", "_build_training_context", "_init_callbacks",
            ])
            self.assertEqual(events, expected, f"expected={expected}, actual={events}")

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
            with self.subTest(trainer=trainer_class.__name__):
                with (
                    patch.object(BaseTrainer, "_setup", autospec=True,
                                 side_effect=partial(_setup_trainer, events=events)),
                    patch.object(BaseTrainer, "_build_model", autospec=True) as build_model_mock,
                    patch(f"{module_name}.memory_profiler.reset",
                          side_effect=partial(_reset_profiler, events=events)),
                    patch(f"{module_name}.memory_profiler.abort", side_effect=partial(events.append, "abort")),
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
                    patch.object(trainer, "on_train_begin", side_effect=partial(events.append, "train_begin")),
                    patch.object(trainer, "on_train_end", side_effect=partial(events.append, "train_end")),
                    patch(f"{module_name}.memory_profiler.stop",
                          side_effect=partial(events.append, "memory_stop")),
                    patch(f"{module_name}.synchronize", side_effect=partial(events.append, "synchronize")),
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
                    patch(f"{module_name}.memory_profiler.abort", side_effect=partial(events.append, "abort")),
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
        base.train_dataloader = MagicMock()
        base.destroy_distributed = lambda: events.append("destroy")
        if trainer_class is BaseTrainer:
            return base

        trainer = trainer_class.__new__(trainer_class)
        trainer.base = base
        return trainer


if __name__ == "__main__":
    unittest.main()
