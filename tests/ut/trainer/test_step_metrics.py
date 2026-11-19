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
"""Regression tests for deferred logging scalars and Trainer optimizer boundaries."""

import gc
import unittest
import weakref
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import torch

from tests.common.mark_utils import arg_mark

from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.config.training import TrainingConfig
from hyper_parallel.trainer.runtime.step_metrics import StepMetricAccumulator
from hyper_parallel.trainer.state import TrainerState
from hyper_parallel.trainer.text_trainer import TextTrainer
from hyper_parallel.trainer.vlm_trainer import VLMTrainer


class TestStepMetricAccumulator(unittest.TestCase):
    """Preserve scalar values while moving host reads out of microbatch execution."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_preserves_python_sum_order_and_mixed_dtypes(self) -> None:
        """Feature: metric precision.
        Description: cancellation and mixed types.
        Expectation: exact legacy sums.
        """
        results = []
        for deferred in (False, True):
            metrics = StepMetricAccumulator(defer=deferred)
            for value in (2**24, 1, -(2**24), 3):
                metrics.add(torch.tensor(value, dtype=torch.float32), {
                    "fp32": torch.tensor(value, dtype=torch.float32),
                    "fp64": torch.tensor(2**40 + 0.5, dtype=torch.float64),
                    "integer": torch.tensor(2**55 + 1, dtype=torch.int64),
                })
            results.append(metrics.finalize())
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], (4.0, {
            "fp32": 4.0, "fp64": 4 * (2**40 + 0.5), "integer": 4 * (2**55 + 1),
        }))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_snapshots_reused_scalar_buffers_without_retaining_graph(self) -> None:
        """Feature: metric lifetime.
        Description: reuse buffers and release loss.
        Expectation: original values.
        """
        for deferred in (False, True):
            with self.subTest(deferred=deferred):
                metrics = StepMetricAccumulator(defer=deferred)
                weight = torch.tensor(2., requires_grad=True)
                loss = weight * 3
                reference = weakref.ref(loss)
                buffer = torch.tensor(5.)
                metrics.add(loss, {"aux": buffer})
                buffer.fill_(99.)
                del loss
                gc.collect()
                self.assertIsNone(reference())
                self.assertEqual(metrics.finalize(), (6.0, {"aux": 5.0}))
                self.assertIsNone(weight.grad)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_deferred_add_avoids_item_and_finalizes_once_per_dtype(self) -> None:
        """Feature: grouped reads.
        Description: two scalar dtypes.
        Expectation: no item reads and two groups.
        """
        metrics = StepMetricAccumulator(defer=True)
        with patch.object(torch.Tensor, "item", side_effect=AssertionError("premature scalar read")):
            for value in (1., 2.):
                metrics.add(torch.tensor(value), {"double": torch.tensor(value, dtype=torch.float64)})
            with patch("hyper_parallel.trainer.runtime.step_metrics.torch.stack", wraps=torch.stack) as stack:
                result = metrics.finalize()
        self.assertEqual(stack.call_count, 2)
        self.assertEqual(result, (3.0, {"double": 3.0}))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_finalize_is_idempotent_and_accepts_later_metrics(self) -> None:
        """Feature: consumption.
        Description: finalize repeatedly then append.
        Expectation: no double counting.
        """
        for deferred in (False, True):
            metrics = StepMetricAccumulator(defer=deferred)
            self.assertEqual(metrics.finalize(), (0.0, {}))
            metrics.add(torch.tensor(2.), {"aux": torch.tensor(3.)})
            self.assertEqual(metrics.finalize(), (2.0, {"aux": 3.0}))
            self.assertEqual(metrics.finalize(), (2.0, {"aux": 3.0}))
            metrics.add(torch.tensor(4.), {"aux": torch.tensor(5.)})
            self.assertEqual(metrics.finalize(), (6.0, {"aux": 8.0}))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_rejects_invalid_defer_flag_and_non_scalar_metrics(self) -> None:
        """Feature: input validation.
        Description: invalid flag or shape.
        Expectation: descriptive errors.
        """
        for value in (None, 1, "true"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "boolean"):
                    StepMetricAccumulator(defer=value)
        for value in (1., torch.ones(2), torch.empty(0)):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "scalar tensors"):
                    StepMetricAccumulator(defer=True).add(value, {})


class TestDeferredTrainerMetrics(unittest.TestCase):
    """Exercise actual optimizer updates through all three Trainer loops."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_all_trainers_preserve_updates_and_read_after_optimizer_when_enabled(self) -> None:
        """Feature: optimizer boundary.
        Description: two backward passes.
        Expectation: identical updates and logs.
        """
        for trainer_class in (BaseTrainer, TextTrainer, VLMTrainer):
            for deferred in (False, True):
                with self.subTest(trainer=trainer_class.__name__, deferred=deferred):
                    self._assert_step(trainer_class, deferred)

    def _assert_step(self, trainer_class: type, deferred: bool) -> None:
        """Run a two-microbatch CPU SGD step and capture the host-read boundary."""
        base = BaseTrainer.__new__(BaseTrainer)
        base.config = SimpleNamespace(training=TrainingConfig(defer_microbatch_metrics=deferred, max_grad_norm=0))
        base.state = TrainerState()
        base.model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            base.model.weight.fill_(2.)
        base.optimizer = torch.optim.SGD(base.model.parameters(), lr=0.1, foreach=False, fused=False)
        base.lr_scheduler = None
        base.model_integration = Mock(runtime_enabled=False)
        base.fsdp_runtime_diagnostics = None
        base.num_micro_batches = 2
        base.loss_names = ("foundation_loss",)
        base._loss_denominator = Mock(return_value=4.)
        base.model_reshard = Mock()
        base._configure_fsdp_gradient_sync = Mock()
        base.configure_fsdp_gradient_sync = Mock()
        base.begin_fsdp_runtime_diagnostics = Mock()
        base.on_step_begin = Mock()
        base.on_step_end = Mock()
        base.on_micro_step_end = Mock()
        base._end_model_integration_step = Mock()

        def backward(batch: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            """Accumulate a real CPU gradient while preserving the scalar log value."""
            if not isinstance(batch, dict):
                batch = next(batch)
            loss = base.model.weight.reshape(()) * batch["value"]
            loss.backward()
            return loss, {"foundation_loss": loss, "diagnostic": torch.tensor(float(batch["value"]))}

        base.forward_backward_step = backward
        if trainer_class is BaseTrainer:
            trainer = base
        else:
            trainer = trainer_class.__new__(trainer_class)
            trainer.base = base
            trainer.forward_backward_step = backward
        samples = [{"value": value, "labels": torch.ones((1, 2), dtype=torch.long)} for value in (1, 3)]
        iterator = iter([samples]) if trainer_class is BaseTrainer else iter(samples)
        events = []
        original_step = base.optimizer.step
        original_finalize = StepMetricAccumulator.finalize

        def update() -> None:
            """Record the update boundary before applying CPU SGD."""
            events.append("update")
            original_step()

        def finalize(accumulator: StepMetricAccumulator) -> tuple[float, dict[str, float]]:
            """Record when detached logging values are read back."""
            events.append("read")
            return original_finalize(accumulator)

        module = trainer_class.__module__
        with (
            patch.object(base.optimizer, "step", side_effect=update),
            patch.object(StepMetricAccumulator, "finalize", finalize),
            patch(f"{module}.synchronize"),
            patch(f"{module}.memory_profiler.step"),
        ):
            result = trainer.train_step(iterator)
        self.assertEqual(events, ["update", "read"] if deferred else ["read", "update"])
        self.assertEqual(result, {"loss": 2., "grad_norm": 0.})
        torch.testing.assert_close(base.model.weight, torch.tensor([[1.9]]))
        base._loss_denominator.assert_called_once_with()
        base.on_step_end.assert_called_once_with(
            loss=2., loss_dict={"foundation_loss": 2., "diagnostic": 4.}, grad_norm=0.,
        )
