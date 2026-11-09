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
"""Regression tests for streaming VLM gradient-accumulation inputs."""

from collections.abc import Iterator
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

import torch

from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.callbacks.environ_meter_callback import EnvironMeterCallback
from hyper_parallel.trainer.callbacks.logging_callback import LoggingCallback
from hyper_parallel.trainer.state import TrainerState
from hyper_parallel.trainer.vlm_trainer import VLMTrainer


class TestVLMTrainer(unittest.TestCase):
    """Verify input lifetime, accumulation, and metrics without an accelerator."""

    def setUp(self) -> None:
        """Keep profiling, synchronization, and metric reductions on the CPU."""
        patches = (
            patch("hyper_parallel.trainer.vlm_trainer.memory_profiler.step"),
            patch("hyper_parallel.trainer.vlm_trainer.synchronize"),
            patch("hyper_parallel.trainer.callbacks.environ_meter_callback.get_world_size_safe", return_value=1),
            patch("hyper_parallel.trainer.callbacks.environ_meter_callback.get_device_type", return_value="cpu"),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def _make_trainer(num_micro_steps: int) -> VLMTrainer:
        """Build the step dependencies while retaining real callback dispatch."""
        trainer = VLMTrainer.__new__(VLMTrainer)
        base = BaseTrainer.__new__(BaseTrainer)
        trainer.base = base
        base.num_micro_batches = num_micro_steps
        base.state = TrainerState(global_step=0, epoch=0)
        base.mesh = SimpleNamespace(dp_cp_mesh=None, cp_size=1)
        base.optimizer = SimpleNamespace(param_groups=[{"lr": 0.001}])
        base.lr_scheduler = None
        base.model_integration = SimpleNamespace(begin_step=Mock())
        base.model_reshard = Mock()
        base.configure_fsdp_gradient_sync = Mock()
        base.begin_fsdp_runtime_diagnostics = Mock()
        base.model = torch.nn.Linear(1, 1)
        base.prepare_optimizer_step = Mock(side_effect=lambda loss, metrics: (loss, 0.5))
        base.step_optimizers_and_schedulers = Mock()
        base._end_model_integration_step = Mock()  # pylint: disable=protected-access
        meter = EnvironMeterCallback(base, microbatch_mean_metrics=("router_aux_loss",))
        base._callbacks = [meter]
        return trainer

    def test_router_aux_mean_reaches_shared_metrics_and_logger(self) -> None:
        """Publish the AUX mean through VLM callbacks while keeping the training loss sum."""
        trainer = self._make_trainer(2)
        trainer.base.config = SimpleNamespace(training=SimpleNamespace(logging_steps=1))
        trainer.base._callbacks.append(LoggingCallback(trainer.base))
        trainer.base.get_batch = Mock(return_value=(
            {"input_ids": torch.ones((1, 4), dtype=torch.long)},
            {"labels": torch.ones((1, 4), dtype=torch.long)},
        ))
        trainer.base.forward_backward_step = Mock(side_effect=[
            (torch.tensor(0.5), {"foundation_loss": torch.tensor(0.5), "router_aux_loss": torch.tensor(aux)})
            for aux in (1.0, 3.0)
        ])

        with patch("hyper_parallel.trainer.callbacks.logging_callback.logger.info") as log:
            result = trainer.train_step(iter(range(2)))

        metrics = getattr(trainer.base, "step_env_metrics")
        expected = {"training/router_aux_loss": 2.0, "training/foundation_loss": 1.0, "training/total_loss": 1.0}
        observed = {name: metrics[name] for name in expected}
        self.assertEqual(observed, expected, f"expected step metrics={expected}, got={observed}")
        self.assertEqual(result["loss"], 1.0, f"expected backward loss sum=1.0, got={result['loss']}")
        log.assert_called_once()
        message = log.call_args.args[1]
        self.assertIn("training/router_aux_loss=2", message.split(), f"expected terminal AUX=2, got={message}")

    def test_micro_steps_release_images_before_fetching_next_batch(self) -> None:
        """Bound image residency without changing accumulation or step metrics.

        Feature: Streaming VLM micro-batches.
        Description: Fetch image tensors lazily and run real CPU backward over one or eight micro-steps.
        Expectation: Only one image payload lives at a time; gradients, losses, and counters accumulate.
        """
        for num_micro_steps in (1, 8):
            with self.subTest(num_micro_steps=num_micro_steps):
                self._assert_micro_step_lifetime(num_micro_steps)

    def _assert_micro_step_lifetime(self, num_micro_steps: int) -> None:
        """Check one accumulation window with callbacks bound to its own state."""
        trainer = self._make_trainer(num_micro_steps)
        image_refs = []
        events = []
        weight = torch.tensor(2.0, requires_grad=True)

        def get_batch(
                data_iterator: Iterator[int],
        ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
            """Allocate the next image only after the previous one is released."""
            index = next(data_iterator)
            alive = sum(reference() is not None for reference in image_refs)
            self.assertEqual(alive, 0, f"expected no previous image payload, got={alive}")
            events.append(("fetch", index))
            pixels = torch.full((2, 3, 2, 2), float(index))
            image_refs.append(weakref.ref(pixels))
            labels = torch.tensor([[-100, 1, 2, -100]])
            return (
                {"input_ids": torch.full((1, 4), index), "pixel_values": pixels, "labels": labels},
                {
                    "labels": labels,
                    "loss_mask": torch.tensor([[0, 1, index % 2, 0]], dtype=torch.bool),
                },
            )

        def forward_backward(
                model_inputs: dict[str, torch.Tensor], loss_inputs: dict[str, torch.Tensor],
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            """Accumulate a real gradient while checking the loss-token scale."""
            del loss_inputs
            index = int(model_inputs["input_ids"][0, 0])
            expected_tokens = 1 + index % 2
            counts = trainer.base.current_token_counts
            observed_tokens = int(counts["foundation_tokens"])
            self.assertEqual(
                observed_tokens, expected_tokens,
                f"expected local tokens={expected_tokens}, got={observed_tokens}",
            )
            loss = weight * model_inputs["pixel_values"].mean() / num_micro_steps
            loss.backward()
            events.append(("backward", index))
            return loss.detach(), {"foundation_loss": loss.detach()}

        trainer.base.get_batch = get_batch
        trainer.base.forward_backward_step = forward_backward
        result = trainer.train_step(iter(range(1, num_micro_steps + 1)))

        expected_events = [
            (phase, index)
            for index in range(1, num_micro_steps + 1)
            for phase in ("fetch", "backward")
        ]
        self.assertEqual(events, expected_events, f"expected={expected_events}, got={events}")
        alive = sum(reference() is not None for reference in image_refs)
        self.assertEqual(alive, 0, f"expected all image inputs released, got={alive}")
        torch.testing.assert_close(weight.grad, torch.tensor((num_micro_steps + 1) / 2))
        expected_metrics = {"loss": float(num_micro_steps + 1), "grad_norm": 0.5}
        self.assertEqual(result, expected_metrics, f"expected={expected_metrics}, got={result}")
        metrics = getattr(trainer.base, "step_env_metrics")
        expected_tokens = 2 * num_micro_steps
        self.assertEqual(
            metrics["data/step_tokens"], expected_tokens,
            f"expected step tokens={expected_tokens}, got={metrics['data/step_tokens']}",
        )
        self.assertEqual(
            metrics["data/step_samples"], num_micro_steps,
            f"expected samples={num_micro_steps}, got={metrics['data/step_samples']}",
        )
        self.assertEqual(
            trainer.base.state.global_step, 1,
            f"expected completed global step=1, got={trainer.base.state.global_step}",
        )
        trainer.base.model_integration.begin_step.assert_called_once_with(1)
        trainer.base.prepare_optimizer_step.assert_called_once()
        trainer.base.step_optimizers_and_schedulers.assert_called_once_with()

    def test_backward_failure_does_not_fetch_later_micro_batches(self) -> None:
        """Stop reading input when the current micro-step fails.

        Feature: Streaming VLM failure handling.
        Description: Fail the first backward in an eight-micro-step optimizer update.
        Expectation: Only one batch is fetched and neither optimizer nor global step advances.
        """
        trainer = self._make_trainer(8)
        data_iterator = iter(range(8))
        trainer.base.get_batch = Mock(return_value=(
            {"input_ids": torch.ones((1, 4), dtype=torch.long)},
            {"labels": torch.ones((1, 4), dtype=torch.long)},
        ))
        trainer.base.forward_backward_step = Mock(side_effect=RuntimeError("backward failed"))

        with self.assertRaisesRegex(RuntimeError, "backward failed"):
            trainer.train_step(data_iterator)

        trainer.base.get_batch.assert_called_once_with(data_iterator)
        trainer.base.prepare_optimizer_step.assert_not_called()
        trainer.base.step_optimizers_and_schedulers.assert_not_called()
        self.assertEqual(
            trainer.base.state.global_step, 0,
            f"expected unchanged global step=0, got={trainer.base.state.global_step}",
        )


if __name__ == "__main__":
    unittest.main()
