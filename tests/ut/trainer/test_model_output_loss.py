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
"""Default model losses honor the trainer's aligned, masked token count."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.compile import GraphCompiler, PassConfig
from hyper_parallel.components.losses.model_output import ModelOutputLoss
from hyper_parallel.data.batching.get_batch import TextParallelBatch
from hyper_parallel.trainer.base import BaseTrainer
from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token


class CausalLossModel(torch.nn.Linear):
    """Small causal CE model over flattened local tokens and shifted targets."""

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor, position_ids: torch.Tensor,
        shift_labels: torch.Tensor | None = None, use_cache: bool = False,
    ) -> SimpleNamespace:
        """Compute the same aligned-target CE as a causal language model."""
        del position_ids, use_cache
        targets = shift_labels
        if targets is None:
            targets = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
        loss = torch.nn.functional.cross_entropy(super().forward(input_ids), targets)
        return SimpleNamespace(loss=loss)


class TestModelOutputLoss(unittest.TestCase):
    """Exercise the default loss through real eager and joint-graph training."""

    def test_aligned_and_empty_targets_match_explicit_ce(self) -> None:
        """Feature: Default Trainer loss with pre-aligned labels.
        Description: Keep a first-position target, then vary padding, masks and label alignment.
        Expectation: Losses/gradients match valid-token CE and empty shards remain zero.
        """
        for mode in ("eager", "static", "dynamic"):
            for aligned in (False, True):
                with self.subTest(mode=mode, aligned=aligned):
                    self._assert_training(mode, aligned)

    def _assert_training(self, mode: str, aligned: bool) -> None:
        """Compare a sequence of default-loss microsteps with an explicit objective."""
        trainer = BaseTrainer.__new__(BaseTrainer)
        trainer.device = torch.device("cpu")
        trainer.model = CausalLossModel(4, 4)
        trainer.model_integration = Mock(spec=["record_batch"])
        trainer.model_fwd_context = nullcontext()
        trainer.model_bwd_context = nullcontext()
        trainer.config = SimpleNamespace(training=SimpleNamespace(empty_cache_before_backward=False))
        trainer.mesh = SimpleNamespace(dp_cp_mesh=None, dp_size=1, cp_size=1, sequence_parallel=False)
        trainer.loss_fn = ModelOutputLoss()
        compiler = GraphCompiler(
            trainer.model, trainer._graph_train_fn,  # pylint: disable=protected-access
            pass_config=PassConfig(fsdp_enabled=False), dynamic=mode == "dynamic",
        )
        trainer.graph_compiler = None if mode == "eager" else compiler
        get_batch = TextParallelBatch.__new__(TextParallelBatch)
        get_batch.labels_are_shifted = aligned
        cases = (
            ([1, -100, -100], [1, 1, 1]),
            ([-100, -100, -100], [1, 1, 1]),
            ([-100, 1, 2], [1, 1, 1]),
            ([1, 2, 3], [0, 0, 0]),
        )
        with patch.object(compiler, "compile", wraps=compiler.compile) as compile_spy:
            for label_values, mask_values in cases:
                labels = torch.tensor(label_values)
                batch = {
                    "input_ids": torch.randn(3, 4),
                    "labels": labels,
                    "position_ids": torch.arange(3),
                    "loss_mask": torch.tensor(mask_values),
                }
                model_inputs, loss_inputs = get_batch._split_model_and_loss_inputs(batch)  # pylint: disable=protected-access
                counts = count_loss_token(loss_inputs)
                trainer.current_token_counts = counts
                trainer.begin_loss_step()
                trainer.model.zero_grad()
                actual, losses = trainer.forward_backward_step(model_inputs, loss_inputs)
                logged = {name: value.item() for name, value in losses.items()}
                actual = trainer.loss_normalize(actual.item(), logged)
                targets = labels if aligned else torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
                valid = targets.ne(-100)
                if counts["foundation_tokens"].item() == 0:
                    self.assertEqual(actual, 0.0)
                    for parameter in trainer.model.parameters():
                        torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))
                else:
                    logits = torch.nn.functional.linear(
                        batch["input_ids"], trainer.model.weight, trainer.model.bias,
                    )
                    expected = torch.nn.functional.cross_entropy(logits[valid], targets[valid])
                    gradients = torch.autograd.grad(expected, tuple(trainer.model.parameters()))
                    self.assertAlmostEqual(actual, expected.item(), places=6)
                    for parameter, gradient in zip(trainer.model.parameters(), gradients):
                        torch.testing.assert_close(parameter.grad, gradient)
                self.assertEqual(logged["foundation_loss"], actual)
            self.assertEqual(compile_spy.call_count, 0 if mode == "eager" else 1)
