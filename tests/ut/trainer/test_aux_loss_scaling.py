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
"""Trainer token weighting must scale injected MoE and indexer objectives."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
from torch import nn

from hyper_parallel.components.functional.aux_loss import aux_loss_auto_scale
from hyper_parallel.trainer.base import BaseTrainer


class AuxiliaryModel(nn.Module):
    """Synthetic model with two independently injected auxiliary objectives."""

    def __init__(self, enabled: bool, split_branches: bool = False) -> None:
        """Create the scalar parameters for the main and auxiliary branches."""
        super().__init__()
        self.main = nn.Parameter(torch.tensor(2.0))
        self.router = nn.Parameter(torch.tensor(3.0))
        self.indexer = nn.Parameter(torch.tensor(4.0))
        self.enabled = enabled
        self.split_branches = split_branches

    def forward(
        self, input_ids: torch.Tensor, use_cache: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Return the main loss and optional independent MTP-like branch."""
        del use_cache
        output = self.main.square() * input_ids.mean()
        if self.enabled:
            output = aux_loss_auto_scale(output, self.router.square() * input_ids.mean())
            if self.split_branches:
                mtp = aux_loss_auto_scale(self.main * input_ids.mean(), self.indexer.square() * input_ids.mean())
                return {"foundation_loss": output, "foundation_loss_mtp": mtp}
            output = aux_loss_auto_scale(output, self.indexer.square() * input_ids.mean())
        return output


class TestTrainerAuxLossScaling(unittest.TestCase):
    """Exercise the actual BaseTrainer microstep and loss aggregation."""

    def test_unequal_microbatches_match_weighted_explicit_objective(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: Ignore an empty microstep; weight LM and independent MTP/aux branches by 1/4 and 3/4.
        Expectation: The actual Trainer backward matches the explicitly token-weighted reference.
        """
        for enabled, split_branches in ((False, False), (True, False), (True, True)):
            with self.subTest(enabled=enabled, split_branches=split_branches):
                trainer = BaseTrainer.__new__(BaseTrainer)
                trainer.device = torch.device("cpu")
                trainer.model = AuxiliaryModel(enabled, split_branches)
                trainer.model_fwd_context = nullcontext()
                trainer.model_bwd_context = nullcontext()
                trainer.graph_compiler = None
                trainer.model_integration = Mock()
                trainer.config = SimpleNamespace(training=SimpleNamespace(empty_cache_before_backward=False))
                trainer.mesh = SimpleNamespace(dp_cp_mesh=None, sequence_parallel=False, dp_size=1, cp_size=1)
                trainer.loss_fn = lambda model_output, labels: model_output
                trainer.begin_loss_step()
                total_loss = 0.0
                metrics = {"foundation_loss": 0.0}
                for tokens, value in ((0, 99.0), (1, 1.0), (3, 2.0)):
                    trainer.current_token_counts = {"foundation_tokens": torch.tensor(tokens)}
                    loss, losses = trainer.forward_backward_step({"input_ids": torch.tensor([value])})
                    total_loss += loss.item()
                    for name, numerator in losses.items():
                        metrics[name] = metrics.get(name, 0.0) + numerator.item()
                actual_loss = trainer.loss_normalize(total_loss, metrics)
                self.assertEqual(actual_loss, 10.5 if split_branches else 7.0)
                self.assertEqual(metrics["foundation_loss"], 7.0)
                trainer.begin_loss_step()
                self.assertEqual(trainer.step_token_counts, {})
                weighted_input = 0.25 + 0.75 * 2
                torch.testing.assert_close(
                    trainer.model.main.grad, torch.tensor((5 if split_branches else 4) * weighted_input),
                )
                if enabled:
                    torch.testing.assert_close(trainer.model.router.grad, torch.tensor(6 * weighted_input))
                    torch.testing.assert_close(trainer.model.indexer.grad, torch.tensor(8 * weighted_input))
                else:
                    self.assertIsNone(trainer.model.router.grad)
                    self.assertIsNone(trainer.model.indexer.grad)
