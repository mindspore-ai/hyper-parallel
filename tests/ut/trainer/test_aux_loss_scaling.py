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

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from hyper_parallel.components.functional.aux_loss import aux_loss_auto_scale
from hyper_parallel.trainer.base import BaseTrainer
from tests.common.mark_utils import arg_mark


class AuxiliaryModel(nn.Module):
    """Synthetic model with two independently injected auxiliary objectives."""

    def __init__(self, enabled: bool) -> None:
        """Create the scalar parameters for the main and auxiliary branches."""
        super().__init__()
        self.main = nn.Parameter(torch.tensor(2.0))
        self.router = nn.Parameter(torch.tensor(3.0))
        self.indexer = nn.Parameter(torch.tensor(4.0))
        self.enabled = enabled

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """Return the main scalar while attaching optional penalties."""
        del use_cache
        output = self.main.square() * input_ids.mean()
        if self.enabled:
            output = aux_loss_auto_scale(output, self.router.square() * input_ids.mean())
            output = aux_loss_auto_scale(output, self.indexer.square() * input_ids.mean())
        return output


class TestTrainerAuxLossScaling(unittest.TestCase):
    """Exercise the actual BaseTrainer microstep and loss aggregation."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_unequal_microbatches_match_weighted_explicit_objective(self) -> None:
        """
        Feature: Auxiliary-loss training lifecycle.
        Description: Two microsteps contribute 1/4 and 3/4 to every gradient branch.
        Expectation: The actual Trainer backward matches the explicitly token-weighted reference.
        """
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                trainer = BaseTrainer.__new__(BaseTrainer)
                trainer.device = torch.device("cpu")
                trainer.model = AuxiliaryModel(enabled)
                trainer.model_fwd_context = nullcontext()
                trainer.model_bwd_context = nullcontext()
                trainer.model_integration = Mock()
                trainer.config = SimpleNamespace(training=SimpleNamespace(empty_cache_before_backward=False))
                trainer.mesh = SimpleNamespace(dp_cp_mesh=None, sequence_parallel=False, dp_size=1, cp_size=1)
                trainer.loss_fn = lambda model_output, labels: model_output
                trainer.step_token_counts = {"foundation_tokens": torch.tensor(4)}
                prefix = "hyper_parallel.trainer.runtime.metrics.all_reduce"
                with patch(prefix, side_effect=lambda value, **kwargs: value):
                    for tokens, value in ((1, 1.0), (3, 2.0)):
                        trainer.current_token_counts = {"foundation_tokens": torch.tensor(tokens)}
                        trainer.forward_backward_step({"input_ids": torch.tensor([value])})
                weighted_input = 0.25 + 0.75 * 2
                torch.testing.assert_close(trainer.model.main.grad, torch.tensor(4 * weighted_input))
                if enabled:
                    torch.testing.assert_close(trainer.model.router.grad, torch.tensor(6 * weighted_input))
                    torch.testing.assert_close(trainer.model.indexer.grad, torch.tensor(8 * weighted_input))
                else:
                    self.assertIsNone(trainer.model.router.grad)
                    self.assertIsNone(trainer.model.indexer.grad)
