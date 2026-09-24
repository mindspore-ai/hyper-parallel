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
"""Public supervision forwarding and unmodified model-computed objectives."""

from types import SimpleNamespace
import unittest

import torch

from hyper_parallel.components.losses.model_output import ModelOutputLoss
from tests.common.mark_utils import arg_mark


class TestModelOutputLoss(unittest.TestCase):
    """Verify forwarding without renaming, shifting or numerical compensation."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_combined_loss_keeps_auxiliary_and_gradient(self):
        """Feature: Model-owned objective.

        Description: Return an auxiliary objective while all labels are masked.
        Expectation: The returned tensor and its unit derivative are unchanged.
        """
        value = torch.tensor(2., requires_grad=True)
        result = ModelOutputLoss(check_valid_labels=False)(
            model_output=SimpleNamespace(loss=value), labels=torch.full((1, 2), -100))
        self.assertIs(result, value)
        result.backward()
        self.assertEqual(value.grad.item(), 1.)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_opt_in_supervision_keeps_public_names_and_identity(self):
        """Feature: Public forward fields.

        Description: Supply separate model and loss dictionaries, including shared labels.
        Expectation: Opt-in forwarding preserves tensors and metadata without mutating either input.
        """
        labels, mask = torch.tensor([[3, -100]]), torch.tensor([[0.5, 0.]])
        model_inputs = {"labels": labels, "position_ids": torch.tensor([[0, 1]])}
        loss_inputs = {"labels": labels, "shift_labels": labels, "loss_mask": mask}
        default = ModelOutputLoss().prepare_model_inputs(model_inputs, loss_inputs)
        self.assertEqual(set(default), set(model_inputs))
        result = ModelOutputLoss(pass_loss_inputs=True).prepare_model_inputs(model_inputs, loss_inputs)
        self.assertEqual(set(result), set(model_inputs) | set(loss_inputs))
        self.assertIs(result["shift_labels"], labels)
        self.assertIs(result["loss_mask"], mask)
        self.assertIs(result["position_ids"], model_inputs["position_ids"])
        self.assertNotIn("loss_mask", model_inputs)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            ModelOutputLoss(pass_loss_inputs=True).prepare_model_inputs(
                model_inputs, {"labels": labels.clone()})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_default_causal_label_check_and_gradient(self):
        """Feature: Existing causal objective behavior.

        Description: Supply valid, ignored and first-position-only causal labels.
        Expectation: Only a valid shifted target retains the loss and its gradient.
        """
        for labels, expected in (([[1, 2]], 2.), ([[-100, -100]], 0.), ([[1, -100]], 0.)):
            with self.subTest(labels=labels):
                value = torch.tensor(2., requires_grad=True)
                result = ModelOutputLoss()(model_output=SimpleNamespace(loss=value), labels=torch.tensor(labels))
                self.assertEqual(result.item(), expected)
                result.backward()
                self.assertEqual(value.grad.item(), expected / 2.)
