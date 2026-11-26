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
"""Separate-matrix SiTU ownership, activation gradients and supported topology."""

import unittest
from types import SimpleNamespace

import torch

from hyper_parallel.components.modules import SiTUMLP


class TestSiTUMLP(unittest.TestCase):
    """Compare public activation equations and retained optimizer matrix identities."""

    def test_source_adoption_preserves_three_matrices_and_activation_gradients(self) -> None:
        """The adopted family keeps state keys and matches an independent saturated-gate expression."""
        config = SimpleNamespace(hidden_size=8, intermediate_size=12, hidden_act="situ", mlp_bias=False,
                                 activation_situ_beta=1.5, activation_situ_linear_beta=2.0)
        source = SiTUMLP(config=config)
        member = SiTUMLP(module=source)
        self.assertEqual(tuple(member.state_dict()), tuple(source.state_dict()))
        for name, parameter in source.named_parameters():
            self.assertIs(member.get_parameter(name), parameter)
        values = torch.randn(2, 3, 8, requires_grad=True)
        gate, up = source.gate_proj(values).float(), source.up_proj(values).float()
        expected = source.down_proj(1.5 * torch.tanh(gate / 1.5) * torch.sigmoid(gate)
                                    * (2.0 * torch.tanh(up / 2.0)))
        actual = member(x=values)
        torch.testing.assert_close(actual, expected)
        parameters = (values, *tuple(member.parameters()))
        expected_gradients = torch.autograd.grad(expected.square().sum(), parameters)
        actual_gradients = torch.autograd.grad(actual.square().sum(), parameters)
        for gradient, target in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(gradient, target)
        with self.assertRaisesRegex(NotImplementedError, "TP1"):
            SiTUMLP(config=config, context={"tp_size": 2})
