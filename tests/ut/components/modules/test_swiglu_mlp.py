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
"""Separate SwiGLU source ownership and Muon update parity."""

import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from hyper_parallel.components.modules.common.swiglu_mlp import SwiGLUMLP
from hyper_parallel.core.optimizer.muon import Muon


class SourceMLP(nn.Module):
    """Independent source math with three optimizer matrices."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_act="silu")
        self.gate_proj = nn.Linear(8, 12, bias=False)
        self.up_proj = nn.Linear(8, 12, bias=False)
        self.down_proj = nn.Linear(12, 8, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Evaluate the source expression."""
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(value)) * self.up_proj(value))


class TestSeparateSwiGLU(unittest.TestCase):
    """Test optimizer semantics and source ownership of the public member."""

    def test_muon_updates_match_source(self) -> None:
        """Feature: Separate SwiGLU Muon parity.
        Description: Train independent source and adopted family matrices for ten CPU steps.
        Expectation: Every gradient, parameter and momentum buffer matches exactly.
        """
        torch.manual_seed(42)
        source = SourceMLP()
        adopted = copy.deepcopy(source)
        target = SwiGLUMLP(module=adopted, state_layout="separate")
        self.assertIs(target.gate_proj.weight, adopted.gate_proj.weight)
        self.assertEqual(target.make_transforms(), [])
        optimizers = [Muon(model.parameters()) for model in (source, target)]
        generator = torch.Generator().manual_seed(43)
        for _ in range(10):
            value = torch.randn(2, 3, 8, generator=generator)
            expected = torch.randn(2, 3, 8, generator=generator)
            for model, optimizer in zip((source, target), optimizers):
                optimizer.zero_grad(set_to_none=True)
                loss = (model(value) - expected).square().mean()
                loss.backward()
            for left, right in zip(source.parameters(), target.parameters()):
                torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
            for optimizer in optimizers:
                optimizer.step()
            for left, right in zip(source.parameters(), target.parameters()):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
                torch.testing.assert_close(
                    optimizers[0].state[left]["momentum_buffer"],
                    optimizers[1].state[right]["momentum_buffer"],
                    rtol=0,
                    atol=0,
                )
