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
"""Unit tests for the public MegaGate module."""

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore.modules.mega_gate.module import MegaGate


class TestMegaGate(unittest.TestCase):
    """Verify standalone parameter ownership and initialization."""

    @staticmethod
    def _create_gate(
        *,
        vision_enabled: bool,
        initializer_range: float = 0.02,
    ) -> MegaGate:
        """Create one small standalone router."""
        return MegaGate(
            hidden_size=128,
            num_experts=16,
            top_k=6,
            scoring_func="sqrtsoftplus",
            routed_scaling_factor=1.5,
            vision_enabled=vision_enabled,
            initializer_range=initializer_range,
        )

    def test_parameter_initialization(self) -> None:
        """Initialize projection with Normal and correction biases with zero."""
        with patch(
            "torch.nn.init.normal_",
            wraps=torch.nn.init.normal_,
        ) as normal:
            gate = self._create_gate(vision_enabled=True, initializer_range=0.035)

        normal.assert_called_once_with(gate.weight, mean=0.0, std=0.035)
        self.assertEqual(set(gate.state_dict()), {"weight", "bias", "bias_vl"})
        self.assertEqual(tuple(gate.weight.shape), (16, 128))
        self.assertTrue(torch.isfinite(gate.weight).all())
        self.assertGreater(torch.count_nonzero(gate.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(gate.bias).item(), 0)
        self.assertEqual(torch.count_nonzero(gate.bias_vl).item(), 0)

    def test_meta_materialization_and_reset(self) -> None:
        """Initialize parameters after materializing a meta module."""
        with torch.device("meta"):
            gate = self._create_gate(vision_enabled=False)

        gate.to_empty(device="cpu")
        gate.reset_parameters()

        self.assertTrue(torch.isfinite(gate.weight).all())
        self.assertGreater(torch.count_nonzero(gate.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(gate.bias).item(), 0)
        self.assertIsNone(gate.bias_vl)
