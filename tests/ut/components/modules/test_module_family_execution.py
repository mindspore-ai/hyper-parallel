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
"""Standalone family execution without an operator-selection framework."""
import unittest

import torch

from examples.torch.module_family.config import RMSNormConfig, SwiGLUMLPConfig, SharedExpertConfig
from hyper_parallel.components.modules.common.rms_norm import RMSNorm, OffsetRMSNorm
from hyper_parallel.components.modules.swiglu_mlp_family import SwiGLUMLP
from hyper_parallel.components.modules.shared_expert_family import SharedExpert


class TestModuleFamilyExecution(unittest.TestCase):
    """Exercise independently usable public components with CPU autograd."""

    def test_native_components_forward_and_backward(self):
        """Direct construction executes and differentiates without a global operator runtime."""
        modules = (
            RMSNorm(config=RMSNormConfig(8), use_fused_ops=False),
            OffsetRMSNorm(config=RMSNormConfig(8), use_fused_ops=False),
            SwiGLUMLP(config=SwiGLUMLPConfig(8, 12)),
            SharedExpert(config=SharedExpertConfig(8, 12, swiglu_limit=0.3)),
            SharedExpert(config=SharedExpertConfig(8, 12, swiglu_limit=0.3, activation_dtype="input")),
        )
        for module in modules:
            with self.subTest(module=type(module).__name__):
                values = torch.randn(3, 8, requires_grad=True)
                module(values).square().mean().backward()
                self.assertTrue(torch.isfinite(values.grad).all())
                self.assertTrue(all(parameter.grad is not None for parameter in module.parameters()))
