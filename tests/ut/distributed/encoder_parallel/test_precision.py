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
"""Precision contracts for the DeepSeek visual units and unchanged TP sum boundary."""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch
from torch import nn

from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.distributed._builder.tp_collective_lowering import TPExecutionOp
from hyper_parallel.models.deepseek_v41 import deepseek_v41_spec as registration  # pylint: disable=unused-import


class TestEncoderPrecision(TestCase):
    """Check visual FSDP selection and the native TP collective dtype."""

    def test_visual_units_are_selected_without_precision_override(self):
        """The adapter selects visual FSDP units without choosing their dtype."""
        model = nn.Module()
        model.config = SimpleNamespace(model_type="deepseek_v41")
        model.model = nn.Module()
        model.model.vision = nn.Module()
        model.model.vision.patch_embed = nn.Linear(2, 2)
        model.model.vision.blocks = nn.ModuleList([nn.Linear(2, 2)])
        model.model.vision.norm = nn.LayerNorm(2)
        model.model.aligner = nn.Linear(2, 2)
        selected = FSDP2Manager._find_adapter_declared_wrap_modules(model)  # pylint: disable=protected-access
        self.assertEqual(
            {unit.fqn for unit in selected},
            {
                "model.vision.patch_embed",
                "model.vision.blocks.0",
                "model.vision.norm",
                "model.aligner",
            },
        )

    def test_bf16_partial_sum_keeps_collective_dtype_and_preserves_grad(self):
        """BF16 inputs retain their dtype at the shared TP collective."""
        seen_dtypes = []

        def reduce_sum(tensor: torch.Tensor, unused_reduce_op: object, unused_group: object) -> torch.Tensor:
            """Capture the collective input dtype and return a scaled tensor."""
            seen_dtypes.append(tensor.dtype)
            return tensor * 2

        operation = TPExecutionOp("all_reduce", object(), 2, 0)
        value = torch.tensor([1.0, 2.0], dtype=torch.bfloat16, requires_grad=True)
        with patch(
            "hyper_parallel.distributed._builder.tp_collective_lowering."
            "communication.differentiable_all_reduce", side_effect=reduce_sum
        ):
            result = operation.execute(value)
            result.float().sum().backward()
        self.assertEqual(seen_dtypes, [torch.bfloat16])
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(value.grad, torch.full_like(value, 2)))
