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
"""Arithmetic contract of the standalone parallelism-invariant reference operators."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from hyper_parallel.components.parallelism_invariant.row_parallel_linear import PikContext, PikPlan, linear
from hyper_parallel.components.parallelism_invariant.row_parallel_linear.gemm import batched_leaves
from hyper_parallel.components.parallelism_invariant.row_parallel_linear.reduction import fixed_tree


class TestPik(unittest.TestCase):
    """Validate contracts without initializing an NPU or a process group."""

    def test_plan_and_rank_alignment(self):
        """Each supported TP owns a power-of-two complete subtree."""
        plan = PikPlan(8)
        for world in (1, 2, 4, 8):
            plan.validate(4096, world)
        for leaves in (0, -1, 3, True):
            with self.assertRaises(ValueError):
                PikPlan(leaves)
        for k, world in ((4097, 2), (4096, 3), (4096, 16)):
            with self.assertRaises(ValueError):
                plan.validate(k, world)
        with self.assertRaises(ValueError):
            PikPlan(8, "auto")
        self.assertEqual(plan.contract()["tree_dtype"], "float32")

    def test_leaf_layouts_preserve_logical_k_intervals(self):
        """Strided and packed BMM inputs describe exactly the same consecutive leaves."""
        x = torch.arange(3 * 32, dtype=torch.float32).reshape(3, 32)[:, ::2]
        weight = torch.arange(5 * 32, dtype=torch.float32).reshape(5, 32)[:, ::2]
        expected = torch.stack([x[:, i * 4:(i + 1) * 4] @ weight[:, i * 4:(i + 1) * 4].T
                                for i in range(4)])
        for layout in ("strided", "packed"):
            result = batched_leaves(x, weight, plan=PikPlan(8, layout), local_leaves=4)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_tree_fake_metadata_and_single_leaf(self):
        """The compiler sees FP32 internal nodes without loading Triton."""
        with FakeTensorMode():
            output = fixed_tree(torch.empty(8, 7, 13, dtype=torch.bfloat16))
            self.assertEqual(output.shape, (7, 13))
            self.assertEqual(output.dtype, torch.float32)
        value = torch.randn(1, 3, 5).bfloat16()
        torch.testing.assert_close(fixed_tree(value), value[0].float(), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            fixed_tree(torch.empty(3, 2, 4))

    def test_input_contract_rejects_shape_and_axis_errors(self):
        """Invalid local geometry fails before any data collective."""
        device = SimpleNamespace(type="npu", index=0)
        context = PikContext(PikPlan(), 64, 16, None, 2, 0, 0, True, 1, "all_to_all")
        value = SimpleNamespace(shape=(2, 6, 32), ndim=3, dtype=torch.bfloat16, device=device)
        weight = SimpleNamespace(shape=(16, 32), ndim=2, dtype=torch.bfloat16, device=device)
        self.assertEqual(context.token_axis(value, weight, None), 1)
        value.shape = (2, 5, 32)
        with self.assertRaisesRegex(ValueError, "SP sequence"):
            context.token_axis(value, weight, None)
        value.shape = (2, 6, 31)
        with self.assertRaisesRegex(ValueError, "local shapes"):
            context.token_axis(value, weight, None)

    def test_forward_root_cast_precedes_bias(self):
        """Rank roots stay FP32 until the global root, then BF16 rounding precedes bias."""
        context = PikContext(PikPlan(), 8, 2, None, 1, 0, 0)
        value, weight = torch.ones(3, 8), torch.ones(2, 8)
        root = torch.tensor([[1.003, -1.003]]).expand(3, 2)
        bias = torch.tensor([0.001, -0.001])
        with patch.object(linear, "local_gemm", return_value=root):
            actual = linear._forward(value, weight, bias, context, 0)
        expected = root.bfloat16() + bias.bfloat16()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(expected, (root + bias).bfloat16()))
