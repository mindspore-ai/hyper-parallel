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
"""CPU reference checks for route ownership, ordered reduction and gradients."""

import unittest

import torch

from rl.unified_execution.ops.moe.combine import ordered_combine, owner_routes


class TestMoECombine(unittest.TestCase):
    """Check mathematical results independently of NPU or distributed setup."""

    def test_owner_selection_does_not_sum_other_ranks(self) -> None:
        received = torch.full((3, 2, 2, 4), float("nan"))
        ids = torch.tensor([[0, 3], [5, -1]])
        expected = torch.tensor([[[1.] * 4, [2.] * 4], [[3.] * 4, [0.] * 4]])
        received[0, 0, 0] = 1
        received[1, 0, 1] = 2
        received[2, 1, 0] = 3
        torch.testing.assert_close(owner_routes(received, ids, 2), expected, rtol=0, atol=0)

    def test_ordered_combine_autograd(self) -> None:
        """Compare input and router-weight derivatives with a simple linear expert bank."""
        x = torch.tensor([[2., 3.], [4., 5.]], requires_grad=True)
        ids = torch.tensor([[0, 1], [1, 0]])
        weights = torch.tensor([[.25, .75], [.6, .4]], requires_grad=True)

        def experts(values: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
            """Independent per-expert linear map with a known derivative."""
            return values * (selected[:, None] + 1)

        routes = experts(x.repeat_interleave(2, 0), ids.flatten()).view(2, 2, 2)
        result = ordered_combine(routes, weights)
        expected = x * ((ids + 1) * weights).sum(-1, keepdim=True)
        torch.testing.assert_close(result, expected)
        actual_grads = torch.autograd.grad(result.sum(), (x, weights), retain_graph=True)
        expected_grads = torch.autograd.grad(expected.sum(), (x, weights))
        for actual, reference in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(actual, reference)

    def test_slot_order_and_final_rounding(self) -> None:
        routes = torch.tensor([[[1e8], [1.], [-1e8], [3.]]])
        self.assertEqual(ordered_combine(routes, torch.ones(1, 4)).item(), 3.)
        with self.assertRaisesRegex(ValueError, "combine requires"):
            ordered_combine(routes, torch.ones(1, 3))

    def test_bfloat16_cast_happens_after_fp32_accumulation(self) -> None:
        """A small weighted contribution survives cancellation before the final cast."""
        routes = torch.tensor([[[256.], [1.], [-256.]]], dtype=torch.bfloat16)
        weights = torch.tensor([[1., .001, 1.]], dtype=torch.float32)
        expected = (torch.tensor(256., dtype=torch.float32) + .001 - 256.).bfloat16()
        actual = ordered_combine(routes, weights)
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual.flatten()[0], expected, rtol=0, atol=0)
        self.assertNotEqual(actual.item(), 0)
