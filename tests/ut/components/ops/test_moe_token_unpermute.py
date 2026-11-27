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

"""Native moe_token_unpermute forward and gradient precision contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components import ops as operators
from hyper_parallel.components.ops.impl import moe_token_unpermute
from tests.common.mark_utils import arg_mark
from tests.common.op_precision import check_case
from tests.common.op_precision_suite import precision_cases


class TestMoeTokenUnpermutePrecision(unittest.TestCase):
    """Verify weighted restoration precision and derivatives."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_native_forward_and_gradients(self) -> None:
        """Feature: Native weighted token unpermutation.
        Description: Compare all variants and training dtypes with independent formulas.
        Expectation: Outputs, gradients, discrete metadata and input immutability match.
        """
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for case in precision_cases("moe_token_unpermute", dtype):
                with self.subTest(dtype=dtype, variant=case.variant, case=case.name):
                    check_case(case)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_restore_preserves_fp32_weights_with_low_precision_states(self) -> None:
        """Feature: Mixed-precision weighted restoration.
        Description: Opposite expert outputs expose premature FP32 route-weight rounding.
        Expectation: FP32 weighting and both input gradients match the independent formula.
        """
        self.addCleanup(operators.ops.set_backends, operators.ops.get_backends())
        operators.ops.set_backends({"moe_token_unpermute": "native"})
        positions = torch.tensor([0, 2, 1, 3])
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                states = torch.tensor([[256, -128], [64, 32], [-256, 128], [-64, -32]], dtype=dtype,
                                      requires_grad=True)
                weights = torch.tensor([[0.5001, 0.4999], [0.5001, 0.4999]], requires_grad=True)
                saved_weights = weights.detach().clone()
                upstream = torch.tensor([[0.0313, -0.1251], [0.2, -0.3]], dtype=dtype)
                selected = states.detach()[positions.reshape_as(weights)].float()
                expected = (selected * weights.detach()[..., None]).sum(1).to(dtype)

                output = operators.moe_token_unpermute(states, positions, weights)
                torch.testing.assert_close(output, expected, atol=0, rtol=0)
                output.backward(upstream)

                expected_states_grad = torch.empty_like(states)
                expected_states_grad[positions] = (
                    weights.detach()[..., None] * upstream.float()[:, None, :]
                ).reshape_as(states).to(dtype)
                expected_weights_grad = (selected * upstream.float()[:, None, :]).sum(-1)
                torch.testing.assert_close(states.grad, expected_states_grad, atol=0, rtol=0)
                torch.testing.assert_close(weights.grad, expected_weights_grad, atol=0, rtol=0)
                torch.testing.assert_close(weights, saved_weights, atol=0, rtol=0)

                rounded = operators.moe_token_unpermute(states.detach(), positions, weights.detach().to(dtype))
                torch.testing.assert_close(rounded, torch.zeros_like(output), atol=0, rtol=0)
                self.assertFalse(torch.equal(output, rounded))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_restore_fp64_gradcheck_and_empty_routes(self) -> None:
        """Feature: Weighted restoration derivatives and empty inputs.
        Description: Check FP64 finite differences and empty low-precision routes with FP32 weights.
        Expectation: Both gradients pass gradcheck and empty gradients retain their input dtypes.
        """
        self.addCleanup(operators.ops.set_backends, operators.ops.get_backends())
        operators.ops.set_backends({"moe_token_unpermute": "native"})
        generator = torch.Generator().manual_seed(61)
        states = torch.randn(6, 4, generator=generator, dtype=torch.float64, requires_grad=True)
        weights = torch.randn(3, 2, generator=generator, dtype=torch.float64, requires_grad=True)
        positions = torch.tensor([4, 0, 2, 5, 1, 3])
        self.assertTrue(torch.autograd.gradcheck(operators.moe_token_unpermute, (states, positions, weights)))

        empty_states = torch.empty(0, 4, dtype=torch.bfloat16, requires_grad=True)
        empty_weights = torch.empty(0, 2, requires_grad=True)
        output = operators.moe_token_unpermute(empty_states, torch.empty(0, dtype=torch.int64), empty_weights)
        self.assertEqual(output.shape, (0, 4))
        self.assertEqual(output.dtype, empty_states.dtype)
        output.sum().backward()
        torch.testing.assert_close(empty_states.grad, torch.zeros_like(empty_states), atol=0, rtol=0)
        torch.testing.assert_close(empty_weights.grad, torch.zeros_like(empty_weights), atol=0, rtol=0)

    def test_npu_loader_requires_only_unpermute(self) -> None:
        """Unpermutation loads without permutation and passes route weights unchanged."""
        states = torch.randn(2, 4, dtype=torch.bfloat16)
        positions = torch.tensor([1, 0])
        weights = torch.ones(2, 1)
        kernel = Mock(return_value=states.float())
        with patch.object(moe_token_unpermute, "require_package", return_value=SimpleNamespace(
            npu_moe_token_unpermute=kernel,
        )):
            operation = moe_token_unpermute._load_npu()  # pylint: disable=protected-access
        output = operation(states, positions, weights)
        self.assertEqual(output.dtype, states.dtype)
        self.assertEqual(kernel.call_args.args[1].dtype, torch.int32)
        self.assertIs(kernel.call_args.args[2], weights)
