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

"""Native moe_token_permute forward and gradient precision contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components import ops as operators
from hyper_parallel.components.ops.impl import moe_token_permute
from tests.common.mark_utils import arg_mark
from tests.common.op_precision import check_case
from tests.common.op_precision_suite import precision_cases


class TestMoeTokenPermutePrecision(unittest.TestCase):
    """Verify every variant against independent formulas in all training dtypes."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_native_forward_and_gradients(self) -> None:
        """Feature: Native token reordering.
        Description: Compare all variants and training dtypes with independent formulas.
        Expectation: Outputs, gradients, discrete metadata and input immutability match.
        """
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for case in precision_cases("moe_token_permute", dtype):
                with self.subTest(dtype=dtype, variant=case.variant, case=case.name):
                    check_case(case)


    def test_independent_backend_selection(self) -> None:
        """Selecting an unavailable backend for either family leaves the other usable."""
        self.addCleanup(operators.ops.set_backends, operators.ops.get_backends())
        states = torch.randn(3, 4)
        indices = torch.tensor([[2, 0], [1, 2], [0, 1]])
        weights = torch.ones(3, 2)
        operators.ops.set_backends({"moe_token_permute": "native", "moe_token_unpermute": "torch_npu"})
        permuted, _, positions = operators.moe_token_permute(states, indices)
        with self.assertRaisesRegex(RuntimeError, "moe_token_unpermute"):
            operators.moe_token_unpermute(permuted, positions, weights)
        operators.ops.set_backends({"moe_token_permute": "torch_npu", "moe_token_unpermute": "native"})
        restored = operators.moe_token_unpermute(permuted, positions, weights)
        torch.testing.assert_close(restored, states * 2)
        with self.assertRaisesRegex(RuntimeError, "moe_token_permute"):
            operators.moe_token_permute(states, indices)

    def test_npu_loader_requires_only_permute(self) -> None:
        """The permutation kernel loads without an unpermutation symbol installed."""
        states = torch.randn(2, 4)
        indices = torch.tensor([[1], [0]])
        positions = torch.tensor([1, 0], dtype=torch.int32)
        kernel = Mock(return_value=(states.flip(0), positions))
        with patch.object(moe_token_permute, "require_package", return_value=SimpleNamespace(
            npu_moe_token_permute=kernel,
        )):
            operation = moe_token_permute._load_npu()  # pylint: disable=protected-access
        actual, order, inverse = operation(states, indices)
        torch.testing.assert_close(actual, states.flip(0))
        torch.testing.assert_close(order, positions.long())
        torch.testing.assert_close(inverse, positions.long())
        self.assertEqual(kernel.call_args.args[1].dtype, torch.int32)
