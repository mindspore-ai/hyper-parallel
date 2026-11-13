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
"""Exact receive permutations and gradients, independent of expert computation."""
from functools import partial
import unittest

import torch

from hyper_parallel.core.expert_parallel.expert_parallel import _permute, _unpermute
from hyper_parallel.core.expert_parallel.row_permutation import permute_rows


class TestReceiveRowPermutation(unittest.TestCase):
    """A bijection must preserve rows and gradients without reduction."""

    def test_mapping_and_strided_gradient(self):
        """
        Feature: mapping and strided gradient
        Description: Cover several EP sizes, empty owners and vector/multidimensional payloads.
        Expectation: The documented output, gradient and state assertions pass.
        """
        for ep in (2, 4, 8):
            for empty in (False, True):
                counts = torch.tensor([(i * 7 + 2) % 5 for i in range(ep * 3)])
                if empty:
                    counts.zero_()
                rows = int(counts.sum())
                starts = (counts.cumsum(0) - counts).tolist()
                expected_rows = []
                for expert in range(3):
                    for rank in range(ep):
                        block = rank * 3 + expert
                        expected_rows.extend(range(starts[block], starts[block] + int(counts[block])))
                expected_rows = torch.tensor(expected_rows, dtype=torch.long)
                for features in ((), (5,), (2, 3)):
                    with self.subTest(ep=ep, empty=empty, features=features):
                        values = torch.randn((rows, *features), dtype=torch.float64, requires_grad=True)
                        shape, actual, indices, _, inverse = _permute(values, counts, ep, 3)
                        self.assertTrue(torch.equal(indices, expected_rows))
                        self.assertTrue(torch.equal(actual, values[expected_rows]))
                        restored = _unpermute(actual, shape, indices, inverse)
                        self.assertTrue(torch.equal(restored, values))
                        gradient = torch.randn((*actual.shape, 2), dtype=actual.dtype)[..., 0]
                        actual.backward(gradient)
                        expected_gradient = torch.empty_like(values)
                        for destination, source in enumerate(expected_rows.tolist()):
                            expected_gradient[source] = gradient[destination]
                        self.assertTrue(torch.equal(values.grad, expected_gradient))

    def test_inverse_and_higher_derivatives(self):
        """
        Feature: inverse and higher derivatives
        Description: Keep the generic dispatcher differentiable through the inverse map.
        Expectation: The documented output, gradient and state assertions pass.
        """
        values = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        indices = torch.tensor([3, 0, 4, 1, 2])
        inverse = torch.tensor([1, 3, 4, 0, 2])
        function = partial(permute_rows, indices=inverse, inverse=indices)
        self.assertTrue(torch.autograd.gradcheck(function, (values,)))
        self.assertTrue(torch.autograd.gradgradcheck(function, (values,)))
