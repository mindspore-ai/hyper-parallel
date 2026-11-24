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
"""CPU regressions for the small-op grouped GEMM reference."""

import unittest

import torch

from hyper_parallel.components.ops import grouped_matmul, ops


class TestGroupedMatmulReference(unittest.TestCase):
    """Verify grad propagation for expert-major rows and unused experts."""

    def setUp(self) -> None:
        """Pin the baseline independently of other tests' process-wide settings."""
        self.addCleanup(ops.set_backends, ops.get_backends())
        ops.set_backends({"grouped_matmul": "native"})

    def test_reference_keeps_empty_expert_gradients(self) -> None:
        """Include an all-empty batch and a nonempty batch with unused experts."""
        for counts in ([0, 0, 0], [2, 0, 3]):
            with self.subTest(counts=counts):
                counts_tensor = torch.tensor(counts)
                x = torch.randn(sum(counts), 4, requires_grad=True)
                weight = torch.randn(3, 4, 6, requires_grad=True)
                expert_ids = torch.arange(3).repeat_interleave(counts_tensor)
                expected = torch.einsum("tk,tkn->tn", x, weight[expert_ids])

                result = grouped_matmul(x, weight, counts_tensor.cumsum(0))
                gradient = torch.randn_like(expected)
                actual_grads = torch.autograd.grad(result, (x, weight), gradient)
                expected_grads = torch.autograd.grad(expected, (x, weight), gradient)

                torch.testing.assert_close(result, expected)
                for actual, reference in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual, reference)
