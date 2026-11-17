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
"""CPU regressions for CUDA grouped expert weight views and autograd."""

from itertools import product
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.components.modules import moe
from hyper_parallel.components.ops.families import grouped_swiglu as grouped_swiglu_family


class TestGroupedExperts(unittest.TestCase):
    """Exercise the CUDA helper with native grouped matmul's CPU fallback."""

    @staticmethod
    def _make_inputs(counts, dtype=torch.float32):
        """Build reproducible local expert weights and routed token inputs."""
        generator = torch.Generator().manual_seed(7)
        shapes = ((len(counts), 32, 16), (len(counts), 16, 32), (len(counts), 32, 16), (sum(counts), 16))
        return tuple(
            (torch.randn(shape, generator=generator, dtype=dtype) * 0.1).requires_grad_()
            for shape in shapes
        )

    def test_grouped_mm_shares_weight_storage_across_pending_forwards(self) -> None:
        """Feature: CUDA grouped expert weight memory.

        Description: Keep two BF16 forwards pending and inspect native grouped matmul inputs.
        Expectation: All three projections share the original parameter storage on every call.
        """
        counts = torch.tensor([8, 0, 8])
        inputs = self._make_inputs(counts.tolist(), dtype=torch.bfloat16)
        original_weights = (inputs[0], inputs[2], inputs[1])
        native_grouped_mm = torch._grouped_mm

        with patch.object(moe.torch, "_grouped_mm", wraps=native_grouped_mm) as grouped_mm:
            outputs = [grouped_swiglu_family._cuda(inputs[3], *inputs[:3], counts) for _ in range(2)]

        self.assertEqual(grouped_mm.call_count, 6, f"Expected 6 GEMMs, got {grouped_mm.call_count}")
        for index, call in enumerate(grouped_mm.call_args_list):
            weight = call.args[1]
            original = original_weights[index % 3]
            actual_ptr = weight.untyped_storage().data_ptr()
            expected_ptr = original.untyped_storage().data_ptr()
            self.assertEqual(actual_ptr, expected_ptr, f"Expected storage {expected_ptr}, got {actual_ptr}")
            torch.testing.assert_close(call.kwargs["offs"], torch.tensor([8, 8, 16], dtype=torch.int32))

        torch.autograd.backward(outputs, [torch.ones_like(output) for output in outputs])

    def test_grouped_mm_matches_loop_forward_and_backward(self) -> None:
        """Feature: CUDA grouped expert numerical parity.

        Description: Compare grouped and loop execution with routing weights and empty experts.
        Expectation: FP32 and BF16 outputs and all input gradients match the reference.
        """
        cases = product(([16], [8, 8], [8, 0, 8]), (False, True), (torch.float32, torch.bfloat16))
        for counts_list, weighted, dtype in cases:
            with self.subTest(counts=counts_list, weighted=weighted, dtype=dtype):
                counts = torch.tensor(counts_list)
                inputs = self._make_inputs(counts_list, dtype=dtype)
                scores = (
                    torch.linspace(0.1, 0.9, sum(counts_list), dtype=dtype, requires_grad=True)
                    if weighted else None
                )
                actual_inputs = inputs + ((scores,) if weighted else ())
                reference_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in actual_inputs)
                actual = grouped_swiglu_family._cuda(inputs[3], *inputs[:3], counts, scores)
                reference_scores = reference_inputs[4] if weighted else None
                expected = grouped_swiglu_family.native_grouped_swiglu(
                    reference_inputs[3], *reference_inputs[:3], counts, reference_scores,
                )

                torch.testing.assert_close(actual, expected)
                grad_output = torch.linspace(-0.5, 0.5, actual.numel(), dtype=dtype).reshape_as(actual)
                actual_grads = torch.autograd.grad(actual, actual_inputs, grad_outputs=grad_output)
                expected_grads = torch.autograd.grad(expected, reference_inputs, grad_outputs=grad_output)
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual_grad, expected_grad)
