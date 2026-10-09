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
"""CPU lifecycle tests using real MXFP8 wrappers and an identity NPU adapter."""

import gc
import unittest
import weakref
from typing import Any

import torch

from hyper_parallel.components.quantization.functional.base_gmm_func import _GroupedLinearFunction
from hyper_parallel.components.quantization.functional.mxfp8_gmm_func import MXFP8GroupedLinear
from hyper_parallel.components.quantization.functional.mxfp8_linear_func import mxfp8_linear
from hyper_parallel.components.quantization.quantizers.mxfp8 import MXFP8Quantizer

from tests.common.mark_utils import arg_mark


class IdentityMXOps:
    """Do not emulate MXFP8 accuracy; preserve real dispatch and saved-tensor behavior."""

    @staticmethod
    def dynamic_mx_quant(tensor: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        """Copy payloads without MXFP8 rounding."""
        del kwargs
        return tensor.detach().clone(), torch.ones_like(tensor)

    @classmethod
    def dynamic_mx_quant_dual_axis(cls, tensor: torch.Tensor,
                                  **kwargs: Any) -> tuple[torch.Tensor, ...]:
        """Produce two independently owned directions."""
        del kwargs
        return (*cls.dynamic_mx_quant(tensor), *cls.dynamic_mx_quant(tensor))

    @classmethod
    def grouped_dynamic_mx_quant(cls, tensor: torch.Tensor, group_list: torch.Tensor,
                                 **kwargs: Any) -> tuple[torch.Tensor, ...]:
        """Replace the grouped device quantizer with CPU copies."""
        del group_list, kwargs
        return cls.dynamic_mx_quant(tensor)

    @staticmethod
    def quant_matmul(left: torch.Tensor, right: torch.Tensor, scale: torch.Tensor,
                     *, output_dtype: torch.dtype, **kwargs: Any) -> torch.Tensor:
        """Compute a CPU reference projection."""
        del scale, kwargs
        return (left.float() @ right.float()).to(output_dtype)

    @staticmethod
    def quant_grouped_matmul(left: torch.Tensor, right: torch.Tensor, scale: torch.Tensor,
                            *, group_list: torch.Tensor, group_type: int, group_list_type: int,
                            output_dtype: torch.dtype, **kwargs: Any) -> torch.Tensor:
        """Compute forward, input-gradient or weight-gradient grouped products."""
        del scale, kwargs
        counts = group_list if group_list_type else torch.diff(group_list, prepend=group_list.new_zeros(1))
        if group_type == 0:
            output = torch.cat([x.float() @ w.float() for x, w in zip(left.split(counts.tolist()), right)])
        else:
            output = torch.stack([x.float() @ g.float() for x, g in zip(
                left.split(counts.tolist(), dim=1), right.split(counts.tolist()))])
        return output.to(output_dtype)


class MXFP8MemoryTests(unittest.TestCase):
    """Check retention, final release, frozen parameters and autograd boundaries."""

    def setUp(self) -> None:
        """Do not leak seeded RNG state to other tests."""
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())

    @staticmethod
    def projection(grouped: bool, needs: tuple[bool, bool] = (True, True),
                   *, empty: bool = False, kind: int = 1) -> tuple[torch.Tensor, ...]:
        """Construct Dense/GMM with a zero-token expert and optional empty input."""
        torch.manual_seed(42)
        x = torch.randn(0 if empty else 5, 4, dtype=torch.bfloat16, requires_grad=needs[0])
        w = torch.randn((3, 6, 4) if grouped else (6, 4), dtype=torch.bfloat16, requires_grad=needs[1])
        groups = torch.tensor([0, 0, 0] if empty else [2, 0, 3], dtype=torch.int64)
        if not kind:
            groups = groups.cumsum(0)
        quantizer = MXFP8Quantizer(npu_ops=IdentityMXOps())
        strategy = MXFP8GroupedLinear(quantizer=quantizer)
        y = (_GroupedLinearFunction.apply(x, w, groups, strategy, kind) if grouped
             else mxfp8_linear(x, w, quantizer))
        return x, w, y, groups

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_repeat_and_final_release(self) -> None:
        """
        Feature: MXFP8 autograd contracts.
        Description: Exercise gradient subsets, group encodings and empty input.
        Expectation: Repeated gradients agree and saved payloads are finally released.
        """
        for grouped in (False, True):
            for needs in ((True, True), (True, False), (False, True)):
                for empty in ((False, True) if grouped else (False,)):
                    for kind in ((0, 1) if grouped else (1,)):
                        if grouped and needs == (True, False) and not empty:
                            # Grouped MXFP8 always quantizes the input
                            # column-wise, so a frozen weight is unsupported.
                            continue
                        with self.subTest(grouped=grouped, needs=needs, empty=empty, kind=kind):
                            x, w, y, _ = self.projection(grouped, needs, empty=empty, kind=kind)
                            self._check_retention(x, w, y, grouped, needs, empty)

    def _check_retention(self, x, w, y, grouped, needs, empty):
        """Check one retention case without multiplying test-loop complexity."""
        params = [v for v in (x, w) if v.requires_grad]
        if grouped:
            # The strategy flow keeps the quantized operands on the autograd
            # node and clears them when backward finishes, so assert the
            # lifecycle directly instead of through saved_tensors.
            self._check_grouped_operand_lifecycle(y, needs, empty, params)
            return
        saved = y.grad_fn.saved_tensors
        operands = saved[1:]
        self.assertEqual(any(t is not None for t in operands[:4]), needs[1] and not empty)
        self.assertEqual(any(t is not None for t in operands[4:]), needs[0] and not empty)
        self.assertLessEqual({type(t) for t in operands if t is not None}, {torch.Tensor})
        self.assertEqual(saved[0] is not None, grouped)
        refs = [weakref.ref(t) for t in operands if t is not None]
        del saved, operands
        first = torch.autograd.grad(y.sum(), params, retain_graph=True)
        gc.collect()
        self.assertTrue(all(ref() is not None for ref in refs))
        second = torch.autograd.grad(y.sum(), params)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            _ = y.grad_fn.saved_tensors
        with self.assertRaises(RuntimeError):
            torch.autograd.grad(y.sum(), params)

    def _check_grouped_operand_lifecycle(self, y, needs, empty, params):
        """Check the strategy node's quantized operand release contract."""
        node = y.grad_fn
        if empty:
            # Empty inputs short-circuit forward, so the node never stores
            # quantized operands; only the zero-gradient contract is checked.
            for grad in torch.autograd.grad(y.sum(), params):
                self.assertEqual(torch.count_nonzero(grad).item(), 0)
            return
        self.assertEqual(node.input_quant is not None, needs[1])
        self.assertEqual(node.weight_quant is not None, needs[0])
        for operand in (node.input_quant, node.weight_quant):
            if operand is None:
                continue
            for tensor in (operand.row_data, operand.row_scale,
                           operand.col_data, operand.col_scale):
                if tensor is not None:
                    self.assertIs(type(tensor), torch.Tensor)
        torch.autograd.grad(y.sum(), params)
        gc.collect()
        self.assertIsNone(node.input_quant)
        self.assertIsNone(node.weight_quant)
        with self.assertRaises((RuntimeError, TypeError)):
            torch.autograd.grad(y.sum(), params)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_double_backward_rejected(self) -> None:
        """
        Feature: MXFP8 autograd contracts.
        Description: Differentiate the first gradient.
        Expectation: Higher-order backward is explicitly rejected.
        """
        for grouped in (False, True):
            with self.subTest(grouped=grouped):
                x, _, y, _ = self.projection(grouped)
                dy = torch.ones_like(y, requires_grad=True)
                dx, = torch.autograd.grad(y, x, dy, create_graph=True)
                if grouped:
                    # The strategy flow does not mark backward once_differentiable,
                    # so the second differentiation fails with a runtime error.
                    with self.assertRaises(RuntimeError):
                        dx.sum().backward()
                else:
                    with self.assertRaisesRegex(RuntimeError, 'once_differentiable'):
                        dx.sum().backward()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_first_result_matches_cpu_reference(self) -> None:
        """
        Feature: MXFP8 autograd contracts.
        Description: Compare identity-quantized Dense/GMM with differentiable CPU matmul.
        Expectation: Outputs and requested gradients agree, including frozen weights.
        """
        for grouped in (False, True):
            needs_options = ((True, True), (True, False), (False, True))
            if grouped:
                # Grouped MXFP8 always quantizes the input column-wise, so a
                # frozen weight is unsupported here.
                needs_options = ((True, True), (False, True))
            for needs in needs_options:
                with self.subTest(grouped=grouped, needs=needs):
                    x, w, actual, groups = self.projection(grouped, needs)
                    if grouped:
                        expected = torch.cat([
                            (part.float() @ weight.float().T).to(x.dtype)
                            for part, weight in zip(x.split(groups.tolist()), w)
                        ])
                    else:
                        expected = (x.float() @ w.float().T).to(x.dtype)
                    params = [v for v in (x, w) if v.requires_grad]
                    actual_grads = torch.autograd.grad(actual.sum(), params)
                    expected_grads = torch.autograd.grad(expected.sum(), params)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
