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

"""MOME composition contracts over the selectable aggregate-hidden primitive."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.components import ops as operators
from hyper_parallel.components.functional.mome import mome
from hyper_parallel.components.ops.impl import aggregate_hidden as implementation
from tests.common.op_precision import check_case
from tests.common.op_precision_experts import mome_cases


class TestMome(unittest.TestCase):
    """Validate the public composition without introducing a separate operator family."""

    def setUp(self) -> None:
        """Restore the process configuration after every comparison."""
        self.addCleanup(operators.ops.set_backends, operators.ops.get_backends())
        operators.ops.set_backends({"aggregate_hidden": "native"})

    def test_native_forward_and_gradients(self) -> None:
        """Compare masks, bias and all input/parameter VJPs against independent formulas."""
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for case in mome_cases(dtype):
                with self.subTest(dtype=dtype, case=case.name):
                    check_case(case)

    def test_bias_follows_aggregation_output_rounding(self) -> None:
        """Low-precision bias is added after the aggregate result has restored the input dtype."""
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                value = 1.0 + torch.finfo(dtype).eps
                x = torch.tensor([value, 0.0], dtype=dtype).reshape(1, 2, 1)
                weight = torch.tensor([value, 0.0], dtype=dtype).reshape(1, 1, 2)
                bias = -torch.tensor([value * value], dtype=dtype)
                result = mome(x, torch.ones(1, 2, dtype=torch.bool), weight, bias)
                self.assertEqual(result[0, 1, 0].item(), 0)

    def test_mask_retains_residual_and_source_history(self) -> None:
        """A masked token remains unchanged but contributes to later convolution updates."""
        x = torch.tensor([1.0, 2.0, 3.0]).reshape(1, 3, 1).requires_grad_()
        weight = torch.tensor([10.0, 1.0]).reshape(1, 1, 2).requires_grad_()
        result = mome(x, torch.tensor([[True, False, True]]), weight)
        torch.testing.assert_close(result.flatten(), torch.tensor([2.0, 2.0, 26.0]))
        result.sum().backward()
        torch.testing.assert_close(x.grad.flatten(), torch.tensor([2.0, 11.0, 2.0]))
        torch.testing.assert_close(weight.grad.flatten(), torch.tensor([2.0, 4.0]))

    def test_backend_switch_is_controlled_by_aggregate_hidden(self) -> None:
        """Functional composition selects through the public primitive for every call."""
        custom = SimpleNamespace(npu_aggregate_hidden=Mock(), npu_aggregate_hidden_grad=Mock())
        kernel = Mock(side_effect=implementation.native_aggregate_hidden)
        fake_device = property(lambda _tensor: torch.device("npu:0"))
        for backend in ("native", "omni", "native"):
            operators.ops.set_backends({"aggregate_hidden": backend})
            x = torch.ones(1, 3, 2, dtype=torch.bfloat16, requires_grad=True)
            weight = torch.ones(2, 1, 1, dtype=torch.bfloat16, requires_grad=True)
            bias = torch.ones(2, dtype=torch.bfloat16, requires_grad=True)
            mask = torch.tensor([[True, False, True]])
            with patch.object(torch.Tensor, "device", fake_device), \
                    patch.object(implementation, "require_package"), patch.object(torch.ops, "custom", custom), \
                    patch.object(implementation._AggregateHidden, "apply", kernel):  # pylint: disable=protected-access
                output = mome(x, mask, weight, bias)
                output.sum().backward()
            torch.testing.assert_close(output.flatten(), torch.tensor([3, 3, 1, 1, 3, 3], dtype=x.dtype))
            torch.testing.assert_close(x.grad.flatten(), torch.tensor([2, 2, 1, 1, 2, 2], dtype=x.dtype))
            torch.testing.assert_close(weight.grad, torch.full_like(weight, 2))
            torch.testing.assert_close(bias.grad, torch.full_like(bias, 2))
            self.assertEqual(operators.ops.used_backends(), (("aggregate_hidden", "standard", backend),))
        kernel.assert_called_once()

    def test_mome_has_no_operator_family_or_export(self) -> None:
        """Only the underlying primitive accepts backend configuration."""
        self.assertNotIn("mome", operators.__all__)
        self.assertFalse(hasattr(operators, "mome"))
        with self.assertRaisesRegex(ValueError, "Unknown operator family"):
            operators.ops.set_backends({"mome": "native"})

    def test_compiled_forward_and_gradients(self) -> None:
        """AOT eager tracing preserves primitive selection and the composition's gradients."""
        case = mome_cases(torch.float32)[0]
        x, mask, weight, bias = case.args
        expected = case.reference(*case.args)
        actual = torch.compile(mome, backend="aot_eager")(x, mask, weight, bias)
        torch.testing.assert_close(actual, expected)
        actual_grads = torch.autograd.grad(actual.sum(), (x, weight))
        expected_grads = torch.autograd.grad(expected.sum(), (x, weight))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_invalid_composition_shapes(self) -> None:
        """Reject mismatched masks, non-depthwise filters, empty filters and invalid biases."""
        x, mask, weight, _ = mome_cases(torch.float32)[0].args
        for args in ((x[0], mask, weight), (x, mask[:, :1], weight), (x, mask, weight.squeeze(1)),
                     (x, mask, weight[..., :0]), (x, mask, weight, torch.ones(1))):
            with self.subTest(shapes=[arg.shape for arg in args]), self.assertRaisesRegex(ValueError, "mome"):
                mome(*args)
