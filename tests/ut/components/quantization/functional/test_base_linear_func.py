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
"""CPU contracts for the shared Dense low-precision strategy lifecycle."""

import inspect
import unittest
from unittest import mock

import torch
from torch.nn import functional

import hyper_parallel.components.quantization.functional.hifloat8_linear_func as hifloat8_impl
import hyper_parallel.components.quantization.functional.mxfp8_linear_func as mxfp8_impl
from hyper_parallel.components.quantization.functional.base_linear_func import (
    LinearStrategy,
    _LinearFunction,
)
from hyper_parallel.components.quantization.functional._saved_quantized import (
    restore_quantized_operands,
    save_quantized_operands,
)
from hyper_parallel.components.quantization.tensor import QuantizedTensorStorage
from hyper_parallel.components.quantization.tensor.hifloat8_tensor import HiFloat8Tensor
from tests.common.mark_utils import arg_mark


cpu_test = arg_mark(
    plat_marks=["cpu_linux", "cpu_macos"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)


class _DenseStorage(QuantizedTensorStorage):
    """Retain a full-precision tensor while recording directional releases."""

    def __init__(
        self,
        tensor: torch.Tensor,
        *,
        role: str,
        rowwise: bool,
        colwise: bool,
    ) -> None:
        """Store one logical value and its available directions."""

        self.tensor = tensor
        self.role = role
        self.rowwise = rowwise
        self.colwise = colwise
        self.usage_updates: list[tuple[bool, bool]] = []

    def update_usage(self, rowwise: bool = True, colwise: bool = True) -> None:
        """Record and apply one directional retention request."""

        self.usage_updates.append((rowwise, colwise))
        self.rowwise = rowwise
        self.colwise = colwise

    def is_rowwise(self) -> bool:
        """Return whether the row view remains available."""

        return self.rowwise

    def is_colwise(self) -> bool:
        """Return whether the column view remains available."""

        return self.colwise

    def get_metadata(self) -> dict[str, object]:
        """Return the values needed by assertions in this test."""

        return {
            "role": self.role,
            "rowwise": self.rowwise,
            "colwise": self.colwise,
        }


class _DenseLinearStrategy(LinearStrategy):
    """Use full-precision CPU tensors behind the production strategy seam."""

    def __init__(self) -> None:
        """Create empty call and storage traces."""

        self.quantize_calls: list[tuple[str, bool, bool, tuple[int, ...]]] = []
        self.matmul_calls: list[tuple[str, str, str, torch.dtype]] = []
        self.storages: list[_DenseStorage] = []

    def _quantize(
        self,
        role: str,
        tensor: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
    ) -> _DenseStorage:
        self.quantize_calls.append((role, rowwise, colwise, tuple(tensor.shape)))
        storage = _DenseStorage(
            tensor,
            role=role,
            rowwise=rowwise,
            colwise=colwise,
        )
        self.storages.append(storage)
        return storage

    def quantize_input(
        self,
        inputs: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
    ) -> _DenseStorage:
        """Wrap the input and record its requested directions."""

        return self._quantize(
            "input",
            inputs,
            rowwise=rowwise,
            colwise=colwise,
        )

    def quantize_weight(
        self,
        weight: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
    ) -> _DenseStorage:
        """Wrap the weight and record its requested directions."""

        return self._quantize(
            "weight",
            weight,
            rowwise=rowwise,
            colwise=colwise,
        )

    def quantize_grad_output(
        self,
        grad_output: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
    ) -> _DenseStorage:
        """Wrap the gradient output and record its requested directions."""

        return self._quantize(
            "grad_output",
            grad_output,
            rowwise=rowwise,
            colwise=colwise,
        )

    def matmul(
        self,
        left: _DenseStorage,
        right: _DenseStorage,
        *,
        layout: str,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Execute the requested layout with ordinary CPU matrix multiply."""

        self.matmul_calls.append((layout, left.role, right.role, output_dtype))
        left_tensor = left.tensor.transpose(-1, -2) if layout[0] == "T" else left.tensor
        right_tensor = right.tensor.transpose(-1, -2) if layout[1] == "T" else right.tensor
        return (left_tensor @ right_tensor).to(output_dtype)


class _DenseQuantizer:
    """Return CPU storage while retaining the concrete strategy call trace."""

    def __init__(self, *roles: str) -> None:
        """Bind the logical role expected for each quantization call."""

        self._roles = iter(roles)
        self.calls: list[tuple[str, bool, bool, tuple[int, ...]]] = []

    def quantize(
        self,
        tensor: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
    ) -> _DenseStorage:
        """Wrap a tensor and record the directions requested by production code."""

        role = next(self._roles)
        self.calls.append((role, rowwise, colwise, tuple(tensor.shape)))
        return _DenseStorage(
            tensor,
            role=role,
            rowwise=rowwise,
            colwise=colwise,
        )


def _dense_matmul(
    left: _DenseStorage,
    right: _DenseStorage,
    *,
    layout: str,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Emulate the NPU MM adapter for concrete-strategy CPU contracts."""

    left_tensor = left.tensor.transpose(-1, -2) if layout[0] == "T" else left.tensor
    right_tensor = right.tensor.transpose(-1, -2) if layout[1] == "T" else right.tensor
    return (left_tensor @ right_tensor).to(output_dtype)


class TestLinearStrategyFlow(unittest.TestCase):
    """The base strategy owns one complete Dense forward/backward lifecycle."""

    @cpu_test
    def test_saved_quantized_keeps_physical_operands(self):
        """The compatibility helper saves payloads, not only wrapper objects."""

        class _Context:
            def save_for_backward(self, *tensors):
                self.saved_tensors = tensors

        quantizer = object()
        row_data = torch.ones(2, 2, dtype=torch.uint8)
        row_scale = torch.ones(2, 1, dtype=torch.float32)
        col_data = torch.zeros(2, 2, dtype=torch.uint8)
        col_scale = torch.ones(1, 2, dtype=torch.float32)
        operand = HiFloat8Tensor(
            (2, 2),
            torch.bfloat16,
            quantizer=quantizer,
            row_data=row_data,
            row_scale=row_scale,
            col_data=col_data,
            col_scale=col_scale,
        )
        group_list = torch.tensor([2, 0], dtype=torch.int64)
        context = _Context()

        save_quantized_operands(context, operand, None, group_list)
        restored_group, restored_input, restored_weight = restore_quantized_operands(context)

        self.assertIs(restored_group, group_list)
        self.assertIsNone(restored_weight)
        self.assertIsInstance(restored_input, HiFloat8Tensor)
        self.assertIs(restored_input.quantizer, quantizer)
        self.assertIs(restored_input.row_data, row_data)
        self.assertIs(restored_input.col_data, col_data)

    @cpu_test
    def test_shared_flow_matches_dense_for_leading_dimensions(self):
        """The common bridge preserves output, gradients, layouts, and releases."""

        inputs = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
        weight = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
        grad_output = torch.randn(2, 3, 5, dtype=torch.float64)
        reference_inputs = inputs.detach().clone().requires_grad_(True)
        reference_weight = weight.detach().clone().requires_grad_(True)
        reference_output = functional.linear(reference_inputs, reference_weight)
        reference_output.backward(grad_output)
        strategy = _DenseLinearStrategy()

        output = _LinearFunction.apply(inputs, weight, strategy)
        output.backward(grad_output)

        torch.testing.assert_close(output, reference_output)
        torch.testing.assert_close(inputs.grad, reference_inputs.grad)
        torch.testing.assert_close(weight.grad, reference_weight.grad)
        self.assertEqual(
            strategy.quantize_calls,
            [
                ("input", True, True, (6, 4)),
                ("weight", True, True, (5, 4)),
                ("grad_output", True, True, (6, 5)),
            ],
        )
        self.assertEqual(
            [call[:3] for call in strategy.matmul_calls],
            [
                ("NT", "input", "weight"),
                ("NN", "grad_output", "weight"),
                ("TN", "grad_output", "input"),
            ],
        )
        self.assertEqual(strategy.storages[0].usage_updates, [(False, True)])
        self.assertEqual(strategy.storages[1].usage_updates, [(False, True)])
        self.assertEqual(strategy.storages[2].usage_updates, [(False, False)])

    @cpu_test
    def test_shared_flow_builds_only_required_backward_directions(self):
        """requires_grad controls quantization directions and MM branches."""

        cases = (
            (
                "input_only",
                True,
                False,
                [("input", True, False), ("weight", True, True), ("grad_output", True, False)],
                ["NT", "NN"],
            ),
            (
                "weight_only",
                False,
                True,
                [("input", True, True), ("weight", True, False), ("grad_output", False, True)],
                ["NT", "TN"],
            ),
        )
        for name, need_input, need_weight, expected_quant, expected_layouts in cases:
            with self.subTest(name=name):
                inputs = torch.randn(3, 4, dtype=torch.float64, requires_grad=need_input)
                weight = torch.randn(5, 4, dtype=torch.float64, requires_grad=need_weight)
                strategy = _DenseLinearStrategy()

                _LinearFunction.apply(inputs, weight, strategy).sum().backward()

                self.assertEqual(
                    [call[:3] for call in strategy.quantize_calls],
                    expected_quant,
                )
                self.assertEqual(
                    [call[0] for call in strategy.matmul_calls],
                    expected_layouts,
                )

    @cpu_test
    def test_concrete_strategies_implement_format_hooks(self):
        """Both concrete classes are complete and route each role correctly."""

        self.assertTrue(issubclass(mxfp8_impl.MXFP8LinearStrategy, LinearStrategy))
        self.assertTrue(issubclass(hifloat8_impl.HiFloat8LinearStrategy, LinearStrategy))
        self.assertFalse(inspect.isabstract(mxfp8_impl.MXFP8LinearStrategy))
        self.assertFalse(inspect.isabstract(hifloat8_impl.HiFloat8LinearStrategy))
        self.assertFalse(hasattr(mxfp8_impl, "_MXFP8LinearFunction"))
        self.assertFalse(hasattr(hifloat8_impl, "_HiFloat8LinearFunction"))

        tensor = torch.ones(2, 2)
        mxfp_quantizer = mock.Mock()
        mxfp_quantizer.quantize.side_effect = ("mx-input", "mx-weight", "mx-grad")
        mxfp_strategy = mxfp8_impl.MXFP8LinearStrategy(mxfp_quantizer)
        self.assertEqual(mxfp_strategy.quantize_input(tensor, rowwise=True, colwise=False), "mx-input")
        self.assertEqual(mxfp_strategy.quantize_weight(tensor, rowwise=True, colwise=True), "mx-weight")
        self.assertEqual(mxfp_strategy.quantize_grad_output(tensor, rowwise=False, colwise=True), "mx-grad")
        self.assertEqual(mxfp_quantizer.quantize.call_count, 3)
        left = object()
        right = object()
        with mock.patch.object(mxfp8_impl, "mxfp8_matmul", return_value=tensor) as matmul:
            result = mxfp_strategy.matmul(
                left,
                right,
                layout="NT",
                output_dtype=torch.float32,
            )
        self.assertIs(result, tensor)
        matmul.assert_called_once_with(
            left,
            right,
            layout="NT",
            output_dtype=torch.float32,
        )

        input_quantizer = mock.Mock()
        weight_quantizer = mock.Mock()
        grad_quantizer = mock.Mock()
        input_quantizer.quantize.return_value = "hif-input"
        weight_quantizer.quantize.return_value = "hif-weight"
        grad_quantizer.quantize.return_value = "hif-grad"
        hif_strategy = hifloat8_impl.HiFloat8LinearStrategy(
            input_quantizer,
            weight_quantizer,
            grad_quantizer,
        )
        self.assertEqual(hif_strategy.quantize_input(tensor, rowwise=True, colwise=False), "hif-input")
        self.assertEqual(hif_strategy.quantize_weight(tensor, rowwise=True, colwise=True), "hif-weight")
        self.assertEqual(hif_strategy.quantize_grad_output(tensor, rowwise=False, colwise=True), "hif-grad")
        input_quantizer.quantize.assert_called_once()
        weight_quantizer.quantize.assert_called_once()
        grad_quantizer.quantize.assert_called_once()
        with mock.patch.object(hifloat8_impl, "hifloat8_matmul", return_value=tensor) as matmul:
            result = hif_strategy.matmul(
                left,
                right,
                layout="TN",
                output_dtype=torch.bfloat16,
            )
        self.assertIs(result, tensor)
        matmul.assert_called_once_with(
            left,
            right,
            layout="TN",
            output_dtype=torch.bfloat16,
        )

    @cpu_test
    def test_concrete_strategies_run_complete_shared_forward_and_backward(self):
        """Both production strategies connect their hooks to the single bridge."""

        inputs = torch.randn(2, 3, 4, dtype=torch.float64)
        weight = torch.randn(5, 4, dtype=torch.float64)
        grad_output = torch.randn(2, 3, 5, dtype=torch.float64)
        cases = []

        mxfp_quantizer = _DenseQuantizer("input", "weight", "grad_output")
        cases.append(
            (
                "mxfp8",
                mxfp8_impl.MXFP8LinearStrategy(mxfp_quantizer),
                mock.patch.object(
                    mxfp8_impl,
                    "mxfp8_matmul",
                    side_effect=_dense_matmul,
                ),
                [mxfp_quantizer],
            )
        )
        hifloat8_quantizers = (
            _DenseQuantizer("input"),
            _DenseQuantizer("weight"),
            _DenseQuantizer("grad_output"),
        )
        cases.append(
            (
                "hifloat8",
                hifloat8_impl.HiFloat8LinearStrategy(*hifloat8_quantizers),
                mock.patch.object(
                    hifloat8_impl,
                    "hifloat8_matmul",
                    side_effect=_dense_matmul,
                ),
                list(hifloat8_quantizers),
            )
        )

        for name, strategy, matmul_patch, quantizers in cases:
            with self.subTest(name=name), matmul_patch as matmul:
                actual_inputs = inputs.clone().requires_grad_(True)
                actual_weight = weight.clone().requires_grad_(True)
                expected_inputs = inputs.clone().requires_grad_(True)
                expected_weight = weight.clone().requires_grad_(True)
                expected_output = functional.linear(expected_inputs, expected_weight)
                expected_output.backward(grad_output)

                actual_output = _LinearFunction.apply(
                    actual_inputs,
                    actual_weight,
                    strategy,
                )
                actual_output.backward(grad_output)

                torch.testing.assert_close(actual_output, expected_output)
                torch.testing.assert_close(actual_inputs.grad, expected_inputs.grad)
                torch.testing.assert_close(actual_weight.grad, expected_weight.grad)
                self.assertEqual(
                    [call.kwargs["layout"] for call in matmul.call_args_list],
                    ["NT", "NN", "TN"],
                )
                self.assertEqual(
                    [call for quantizer in quantizers for call in quantizer.calls],
                    [
                        ("input", True, True, (6, 4)),
                        ("weight", True, True, (5, 4)),
                        ("grad_output", True, True, (6, 5)),
                    ],
                )

    @cpu_test
    def test_hifloat8_default_role_quantizers_use_15_15_224(self):
        """The default strategy freezes the intended recipe per tensor role."""

        strategy = hifloat8_impl.HiFloat8LinearStrategy()

        self.assertEqual(strategy.input_quantizer.fp8_max, 15.0)
        self.assertEqual(strategy.weight_quantizer.fp8_max, 15.0)
        self.assertEqual(strategy.grad_output_quantizer.fp8_max, 224.0)
        self.assertIsNot(strategy.input_quantizer, strategy.weight_quantizer)
        self.assertIsNot(strategy.input_quantizer, strategy.grad_output_quantizer)
        self.assertIsNot(strategy.weight_quantizer, strategy.grad_output_quantizer)

    @cpu_test
    def test_format_modules_expose_strategies_without_linear_wrappers(self):
        """Format modules define strategies without a second execution path."""

        cases = (
            (mxfp8_impl, "MXFP8LinearStrategy", "mxfp8_linear"),
            (hifloat8_impl, "HiFloat8LinearStrategy", "hifloat8_linear"),
        )
        for module, strategy_name, wrapper_name in cases:
            with self.subTest(module=module.__name__):
                self.assertIn(strategy_name, module.__all__)
                self.assertNotIn(wrapper_name, module.__all__)
                self.assertFalse(hasattr(module, wrapper_name))
                self.assertFalse(hasattr(module, "_LinearFunction"))


if __name__ == "__main__":
    unittest.main()
