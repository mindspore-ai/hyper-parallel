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
"""CPU tests for grouped-linear strategy dispatch and shared autograd."""

from typing import Any
import unittest
from unittest import mock

import torch

from hyper_parallel.components.quantization.config import LowPrecisionDtypeScheme
from hyper_parallel.components.quantization.functional import (
    hifloat8_gmm_func as hifloat8_impl,
)
from hyper_parallel.components.quantization.functional.base_gmm_func import (
    GroupedLinear,
    _GroupedLinearFunction,
    validate_grouped_linear_inputs,
)
from hyper_parallel.components.quantization.functional.fake_w4a8_gmm_func import (
    FakeW4A8GroupedLinear,
)
from hyper_parallel.components.quantization.functional.hifloat8_gmm_func import (
    HiFloat8GroupedLinear,
)
from hyper_parallel.components.quantization.functional.mxfp8_gmm_func import (
    MXFP8GroupedLinear,
)
from hyper_parallel.components.quantization.functional.strategy_factory import (
    build_low_precision_strategy,
)
from hyper_parallel.components.quantization.functional.w4a8_gmm_func import (
    W4A8GroupedLinear,
)
from tests.common.mark_utils import arg_mark


cpu_test = arg_mark(
    plat_marks=["cpu_linux", "cpu_macos"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)


class _DenseStorage:
    """Minimal directional storage carrying high-precision CPU data."""

    def __init__(
        self,
        value: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
    ) -> None:
        """Store one dense value and its available directional views."""

        self.value = value
        self.rowwise = rowwise
        self.colwise = colwise

    def update_usage(
        self,
        rowwise: bool = True,
        colwise: bool = True,
    ) -> None:
        """Release directional views no longer needed by the lifecycle."""

        self.rowwise = self.rowwise and rowwise
        self.colwise = self.colwise and colwise


class _RecordingQuantizer:
    """Dense quantizer double recording role-specific direction requests."""

    def __init__(self) -> None:
        """Create empty call and storage histories."""

        self.calls: list[tuple[torch.Tensor, dict[str, Any]]] = []
        self.storages: list[_DenseStorage] = []

    def quantize(self, tensor: torch.Tensor, **kwargs: Any) -> _DenseStorage:
        """Return dense storage while retaining arguments for assertions."""

        storage = _DenseStorage(
            tensor,
            rowwise=kwargs["rowwise"],
            colwise=kwargs["colwise"],
        )
        self.calls.append((tensor.detach().clone(), dict(kwargs)))
        self.storages.append(storage)
        return storage


class _DenseGroupedLinear(GroupedLinear):
    """High-precision test double that exercises the production lifecycle."""

    FORMAT_NAME = "DenseTest"

    def normalize_group_list(
        self,
        group_list: torch.Tensor,
        group_list_type: int,
    ) -> tuple[torch.Tensor, int]:
        """Keep count/offset metadata unchanged."""

        return group_list, group_list_type

    def output_features(self, weight: torch.Tensor) -> int:
        """Read output width from the live weight layout."""

        return weight.shape[-2]

    def weight_quantization_directions(
        self,
        needs_grad_input: bool,
    ) -> tuple[bool, bool]:
        """Request the same directional views as MXFP8/HiFloat8."""

        return needs_grad_input, True

    def retain_weight_backward(
        self,
        weight_quant: _DenseStorage,
        needs_grad_input: bool,
    ) -> None:
        """Retain only the view consumed by dense dgrad."""

        weight_quant.update_usage(rowwise=needs_grad_input, colwise=False)

    @staticmethod
    def _store(
        value: torch.Tensor,
        rowwise: bool,
        colwise: bool,
    ) -> _DenseStorage:
        return _DenseStorage(value, rowwise=rowwise, colwise=colwise)

    def quantize_input(
        self,
        inputs: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
        **kwargs: Any,
    ) -> _DenseStorage:
        """Wrap the dense input without changing its values."""

        del kwargs
        return self._store(inputs, rowwise, colwise)

    def quantize_weight(
        self,
        weight: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
    ) -> _DenseStorage:
        """Wrap the dense weight without changing its values."""

        return self._store(weight, rowwise, colwise)

    def quantize_grad_output(
        self,
        grad_output: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
        **kwargs: Any,
    ) -> _DenseStorage:
        """Wrap the dense output gradient without changing its values."""

        del kwargs
        return self._store(grad_output, rowwise, colwise)

    @staticmethod
    def _counts(group_list: torch.Tensor, group_list_type: int) -> list[int]:
        if group_list_type == 1:
            return group_list.tolist()
        boundaries = group_list.tolist()
        return [end - (boundaries[index - 1] if index else 0)
                for index, end in enumerate(boundaries)]

    def grouped_matmul(
        self,
        left: _DenseStorage,
        right: _DenseStorage,
        *,
        layout: str,
        group_list: torch.Tensor,
        group_type: int,
        group_list_type: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Execute grouped dense matmul for the requested layout."""

        del group_type
        counts = self._counts(group_list, group_list_type)
        left_parts = left.value.split(counts, dim=0)
        if layout == "NN":
            result = [part @ right.value[index] for index, part in enumerate(left_parts)]
            return torch.cat(result, dim=0).to(output_dtype)
        if layout == "NT":
            result = [part @ right.value[index].transpose(-2, -1)
                      for index, part in enumerate(left_parts)]
            return torch.cat(result, dim=0).to(output_dtype)
        if layout == "TN":
            right_parts = right.value.split(counts, dim=0)
            result = [
                left_part.transpose(-2, -1) @ right_part
                for left_part, right_part in zip(left_parts, right_parts)
            ]
            return torch.stack(result, dim=0).to(output_dtype)
        raise AssertionError(f"unexpected layout: {layout}")


def _dense_reference(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    """Reference grouped linear for live [E, O, K] weights."""
    parts = inputs.split(counts.tolist(), dim=0)
    return torch.cat(
        [part @ weight[index].transpose(-2, -1) for index, part in enumerate(parts)],
        dim=0,
    )


class TestStrategyFactory(unittest.TestCase):
    """Policy-to-strategy dispatch without importing an NPU runtime."""

    @staticmethod
    def _patch_probes():
        module = "hyper_parallel.components.quantization.functional.strategy_factory"
        return (
            mock.patch(f"{module}.validate_npu_gmm_runtime"),
            mock.patch(f"{module}.validate_w4a8_gmm_runtime"),
            mock.patch(f"{module}.validate_fake_w4a8_gmm_runtime"),
            mock.patch(f"{module}.validate_hifloat8_gmm_runtime"),
        )

    @cpu_test
    def test_dispatches_all_supported_grouped_linear_strategies(self):
        """Every supported policy chooses its independent strategy."""
        aligned32 = ((2, 64, 32), (2, 32, 64))
        aligned128 = ((2, 256, 128), (2, 128, 256))
        p_mx, p_native, p_fake, p_hifloat = self._patch_probes()
        with (
            p_mx as mx_probe,
            p_native as native_probe,
            p_fake as fake_probe,
            p_hifloat as hifloat_probe,
        ):
            w8a8 = build_low_precision_strategy(None, tile_shapes=aligned32)
            native = build_low_precision_strategy(
                LowPrecisionDtypeScheme(
                    weight_format="mxfp4", act_format="mxfp8", block_size=128
                ),
                tile_shapes=aligned128,
            )
            fake = build_low_precision_strategy(
                LowPrecisionDtypeScheme(
                    is_fake_quantize=True,
                    weight_format="mxfp4",
                    act_format="mxfp8",
                    block_size=32,
                ),
                tile_shapes=aligned32,
            )
            hifloat = build_low_precision_strategy(
                LowPrecisionDtypeScheme(
                    weight_format="hif8",
                    act_format="hif8",
                ),
                # HiFloat8 current scaling has no MX block-alignment gate.
                tile_shapes=((2, 63, 31), (2, 31, 63)),
            )

        self.assertIsInstance(w8a8, MXFP8GroupedLinear)
        self.assertIsInstance(native, W4A8GroupedLinear)
        self.assertEqual(native.block_size, 128)
        self.assertIsInstance(fake, FakeW4A8GroupedLinear)
        self.assertIsInstance(hifloat, HiFloat8GroupedLinear)
        mx_probe.assert_called_once_with()
        native_probe.assert_called_once_with()
        fake_probe.assert_called_once_with()
        hifloat_probe.assert_called_once_with()

    @cpu_test
    def test_alignment_and_unsupported_policies_fail_at_factory(self):
        """Bad tiles and unimplemented family combinations fail before replacement."""
        module = "hyper_parallel.components.quantization.functional.strategy_factory"
        with mock.patch(f"{module}.validate_w4a8_gmm_runtime"):
            with self.assertRaisesRegex(ValueError, "not aligned"):
                build_low_precision_strategy(
                    LowPrecisionDtypeScheme(
                        weight_format="mxfp4", act_format="mxfp8"
                    ),
                    tile_shapes=((2, 63, 32), (2, 32, 64)),
                )
        with self.assertRaisesRegex(NotImplementedError, "fake QAT currently supports"):
            build_low_precision_strategy(
                LowPrecisionDtypeScheme(is_fake_quantize=True),
                tile_shapes=((2, 64, 32), (2, 32, 64)),
            )
        with self.assertRaisesRegex(NotImplementedError, "hifloat combination"):
            build_low_precision_strategy(
                LowPrecisionDtypeScheme(weight_format="hif4", act_format="hif8"),
                tile_shapes=((2, 64, 32), (2, 32, 64)),
            )
        with self.assertRaisesRegex(NotImplementedError, "fake HiFloat QAT"):
            build_low_precision_strategy(
                LowPrecisionDtypeScheme(
                    is_fake_quantize=True,
                    weight_format="hif8",
                    act_format="hif8",
                ),
                tile_shapes=((2, 64, 32), (2, 32, 64)),
            )


class TestHiFloat8GroupedLinear(unittest.TestCase):
    """HiFloat8 plugs format hooks into the one shared autograd lifecycle."""

    @cpu_test
    def test_default_role_quantizers_keep_original_recipes(self):
        """Input, weight, and gradient recipes stay independent at 15/15/224."""
        strategy = HiFloat8GroupedLinear()

        self.assertIsInstance(strategy, GroupedLinear)
        self.assertIsNot(strategy.input_quantizer, strategy.weight_quantizer)
        self.assertIsNot(strategy.input_quantizer, strategy.grad_output_quantizer)
        self.assertEqual(strategy.input_quantizer.fp8_max, 15.0)
        self.assertEqual(strategy.weight_quantizer.fp8_max, 15.0)
        self.assertEqual(strategy.grad_output_quantizer.fp8_max, 224.0)

    @cpu_test
    def test_shared_forward_backward_preserves_hifloat8_operator_contract(self):
        """The merged flow keeps quantization directions and NN/NT/TN calls."""
        torch.manual_seed(17)
        input_quantizer = _RecordingQuantizer()
        weight_quantizer = _RecordingQuantizer()
        grad_quantizer = _RecordingQuantizer()
        strategy = HiFloat8GroupedLinear(
            input_quantizer=input_quantizer,
            weight_quantizer=weight_quantizer,
            grad_output_quantizer=grad_quantizer,
        )
        counts = torch.tensor([2, 3], dtype=torch.int64)
        inputs = torch.randn(5, 4, dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(2, 6, 4, dtype=torch.bfloat16, requires_grad=True)
        grad = torch.randn(5, 6, dtype=torch.bfloat16)
        operator_calls = []
        dense_strategy = _DenseGroupedLinear()

        def _dense_hifloat8_gmm(
            left: _DenseStorage,
            right: _DenseStorage,
            **kwargs: Any,
        ) -> torch.Tensor:
            """Record and execute one dense stand-in for a HiFloat8 GMM."""

            operator_calls.append(dict(kwargs))
            return dense_strategy.grouped_matmul(left, right, **kwargs)

        with mock.patch.object(
            hifloat8_impl,
            "hifloat8_grouped_matmul",
            side_effect=_dense_hifloat8_gmm,
        ):
            actual = _GroupedLinearFunction.apply(
                inputs,
                weight,
                counts,
                strategy,
                1,
            )
            actual_grads = torch.autograd.grad(actual, (inputs, weight), grad)

        ref_inputs = inputs.detach().clone().requires_grad_(True)
        ref_weight = weight.detach().clone().requires_grad_(True)
        expected = _dense_reference(ref_inputs, ref_weight, counts)
        expected_grads = torch.autograd.grad(expected, (ref_inputs, ref_weight), grad)

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_grads[0], expected_grads[0])
        torch.testing.assert_close(actual_grads[1], expected_grads[1])
        self.assertEqual(
            [(call["layout"], call["group_type"]) for call in operator_calls],
            [("NN", 0), ("NT", 0), ("TN", 2)],
        )
        self.assertTrue(
            all(call["group_list_type"] == 1 for call in operator_calls)
        )
        self.assertTrue(
            all(call["output_dtype"] == torch.bfloat16 for call in operator_calls)
        )
        self.assertEqual(
            (input_quantizer.calls[0][1]["rowwise"],
             input_quantizer.calls[0][1]["colwise"]),
            (True, True),
        )
        self.assertEqual(
            (weight_quantizer.calls[0][1]["rowwise"],
             weight_quantizer.calls[0][1]["colwise"]),
            (True, True),
        )
        self.assertEqual(
            (grad_quantizer.calls[0][1]["rowwise"],
             grad_quantizer.calls[0][1]["colwise"]),
            (True, True),
        )
        torch.testing.assert_close(
            weight_quantizer.calls[0][0],
            weight.detach().transpose(-2, -1).contiguous(),
        )
        self.assertEqual(
            (input_quantizer.storages[0].rowwise,
             input_quantizer.storages[0].colwise),
            (False, True),
        )
        self.assertEqual(
            (weight_quantizer.storages[0].rowwise,
             weight_quantizer.storages[0].colwise),
            (True, False),
        )
        self.assertEqual(
            (grad_quantizer.storages[0].rowwise,
             grad_quantizer.storages[0].colwise),
            (False, False),
        )

    @cpu_test
    def test_hifloat8_specific_validation_is_retained(self):
        """The shared bridge still rejects invalid HiFloat8 dtype and groups."""
        quantizer = _RecordingQuantizer()
        strategy = HiFloat8GroupedLinear(quantizer, quantizer, quantizer)
        weight = torch.randn(2, 6, 4, dtype=torch.bfloat16)

        with self.assertRaisesRegex(TypeError, "float16 or bfloat16"):
            _GroupedLinearFunction.apply(
                torch.randn(3, 4),
                weight,
                torch.tensor([1, 2], dtype=torch.int64),
                strategy,
                1,
            )
        with self.assertRaisesRegex(TypeError, "torch.int64"):
            _GroupedLinearFunction.apply(
                torch.randn(3, 4, dtype=torch.bfloat16),
                weight,
                torch.tensor([1, 2], dtype=torch.int32),
                strategy,
                1,
            )
        with self.assertRaisesRegex(ValueError, "all-zero group_list"):
            _GroupedLinearFunction.apply(
                torch.empty(0, 4, dtype=torch.bfloat16),
                weight,
                torch.tensor([0, 1], dtype=torch.int64),
                strategy,
                1,
            )


class TestGroupedLinearLifecycle(unittest.TestCase):
    """The shared autograd bridge preserves dense forward/dgrad/wgrad semantics."""

    @cpu_test
    def test_forward_and_both_gradients_match_dense_reference(self):
        """Live [E,O,K] weights are transposed only at the GMM seam."""
        torch.manual_seed(7)
        counts = torch.tensor([2, 3], dtype=torch.int64)
        inputs = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
        weight = torch.randn(2, 6, 4, dtype=torch.float64, requires_grad=True)
        grad = torch.randn(5, 6, dtype=torch.float64)

        actual = _GroupedLinearFunction.apply(
            inputs, weight, counts, _DenseGroupedLinear(), 1
        )
        actual_grads = torch.autograd.grad(actual, (inputs, weight), grad)

        ref_inputs = inputs.detach().clone().requires_grad_(True)
        ref_weight = weight.detach().clone().requires_grad_(True)
        expected = _dense_reference(ref_inputs, ref_weight, counts)
        expected_grads = torch.autograd.grad(expected, (ref_inputs, ref_weight), grad)

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_grads[0], expected_grads[0])
        torch.testing.assert_close(actual_grads[1], expected_grads[1])

    @cpu_test
    def test_empty_input_has_shaped_output_and_zero_gradients(self):
        """Experts receiving zero total tokens remain valid in forward/backward."""
        inputs = torch.empty(0, 4, requires_grad=True)
        weight = torch.randn(2, 6, 4, requires_grad=True)
        counts = torch.tensor([0, 0], dtype=torch.int64)

        output = _GroupedLinearFunction.apply(
            inputs, weight, counts, _DenseGroupedLinear(), 1
        )
        output.sum().backward()

        self.assertEqual(tuple(output.shape), (0, 6))
        torch.testing.assert_close(inputs.grad, torch.zeros_like(inputs))
        torch.testing.assert_close(weight.grad, torch.zeros_like(weight))

    @cpu_test
    def test_shared_shape_and_group_validation(self):
        """Malformed geometry is rejected consistently for every strategy."""
        inputs = torch.randn(3, 4)
        weight = torch.randn(2, 6, 4)
        groups = torch.tensor([1, 2])
        validate_grouped_linear_inputs(
            inputs, weight, groups, 1, weight_input_dim=-1, name="test"
        )
        with self.assertRaisesRegex(ValueError, "contracting dimensions differ"):
            validate_grouped_linear_inputs(
                inputs, torch.randn(2, 6, 5), groups, 1,
                weight_input_dim=-1, name="test"
            )
        with self.assertRaisesRegex(ValueError, "one group per expert"):
            validate_grouped_linear_inputs(
                inputs, weight, torch.tensor([3]), 1,
                weight_input_dim=-1, name="test"
            )


if __name__ == "__main__":
    unittest.main()
