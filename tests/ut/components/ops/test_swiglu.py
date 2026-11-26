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

"""SwiGLU precision and optional-backend ABI contracts."""

from typing import Any

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from hyper_parallel.components.ops import (
    ops, swiglu, swiglu_clamped, swiglu_forward, swiglu_backward, swiglu_forward_and_per_token_cast,
)
from hyper_parallel.components.ops.impl.swiglu import (
    SWIGLU, SWIGLU_CLAMPED, SWIGLU_FORWARD, SWIGLU_BACKWARD, SWIGLU_QUANTIZED, SWIGLU_QUANTIZED_BACKWARD,
    _load_tile_kernels, _load_clamped, _load_routed, _load_explicit, register_swiglu,
)
from hyper_parallel.components.ops.registry import ImplementationUnavailable, OpRegistry
from tests.common.op_precision_cases import (
    _routed_swiglu_reference, _swiglu_backward_reference, _quantized_swiglu_reference,
    _quantized_swiglu_backward_reference,
)
from tests.common.op_precision import check_case
from tests.common.op_precision_suite import precision_cases


class TestSwigluPrecision(unittest.TestCase):
    """Verify every variant against independent formulas in all training dtypes."""

    def test_native_forward_and_gradients(self) -> None:
        """Check outputs, all differentiable arguments, discrete metadata and input immutability."""
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for case in precision_cases("swiglu", dtype):
                with self.subTest(dtype=dtype, variant=case.variant, case=case.name):
                    check_case(case)


class TestTileKernelsSwiGLU(unittest.TestCase):
    """Exercise selection and autograd through a CPU-only upstream ABI stub."""

    def setUp(self) -> None:
        """Keep global configuration and the production backend inventory isolated."""
        previous = ops.get_backends()
        self.addCleanup(ops.set_backends, previous)
        self.calls = []

        def forward(x: Any, fmt: Any, **options: Any) -> Any:
            """Execute the selected backend and retain the tensors needed for backward."""
            self.assertEqual(x.ndim, 2)
            self.assertEqual(x.shape[-1] % 256, 0)
            self.assertIn(x.dtype, (torch.float32, torch.bfloat16))
            self.calls.append(("forward", fmt, tuple(x.shape), x.is_contiguous()))
            count = options.get("clamped_count")
            if count is not None:
                gate, up = x.detach().chunk(2, -1)
                limit = options["clamp_value"]
                count.add_(torch.stack(((gate > limit).sum(), (up > limit).sum(), (up < -limit).sum(),
                                       gate.new_tensor(gate.numel(), dtype=torch.int64))))
            result = _routed_swiglu_reference(x, fmt, **options)
            if options.get("psum_num_tokens_per_expert") is not None:
                result[3:] = float("nan")
            return result

        def backward(x: Any, grad: Any, fmt: Any, **options: Any) -> Any:
            """Return the input gradients required by the operator contract."""
            x = x.reshape(-1, x.shape[-1])
            grad = grad.reshape(x.shape[0], x.shape[-1] // 2)
            self.assertEqual(grad.shape, (x.shape[0], x.shape[1] // 2))
            self.assertEqual(grad.dtype, x.dtype)
            self.calls.append(("backward", fmt, tuple(grad.shape), grad.is_contiguous()))
            if options:
                return _swiglu_backward_reference(x, grad, fmt, **options)
            gate, up = x.float().chunk(2, -1)
            sigmoid = gate.sigmoid()
            dgate = grad.float() * up * sigmoid * (1 + gate * (1 - sigmoid))
            dup = grad.float() * gate * sigmoid
            return (torch.cat((dgate, dup), -1).to(x.dtype),)

        def quantized(x: Any, fmt: Any, channels: Any, **options: Any) -> Any:
            """Quantized."""
            data, scale = _quantized_swiglu_reference(x, fmt, channels, **options)
            if options.get("psum_num_tokens_per_expert") is not None:
                data[3:] = 127
                scale[3:] = float("nan")
            return data.view(torch.float8_e4m3fn), scale

        def quantized_backward(x: Any, gradient: Any, fmt: Any, channels: Any, **options: Any) -> Any:
            """Quantized backward."""
            result = _quantized_swiglu_backward_reference(x, gradient, fmt, channels, **options)
            return result[0], (result[1][0].view(torch.float8_e4m3fn), result[1][1]), *result[2:]

        self.kernels = SimpleNamespace(get_token_alignment=lambda: 128,
                                       swiglu_forward=forward, swiglu_backward=backward,
                                       swiglu_forward_and_per_token_cast=quantized,
                                       swiglu_backward_and_per_token_cast=quantized_backward)
        dependency = patch("hyper_parallel.components.ops.impl.swiglu.require_package", return_value=self.kernels)
        dependency.start()
        self.addCleanup(dependency.stop)
        registry = OpRegistry()
        register_swiglu(registry)
        # CPU registration is test-only; production declares NPU support only.
        registry.register_impl(SWIGLU, name="tile_kernels", devices=("cpu",), load=_load_tile_kernels)
        for spec, loader in (
            (SWIGLU_CLAMPED, _load_clamped), (SWIGLU_FORWARD, _load_routed),
            (SWIGLU_BACKWARD, lambda: _load_explicit("swiglu_backward")),
            (SWIGLU_QUANTIZED, lambda: _load_explicit("swiglu_forward_and_per_token_cast")),
            (SWIGLU_QUANTIZED_BACKWARD, lambda: _load_explicit("swiglu_backward_and_per_token_cast")),
        ):
            registry.register_impl(spec, name="tile_kernels", devices=("cpu",), load=loader)
            registry.set_auto_order(spec, device="cpu", candidates=("native",))
        registry.set_auto_order(SWIGLU, device="cpu", candidates=("native",))
        registry.seal()
        catalog = patch("hyper_parallel.components.ops.factory.builtin_registry", registry)
        catalog.start()
        self.addCleanup(catalog.stop)
        ops.set_backends({"swiglu": "tile_kernels"})

    def test_packed_axes_shapes_and_gradients(self) -> None:
        """Restore arbitrary packed axes, noncontiguous inputs, empty rows and VJPs."""
        generator = torch.Generator().manual_seed(23)
        for shape, dim, transpose in (
            ((2, 3, 256), -1, False), ((2, 256, 3), 1, True),
            ((2, 256, 3), -2, False), ((256,), 0, False), ((0, 256), -1, False),
        ):
            with self.subTest(shape=shape, dim=dim):
                x = torch.randn(shape, generator=generator).requires_grad_()
                if transpose:
                    x = x.transpose(0, 2).detach().requires_grad_()
                reference = x.detach().clone().requires_grad_()
                before = x.detach().clone()
                actual = swiglu(x, dim)
                gate, up = reference.chunk(2, dim)
                expected = F.silu(gate) * up
                grad = torch.randn(actual.numel() * 2, generator=generator)[::2].reshape(actual.shape)
                actual.backward(grad)
                expected.backward(grad)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(x.grad, reference.grad)
                torch.testing.assert_close(x.detach(), before, rtol=0, atol=0)
        self.assertTrue(all(call[3] for call in self.calls))
        self.assertIn(("swiglu", "standard", "tile_kernels"), ops.used_backends())

    def test_rejects_unsupported_calls_after_cached_success(self) -> None:
        """A previous selection must not admit unaligned shapes or unsupported dtypes."""
        swiglu(torch.ones(2, 256))
        with self.assertRaises(AssertionError):
            swiglu(torch.ones(2, 128))
        with self.assertRaises(AssertionError):
            swiglu(torch.ones(2, 256, dtype=torch.float16))
        with self.assertRaises(AssertionError):
            swiglu_clamped(torch.ones(2, 64), torch.ones(2, 64), 1.0)

    def test_missing_backward_symbol_is_rejected(self) -> None:
        """A discoverable dependency without its backward is not a training backend."""
        with patch("hyper_parallel.components.ops.impl.swiglu.require_package",
                   return_value=SimpleNamespace(swiglu_forward=self.kernels.swiglu_forward)):
            with self.assertRaisesRegex(ImplementationUnavailable, "missing operator swiglu_backward"):
                _load_tile_kernels()

    def test_training_dtypes_match_shared_precision_cases(self) -> None:
        """Check the adapter in BF16/FP32 with the existing forward and VJP tolerances."""
        for dtype in (torch.bfloat16, torch.float32):
            for case in precision_cases("swiglu", dtype, hardware=True, backend="tile_kernels"):
                with self.subTest(dtype=dtype, name=case.name):
                    check_case(case)
                    check_case(case, backend="tile_kernels")

    def test_clamp_boundaries_broadcast_and_mixed_output_dtype(self) -> None:
        """Boundary derivatives and mixed output dtypes retain the native graph semantics."""
        for dtype, fmt in ((torch.bfloat16, "fp32"), (torch.float32, "bf16")):
            x = torch.tensor([-2.0, -1.25, 0.0, 1.25, 2.0], dtype=dtype).repeat(2, 1, 128)
            x = x.transpose(1, 2).contiguous().reshape(2, -1)[:, :256].detach().requires_grad_()
            weights = torch.tensor([0.5, 0.0, 1.5, 0.0])[::2].requires_grad_()
            reference, ref_weights = x.detach().clone().requires_grad_(), weights.detach().clone().requires_grad_()
            actual = swiglu_forward(x, fmt, topk_weights=weights, clamp_value=1.25)
            expected = _routed_swiglu_reference(reference, fmt, topk_weights=ref_weights, clamp_value=1.25)
            gradient = torch.randn_like(actual)
            actual.backward(gradient)
            expected.backward(gradient)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(x.grad, reference.grad, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(weights.grad, ref_weights.grad, atol=1e-4, rtol=1e-4)
        gate = torch.randn(2, 128, requires_grad=True)
        up = torch.randn(1, 128, requires_grad=True)
        actual = swiglu_clamped(gate, up, 1.25)
        expected = F.silu(gate.clamp(max=1.25)) * up.clamp(-1.25, 1.25)
        expected_gradients = torch.autograd.grad(expected.sum(), (gate, up))
        actual.sum().backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(gate.grad, expected_gradients[0])
        torch.testing.assert_close(up.grad, expected_gradients[1])

    def test_explicit_backward_optional_return_fields(self) -> None:
        """No weight or recompute field is added unless requested by the caller."""
        x, gradient = torch.randn(2, 256), torch.randn(2, 128)
        for weighted in (False, True):
            for recompute in (False, True):
                options = {"do_recompute": recompute}
                if weighted:
                    options["topk_weights"] = torch.ones(2)
                result = swiglu_backward(x, gradient, "fp32", **options)
                self.assertEqual(len(result), 1 + weighted + recompute)
                expected = _swiglu_backward_reference(x, gradient, "fp32", **options)
                for actual, reference in zip(result, expected):
                    torch.testing.assert_close(actual, reference)

    def test_padding_alignment_and_flattened_explicit_vjp(self) -> None:
        """Do not silently reinterpret expert prefixes when upstream alignment differs."""
        x, gradient = torch.randn(2, 2, 256), torch.randn(2, 2, 128)
        for backend in ("native", "tile_kernels"):
            ops.set_backends({"swiglu": backend})
            result = swiglu_backward(x, gradient, "fp32")
            expected = _swiglu_backward_reference(x.flatten(0, 1), gradient.flatten(0, 1), "fp32")
            torch.testing.assert_close(result[0], expected[0])
            self.assertEqual(result[0].shape, (4, 256))
        with self.assertRaisesRegex(ValueError, "token_alignment must match"):
            swiglu_forward(x.flatten(0, 1), "fp32", token_alignment=256,
                           psum_num_tokens_per_expert=torch.tensor([4], dtype=torch.int32))

    def test_quantized_padding_preserves_scale_layouts(self) -> None:
        """Clear unwritten padding without decoding or rebuilding valid packed scales."""
        for layout in ("row", "packed_row", "packed_column"):
            data = torch.full((128, 128), 0.5).to(torch.float8_e4m3fn)
            if layout == "row":
                scale = torch.full((128, 4), 4.0)
            else:
                exponent = torch.full((128, 4), 129, dtype=torch.uint8)
                scale = (exponent.view(torch.int16) if layout == "packed_row"
                         else exponent.T.contiguous().view(torch.int16).T)
            stride = scale.stride()
            with patch.object(self.kernels, "swiglu_forward_and_per_token_cast", return_value=(data, scale)):
                actual = swiglu_forward_and_per_token_cast(torch.zeros(128, 256), "e4m3", 32,
                                                          psum_num_tokens_per_expert=torch.tensor([3, 128],
                                                                                                dtype=torch.int32))
            self.assertEqual(actual[1].stride(), stride)
            torch.testing.assert_close(actual[0][:3].float(), torch.full((3, 128), 0.5), rtol=0, atol=0)
            torch.testing.assert_close(actual[0][3:].float(), torch.zeros(125, 128), rtol=0, atol=0)
            if layout == "row":
                expected_scale = torch.ones(128, 4)
                expected_scale[:3] = 4
                torch.testing.assert_close(actual[1], expected_scale, rtol=0, atol=0)
            else:
                logical = scale.view(torch.uint8) if layout == "packed_row" else scale.T.view(torch.uint8).T
                expected_exponent = torch.full((128, 4), 127, dtype=torch.uint8)
                expected_exponent[:3] = 129
                torch.testing.assert_close(logical, expected_exponent, rtol=0, atol=0)
