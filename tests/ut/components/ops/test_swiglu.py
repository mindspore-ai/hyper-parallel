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

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from hyper_parallel.components.ops import ops, swiglu, swiglu_clamped
from hyper_parallel.components.ops.impl.swiglu import (
    SWIGLU, _load_tile_kernels, register_swiglu,
)
from hyper_parallel.components.ops.registry import ImplementationUnavailable, OpRegistry
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

        def forward(x, fmt):
            self.assertEqual(x.ndim, 2)
            self.assertEqual(x.shape[-1] % 256, 0)
            self.assertEqual(x.dtype, {"fp32": torch.float32, "bf16": torch.bfloat16}[fmt])
            self.calls.append(("forward", fmt, tuple(x.shape), x.is_contiguous()))
            gate, up = x.float().chunk(2, -1)
            return (gate.sigmoid() * gate * up).to(x.dtype)

        def backward(x, grad, fmt):
            self.assertEqual(grad.shape, (x.shape[0], x.shape[1] // 2))
            self.assertEqual(grad.dtype, x.dtype)
            self.calls.append(("backward", fmt, tuple(grad.shape), grad.is_contiguous()))
            gate, up = x.float().chunk(2, -1)
            sigmoid = gate.sigmoid()
            dgate = grad.float() * up * sigmoid * (1 + gate * (1 - sigmoid))
            dup = grad.float() * gate * sigmoid
            return (torch.cat((dgate, dup), -1).to(x.dtype),)

        self.kernels = SimpleNamespace(swiglu_forward=forward, swiglu_backward=backward)
        dependency = patch("hyper_parallel.components.ops.impl.swiglu.require_package", return_value=self.kernels)
        dependency.start()
        self.addCleanup(dependency.stop)
        registry = OpRegistry()
        register_swiglu(registry)
        # CPU registration is test-only; production declares NPU support only.
        registry.register_impl(SWIGLU, name="tile_kernels", devices=("cpu",), load=_load_tile_kernels)
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
        with self.assertRaisesRegex(RuntimeError, "not registered for this contract"):
            swiglu_clamped(torch.ones(2, 128), torch.ones(2, 128), 1.0)

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
                if case.variant == "standard":
                    with self.subTest(dtype=dtype, name=case.name):
                        check_case(case)
                        check_case(case, backend="tile_kernels")
