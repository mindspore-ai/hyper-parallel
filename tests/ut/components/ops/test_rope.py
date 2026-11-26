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

"""Native rope forward and gradient precision contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.components.ops import ops, rope, rope_interleaved
from hyper_parallel.components.ops.impl.rope import (
    ROPE_INTERLEAVED, _load_tile_kernels_interleaved, _tile_kernels_dtype, register_rope,
)
from hyper_parallel.components.ops.registry import ImplementationUnavailable, OpRegistry
from tests.common.op_precision import check_case
from tests.common.op_precision_suite import precision_cases


class TestRopePrecision(unittest.TestCase):
    """Verify every variant against independent formulas in all training dtypes."""

    def test_native_forward_and_gradients(self) -> None:
        """Check outputs, all differentiable arguments, discrete metadata and input immutability."""
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for case in precision_cases("rope", dtype):
                with self.subTest(dtype=dtype, variant=case.variant, case=case.name):
                    check_case(case)


class TestTileKernelsRope(unittest.TestCase):
    """CPU ABI tests for the upstream in-place kernel, not hardware validation."""

    def setUp(self) -> None:
        """Use an isolated CPU registration while keeping production NPU-only."""
        self.addCleanup(ops.set_backends, ops.get_backends())
        self.calls = []

        def apply_rotary(query, cache, *, interleaved=False, conjugate=False):
            self.assertTrue(interleaved)
            self.assertEqual(query.ndim, 3)
            self.assertEqual(cache.shape, (query.shape[0], query.shape[-1]))
            self.assertEqual(cache.dtype, torch.float32)
            self.assertEqual(query.stride(-1), 1)
            self.assertIn(query.shape[-1], (64, 128))
            self.calls.append(conjugate)
            cos, sin = cache[:, None].chunk(2, dim=-1)
            sin = -sin if conjugate else sin
            even, odd = query.float()[..., 0::2], query.float()[..., 1::2]
            result = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
            query.copy_(result.flatten(-2).to(query.dtype))

        kernels = SimpleNamespace(apply_rotary=apply_rotary)
        dependency = patch("hyper_parallel.components.ops.impl.rope.require_package", return_value=kernels)
        dependency.start()
        self.addCleanup(dependency.stop)
        registry = OpRegistry()
        register_rope(registry)
        registry.register_impl(ROPE_INTERLEAVED, name="tile_kernels", devices=("cpu",),
                               load=_load_tile_kernels_interleaved, check_compatibility=_tile_kernels_dtype)
        registry.set_auto_order(ROPE_INTERLEAVED, device="cpu", candidates=("native",))
        registry.seal()
        catalog = patch("hyper_parallel.components.ops.factory.builtin_registry", registry)
        catalog.start()
        self.addCleanup(catalog.stop)
        ops.set_backends({"rope": "tile_kernels"})

    def test_shared_forward_and_gradient_cases(self) -> None:
        """Cover both layouts, partial tails, identity and broadcast frequency gradients."""
        for dtype in (torch.float32, torch.bfloat16):
            for case in precision_cases("rope", dtype, hardware=True):
                if case.variant == "interleaved":
                    with self.subTest(dtype=dtype, name=case.name):
                        check_case(case, backend="tile_kernels")
        self.assertIn(False, self.calls)
        self.assertIn(True, self.calls)

    def test_noncontiguous_input_and_gradients_leave_inputs_unchanged(self) -> None:
        """The public call must never mutate caller-owned Q/K or the upstream VJP."""
        generator = torch.Generator().manual_seed(29)
        query = torch.randn(2, 4, 3, 256, generator=generator)[..., ::2].requires_grad_()
        key = torch.randn(2, 2, 3, 256, generator=generator)[..., ::2].requires_grad_()
        cos = torch.randn(1, 3, 64, generator=generator).requires_grad_()
        sin = torch.randn(1, 3, 64, generator=generator).requires_grad_()
        args = (query, key, cos, sin)
        references = tuple(t.detach().clone().requires_grad_() for t in args)
        before = tuple(t.detach().clone() for t in args)
        actual = rope_interleaved(*args)
        gradients = tuple(torch.randn(t.numel() * 2, generator=generator)[::2].reshape(t.shape) for t in actual)
        grad_before = tuple(t.clone() for t in gradients)
        torch.autograd.backward(actual, gradients)
        ops.set_backends({"rope": "native"})
        expected = rope_interleaved(*references)
        torch.autograd.backward(expected, gradients)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right)
        for arg, reference, original in zip(args, references, before):
            torch.testing.assert_close(arg.grad, reference.grad)
            torch.testing.assert_close(arg.detach(), original, rtol=0, atol=0)
        for grad, original in zip(gradients, grad_before):
            torch.testing.assert_close(grad, original, rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(cos.grad[..., 32:]).item(), 0)
        self.assertEqual(torch.count_nonzero(sin.grad[..., 32:]).item(), 0)

    def test_cached_selection_rechecks_shape_and_frequency_dtype(self) -> None:
        """A successful call must not admit unsupported dimensions or FP64 tables later."""
        query = torch.ones(1, 2, 3, 128)
        key = torch.ones(1, 1, 3, 128)
        rope_interleaved(query, key, torch.ones(1, 3, 64), torch.zeros(1, 3, 64))
        with self.assertRaisesRegex(RuntimeError, "rotary dimension of 64 or 128"):
            rope_interleaved(query, key, torch.ones(1, 3, 32), torch.zeros(1, 3, 32))
        with self.assertRaisesRegex(RuntimeError, "floating-point cos/sin"):
            rope_interleaved(query, key, torch.ones(1, 3, 64, dtype=torch.float64),
                             torch.zeros(1, 3, 64, dtype=torch.float64))
        with self.assertRaisesRegex(RuntimeError, "only BF16 and FP32"):
            rope_interleaved(query.half(), key, torch.ones(1, 3, 64), torch.zeros(1, 3, 64))
        with self.assertRaisesRegex(RuntimeError, "not registered for this contract"):
            rope(query, torch.ones(1, 1, 3, 64), torch.zeros(1, 1, 3, 64))

    def test_missing_upstream_symbol_is_rejected(self) -> None:
        """A discoverable dependency without apply_rotary must not become a selected backend."""
        with patch("hyper_parallel.components.ops.impl.rope.require_package", return_value=SimpleNamespace()):
            with self.assertRaisesRegex(ImplementationUnavailable, "missing operator apply_rotary"):
                _load_tile_kernels_interleaved()
