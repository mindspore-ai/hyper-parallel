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

"""Native mhc_post forward and gradient precision contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.components.ops import ops
from hyper_parallel.components.ops.impl.mhc_post import MHC_POST, _load_tile_kernels, register_mhc_post
from hyper_parallel.components.ops.registry import ImplementationUnavailable, OpRegistry
from tests.common.op_precision import check_case
from tests.common.op_precision_suite import precision_cases


class TestMhcPostPrecision(unittest.TestCase):
    """Verify every variant against independent formulas in all training dtypes."""

    def test_native_forward_and_gradients(self) -> None:
        """Check outputs, all differentiable arguments, discrete metadata and input immutability."""
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for case in precision_cases("mhc_post", dtype):
                with self.subTest(dtype=dtype, variant=case.variant, case=case.name):
                    check_case(case)

    def test_tile_kernels_cpu_abi(self) -> None:
        """Simulate upstream's explicit stream layout and four-gradient backward on CPU."""
        calls = []

        def reference(x, residual, post, combine):
            return (x.float().unsqueeze(-2) * post
                    + torch.einsum("bsij,bsid->bsjd", combine, residual.float())).to(x.dtype)

        def forward(x, residual, post, combine):
            self.assertEqual(residual.ndim, 4)
            self.assertEqual(post.shape, (*residual.shape[:-1], 1))
            self.assertEqual(x.dtype, torch.bfloat16)
            self.assertTrue(all(t.is_contiguous() for t in (x, residual, post, combine)))
            calls.append("forward")
            return reference(x, residual, post, combine)

        def backward(x, residual, post, combine, gradient):
            calls.append("backward")
            with torch.enable_grad():
                leaves = [t.detach().requires_grad_() for t in (x, residual, post, combine)]
                return torch.autograd.grad(reference(*leaves), leaves, gradient)

        registry = OpRegistry()
        register_mhc_post(registry)
        registry.register_impl(MHC_POST, name="tile_kernels", devices=("cpu",), load=_load_tile_kernels)
        registry.set_auto_order(MHC_POST, device="cpu", candidates=("native",))
        registry.seal()
        self.addCleanup(ops.set_backends, ops.get_backends())
        kernels = SimpleNamespace(mhc_post_fwd=forward, mhc_post_bwd=backward)
        with patch("hyper_parallel.components.ops.impl.mhc_post.require_package", return_value=kernels), \
                patch("hyper_parallel.components.ops.factory.builtin_registry", registry):
            for case in precision_cases("mhc_post", torch.bfloat16, backend="tile_kernels"):
                with self.subTest(case=case.name):
                    check_case(case)
                    check_case(case, backend="tile_kernels")
        self.assertEqual(calls, ["forward", "backward"] * 2)

    def test_tile_kernels_requires_backward_symbol(self) -> None:
        """An installed package without the matching backward must fail selection."""
        kernels = SimpleNamespace(mhc_post_fwd=lambda *args: None)
        with patch("hyper_parallel.components.ops.impl.mhc_post.require_package", return_value=kernels):
            with self.assertRaisesRegex(ImplementationUnavailable, "missing operator mhc_post_bwd"):
                _load_tile_kernels()
