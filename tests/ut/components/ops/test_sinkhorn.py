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

"""Native sinkhorn forward and gradient precision contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.components.ops import ops
from hyper_parallel.components.ops.impl.sinkhorn import SINKHORN, _load_tile_kernels, register_sinkhorn
from hyper_parallel.components.ops.registry import ImplementationUnavailable, OpRegistry
from tests.common.op_precision import check_case
from tests.common.op_precision_cases import _sinkhorn
from tests.common.op_precision_suite import precision_cases


class TestSinkhornPrecision(unittest.TestCase):
    """Verify every variant against independent formulas in all training dtypes."""

    def test_native_forward_and_gradients(self) -> None:
        """Check outputs, all differentiable arguments, discrete metadata and input immutability."""
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for case in precision_cases("sinkhorn", dtype):
                with self.subTest(dtype=dtype, variant=case.variant, case=case.name):
                    check_case(case)

    def test_tile_kernels_cpu_abi(self) -> None:
        """Simulate the raw buffer-writing forward/backward, not the NPU kernel."""
        calls = []

        def forward(x, output, iterations, eps):
            self.assertEqual(x.shape, (10, 4, 4))
            self.assertTrue(x.is_contiguous())
            calls.append(("forward", iterations, eps))
            output.copy_(_sinkhorn(x, iterations, eps))

        def backward(gradient, x, result, iterations, eps):
            self.assertTrue(gradient.is_contiguous())
            calls.append(("backward", iterations, eps))
            with torch.enable_grad():
                leaf = x.detach().requires_grad_()
                result.copy_(torch.autograd.grad(_sinkhorn(leaf, iterations, eps), leaf, gradient)[0])

        registry = OpRegistry()
        register_sinkhorn(registry)
        registry.register_impl(SINKHORN, name="tile_kernels", devices=("cpu",), load=_load_tile_kernels)
        registry.set_auto_order(SINKHORN, device="cpu", candidates=("native",))
        registry.seal()
        self.addCleanup(ops.set_backends, ops.get_backends())
        kernels = SimpleNamespace(mhc_sinkhorn_fwd=forward, mhc_sinkhorn_bwd=backward)
        with patch("hyper_parallel.components.ops.impl.sinkhorn.require_package", return_value=kernels), \
                patch("hyper_parallel.components.ops.factory.builtin_registry", registry):
            for case in precision_cases("sinkhorn", torch.float32, backend="tile_kernels"):
                with self.subTest(case=case.name):
                    check_case(case)
                    check_case(case, backend="tile_kernels")
            with self.assertRaisesRegex(ValueError, "4-by-4"):
                _load_tile_kernels()(torch.ones(2, 3, 3))
        self.assertEqual(calls[::2], [("forward", n, eps) for n, eps in ((1, 1e-6), (3, 1e-4), (20, 1e-6))])
        self.assertEqual([entry[1:] for entry in calls[::2]], [entry[1:] for entry in calls[1::2]])

    def test_tile_kernels_requires_backward_symbol(self) -> None:
        """Do not treat a package with a missing backward as an available implementation."""
        kernels = SimpleNamespace(mhc_sinkhorn_fwd=lambda *args: None)
        with patch("hyper_parallel.components.ops.impl.sinkhorn.require_package", return_value=kernels):
            with self.assertRaisesRegex(ImplementationUnavailable, "missing operator mhc_sinkhorn_bwd"):
                _load_tile_kernels()
