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

"""Native moe_router forward and gradient precision contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.components.ops import ops
from hyper_parallel.components.ops.impl.moe_router import MOE_ROUTER, _load_tile_kernels, register_moe_router
from hyper_parallel.components.ops.registry import ImplementationUnavailable, OpRegistry
from tests.common.op_precision import check_case
from tests.common.op_precision_suite import precision_cases


class TestMoeRouterPrecision(unittest.TestCase):
    """Verify every variant against independent formulas in all training dtypes."""

    def test_native_forward_and_gradients(self) -> None:
        """Check outputs, all differentiable arguments, discrete metadata and input immutability."""
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for case in precision_cases("moe_router", dtype):
                with self.subTest(dtype=dtype, variant=case.variant, case=case.name):
                    check_case(case)

    def test_tile_kernels_cpu_abi(self) -> None:
        """Check Top-K argument forwarding while keeping the projection and weight gradients."""
        calls = []

        def topk_gate(scores, top_k):
            self.assertEqual(scores.ndim, 2)
            self.assertEqual(scores.dtype, torch.float32)
            self.assertTrue(scores.is_contiguous())
            calls.append(top_k)
            return scores.argsort(dim=-1, descending=True, stable=True)[:, :top_k]

        registry = OpRegistry()
        register_moe_router(registry)
        registry.register_impl(MOE_ROUTER, name="tile_kernels", devices=("cpu",), load=_load_tile_kernels)
        registry.set_auto_order(MOE_ROUTER, device="cpu", candidates=("native",))
        registry.seal()
        self.addCleanup(ops.set_backends, ops.get_backends())
        with patch("hyper_parallel.components.ops.impl.moe_router.require_package",
                   return_value=SimpleNamespace(topk_gate=topk_gate)), \
                patch("hyper_parallel.components.ops.factory.builtin_registry", registry):
            for dtype in (torch.float32, torch.bfloat16):
                for case in precision_cases("moe_router", dtype, backend="tile_kernels"):
                    with self.subTest(dtype=dtype, case=case.name):
                        check_case(case)
                        check_case(case, backend="tile_kernels")
        self.assertEqual(calls, [1, 8, 8, 8] * 2)

    def test_tile_kernels_requires_topk_symbol(self) -> None:
        """A missing Top-K export must not silently fall back to native."""
        with patch("hyper_parallel.components.ops.impl.moe_router.require_package", return_value=SimpleNamespace()):
            with self.assertRaisesRegex(ImplementationUnavailable, "missing operator topk_gate"):
                _load_tile_kernels()
