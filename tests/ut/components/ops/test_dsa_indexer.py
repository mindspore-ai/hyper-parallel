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

"""Native dsa_indexer forward and gradient precision contracts."""

from typing import Any

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.components.ops import ops, deep_select_topk
from hyper_parallel.components.ops.impl.dsa_indexer import (
    DSA_INDEXER, DSA_PACKED_INDEXER, DSA_TOPK, _load_select, _load_select_indexer, _native_topk, register_dsa_indexer,
)
from hyper_parallel.components.ops.registry import OpRegistry
from tests.common.op_precision import check_case
from tests.common.op_precision_suite import precision_cases


class TestDsaIndexerPrecision(unittest.TestCase):
    """Verify every variant against independent formulas in all training dtypes."""

    def test_native_forward_and_gradients(self) -> None:
        """Check outputs, all differentiable arguments, discrete metadata and input immutability."""
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for case in precision_cases("dsa_indexer", dtype):
                with self.subTest(dtype=dtype, variant=case.variant, case=case.name):
                    check_case(case)


class TestDeepSelect(unittest.TestCase):
    """CPU ABI emulation verifies alignment and dispatch, not native device kernels."""

    def setUp(self) -> None:
        """Install isolated backend fixtures and restore process state on cleanup."""
        self.addCleanup(ops.set_backends, ops.get_backends())
        self.calls = []

        def topk(x: Any, count: Any, **options: Any) -> Any:
            """Topk."""
            self.assertEqual(x.stride(0) * x.element_size() % 32, 0)
            self.calls.append(options)
            translate = {"output_idx": "output_indices", "output_idx_offset": "output_indices_offset",
                         "idx_oob_fill_value": "index_out_of_bounds_fill_value",
                         "value_oob_fill_value": "value_out_of_bounds_fill_value"}
            return _native_topk(x, count, **{translate.get(k, k): v for k, v in options.items()})

        dependency = patch("hyper_parallel.components.ops.impl.dsa_indexer.require_package",
                           return_value=SimpleNamespace(topk=topk, get_stride_requirement=lambda: (32, 32)))
        dependency.start()
        self.addCleanup(dependency.stop)
        registry = OpRegistry()
        register_dsa_indexer(registry)
        for spec, loader in ((DSA_TOPK, _load_select), (DSA_INDEXER, _load_select_indexer),
                             (DSA_PACKED_INDEXER, lambda: _load_select_indexer(packed=True))):
            registry.register_impl(spec, name="deep_select", devices=("cpu",), load=loader)
            registry.set_auto_order(spec, device="cpu", candidates=("native",))
        registry.seal()
        catalog = patch("hyper_parallel.components.ops.factory.builtin_registry", registry)
        catalog.start()
        self.addCleanup(catalog.stop)

    def test_all_indexer_variants(self) -> None:
        for dtype in (torch.float32, torch.bfloat16):
            for case in precision_cases("dsa_indexer", dtype, backend="deep_select"):
                with self.subTest(dtype=dtype, variant=case.variant, name=case.name):
                    check_case(case, backend="deep_select")
        self.assertTrue(self.calls)

    def test_output_buffer_and_unsupported_reserved_arguments(self) -> None:
        ops.set_backends({"dsa_indexer": "deep_select"})
        values = torch.randn(2, 17)
        out = torch.empty(2, 4, dtype=torch.int64)
        _, actual = deep_select_topk(values, 4, indices_type=torch.int64, output_indices=out, sorted=True)
        self.assertIs(actual, out)
        self.assertEqual(self.calls[-1]["output_idx"].data_ptr(), out.data_ptr())
        for options in ({"begin": torch.zeros(2, dtype=torch.int32)}, {"hint": torch.zeros(2)}):
            with self.assertRaisesRegex(NotImplementedError, "begin/hint"):
                deep_select_topk(values, 4, **options)
