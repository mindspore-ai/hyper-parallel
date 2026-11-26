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
"""FP8/FP4 representations and CPU-only TileKernels conversion ABI checks."""

from typing import Any

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_parallel.components.ops import cast_back, ops, per_block_cast, per_token_cast
from hyper_parallel.components.ops.impl.quantization import (
    CAST_BACK, PER_BLOCK_CAST, PER_TOKEN_CAST, _load_symbol,
    _native_cast_back, _native_per_block, _native_per_token, register_quantization,
)
from hyper_parallel.components.ops.registry import OpRegistry
from tests.common.op_precision import check_case
from tests.common.op_precision_suite import precision_cases


class TestQuantizationPrecision(unittest.TestCase):
    """Native values, bytes and scales are checked against independent references."""

    def setUp(self) -> None:
        """Install isolated backend fixtures and restore process state on cleanup."""
        self.addCleanup(ops.set_backends, ops.get_backends())
        ops.set_backends({"quantization": "native"})

    def test_native_reference(self) -> None:
        for dtype in (torch.float32, torch.bfloat16):
            for case in precision_cases("quantization", dtype):
                with self.subTest(dtype=dtype, variant=case.variant, case=case.name):
                    check_case(case)

    def test_fp4_ties_and_signed_zero_have_known_bytes(self) -> None:
        x = torch.zeros(1, 128)
        x[0, :16] = torch.tensor([0., .25, .5, .75, 1., 1.25, 1.5, 1.75, 2., 2.5, 3., 3.5, 4., 5., 6., -0.])
        data, scale = per_token_cast(x, "e2m1", 128, round_sf=True)
        expected = torch.tensor([[0x00, 0x21, 0x22, 0x43, 0x44, 0x65, 0x66, 0x87]], dtype=torch.uint8)
        torch.testing.assert_close(data.view(torch.uint8)[:, :8], expected, rtol=0, atol=0)
        torch.testing.assert_close(scale, torch.ones_like(scale), rtol=0, atol=0)

    def test_packed_scale_layouts_and_requantization(self) -> None:
        x = torch.randn(4, 256)
        for column_major in (False, True):
            for packed in (False, True):
                with self.subTest(column_major=column_major, packed=packed):
                    pair = per_token_cast(x, "e4m3", 128, round_sf=True, use_packed_ue8m0=packed,
                                          use_tma_aligned_col_major_sf=column_major)
                    restored = cast_back(pair, "fp32", (1, 128))
                    self.assertEqual(pair[0].dtype, torch.float8_e4m3fn)
                    self.assertEqual(pair[1].dtype, torch.int16 if packed else torch.float32)
                    torch.testing.assert_close(restored, x, atol=0.15, rtol=0.07)
                    recast = per_token_cast(pair, "e2m1", 128, x_block_size=(1, 128))
                    self.assertEqual(recast[0].dtype, torch.int8)

    def test_no_straight_through_gradient(self) -> None:
        x = torch.randn(4, 128, requires_grad=True)
        pair = per_token_cast(x, "e4m3", 128)
        self.assertFalse(any(value.requires_grad for value in pair))
        self.assertFalse(cast_back(pair, "fp32", (1, 128)).requires_grad)


class TestTileKernelsQuantization(unittest.TestCase):
    """Conversion selection preserves the raw data/scale ABI and optional buffers."""

    def setUp(self) -> None:
        """Install isolated backend fixtures and restore process state on cleanup."""
        self.addCleanup(ops.set_backends, ops.get_backends())
        self.calls = []

        def wrap(function: Any, name: Any) -> Any:
            """Wrap."""
            def run(*args: Any, **options: Any) -> Any:
                """Apply the operator contract using the selected implementation."""
                self.calls.append((name, args, options))
                return function(*args, **options)
            return run

        kernels = SimpleNamespace(per_token_cast=wrap(_native_per_token, "per_token"),
                                  per_block_cast=wrap(_native_per_block, "per_block"),
                                  cast_back=wrap(_native_cast_back, "cast_back"))
        dependency = patch("hyper_parallel.components.ops.impl.quantization.require_package", return_value=kernels)
        dependency.start()
        self.addCleanup(dependency.stop)
        registry = OpRegistry()
        register_quantization(registry)
        for spec, name in ((PER_TOKEN_CAST, "per_token_cast"), (PER_BLOCK_CAST, "per_block_cast"),
                           (CAST_BACK, "cast_back")):
            registry.register_impl(spec, name="tile_kernels", devices=("cpu",),
                                   load=lambda symbol=name: _load_symbol(symbol))
            registry.set_auto_order(spec, device="cpu", candidates=("native",))
        registry.seal()
        catalog = patch("hyper_parallel.components.ops.factory.builtin_registry", registry)
        catalog.start()
        self.addCleanup(catalog.stop)

    def test_shared_conversion_cases(self) -> None:
        for case in precision_cases("quantization", torch.float32, backend="tile_kernels"):
            with self.subTest(variant=case.variant, case=case.name):
                check_case(case)
                check_case(case, backend="tile_kernels")
        self.assertEqual({entry[0] for entry in self.calls}, {"per_token", "per_block", "cast_back"})

    def test_output_buffers_and_optional_layout_flags_are_forwarded(self) -> None:
        ops.set_backends({"quantization": "tile_kernels"})
        x = torch.randn(128, 256)
        expected = per_block_cast(x, "e4m3", (128, 128), round_sf=True, use_packed_ue8m0=True)
        buffers = tuple(torch.empty_like(value) for value in expected)
        actual = per_block_cast(x, "e4m3", (128, 128), round_sf=True, use_packed_ue8m0=True, out=buffers)
        self.assertIs(actual, buffers)
        self.assertIs(self.calls[-1][2]["out"], buffers)
        torch.testing.assert_close(actual[0].view(torch.uint8), expected[0].view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        cast_back(actual, "bf16", (128, 128))
        self.assertIs(self.calls[-1][1][0], actual)
