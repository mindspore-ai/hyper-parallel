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
"""Width-matrix round-trip contracts for the three CSA2 packed-cache formats.

Each format's physical byte layout is pinned at every legal width, including
odd block/group counts that exercise the E8M0 scale-pair padding. The expected
packed widths below are derived from the format spec, not from the production
``scale_slots`` formulas, so a divergence in the code fails the test.
"""

import unittest

import torch

from hyper_parallel.components.quantization.ops.npu_mxfp4 import (
    _native_qdq,
    pack_indexer_k,
    unpack_indexer_k,
)
from hyper_parallel.components.quantization.quantizers.main_kv import (
    pack_main_kv,
    unpack_main_kv,
)
from hyper_parallel.components.quantization.quantizers.swa_kv import (
    pack_swa_kv,
    unpack_swa_kv,
)
from hyper_parallel.components.quantization.tensor.csa2_cache_tensor import pack_csa2_cache

# Expected packed byte width per logical width.
#
# Indexer K  = E2M1 data (2/byte) + E8M0 scale (1 per 32-element block, stored
#              in pairs of 2 per 64-element tile: MX_SCALE_SLOTS=2, MX_SCALE_TILE=64).
# Main KV    = E2M1 data (2/byte) + E4M3 scale (1 per 16-element group).
# SWA KV     = E4M3 data (1/byte) + E8M0 scale (1 per 32-element group, pair-padded).
_INDEXER_PACKED = {32: 18, 64: 34, 96: 52, 128: 68, 256: 136, 512: 272}
_MAIN_PACKED = {16: 9, 32: 18, 128: 72, 256: 144, 512: 288}
_SWA_PACKED = {32: 34, 64: 66, 96: 100, 128: 132, 256: 264, 512: 528}
_MAIN_BF16_PACKED = {16: 32, 32: 32, 128: 96, 256: 160, 512: 320}
_SWA_BF16_PACKED = {32: 64, 64: 96, 96: 128, 128: 160, 256: 288, 512: 544}

# Format-inherent FP4/FP8 reconstruction tolerance. E2M1 keeps a 2-bit mantissa,
# so its round-trip error reaches tens of percent for values near a bin edge;
# E4M3 keeps a 4-bit mantissa (≈6%).
_E2M1_RTOL = 0.5
_E2M1_ATOL = 0.2
_E4M3_RTOL = 0.1
_E4M3_ATOL = 0.05


def _npu_available() -> bool:
    """Report whether a real Ascend device and torch_npu are importable."""
    try:
        import torch_npu  # pylint: disable=C0415,W0611
    except (ImportError, OSError, RuntimeError):
        return False
    return torch.npu.is_available()


@unittest.skipUnless(_npu_available(), "requires Ascend NPU")
class TestWidthMatrixNPU(unittest.TestCase):
    """Native torch_npu MX/FP8 operators produce the documented layout at every width."""

    def test_indexer_q_qdq_round_trip_across_widths(self):
        torch.manual_seed(0)
        device = torch.device("npu:0")
        for width in _INDEXER_PACKED:
            query = torch.randn(8, width, dtype=torch.bfloat16, device=device)
            output = _native_qdq(query)
            self.assertEqual(output.shape, query.shape)
            self.assertTrue(torch.isfinite(output).all())
            torch.testing.assert_close(output, query, rtol=_E2M1_RTOL, atol=_E2M1_ATOL)

    def test_indexer_k_byte_count_and_round_trip(self):
        torch.manual_seed(0)
        device = torch.device("npu:0")
        for width, expected in _INDEXER_PACKED.items():
            key = torch.randn(2, 4, width, dtype=torch.bfloat16, device=device)
            packed = pack_indexer_k(key)
            self.assertEqual(packed.shape[-1], expected)
            output = unpack_indexer_k(packed, width, torch.bfloat16)
            self.assertEqual(output.shape, key.shape)
            self.assertTrue(torch.isfinite(output).all())
            torch.testing.assert_close(output, key, rtol=_E2M1_RTOL, atol=_E2M1_ATOL)

    def test_main_kv_byte_count_and_round_trip(self):
        torch.manual_seed(0)
        device = torch.device("npu:0")
        for width, expected in _MAIN_PACKED.items():
            kv = torch.randn(2, 3, width, dtype=torch.bfloat16, device=device)
            packed = pack_main_kv(kv)
            self.assertEqual(packed.shape[-1], expected)
            output = unpack_main_kv(packed, width, kv.dtype)
            self.assertEqual(output.shape, kv.shape)
            self.assertTrue(torch.isfinite(output).all())
            torch.testing.assert_close(output, kv, rtol=_E2M1_RTOL, atol=_E2M1_ATOL)

    def test_swa_kv_byte_count_and_round_trip(self):
        torch.manual_seed(0)
        device = torch.device("npu:0")
        for width, expected in _SWA_PACKED.items():
            kv = torch.randn(2, 3, width, dtype=torch.bfloat16, device=device)
            packed = pack_swa_kv(kv)
            self.assertEqual(packed.shape[-1], expected)
            output = unpack_swa_kv(packed, width)
            self.assertEqual(output.shape, kv.shape)
            self.assertTrue(torch.isfinite(output).all())
            torch.testing.assert_close(output, kv, rtol=_E4M3_RTOL, atol=_E4M3_ATOL)

    def test_bf16_scale_byte_count_and_round_trip(self):
        """E8M7 scales retain packed data, with 32-byte aligned physical rows."""
        torch.manual_seed(0)
        device = torch.device("npu:0")
        for width, expected in _MAIN_BF16_PACKED.items():
            with self.subTest(cache="main_kv", width=width):
                kv = torch.randn(2, 3, width, dtype=torch.bfloat16, device=device)
                packed = pack_main_kv(kv, "e8m7")
                self.assertEqual(packed.shape[-1], expected)
                output = unpack_main_kv(packed, width, kv.dtype, "e8m7")
                torch.testing.assert_close(output, kv, rtol=_E2M1_RTOL, atol=_E2M1_ATOL)
        for width, expected in _SWA_BF16_PACKED.items():
            with self.subTest(cache="swa_kv", width=width):
                kv = torch.randn(2, 3, width, dtype=torch.bfloat16, device=device)
                packed = pack_swa_kv(kv, "e8m7")
                self.assertEqual(packed.shape[-1], expected)
                output = unpack_swa_kv(packed, width, "e8m7")
                torch.testing.assert_close(output, kv, rtol=_E4M3_RTOL, atol=_E4M3_ATOL)

    def test_bf16_scales_and_training_ste(self):
        """Packed BF16 scale bytes encode group maxima; read gradients reach KV."""
        device = torch.device("npu:0")
        for name, maximum, group, expected_width in (
            ("main_kv", 6.0, 16, 32),
            ("swa_kv", 448.0, 32, 96),
        ):
            with self.subTest(cache=name):
                source = torch.cat((torch.full((1, group), maximum, device=device),
                                    torch.full((1, group), maximum / 2, device=device)), dim=-1)
                source = source.to(torch.bfloat16).requires_grad_()
                cache = pack_csa2_cache(source, name, "e8m7")
                self.assertEqual(cache.packed.shape[-1], expected_width)
                scale_offset = source.shape[-1] // (2 if name == "main_kv" else 1)
                scales = cache.packed[..., scale_offset:scale_offset + 4].contiguous().view(torch.bfloat16)
                torch.testing.assert_close(scales, torch.tensor([[1.0, 0.5]], dtype=torch.bfloat16, device=device))
                cache.dequantize().sum().backward()
                torch.testing.assert_close(source.grad, torch.ones_like(source))


if __name__ == "__main__":
    unittest.main()
