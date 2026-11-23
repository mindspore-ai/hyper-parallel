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
"""Numeric and gradient contracts for packed paper-format Main KV."""

import unittest

import torch

from hyper_parallel.components.quantization.config import CSA2MainKVFormat
from hyper_parallel.components.quantization.functional.main_kv_cache_qat import MainKVCacheQATStrategy
from hyper_parallel.components.quantization.quantizers.main_kv import pack_main_kv
from tests.ut.components.quantization.csa2_cpu_reference import install_cache_standins


class TestMainKVQDQ(unittest.TestCase):
    """E2M1 values, E4M3 scales, and the producer-side STE."""

    def setUp(self) -> None:
        """Replace native codecs only inside CPU cache-flow tests."""
        install_cache_standins(self)

    def test_cpu_input_has_no_production_quantizer_fallback(self):
        """A selected format must require NPU execution."""
        with self.assertRaisesRegex(RuntimeError, "requires NPU"):
            pack_main_kv(torch.zeros(1, 16))

    def test_main_kv_is_independent_and_uses_identity_ste(self):
        """Configured Main KV packing preserves producer gradients."""
        kv = torch.randn(2, 3, 32, requires_grad=True)
        with self.assertRaisesRegex(TypeError, "requires a CSA2MainKVFormat"):
            MainKVCacheQATStrategy(None)
        strategy = MainKVCacheQATStrategy(CSA2MainKVFormat("fp4_e2m1", "e4m3"))
        output = strategy.pack(kv).dequantize()
        self.assertEqual(output.shape, kv.shape)
        output.sum().backward()
        self.assertTrue(torch.equal(kv.grad, torch.ones_like(kv)))

    def test_main_kv_rejects_partial_group(self):
        """A channel group cannot spill into the next token."""
        strategy = MainKVCacheQATStrategy(CSA2MainKVFormat("fp4_e2m1", "e4m3"))
        with self.assertRaisesRegex(ValueError, "not divisible"):
            strategy.pack(torch.ones(2, 17))

    def test_main_kv_accepts_empty_sequence(self):
        """An empty CP shard has no group maximum to compute."""
        kv = torch.empty(1, 0, 32)
        strategy = MainKVCacheQATStrategy(CSA2MainKVFormat("fp4_e2m1", "e4m3"))
        self.assertEqual(strategy.pack(kv).dequantize().shape, kv.shape)


if __name__ == "__main__":
    unittest.main()
