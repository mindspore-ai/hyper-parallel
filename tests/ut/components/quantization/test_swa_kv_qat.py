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
"""Numeric and autograd contracts for optional packed SWA FP8 KV."""

import unittest

import torch

from hyper_parallel.components.quantization.config import CSA2SWAKVFormat
from hyper_parallel.components.quantization.functional.swa_kv_qat import SWAKVQATStrategy
from hyper_parallel.components.quantization.quantizers.swa_kv import pack_swa_kv
from tests.ut.components.quantization.csa2_cpu_reference import install_cache_standins


class TestSWAKVQAT(unittest.TestCase):
    """The SWA policy activates independently of Main KV and Indexer Q/K."""

    def setUp(self) -> None:
        """Replace native codecs only inside CPU cache-flow tests."""
        install_cache_standins(self)

    def test_cpu_input_has_no_production_quantizer_fallback(self):
        """A selected SWA format must require NPU execution."""
        with self.assertRaisesRegex(RuntimeError, "requires NPU"):
            pack_swa_kv(torch.zeros(1, 32, dtype=torch.bfloat16))

    def test_fp8_ste(self):
        """Configured E4M3 packing passes read gradients to the producer."""
        kv = torch.randn(2, 3, 32, dtype=torch.bfloat16, requires_grad=True)
        with self.assertRaisesRegex(TypeError, "requires a CSA2SWAKVFormat"):
            SWAKVQATStrategy(None)
        output = SWAKVQATStrategy(CSA2SWAKVFormat("fp8_e4m3")).pack(kv).dequantize()
        self.assertEqual(output.shape, kv.shape)
        output.sum().backward()
        self.assertTrue(torch.equal(kv.grad, torch.ones_like(kv)))

    def test_partial_group_is_rejected(self):
        """SWA FP8 QDQ cannot cross a token boundary to fill its group."""
        with self.assertRaisesRegex(ValueError, "divisible by 32"):
            SWAKVQATStrategy(CSA2SWAKVFormat("fp8_e4m3")).pack(
                torch.zeros(2, 33, dtype=torch.bfloat16)
            )

    def test_empty_local_sequence_avoids_group_reduction(self):
        """A CP shard without tokens can still pass through SWA QDQ."""
        kv = torch.empty(1, 0, 32, dtype=torch.bfloat16)
        output = SWAKVQATStrategy(CSA2SWAKVFormat("fp8_e4m3")).pack(kv).dequantize()
        self.assertEqual(output.shape, kv.shape)


if __name__ == "__main__":
    unittest.main()
