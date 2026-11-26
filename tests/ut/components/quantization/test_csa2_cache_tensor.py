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
"""Physical cache and selected-read contracts for CSA2 training."""

import unittest

import torch

from hyper_parallel.components.quantization.ops.npu_fp8 import (
    allow_fully_written_cache_allocation,
)
from hyper_parallel.components.quantization.tensor.csa2_cache_tensor import pack_csa2_cache
from tests.ut.components.quantization.csa2_cpu_reference import install_cache_standins


class TestCSA2CacheTensor(unittest.TestCase):
    """Packed cache bytes must remain physical until an explicit read."""

    def setUp(self) -> None:
        """Use a CPU codec only for wrapper ownership and gradient tests."""
        install_cache_standins(self)

    def test_empty_and_invalid_selection_keeps_zero_main_gradient(self):
        """A rank with no compressed prefix can still run a sink-only query."""
        main = torch.empty(1, 0, 16, requires_grad=True)
        cache = pack_csa2_cache(main, "main_kv")
        selected = cache.dequantize_selected(torch.full((1, 2, 3), -1))
        self.assertEqual(selected.shape, (1, 2, 3, 16))
        self.assertTrue(torch.equal(selected, torch.zeros_like(selected)))
        self.assertEqual(cache.packed_nbytes, 0)
        selected.sum().backward()
        self.assertEqual(main.grad.numel(), 0)

    def test_selected_indices_require_integer_dtype(self):
        """Fractional or Boolean indices must not silently select a different row."""
        cache = pack_csa2_cache(torch.randn(1, 2, 16), "main_kv")
        for indices in (torch.tensor([[[0.5]]]), torch.tensor([[[True]]])):
            with self.subTest(dtype=indices.dtype):
                with self.assertRaisesRegex(ValueError, "int32 or int64"):
                    cache.dequantize_selected(indices)

    def test_explicit_empty_scale_is_rejected(self):
        """Only an omitted scale format receives the cache default."""
        with self.assertRaisesRegex(ValueError, "Unsupported main_kv scale format"):
            pack_csa2_cache(torch.randn(1, 2, 16), "main_kv", "")

    def test_swa_packed_rows_include_e8m0_scales_and_preserve_gradient(self):
        """Each FP8 group has a scale byte and the read uses identity STE."""
        kv = torch.full((1, 2, 64), 0.125, dtype=torch.bfloat16, requires_grad=True)
        cache = pack_csa2_cache(kv, "swa_kv")
        self.assertEqual(cache.packed.dtype, torch.uint8)
        self.assertEqual(cache.packed_nbytes, 2 * (64 + 2))
        self.assertTrue(torch.equal(cache.packed[..., -2:], torch.full((1, 2, 2), 116, dtype=torch.uint8)))
        torch.testing.assert_close(cache.dequantize(), kv)
        cache.dequantize().sum().backward()
        self.assertTrue(torch.equal(kv.grad, torch.ones_like(kv)))

    def test_e8m7_scale_metadata_survives_sparse_reads(self):
        """E8M7 selection uses the matching row layout and keeps its STE edge."""
        for cache_format, shape, expected_width in (
            ("main_kv", (1, 3, 32), 32),
            ("swa_kv", (1, 1, 3, 32), 64),
        ):
            with self.subTest(cache_format=cache_format):
                value = torch.full(shape, 0.125, dtype=torch.bfloat16, requires_grad=True)
                cache = pack_csa2_cache(value, cache_format, "e8m7")
                self.assertEqual(cache.scale_format, "e8m7")
                self.assertEqual(cache.packed.shape[-1], expected_width)
                selected = cache.dequantize_selected(torch.tensor([[[0, 2, -1]]]))
                self.assertEqual(selected.shape, (1, 1, 3, 32))
                self.assertTrue(torch.equal(selected[0, 0, 2], torch.zeros_like(selected[0, 0, 2])))
                selected.sum().backward()
                self.assertTrue(torch.isfinite(value.grad).all())

    def test_swa_selected_read_handles_ragged_windows_and_repeated_indices(self):
        """Only valid local rows contribute, including a short final window."""
        kv = torch.arange(5 * 32, dtype=torch.float32).reshape(1, 1, 5, 32).to(torch.bfloat16)
        kv.requires_grad_()
        cache = pack_csa2_cache(kv, "swa_kv")
        indices = torch.tensor([[[0, 2, -1, 2], [4, 7, -1, -1]]])
        selected = cache.dequantize_selected(indices)
        dense = cache.dequantize().detach()
        torch.testing.assert_close(selected[0, 0, 0], dense[0, 0, 0])
        torch.testing.assert_close(selected[0, 0, 1], dense[0, 0, 2])
        torch.testing.assert_close(selected[0, 1, 0], dense[0, 0, 4])
        self.assertTrue(torch.equal(selected[0, 0, 2], torch.zeros_like(selected[0, 0, 2])))
        self.assertTrue(torch.equal(selected[0, 1, 1], torch.zeros_like(selected[0, 1, 1])))
        selected.sum().backward()
        expected = torch.zeros_like(kv)
        expected[0, 0, 0] = 1
        expected[0, 0, 2] = 2
        expected[0, 0, 4] = 1
        torch.testing.assert_close(kv.grad, expected)


    def test_fp8_allocation_scope_restores_deterministic_prefill(self):
        """FP8 output allocation never changes the policy after its call."""
        original = torch.utils.deterministic.fill_uninitialized_memory
        try:
            for previous in (True, False):
                torch.utils.deterministic.fill_uninitialized_memory = previous
                with allow_fully_written_cache_allocation():
                    self.assertFalse(torch.utils.deterministic.fill_uninitialized_memory)
                    with allow_fully_written_cache_allocation():
                        self.assertFalse(torch.utils.deterministic.fill_uninitialized_memory)
                self.assertEqual(torch.utils.deterministic.fill_uninitialized_memory, previous)
            torch.utils.deterministic.fill_uninitialized_memory = True
            with self.assertRaisesRegex(RuntimeError, "probe"):
                with allow_fully_written_cache_allocation():
                    raise RuntimeError("probe")
            self.assertTrue(torch.utils.deterministic.fill_uninitialized_memory)
        finally:
            torch.utils.deterministic.fill_uninitialized_memory = original

    def test_cache_clone_and_detach_keep_physical_bytes(self):
        """Recompute helpers cannot silently expand the packed cache."""
        value = torch.randn(1, 3, 16, requires_grad=True)
        cache = pack_csa2_cache(value, "main_kv")
        cloned = cache.clone()
        detached = cache.detach()
        for result in (cloned, detached):
            self.assertEqual(result.shape, cache.shape)
            self.assertEqual(result.packed_nbytes, cache.packed_nbytes)
            self.assertEqual(result.cache_format, "main_kv")
            torch.testing.assert_close(result.packed, cache.packed)
        self.assertNotEqual(cloned.packed.data_ptr(), cache.packed.data_ptr())
        self.assertFalse(detached.requires_grad)


if __name__ == "__main__":
    unittest.main()
