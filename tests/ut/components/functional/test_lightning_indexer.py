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
"""Hardware-free checks for bounded shared-page addressing."""

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.components.functional import lightning_indexer as indexer
from hyper_parallel.components.functional.sparse_flash_mla import csa_tnd_plan


class TestPagedIndexerCache(unittest.TestCase):
    """Test storage accounting and page addresses independently of native kernels."""

    def tearDown(self) -> None:
        """Do not retain this test's cache entries across other tests."""
        indexer._PAGED_CACHE.clear()
        indexer._paged_key_plan.cache_clear()

    def test_cache_evicts_by_storage_bytes_and_skips_oversized_values(self):
        """LRU accounting includes a view's backing storage, without rejecting its use."""
        cache = indexer._PagedTensorCache(32)
        first = cache.get((1,), lambda: torch.arange(4, dtype=torch.int32))
        cache.get((2,), lambda: torch.arange(4, dtype=torch.int32))
        self.assertIs(cache.get((1,), lambda: None), first)
        cache.get((3,), lambda: torch.arange(4, dtype=torch.int32))
        self.assertEqual(tuple(cache.entries), ((1,), (3,)))
        self.assertEqual(cache.retained_bytes, 32)
        large_view = cache.get((4,), lambda: torch.arange(16, dtype=torch.int32)[:1])
        self.assertEqual(large_view.numel(), 1)
        self.assertNotIn((4,), cache.entries)
        self.assertEqual(cache.retained_bytes, 32)
        cache.clear()
        self.assertEqual(cache.retained_bytes, 0)

    def test_page_addresses_preserve_repeated_prefixes_without_cumulative_k(self):
        """r1/r2 share pages and do not form an overflowing sum of logical K lengths."""
        device = torch.device("cpu")
        for ratio in (1, 2):
            with self.subTest(ratio=ratio), patch.object(indexer, "_uses_indexer_metadata", return_value=False):
                segments = ((0, 4, 3, 133, 0), (4, 8, 3, 261, ratio - 1), (8, 12, 263, 393, 0))
                plan = csa_tnd_plan(segments, ratio)
                pages = indexer._paged_key_plan(plan, 400)
                table = indexer._paged_block_table(pages, device)
                torch.testing.assert_close(table, torch.tensor([[0, 1, 0], [0, 1, 2], [3, 4, 0]], dtype=torch.int32))
                bank = torch.arange(400).view(1, 400, 1)
                physical = pages.key(bank)
                for row, (start, end) in enumerate(plan.k_ranges):
                    restored = physical[table[row].long()].reshape(-1)[:end-start]
                    torch.testing.assert_close(restored, bank[0, start:end, 0])
                options, _ = indexer._paged_indexer_metadata(plan, device, 8, 128, 512, ratio)
                self.assertNotIn("block_table", options)
                self.assertNotIn("cu_seqlens_k", options)
                long_plan = csa_tnd_plan(tuple((i, i+1, 0, 524288, ratio-1) for i in range(4097)), ratio)
                lengths = indexer._paged_indexer_lengths(long_plan, device)
                self.assertEqual(int(lengths[0][-1]), 4097)
                self.assertLessEqual(indexer._PAGED_CACHE.retained_bytes, indexer._PAGED_CACHE.maximum_bytes)
