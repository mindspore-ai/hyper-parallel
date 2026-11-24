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
from hyper_parallel.components.functional.dsa_indexer import dsa_indexer_select
from hyper_parallel.components.functional.sparse_flash_mla import csa_tnd_plan


from tests.common.mark_utils import arg_mark

class TestPagedIndexerCache(unittest.TestCase):
    """Test storage accounting and page addresses independently of native kernels."""

    def tearDown(self) -> None:
        """Do not retain this test's cache entries across other tests."""
        indexer._PAGED_CACHE.clear()
        indexer._paged_key_plan.cache_clear()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_cache_evicts_by_storage_bytes_and_skips_oversized_values(self):
        """
        Feature: Shared packed-attention planning and routing.
        Description: LRU accounting includes a view's backing storage, without rejecting its use.
        Expectation: Eviction respects physical bytes and oversized tensors are not retained.
        """
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_page_addresses_preserve_repeated_prefixes_without_cumulative_k(self):
        """
        Feature: Shared packed-attention planning and routing.
        Description: r1/r2 share pages and do not form an overflowing sum of logical K lengths.
        Expectation: Repeated logical prefixes resolve to shared physical pages and correct masks.
        """
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

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard",
              essential_mark="essential")
    def test_dsa_keeps_document_local_order_with_shared_pages(self):
        """Feature: Shared LI packing with different consumer index conventions.

        Description: Decode ordinary and paged keys in a CPU LI stub for repeated
            prefixes, including a second document with nonzero bank origin.
        Expectation: DSA keeps score-ordered document IDs; CSA adds bank offsets
            and sorts IDs. Both storage modes select the same keys.
        """
        def _native(query, key, weights, topk, **options):
            query_ends = options["cu_seqlens_q"].tolist()
            key_ends = options.get("cu_seqlens_k")
            output, key_offset = [], 0
            for row, (first, last) in enumerate(zip(query_ends, query_ends[1:])):
                if options["layout_k"] == "PA_BBND":
                    count = int(options["seqused_k"][row])
                    keys = key[options["block_table"][row].long()].flatten(0, 1)[:count, 0]
                else:
                    count = int(key_ends[row + 1]) - key_offset
                    keys = key[key_offset:key_offset + count, 0]
                    key_offset += count
                scores = (torch.einsum("qhd,kd->qhk", query[first:last], keys).relu()
                          * weights[first:last, :, None]).sum(1)
                visible = torch.arange(count)[None, :] <= torch.arange(last-first)[:, None] + count-(last-first)
                scores = scores.masked_fill(~visible, -torch.inf)
                ids = scores.argsort(dim=-1, descending=True, stable=True)[:, :topk]
                output.append(ids.int().unsqueeze(1))
            return torch.cat(output), None

        generator = torch.Generator().manual_seed(23)
        query = torch.randn(1, 7, 2, 8, generator=generator)
        key = torch.randn(1, 400, 8, generator=generator)
        weights = torch.randn(1, 7, 2, generator=generator)
        segments = ((0, 2, 3, 133, 0), (2, 5, 3, 261, 0), (5, 7, 263, 393, 0))
        results = []
        with patch.object(indexer, "_uses_indexer_metadata", return_value=False), \
                patch.object(indexer, "_load_indexer_op", return_value=_native):
            for paged in (False, True):
                selected = dsa_indexer_select(query, key.unsqueeze(2), weights, segments, 8, paged_key=paged)
                results.append(selected)
                csa = indexer.lightning_indexer(query, key, weights, 1, 8, segments, paged_key=paged)
                offsets = torch.tensor([3, 3, 3, 3, 3, 263, 263], dtype=torch.int32).unsqueeze(1)
                expected = (selected[:, 0] + offsets).sort(-1).values.unsqueeze(0)
                torch.testing.assert_close(csa, expected, rtol=0, atol=0)
        torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
