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
"""Exact bounded token counts, masking and integer fallback."""

import unittest
from unittest.mock import patch

import torch

from hyper_parallel.components.functional.token_histogram import token_histogram
from hyper_parallel.distributed.expert_parallel.experts import _expert_token_counts


class TestTokenHistogram(unittest.TestCase):
    """Compare routing histograms with independent integer counting."""

    def test_empty_concentrated_and_noncontiguous_routes(self) -> None:
        """Preserve empty bins, repeated IDs, dtype and arbitrary input layouts."""
        for bins in (1, 2, 8, 16, 32):
            for dtype in (torch.int32, torch.int64):
                cases = [torch.empty((0, 2), dtype=dtype), torch.zeros((19, 3), dtype=dtype),
                         (torch.arange(120).reshape(12, 10) % bins).to(dtype).transpose(0, 1)]
                for indices in cases:
                    with self.subTest(bins=bins, dtype=dtype, shape=indices.shape):
                        expected = torch.bincount(indices.reshape(-1).long(), minlength=bins)
                        actual = token_histogram(indices, bins)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(_expert_token_counts(indices, bins), expected, rtol=0, atol=0)
                        self.assertEqual(actual.dtype, torch.int64)
                        self.assertEqual(actual.device, indices.device)

    def test_masked_out_of_range_ids_and_single_bin(self) -> None:
        """Ignore masked sentinels without including them in the last valid bin."""
        indices = torch.tensor([[0, -9, 2], [2, 999, 0]]).transpose(0, 1)
        mask = torch.tensor([[True, False, True], [True, False, False]]).transpose(0, 1)
        for limit in (1 << 24, 2):
            with self.subTest(limit=limit), patch(
                    "hyper_parallel.components.functional.token_histogram._FP32_EXACT_INTEGER_LIMIT", limit):
                torch.testing.assert_close(token_histogram(indices, 3, mask), torch.tensor([1, 0, 2]))
                torch.testing.assert_close(token_histogram(indices, 3, torch.zeros_like(mask)), torch.zeros(3).long())
                torch.testing.assert_close(token_histogram(torch.tensor([0, -1]), 1, torch.tensor([True, False])),
                                           torch.tensor([1]))

    def test_shape_limits_select_integer_fallback(self) -> None:
        """Keep exact output when either bin count or entry count exceeds the bound."""
        indices = torch.tensor([0, 0, 0, 3])
        module = "hyper_parallel.components.functional.token_histogram"
        with patch(f"{module}._FP32_EXACT_INTEGER_LIMIT", 4), patch("torch.histc", wraps=torch.histc) as histc:
            torch.testing.assert_close(token_histogram(indices, 4), torch.tensor([3, 0, 0, 1]))
            self.assertEqual(histc.call_count, 1)
            torch.testing.assert_close(token_histogram(indices, 5), torch.tensor([3, 0, 0, 1, 0]))
            torch.testing.assert_close(token_histogram(indices.repeat(2), 4), torch.tensor([6, 0, 0, 2]))
            torch.testing.assert_close(token_histogram(indices, 4, torch.ones_like(indices).bool()),
                                       torch.tensor([3, 0, 0, 1]))
            self.assertEqual(histc.call_count, 1)

    def test_count_above_fp32_exact_integer_limit(self) -> None:
        """Do not round a concentrated bin count at 2**24 + 1."""
        entries = (1 << 24) + 1
        actual = token_histogram(torch.zeros(entries, dtype=torch.int8), 2)
        torch.testing.assert_close(actual, torch.tensor([entries, 0]), rtol=0, atol=0)

    def test_invalid_metadata(self) -> None:
        """Reject malformed metadata before invoking a counting kernel."""
        for bins in (0, -1, True, 1.5):
            with self.subTest(bins=bins), self.assertRaises(ValueError):
                token_histogram(torch.tensor([0]), bins)
        with self.assertRaises(ValueError):
            token_histogram(torch.tensor([0.0]), 2)
        for mask in (torch.ones(2), torch.ones(3).bool()):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                token_histogram(torch.tensor([0, 1]), 2, mask)
