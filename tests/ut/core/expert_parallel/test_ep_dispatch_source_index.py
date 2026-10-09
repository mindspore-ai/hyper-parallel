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
"""Unit tests for the routed-slot token id builder of the EP dispatch.

``_routed_slot_token_ids`` selects between the eager ``repeat_interleave`` and
two kernels that spread over every vector core (``HP_EP_SOURCE_INDEX``).  The
fast paths are only safe if they are element-wise identical to the eager call
for every shape, so that is what these tests pin down.

All tests run on CPU without any distributed setup.
"""
import os
import unittest
from unittest.mock import patch

import torch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from hyper_parallel.distributed.expert_parallel.experts import (  # noqa: E402  pylint: disable=wrong-import-position
    _routed_slot_token_ids,
)


class TestRoutedSlotTokenIds(unittest.TestCase):
    """Pin every mode of the routed-slot token id builder to the eager result."""

    @staticmethod
    def _eager(token_count: int, experts_per_token: int) -> torch.Tensor:
        """Reference mapping: one entry per (token, expert) slot."""
        return torch.arange(token_count).repeat_interleave(experts_per_token)

    def _assert_modes_match(self, token_count: int, experts_per_token: int) -> None:
        """Assert legacy/expand/div agree on shape, dtype, and every element."""
        expected = self._eager(token_count, experts_per_token)
        for mode in ("legacy", "expand", "div"):
            with self.subTest(mode=mode, token_count=token_count, top_k=experts_per_token):
                with patch(
                    "hyper_parallel.distributed.expert_parallel.experts._SOURCE_INDEX_MODE",
                    mode,
                ):
                    actual = _routed_slot_token_ids(token_count, experts_per_token, None)
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, torch.int64)
                self.assertTrue(actual.is_contiguous())
                self.assertTrue(torch.equal(actual, expected))

    def test_top_k_8_at_sequence_length_8192(self):
        """The profiled production shape: T=8192, K=8."""
        self._assert_modes_match(8192, 8)

    def test_single_route_per_token(self):
        """K=1 degenerates to the identity mapping."""
        self._assert_modes_match(64, 1)

    def test_ragged_values(self):
        """Odd token counts and top-k values keep the element order."""
        for token_count, experts_per_token in ((1, 3), (7, 3), (5, 5), (33, 7), (128, 129)):
            with self.subTest(token_count=token_count, top_k=experts_per_token):
                self._assert_modes_match(token_count, experts_per_token)

    def test_first_and_last_slot_boundaries(self):
        """Slot ``t * K + i`` maps to token ``t`` at both ends of the tensor."""
        token_count, experts_per_token = 8192, 8
        with patch(
            "hyper_parallel.distributed.expert_parallel.experts._SOURCE_INDEX_MODE",
            "expand",
        ):
            ids = _routed_slot_token_ids(token_count, experts_per_token, None)
        self.assertEqual(ids[0].item(), 0)
        self.assertEqual(ids[experts_per_token - 1].item(), 0)
        self.assertEqual(ids[experts_per_token].item(), 1)
        self.assertEqual(ids[-1].item(), token_count - 1)

    def test_empty_route_falls_back_to_eager(self):
        """K=0 (no routed slot) still returns the eager empty tensor."""
        with patch(
            "hyper_parallel.distributed.expert_parallel.experts._SOURCE_INDEX_MODE",
            "div",
        ):
            ids = _routed_slot_token_ids(4, 0, None)
        self.assertEqual(ids.numel(), 0)
        self.assertEqual(ids.dtype, torch.int64)

    def test_unknown_mode_raises(self):
        """A typo in the env knob fails loudly instead of silently degrading."""
        with patch(
            "hyper_parallel.distributed.expert_parallel.experts._SOURCE_INDEX_MODE",
            "expnad",
        ):
            with self.assertRaisesRegex(ValueError, "HP_EP_SOURCE_INDEX"):
                _routed_slot_token_ids(8, 2, None)

    def test_indexing_agrees_under_a_dispatch_order(self):
        """The ids select the same tokens the eager ids select, after a permutation."""
        token_count, experts_per_token = 1024, 8
        hidden = torch.arange(token_count, dtype=torch.float32).unsqueeze(1)
        dispatch_order = torch.randperm(token_count * experts_per_token, generator=torch.Generator().manual_seed(0))
        with patch(
            "hyper_parallel.distributed.expert_parallel.experts._SOURCE_INDEX_MODE",
            "expand",
        ):
            fast_ids = _routed_slot_token_ids(token_count, experts_per_token, None)
        eager_ids = self._eager(token_count, experts_per_token)
        self.assertTrue(torch.equal(hidden[fast_ids[dispatch_order]], hidden[eager_ids[dispatch_order]]))


if __name__ == "__main__":
    unittest.main()
