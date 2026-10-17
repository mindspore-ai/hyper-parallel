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
"""The BNSD causal path must not rebuild an [S, S] mask on every call.

A 61-layer stack builds the mask inside the kernel entry point, so it pays 61
constructions and 61 device allocations per training step for a tensor that
never changes. The training shape (``q_len == k_len``, no sliding window) is the
plain left-up causal triangle, which the NPU kernel produces itself from a
constant placeholder under ``sparse_mode=2``.

The guard around that shortcut is what these tests pin: decoding
(``q_len < k_len``, causally shifted right) and sliding windows are NOT that
triangle, and must keep the materialised mask. Swapping them to the left-up mode
would attend to the wrong positions silently -- the symptom would look like a
training-quality problem, not a crash.
"""
# pylint: disable=wrong-import-position

import os
import unittest
from typing import Optional

import torch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from hyper_parallel.components.functional import npu_fusion_attention
from tests.common.mark_utils import arg_mark

_MARK = {
    "plat_marks": ["cpu_linux", "cpu_macos"],
    "level_mark": "level0",
    "card_mark": "allcards",
    "essential_mark": "essential",
}


def _qkv(batch: int, heads: int, query_length: int, key_length: int, dim: int) -> tuple:
    """Build a BNSD q/k/v triple with the given (possibly unequal) lengths."""
    shape_q = (batch, heads, query_length, dim)
    shape_kv = (batch, heads, key_length, dim)
    return (
        torch.zeros(shape_q),
        torch.zeros(shape_kv),
        torch.zeros(shape_kv),
    )


def _prepare(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_packed: bool = False,
    is_causal: bool = True,
    sliding_window: Optional[int] = None,
    sparse_mode: int = 0,
) -> tuple:
    """Call the layout/mask selector and return ``(mask, sparse_mode)``."""
    _, _, _, _, mask, resolved = npu_fusion_attention._prepare_attention_inputs(  # pylint: disable=W0212
        query,
        key,
        value,
        None,
        is_packed=is_packed,
        is_causal=is_causal,
        sliding_window=sliding_window,
        sparse_mode=sparse_mode,
    )
    return mask, resolved


def _cached(device: torch.device) -> torch.Tensor:
    """Return the module's cached placeholder for ``device``."""
    return npu_fusion_attention._compressed_causal_mask(device)  # pylint: disable=W0212


class TestCausalMaskCaching(unittest.TestCase):
    """Cache the training-case mask; keep the materialised one where it matters."""

    def setUp(self) -> None:
        """Start every test from an empty placeholder cache."""
        npu_fusion_attention._COMPRESSED_CAUSAL_MASKS.clear()  # pylint: disable=W0212

    def tearDown(self) -> None:
        """Leave no cached device tensor behind for the next test."""
        npu_fusion_attention._COMPRESSED_CAUSAL_MASKS.clear()  # pylint: disable=W0212

    @arg_mark(**_MARK)
    def test_training_shape_uses_the_causal_sparse_mode(self):
        """Feature: NPU causal sparse mode selection.
        Description: prepare a training-shaped q/k/v (q_len == k_len, no window).
        Expectation: sparse_mode 2 with the left-up placeholder mask.
        """
        query, key, value = _qkv(1, 2, 8, 8, 4)
        mask, sparse_mode = _prepare(query, key, value)
        expected = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1)

        self.assertEqual(
            sparse_mode, 2, f"training shape must take the causal sparse mode, got {sparse_mode}"
        )
        self.assertIs(
            mask.dtype, torch.bool, f"the placeholder must stay bool, got {mask.dtype}"
        )
        self.assertTrue(
            torch.equal(mask, expected),
            f"the placeholder must be the left-up triangle, got {mask.nonzero().shape[0]} "
            f"blocked of {expected.nonzero().shape[0]}",
        )

    @arg_mark(**_MARK)
    def test_the_placeholder_is_shared_across_calls(self):
        """Feature: NPU causal mask caching.
        Description: prepare the same training shape twice.
        Expectation: one allocation, reused by every layer and step.
        """
        query, key, value = _qkv(1, 2, 8, 8, 4)
        first, _ = _prepare(query, key, value)
        second, _ = _prepare(query, key, value)

        self.assertIs(
            first,
            second,
            "two calls with the same shape must share one placeholder, not rebuild it",
        )
        self.assertIs(
            first,
            _cached(query.device),
            "the returned mask must be the cached placeholder for the device",
        )
        self.assertEqual(
            len(npu_fusion_attention._COMPRESSED_CAUSAL_MASKS),  # pylint: disable=W0212
            1,
            f"exactly one placeholder must be cached, got "
            f"{len(npu_fusion_attention._COMPRESSED_CAUSAL_MASKS)}",  # pylint: disable=W0212
        )

    @arg_mark(**_MARK)
    def test_different_sequence_lengths_share_one_placeholder(self):
        """Feature: NPU causal mask caching.
        Description: prepare a short and a long training shape.
        Expectation: the placeholder describes a pattern, not a per-position mask, so it is shared.
        """
        short = _prepare(*_qkv(1, 2, 8, 8, 4))[0]
        long = _prepare(*_qkv(1, 2, 512, 512, 4))[0]

        self.assertIs(
            short,
            long,
            f"sequence length must not change the placeholder identity: short={short.shape}, "
            f"long={long.shape}",
        )

    @arg_mark(**_MARK)
    def test_shifted_decoding_keeps_the_materialised_mask(self):
        """Feature: NPU causal mask selection for decoding.
        Description: prepare a decoding shape (q_len < k_len), which is right-aligned.
        Expectation: mode 2 is not taken; the materialised right-aligned mask is returned.
        """
        query, key, value = _qkv(1, 2, 2, 8, 4)
        mask, sparse_mode = _prepare(query, key, value)

        self.assertEqual(
            sparse_mode, 0, f"decoding must not take the left-up mode, got {sparse_mode}"
        )
        self.assertEqual(
            tuple(mask.shape),
            (2, 8),
            f"the materialised decoding mask must be [q_len, k_len], got {tuple(mask.shape)}",
        )
        self.assertIsNot(
            mask,
            _cached(query.device),
            "the decoding mask is not the left-up placeholder and must not be replaced by it",
        )
        # Row 0 (query position 6 after the right shift) may see keys 0..6.
        self.assertTrue(
            torch.equal(mask[0].bool(), torch.arange(8) > 6),
            f"row 0 must allow keys <= 6, got {mask[0].bool().tolist()}",
        )

    @arg_mark(**_MARK)
    def test_sliding_window_keeps_the_materialised_mask(self):
        """Feature: NPU causal mask selection for sliding windows.
        Description: prepare a training shape with a sliding window.
        Expectation: a band is not a triangle, so the materialised mask is returned.
        """
        query, key, value = _qkv(1, 2, 8, 8, 4)
        mask, sparse_mode = _prepare(query, key, value, sliding_window=2)

        self.assertEqual(
            sparse_mode, 0, f"a sliding window must not take the left-up mode, got {sparse_mode}"
        )
        self.assertEqual(
            tuple(mask.shape),
            (8, 8),
            f"the materialised window mask must be [S, S], got {tuple(mask.shape)}",
        )
        self.assertIsNot(
            mask,
            _cached(query.device),
            "a band cannot be expressed by the left-up placeholder",
        )
        # Position 3 attends to 2 and 3 only: everything above the diagonal and
        # everything below the window is blocked.
        self.assertTrue(
            torch.equal(
                mask[3].bool(),
                torch.tensor([True, True, False, False, True, True, True, True]),
            ),
            f"unexpected window mask row: {mask[3].bool().tolist()}",
        )

    @arg_mark(**_MARK)
    def test_packed_branch_reuses_the_same_placeholder(self):
        """Feature: NPU packed (TND) causal mask.
        Description: prepare a packed training shape.
        Expectation: the TND branch shares the placeholder instead of allocating its own copy.
        """
        query, key, value = _qkv(1, 2, 8, 8, 4)
        mask, sparse_mode = _prepare(query, key, value, is_packed=True)

        self.assertEqual(
            sparse_mode, 3, f"the packed branch keeps mode 3, got {sparse_mode}"
        )
        self.assertIs(
            mask,
            _cached(query.device),
            "the packed branch must reuse the cached placeholder, not rebuild it",
        )

    @arg_mark(**_MARK)
    def test_a_supplied_mask_is_still_converted_and_left_alone(self):
        """Feature: NPU causal mask selection with an explicit mask.
        Description: prepare a training shape together with an attention mask.
        Expectation: the shortcut does not shadow the explicit-mask path.
        """
        query, key, value = _qkv(1, 2, 8, 8, 4)
        allowed = torch.ones(1, 1, 8, 8, dtype=torch.bool)
        _, _, _, _, mask, sparse_mode = npu_fusion_attention._prepare_attention_inputs(  # pylint: disable=W0212
            query,
            key,
            value,
            allowed,
            is_packed=False,
            is_causal=True,
            sliding_window=None,
            sparse_mode=0,
        )

        self.assertEqual(
            sparse_mode, 0, f"an explicit mask keeps the caller's sparse mode, got {sparse_mode}"
        )
        self.assertIs(
            mask.dtype, torch.bool, f"the converted mask must be bool, got {mask.dtype}"
        )
        self.assertFalse(
            bool(mask.any()),
            f"an all-allowed mask must convert to all-False, got {mask.sum()} blocked",
        )


if __name__ == "__main__":
    unittest.main()
