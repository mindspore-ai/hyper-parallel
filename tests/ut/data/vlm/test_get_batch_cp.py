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
"""Context-parallel sharding in the merged Omni batch path.

``OmniParallelBatch`` builds a :class:`CPBatchSharder` with the
``OMNI_CP_TOKEN_FIELDS`` / ``OMNI_CP_PAD_VALUES`` rules: every text/token field is
sliced to this rank's contiguous sequence window (padded up so ``2 * cp_size``
divides the sequence), while modality metadata such as ``pixel_values`` and
``image_grid_thw`` stays complete on every CP rank so the vision tower and the
full-sequence media scatter remain consistent. These tests pin that contract on a
real sharder without a process group.
"""
# pylint: disable=wrong-import-position

import importlib.util
import os
import unittest

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch

from tests.common.mark_utils import arg_mark

from hyper_parallel.data.constants import OMNI_CP_PAD_VALUES, OMNI_CP_TOKEN_FIELDS
from hyper_parallel.data.parallel import CPBatchSharder

# Some CI pythons are built without liblzma: the lazy transformers ->
# torchvision import chain dies on "from _lzma import *". The data facade pulls
# transformers.AutoProcessor, so these tests run only where lzma is available.
_HAS_LZMA = importlib.util.find_spec("_lzma") is not None


class _FakeParallelContext:
    """Minimal context exposing the CP coordinates read by the sharder."""

    def __init__(self, cp_world_size: int, cp_rank: int = 0) -> None:
        """Store the CP topology used by CPBatchSharder."""
        self.cp_world_size = cp_world_size
        self.cp_rank = cp_rank


def _make_batch(seq_len=8, pad_tail=2):
    """Build one Omni batch with a padding tail on the labels."""
    input_ids = torch.arange(seq_len).unsqueeze(0)
    labels = input_ids.clone()
    if pad_tail:
        labels[:, seq_len - pad_tail:] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        "labels": labels,
        "loss_mask": (labels >= 0).to(torch.int64),
        "mm_token_type_ids": torch.zeros(1, seq_len, dtype=torch.long),
        "cu_seq_lens": torch.tensor([0, seq_len], dtype=torch.int32),
        "pixel_values": torch.randn(3, 4, 4),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }


def _build_sharder(cp_size, cp_rank=0):
    """Build the sharder exactly as OmniParallelBatch configures it."""
    token_pad_values = {field: OMNI_CP_PAD_VALUES.get(field, 0) for field in OMNI_CP_TOKEN_FIELDS}
    return CPBatchSharder(_FakeParallelContext(cp_size, cp_rank), token_pad_values=token_pad_values)


@unittest.skipIf(not _HAS_LZMA, "python build lacks liblzma (_lzma)")
class TestOmniBatchCpSharding(unittest.TestCase):
    """Token fields shard per CP rank; modality metadata stays complete."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cp1_keeps_the_batch_untouched(self):
        """Without CP the sharder is a pass-through (regression guard)."""
        batch = _make_batch()
        sharded = _build_sharder(1).shard(batch)

        self.assertTrue(torch.equal(sharded["input_ids"], batch["input_ids"]))
        self.assertTrue(torch.equal(sharded["labels"], batch["labels"]))
        self.assertTrue(torch.equal(sharded["pixel_values"], batch["pixel_values"]))
        self.assertTrue(torch.equal(sharded["image_grid_thw"], batch["image_grid_thw"]))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cp2_slices_token_fields_to_the_rank_window(self):
        """Each rank keeps its own contiguous window of every token field."""
        seq_len = 8
        batch = _make_batch(seq_len=seq_len)
        for cp_rank in (0, 1):
            with self.subTest(cp_rank=cp_rank):
                window = slice(cp_rank * 4, (cp_rank + 1) * 4)
                sharded = _build_sharder(2, cp_rank).shard(batch)

                for field in ("input_ids", "labels", "attention_mask", "mm_token_type_ids"):
                    self.assertEqual(tuple(sharded[field].shape), (1, seq_len // 2), field)
                    self.assertTrue(torch.equal(sharded[field], batch[field][:, window]), field)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cp_keeps_media_fields_complete(self):
        """The vision inputs stay full-length so the media scatter is unsharded."""
        batch = _make_batch()
        sharded = _build_sharder(2, 1).shard(batch)

        self.assertEqual(tuple(sharded["pixel_values"].shape), (3, 4, 4))
        self.assertEqual(tuple(sharded["image_grid_thw"].shape), (1, 3))
        self.assertTrue(torch.equal(sharded["pixel_values"], batch["pixel_values"]))
        self.assertTrue(torch.equal(sharded["image_grid_thw"], batch["image_grid_thw"]))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cp_pads_an_indivisible_sequence_length(self):
        """A sequence that ``2 * cp_size`` does not divide is padded, not rejected."""
        seq_len = 8
        batch = _make_batch(seq_len=seq_len)
        sharded = _build_sharder(3, cp_rank=0).shard(batch)

        # cp_size=3 -> pad to 12, one 4-token chunk per rank.
        self.assertEqual(tuple(sharded["input_ids"].shape), (1, 4))
        self.assertTrue(torch.equal(sharded["input_ids"][0], batch["input_ids"][0, :4]))
        self.assertEqual(int(sharded["cu_seq_lens"][-1]), 12)
        self.assertEqual(sharded["cu_seq_lens"].dtype, torch.int32)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cp_rejects_a_multi_row_batch(self):
        """Configured CP sharding requires the packed single-row layout."""
        batch = _make_batch()
        batch["input_ids"] = torch.cat((batch["input_ids"], batch["input_ids"]), dim=0)
        batch["labels"] = torch.cat((batch["labels"], batch["labels"]), dim=0)
        with self.assertRaisesRegex(ValueError, r"shape \[1, sequence\]"):
            _build_sharder(2).shard(batch)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cp_rejects_token_fields_that_do_not_match_input_ids(self):
        """A token field with the wrong shape is a collator bug, not silent padding."""
        batch = _make_batch()
        batch["labels"] = batch["labels"][:, :4]
        with self.assertRaisesRegex(ValueError, "must match input_ids shape"):
            _build_sharder(2).shard(batch)


if __name__ == "__main__":
    unittest.main()
