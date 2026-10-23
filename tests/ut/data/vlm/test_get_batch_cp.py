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

The Omni collator pads token fields to ``tp_size * cp_size`` before
``OmniParallelBatch`` builds a :class:`CPBatchSharder` configured with
``OMNI_CP_TOKEN_FIELDS``. The sharder selects contiguous token windows and
preserves global packed boundaries and complete modality metadata. These tests
exercise the real collator and sharder on CPU without a process group.
"""

import unittest
from types import SimpleNamespace

import torch

from hyper_parallel.data.batching.build_collate_fn import build_omni_collate_fn
from hyper_parallel.data.constants import OMNI_CP_TOKEN_FIELDS
from hyper_parallel.data.parallel import CPBatchSharder, DataLoaderParallelContext
from tests.common.mark_utils import arg_mark


def _make_batch(seq_len: int = 8, pad_tail: int = 2) -> dict[str, torch.Tensor]:
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


def _build_sharder(cp_size: int, cp_rank: int = 0, tp_size: int = 1) -> CPBatchSharder:
    """Build the sharder exactly as OmniParallelBatch configures it."""
    context = DataLoaderParallelContext(cp_world_size=cp_size, cp_rank=cp_rank, tp_world_size=tp_size)
    return CPBatchSharder(context, token_fields=OMNI_CP_TOKEN_FIELDS)


def _collate_batch(batch: dict[str, torch.Tensor], cp_size: int, tp_size: int = 1) -> dict[str, torch.Tensor]:
    """Align one packed sample through the same collator used by the loader."""
    sample = {field: value.squeeze(0) if field in OMNI_CP_TOKEN_FIELDS else value
              for field, value in batch.items()}
    collator = build_omni_collate_fn(SimpleNamespace(cp_size=cp_size, tp_size=tp_size))
    return collator([sample])


class TestOmniBatchCpSharding(unittest.TestCase):
    """Token fields shard per CP rank; modality metadata stays complete."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_sp_alignment_cp1_and_cp2(self) -> None:
        """Feature: Omni token padding for decoder sequence parallelism.

        Description: Shard an eleven-token packed batch with CP1 and CP2 at TP4.
        Expectation: Collate pads to TP * CP and CP-local lengths are divisible by TP.
        """
        batch = _make_batch(seq_len=11, pad_tail=0)
        for cp_size, expected_length in ((1, 12), (2, 8)):
            aligned = _collate_batch(batch, cp_size, tp_size=4)
            for cp_rank in range(cp_size):
                with self.subTest(cp_size=cp_size, cp_rank=cp_rank):
                    sharded = _build_sharder(cp_size, cp_rank, tp_size=4).shard(aligned)
                    self.assertEqual(sharded["input_ids"].shape[1], expected_length)
                    self.assertEqual(sharded["labels"].shape[1], expected_length)
                    self.assertEqual(int(sharded["cu_seq_lens"][-1]), expected_length * cp_size)
                    self.assertEqual(sharded["cu_seq_lens"].tolist(), [0, 11, expected_length * cp_size])
                    self.assertEqual(sharded["input_ids"].shape[1] % 4, 0)

            self.assertEqual(aligned["labels"][0, 11:].tolist(), [-100] * (expected_length * cp_size - 11))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cp1_keeps_the_batch_untouched(self) -> None:
        """Feature: single-rank CP batching.

        Description: Shard an already aligned batch with CP size one.
        Expectation: Token and media fields remain unchanged.
        """
        batch = _make_batch()
        sharded = _build_sharder(1).shard(batch)

        self.assertTrue(torch.equal(sharded["input_ids"], batch["input_ids"]))
        self.assertTrue(torch.equal(sharded["labels"], batch["labels"]))
        self.assertTrue(torch.equal(sharded["pixel_values"], batch["pixel_values"]))
        self.assertTrue(torch.equal(sharded["image_grid_thw"], batch["image_grid_thw"]))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cp2_slices_token_fields_to_the_rank_window(self) -> None:
        """Feature: CP token-field sharding.

        Description: Shard eight token rows across two CP ranks.
        Expectation: Each rank receives its contiguous window for every token field.
        """
        seq_len = 8
        batch = _make_batch(seq_len=seq_len)
        for cp_rank in (0, 1):
            with self.subTest(cp_rank=cp_rank):
                window = slice(cp_rank * 4, (cp_rank + 1) * 4)
                sharded = _build_sharder(2, cp_rank).shard(batch)

                for field in OMNI_CP_TOKEN_FIELDS.intersection(batch):
                    self.assertEqual(tuple(sharded[field].shape), (1, seq_len // 2))
                    torch.testing.assert_close(sharded[field], batch[field][:, window])
                torch.testing.assert_close(sharded["cu_seq_lens"], batch["cu_seq_lens"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cp_keeps_media_fields_complete(self) -> None:
        """Feature: CP multimodal metadata preservation.

        Description: Shard token fields while retaining image payloads.
        Expectation: Vision tensors and grid metadata remain complete.
        """
        batch = _make_batch()
        sharded = _build_sharder(2, 1).shard(batch)

        self.assertEqual(tuple(sharded["pixel_values"].shape), (3, 4, 4))
        self.assertEqual(tuple(sharded["image_grid_thw"].shape), (1, 3))
        self.assertTrue(torch.equal(sharded["pixel_values"], batch["pixel_values"]))
        self.assertTrue(torch.equal(sharded["image_grid_thw"], batch["image_grid_thw"]))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_collate_pads_before_cp_shards_an_indivisible_sequence_length(self) -> None:
        """Feature: CP sequence alignment.

        Description: Shard eight tokens across three CP ranks.
        Expectation: Collate pads to nine tokens, preserving the real boundary at eight.
        """
        seq_len = 8
        batch = _make_batch(seq_len=seq_len, pad_tail=0)
        aligned = _collate_batch(batch, cp_size=3)
        local_batches = [_build_sharder(3, cp_rank).shard(aligned) for cp_rank in range(3)]

        for local in local_batches:
            self.assertEqual(tuple(local["input_ids"].shape), (1, 3))
            self.assertEqual(local["cu_seq_lens"].tolist(), [0, 8, 9])
            self.assertEqual(local["cu_seq_lens"].dtype, torch.int32)
            torch.testing.assert_close(local["pixel_values"], batch["pixel_values"])
            torch.testing.assert_close(local["image_grid_thw"], batch["image_grid_thw"])
        for field in OMNI_CP_TOKEN_FIELDS.intersection(batch):
            reconstructed = torch.cat([local[field] for local in local_batches], dim=1)
            torch.testing.assert_close(reconstructed, aligned[field])
            torch.testing.assert_close(reconstructed[:, :seq_len], batch[field])
        for field, pad_value in (("input_ids", 0), ("labels", -100), ("attention_mask", 0),
                                 ("loss_mask", 0), ("mm_token_type_ids", -1)):
            self.assertEqual(int(local_batches[-1][field][0, -1]), pad_value)
        self.assertEqual(batch["input_ids"].shape[1], seq_len)
        self.assertEqual(batch["cu_seq_lens"].tolist(), [0, seq_len])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cp_rejects_batches_not_aligned_by_collate(self) -> None:
        """The sharder rejects unaligned tokens instead of changing global boundaries."""
        batch = _make_batch(seq_len=11)
        for cp_size, tp_size in ((1, 4), (2, 4), (3, 1)):
            with self.subTest(cp_size=cp_size, tp_size=tp_size):
                with self.assertRaisesRegex(ValueError, "align token fields in collate_fn"):
                    _build_sharder(cp_size, tp_size=tp_size).shard(batch)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cp_rejects_boundaries_that_do_not_cover_the_global_sequence(self) -> None:
        """Packed boundaries must start at zero and cover the complete pre-shard row."""
        for boundaries in ([1, 8], [0, 7], [0]):
            with self.subTest(boundaries=boundaries):
                batch = _make_batch()
                batch["cu_seq_lens"] = torch.tensor(boundaries, dtype=torch.int32)
                with self.assertRaisesRegex(ValueError, "cover the complete CP input sequence"):
                    _build_sharder(2).shard(batch)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cp_rejects_a_multi_row_batch(self) -> None:
        """Feature: packed Omni batch shape validation.

        Description: Pass a two-row input batch to CP sharding.
        Expectation: The single-row contract rejects it.
        """
        batch = _make_batch()
        batch["input_ids"] = torch.cat((batch["input_ids"], batch["input_ids"]), dim=0)
        batch["labels"] = torch.cat((batch["labels"], batch["labels"]), dim=0)
        with self.assertRaisesRegex(ValueError, r"shape \[1, sequence\]"):
            _build_sharder(2).shard(batch)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cp_rejects_token_fields_that_do_not_match_input_ids(self) -> None:
        """Feature: CP token-field shape validation.

        Description: Provide labels shorter than input IDs.
        Expectation: The mismatched field fails before slicing.
        """
        batch = _make_batch()
        batch["labels"] = batch["labels"][:, :4]
        with self.assertRaisesRegex(ValueError, "must match input_ids shape"):
            _build_sharder(2).shard(batch)


if __name__ == "__main__":
    unittest.main()
