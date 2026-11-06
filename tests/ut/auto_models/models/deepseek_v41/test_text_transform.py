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
"""Model-owned alignment keeps the packing budget equal to the padded row."""

import unittest

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.batching.build_collate_fn import (
    IGNORE_INDEX,
    TextPackingCollator,
)
from hyper_parallel.data.batching.dynamic_batch import TokenBudgetCandidateBuffer
from hyper_parallel.models.deepseek_v41.adapter.data.text_transform import (
    DeepseekV41TextTransform,
)


class _StubTokenizer:
    """Encode characters as ids and expose one end-of-sequence token."""

    eos_token_id = 9

    @staticmethod
    def encode(text, add_special_tokens=False):
        """Map each character to its ordinal token id."""
        del add_special_tokens
        return [ord(char) % 100 for char in text]


class _StubPaddedTokenizer(_StubTokenizer):
    """Prefer a dedicated pad token over the end-of-sequence fallback."""

    pad_token_id = 7


def _sample(length: int) -> dict:
    """Build one encoded online sample with the requested token count."""
    return {
        "input_ids": torch.arange(length),
        "labels": torch.arange(1, length + 1),
    }


class TestDeepseekV41TextTransform(unittest.TestCase):
    """The model-owned transform aligns documents before packing selection."""

    def test_plaintext_transform_pads_document_tails(self) -> None:
        """Odd samples grow to the alignment; labels mask the padding."""
        transform = DeepseekV41TextTransform(
            _StubTokenizer(), max_seq_len=8, sample_alignment_size=2,
        )
        # "abcde" + eos -> one 5-token sample, padded to 6 with the eos id.
        samples = transform({"text": "abcde"})
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["input_ids"].shape[-1], 6)
        self.assertEqual(int(samples[0]["labels"][-2]), 9)
        self.assertEqual(int(samples[0]["labels"][-1]), IGNORE_INDEX)
        self.assertEqual(int(samples[0]["input_ids"][-1]), 9)

    def test_pad_token_id_preferred_over_eos(self) -> None:
        """A dedicated pad token is used for the aligned tail."""
        transform = DeepseekV41TextTransform(
            _StubPaddedTokenizer(), max_seq_len=8, sample_alignment_size=2,
        )
        samples = transform({"text": "abcde"})
        self.assertEqual(int(samples[0]["input_ids"][-1]), 7)
        self.assertEqual(int(samples[0]["labels"][-1]), IGNORE_INDEX)

    def test_alignment_default_keeps_natural_lengths(self) -> None:
        """Default 1 leaves every emitted sample unchanged."""
        transform = DeepseekV41TextTransform(_StubTokenizer(), max_seq_len=4)
        samples = transform({"text": "abcdef"})
        # Windows: [a,b,c,d,e] -> 4 tokens, [e,f,eos] -> 2 tokens.
        self.assertEqual([sample["input_ids"].numel() for sample in samples], [4, 2])

    def test_aligned_overflow_sample_dropped(self) -> None:
        """A sample whose aligned length exceeds max_seq_len is skipped."""
        transform = DeepseekV41TextTransform(
            _StubTokenizer(), max_seq_len=3, sample_alignment_size=4,
        )
        # "ab" + eos -> one 2-token sample; aligned 4 > max_seq_len 3.
        self.assertEqual(transform({"text": "ab"}), [])

    def test_invalid_alignment_rejected(self) -> None:
        """Zero, negative, and boolean alignments fail at construction."""
        for bad_value in (0, -2, True):
            with self.subTest(sample_alignment_size=bad_value):
                with self.assertRaisesRegex(ValueError, "sample_alignment_size"):
                    DeepseekV41TextTransform(
                        _StubTokenizer(), max_seq_len=4,
                        sample_alignment_size=bad_value,
                    )


class TestTextPackingCollator(unittest.TestCase):
    """Aligned samples pack unchanged; the budget counts their real length."""

    def test_packed_boundaries_follow_aligned_samples(self) -> None:
        """Pre-aligned documents pack into even cu_seq_lens boundaries."""
        collator = TextPackingCollator()
        batch = collator([_sample(4), _sample(4), _sample(6)])
        self.assertEqual(batch["input_ids"].shape, (1, 14))
        self.assertEqual(batch["cu_seq_lens"].tolist(), [0, 4, 8, 14])

    def test_budgeted_row_never_exceeds_an_equal_pad_length(self) -> None:
        """Aligned transform output plus budget selection keeps the row bounded."""
        collator = TextPackingCollator(pad_to_length=12)
        buffer = TokenBudgetCandidateBuffer(token_budget=12, min_buffered_samples=1)
        # Emulate the aligned transform: each odd length is already padded.
        for length in (6, 8, 4, 2, 6):
            buffer.put_item(_sample(length))
        while not buffer.empty():
            selected = buffer.get_micro_batch()
            batch = collator(selected)
            self.assertLessEqual(int(batch["input_ids"].shape[-1]), 12)
            self.assertFalse(
                any(boundary % 2 for boundary in batch["cu_seq_lens"].tolist())
            )

    def test_budget_rejects_overflowing_group(self) -> None:
        """A group whose real lengths overflow the budget is split."""
        buffer = TokenBudgetCandidateBuffer(token_budget=12, min_buffered_samples=1)
        buffer.put_item(_sample(6))
        buffer.put_item(_sample(8))
        self.assertEqual(len(buffer.get_micro_batch()), 1)  # 6 + 8 > 12


if __name__ == "__main__":
    unittest.main()
