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
"""Ragged tail windows of chunked plaintext records must be dropped."""

from __future__ import annotations

import unittest

from hyper_parallel.data.text.text_transform import build_text_transform


class _StubTokenizer:
    """Deterministic tokenizer stub: one token per character."""

    eos_token_id = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Map every character to a distinct small id."""
        del add_special_tokens
        return [ord(ch) % 251 for ch in text]


class PlaintextTailWindowTest(unittest.TestCase):
    """Validate window shapes produced by the plaintext chunker."""

    def _transform(self, max_seq_len: int, drop_ragged_tail: bool = False):
        """Build the plaintext transform under test."""
        return build_text_transform(
            data_type="plaintext",
            tokenizer=_StubTokenizer(),
            text_keys="text",
            max_seq_len=max_seq_len,
            drop_ragged_tail=drop_ragged_tail,
        )

    def test_default_keeps_the_ragged_tail_window(self):
        """Every other model keeps the trailing partial window it had before."""
        transform = self._transform(max_seq_len=8)
        samples = transform({"text": "a" * 21})  # 2 full windows + ragged tail
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[-1]["input_ids"].numel(), 4)

    def test_opt_in_drops_the_ragged_tail_window(self):
        """Alignment-sensitive models keep only full windows."""
        transform = self._transform(max_seq_len=8, drop_ragged_tail=True)
        samples = transform({"text": "a" * 21})
        self.assertEqual(len(samples), 2)
        for sample in samples:
            self.assertEqual(sample["input_ids"].numel(), 8)
            self.assertEqual(sample["labels"].numel(), 8)

    def test_single_short_record_is_kept(self):
        """Records shorter than one window still produce a sample."""
        transform = self._transform(max_seq_len=8)
        samples = transform({"text": "abcde"})
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["input_ids"].numel(), 4)


if __name__ == "__main__":
    unittest.main()
