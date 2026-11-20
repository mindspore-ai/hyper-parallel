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
"""Mock GPT samples must carry routing diversity, not one constant token.

A mock batch whose every position holds the same id gives every position the
same hidden state, so a MoE router picks the same experts for the whole batch
and the expert-parallel dispatch lands on a handful of ranks. Loss still moves,
which is what made the original constant filler look healthy while the memory
and dispatch numbers it produced were not the ones a real batch would show.
"""

import unittest

import numpy as np

from hyper_parallel.data.indexed.indexed_data_config import GPTDatasetConfig
from hyper_parallel.data.indexed.indexed_pretrain_dataset import MockGPTDataset


VOCAB_SIZE = 8192
PAD_ID = 2
EOD_ID = 5
SEQUENCE_LENGTH = 32
SAMPLE_COUNT = 8


class _FakeSplit:
    """Stand-in for the split enum: ``__getitem__`` reads ``value``."""

    def __init__(self, value: int) -> None:
        """Store the split value and a readable split name."""
        self.value = value
        self.name = "train"


class _FakeTokenizer:
    """Minimal tokenizer surface the mock Dataset reads."""

    def __init__(self) -> None:
        """Expose the pad/eod ids and an empty special-token map."""
        self.pad = PAD_ID
        self.eod = EOD_ID
        self.eos = None
        self.special_tokens_dict = {}

    def __len__(self) -> int:
        """Return the fake vocabulary size."""
        return VOCAB_SIZE


def _build_dataset(num_samples: int = SAMPLE_COUNT) -> MockGPTDataset:
    """Build a deterministic mock Dataset over a fake tokenizer."""
    config = GPTDatasetConfig(
        random_seed=0,
        sequence_length=SEQUENCE_LENGTH,
        mock=True,
        tokenizer=_FakeTokenizer(),
    )
    return MockGPTDataset(None, None, None, num_samples, _FakeSplit(0), config)


def _body(sample) -> np.ndarray:
    """Return the drawn token body of one sample (``tokens[0]`` is its length)."""
    tokens = sample["tokens"]
    return tokens[1:1 + int(tokens[0])]


class TestMockSampleTokenDiversity(unittest.TestCase):
    """Sampled token ids must span the vocabulary, not repeat one filler id."""

    def test_tokens_are_drawn_from_the_vocabulary(self):
        """A sample mixes many ids instead of repeating a single one."""
        dataset = _build_dataset()
        tokens = dataset[3]["tokens"]
        distinct = np.unique(tokens)

        self.assertGreater(
            distinct.size,
            1,
            f"mock tokens collapsed onto one id: distinct={distinct.tolist()}",
        )
        self.assertGreaterEqual(
            int(tokens.min()),
            0,
            f"mock token ids must stay in the vocabulary, got min={int(tokens.min())}",
        )
        self.assertLess(
            int(tokens.max()),
            VOCAB_SIZE,
            f"mock token ids must stay in the vocabulary, got max={int(tokens.max())}",
        )

    def test_reserved_ids_are_never_forged_by_a_draw(self):
        """Random draws must not fabricate an ``eod`` terminator or pad id."""
        dataset = _build_dataset()
        samples = [dataset[index] for index in range(SAMPLE_COUNT)]
        for index, sample in enumerate(samples):
            with self.subTest(index=index):
                body = _body(sample)
                reserved_hits = np.intersect1d(body, [PAD_ID, EOD_ID])

                self.assertEqual(
                    reserved_hits.size,
                    0,
                    f"sample {index} forged reserved ids={reserved_hits.tolist()}",
                )

    def test_boundary_markers_keep_their_positions(self):
        """``eod`` still closes the drawn body and pad fills the tail."""
        dataset = _build_dataset()
        tokens = dataset[1]["tokens"]
        length = int(tokens[0])

        self.assertEqual(
            int(tokens[length + 1]),
            EOD_ID,
            f"eod must close the body at {length + 1}, got tokens={tokens.tolist()}",
        )
        self.assertTrue(
            np.all(tokens[length + 2:] == PAD_ID),
            f"tail must be pad={PAD_ID}, got tail={tokens[length + 2:].tolist()}",
        )

    def test_sample_shape_is_unchanged(self):
        """Sample geometry stays tokens/labels of one sequence length."""
        dataset = _build_dataset()
        sample = dataset[2]

        self.assertEqual(
            (sample["tokens"].size, sample["labels"].size),
            (SEQUENCE_LENGTH, SEQUENCE_LENGTH),
            f"unexpected sample geometry: tokens={sample['tokens'].size}, "
            f"labels={sample['labels'].size}",
        )

    def test_sampling_stays_deterministic(self):
        """The same split and index keep producing the same sample."""
        first, second = _build_dataset()[5], _build_dataset()[5]

        self.assertTrue(
            np.array_equal(first["tokens"], second["tokens"]),
            f"mock sampling lost determinism: first={first['tokens'].tolist()}, "
            f"second={second['tokens'].tolist()}",
        )

    def test_the_split_covers_a_wide_range_of_ids(self):
        """Sampling across the split exercises the whole vocabulary, not a few ids."""
        dataset = _build_dataset()
        drawn = np.concatenate([_body(dataset[index]) for index in range(SAMPLE_COUNT)])
        distinct = np.unique(drawn)

        self.assertGreater(
            distinct.size,
            16,
            f"mock sampling reuses too few ids: distinct={distinct.tolist()}",
        )


if __name__ == "__main__":
    unittest.main()
