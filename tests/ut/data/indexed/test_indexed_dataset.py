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
"""Exercise native indexed construction through the public Dataset builder."""

from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from hyper_parallel.data.indexed.indexed_blended_dataset import BlendedDataset
from hyper_parallel.data.indexed.indexed_dataset import build_indexed_dataset
from hyper_parallel.data.indexed.indexed_lazy_dataset import LazyDatasetProxy
from hyper_parallel.data.indexed.indexed_pretrain_dataset import GPTDataset, GPTFromMRDataset, MockGPTDataset
from hyper_parallel.data.tools.io import IndexedDatasetBuilder


class _Tokenizer:
    """Describe a fixed vocabulary without external tokenizer files."""

    pad = 0
    eod = 999
    vocab_size = 1000
    unique_identifiers = {"vocab_size": 1000}

    def __len__(self) -> int:
        """Return the vocabulary size checked by GPT construction."""
        return self.vocab_size


def _config(directory: str, **options: object) -> dict:
    return {
        "random_seed": 7, "seq_length": 4, "tokenizer": _Tokenizer(),
        "mock_data": False, "data_lazy_load": False, "split": "1,0,0",
        "distributed_walk": False, "low_level_dataset": GPTDataset, "simple_blend": "no",
        "data_cache_path": str(Path(directory) / "cache"), "dataset_margin": 1.0,
        **options,
    }


def _write_records(directory: str, name: str, records: list[list[int]]) -> str:
    prefix = str(Path(directory) / name)
    writer = IndexedDatasetBuilder(prefix + ".bin")
    for record in records:
        writer.add_document(torch.tensor(record), [len(record)])
    writer.finalize(prefix + ".idx")
    return prefix


class TestIndexedDataset(unittest.TestCase):
    """Preserve native packing, lazy loading and weighted sample counts."""

    def test_gpt_and_precut_records_with_eager_and_lazy_loading(self) -> None:
        """Feature: Indexed Dataset storage.
        Description: Read real files through both Dataset types and eager/lazy loading.
        Expectation: Both modes preserve tokens, labels, lengths and sample counts.
        """
        records = [list(range(1 + i * 5, 6 + i * 5)) for i in range(6)]
        with TemporaryDirectory() as directory:
            prefix = _write_records(directory, "text", records)
            for precut in (False, True):
                baseline = None
                for lazy in (False, True):
                    with self.subTest(precut=precut, lazy=lazy):
                        expected_type = GPTFromMRDataset if precut else GPTDataset
                        train, valid, test = build_indexed_dataset(
                            data_path=prefix,
                            data_config=_config(
                                directory, low_level_dataset=expected_type, data_lazy_load=lazy,
                                simple_blend="intra" if precut else "no",
                            ),
                            train_valid_test_num_samples=(3, 0, 0),
                        )
                        self.assertIsNone(valid)
                        self.assertIsNone(test)
                        if lazy:
                            self.assertIsInstance(train, LazyDatasetProxy)
                            self.assertIsNone(train._dataset)
                        else:
                            self.assertIsInstance(train, expected_type)
                        rows = [train[i] for i in range(len(train))]
                        actual = []
                        for row in rows:
                            self.assertEqual(row["input_ids"].shape, (4,))
                            self.assertEqual(row["labels"].shape, (4,))
                            np.testing.assert_array_equal(row["input_ids"][1:], row["labels"][:-1])
                            actual.append((row["input_ids"].tolist(), row["labels"].tolist()))
                        if precut:
                            self.assertEqual(actual, [(record[:-1], record[1:]) for record in records[:3]])
                        if baseline is None:
                            baseline = actual
                        self.assertEqual(actual, baseline)
                        if lazy:
                            self.assertIsInstance(train._dataset, expected_type)

    def test_weighted_blend_preserves_requested_counts_without_repeating_sources(self) -> None:
        """Feature: Weighted indexed Dataset blending.
        Description: Select a short training plan from two larger corpora.
        Expectation: Counts follow blend weights without repeating source records.
        """
        with TemporaryDirectory() as directory:
            first = _write_records(directory, "first", [list(range(i, i + 5)) for i in range(1, 31, 5)])
            second = _write_records(directory, "second", [list(range(i, i + 5)) for i in range(101, 146, 5)])
            for lazy in (False, True):
                with self.subTest(lazy=lazy):
                    train, _, _ = build_indexed_dataset(
                        data_path=["1", first, "2", second],
                        data_config=_config(directory, data_lazy_load=lazy),
                        train_valid_test_num_samples=(6, 0, 0),
                    )
                    self.assertIsInstance(train, BlendedDataset)
                    self.assertEqual(len(train), 6)
                    rows = [train[i] for i in range(len(train))]
                    self.assertEqual(Counter(row["dataset_id"] for row in rows), {0: 2, 1: 4})
                    self.assertCountEqual(
                        list(zip(train.dataset_index.tolist(), train.dataset_sample_index.tolist())),
                        [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (1, 3)],
                    )
                    self.assertEqual(len({tuple(row["input_ids"].tolist()) for row in rows}), 6)
                    with self.assertRaisesRegex(ValueError, "blend requested"):
                        BlendedDataset(
                            train.datasets, [1, 1], sum(len(dataset) for dataset in train.datasets) + 1, train.config,
                        )

    def test_mock_splits_need_no_files(self) -> None:
        """Feature: Mock Dataset splits.
        Description: Build training and validation without indexed files.
        Expectation: Each split honors its independently requested sample count.
        """
        with TemporaryDirectory() as directory:
            train, valid, test = build_indexed_dataset(
                data_config=_config(directory, mock_data=True, low_level_dataset=MockGPTDataset),
                train_valid_test_num_samples=(3, 2, 0),
            )
            self.assertIsInstance(train, MockGPTDataset)
            self.assertIsInstance(valid, MockGPTDataset)
            self.assertEqual((len(train), len(valid)), (3, 2))
            self.assertEqual(train[0]["input_ids"].shape, (4,))
            self.assertIsNone(test)
