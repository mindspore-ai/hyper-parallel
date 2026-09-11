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
"""Tests for Indexed source metadata and built-in text construction."""

import unittest

import numpy as np

from hyper_parallel.distributed_data import (
    DistributedDatasetConfig,
    SampleMetadata,
    build_distributed_dataloader,
    collate_indexed_text_sequences,
    pack_indexed_text_samples,
)
from tests.common.mark_utils import arg_mark


class _StandaloneMesh:
    mesh_shape = (1,)
    mesh_dim_names = ("dp",)
    rank_list = (0,)


class _MetadataDataset:
    """Provide aligned metadata through the Dataset protocol."""

    requires_distributed_packing = True

    def __init__(self) -> None:
        """Create two samples that exactly fill one packed row."""
        self.samples = [
            {"input_ids": np.asarray([1, 2]), "labels": np.asarray([2, 9])},
            {"input_ids": np.asarray([3, 4, 5, 6]), "labels": np.asarray([4, 5, 6, 9])},
        ]
        self.metadata_reads = 0
        self.payload_reads = 0

    def __len__(self) -> int:
        """Return the aligned sample and metadata length."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        """Materialize one source payload."""
        self.payload_reads += 1
        return self.samples[index]

    def get_sample_metadata(self, index: int) -> SampleMetadata:
        """Return the metadata token count without reading the payload."""
        self.metadata_reads += 1
        return SampleMetadata(pack_tokens=len(self.samples[index]["input_ids"]), sample_id=index)


class TestIndexedTextConstruction(unittest.TestCase):
    """Verify fixed row shapes and sample boundaries after dynamic packing."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_pack_and_collate_preserve_boundaries_and_mask_padding(self) -> None:
        """Feature: Built-in Indexed text Data Constructor.
        Description: Pack two bins with and without a padding tail.
        Expectation: Output is fixed-shape and boundaries cover every real sample and padded row.
        """
        first = pack_indexed_text_samples(
            [
                {"input_ids": [1, 2], "labels": [2, 9]},
                {"input_ids": [3], "labels": [9]},
            ],
            seq_len=5,
        )
        second = pack_indexed_text_samples(
            [{"input_ids": [4, 5, 6, 7, 8], "labels": [5, 6, 7, 8, 9]}],
            seq_len=5,
        )

        batch = collate_indexed_text_sequences([first, second])

        self.assertEqual(batch["input_ids"].tolist(), [[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
        self.assertEqual(batch["labels"].tolist(), [[2, 9, 9, -100, -100], [5, 6, 7, 8, 9]])
        self.assertEqual(batch["cu_seq_lens"].tolist(), [0, 2, 3, 5, 10])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
    def test_builder_infers_dataset_metadata(self) -> None:
        """Feature: Unified distributed DataLoader API.
        Description: Omit metadata arguments for a Dataset exposing get_sample_metadata.
        Expectation: Planning is metadata-first and the selected payloads are read directly.
        """
        dataset = _MetadataDataset()
        loader = build_distributed_dataloader(
            dataset,
            _StandaloneMesh(),
            DistributedDatasetConfig(seq_len=6, local_batch_size=1),
            pack_fn=pack_indexed_text_samples,
            collate_fn=collate_indexed_text_sequences,
        )

        batch = next(loader)

        self.assertEqual(sorted(batch["input_ids"].flatten().tolist()), [1, 2, 3, 4, 5, 6])
        self.assertEqual(batch["cu_seq_lens"].tolist(), [0, 4, 6])
        self.assertEqual(dataset.metadata_reads, 2)
        self.assertEqual(dataset.payload_reads, 2)
        self.assertTrue(loader.collective_source)
