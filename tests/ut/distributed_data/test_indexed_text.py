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
from unittest.mock import Mock

import numpy as np

from hyper_parallel.data.parallel import build_dataset_batch_sampler
from hyper_parallel.distributed_data import (
    DistributedDatasetConfig,
    SampleMetadata,
    build_distributed_dataloader,
    collate_indexed_text_sequences,
    pack_indexed_text_samples,
)


class _StandaloneMesh:
    mesh_shape = (1,)
    mesh_dim_names = ("dp",)
    rank_list = (0,)


class _MetadataDataset:
    """Provide variable-length text records through mapping access."""

    def __init__(self) -> None:
        """Create two samples that exactly fill one packed row."""
        self.samples = [
            {"input_ids": np.asarray([1, 2]), "labels": np.asarray([2, 9])},
            {"input_ids": np.asarray([3, 4, 5, 6]), "labels": np.asarray([4, 5, 6, 9])},
        ]
        self.payload_reads = 0

    def __len__(self) -> int:
        """Return the aligned sample and metadata length."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        """Materialize one source payload."""
        self.payload_reads += 1
        return self.samples[index]


class TestIndexedTextConstruction(unittest.TestCase):
    """Verify fixed row shapes and sample boundaries after dynamic packing."""

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

    def test_builder_uses_explicit_metadata(self) -> None:
        """Feature: Unified distributed DataLoader API.
        Description: Supply aligned metadata separately from a mapping Dataset.
        Expectation: Planning is metadata-first and the selected payloads are read directly.
        """
        dataset = _MetadataDataset()
        loader = build_distributed_dataloader(
            dataset,
            _StandaloneMesh(),
            DistributedDatasetConfig(seq_len=6, local_batch_size=2, metadata_mode=True),
            metadata=[SampleMetadata(2, sample_id=0), SampleMetadata(4, sample_id=1)],
            device="cpu", cost_model=lambda metadata: metadata.cost,
            batch_sampler=build_dataset_batch_sampler(
                total_samples=2, micro_batch_size=2, global_batch_size=2, dp_world_size=1, dp_rank=0,
            ),
        )

        batch = next(loader)

        self.assertEqual(sorted(sample["input_ids"].tolist() for sample in batch), [[1, 2], [3, 4, 5, 6]])
        self.assertEqual(dataset.payload_reads, 2)

    def test_builder_requires_explicit_metadata(self) -> None:
        """Feature: Explicit offline metadata source.
        Description: Omit metadata for a mapping Dataset in offline mode.
        Expectation: Construction fails without reading payloads.
        """
        dataset = _MetadataDataset()
        with self.assertRaisesRegex(ValueError, "metadata_mode=True requires metadata"):
            build_distributed_dataloader(
                dataset, _StandaloneMesh(),
                DistributedDatasetConfig(seq_len=6, local_batch_size=2, metadata_mode=True),
                device="cpu", cost_model=lambda metadata: metadata.cost,
                batch_sampler=build_dataset_batch_sampler(
                    total_samples=2, micro_batch_size=2, global_batch_size=2, dp_world_size=1, dp_rank=0,
                ),
            )
        self.assertEqual(dataset.payload_reads, 0)

    def test_builder_calls_metadata_fn_with_selected_samples(self) -> None:
        """Feature: Explicit online metadata source.
        Description: Supply a metadata callback for a mapping Dataset.
        Expectation: The callback receives the selected payloads after they are read.
        """
        dataset = _MetadataDataset()
        metadata_fn = Mock(side_effect=lambda sample: SampleMetadata(len(sample["input_ids"])))
        loader = build_distributed_dataloader(
            dataset, _StandaloneMesh(), DistributedDatasetConfig(seq_len=6, local_batch_size=2),
            metadata_fn=metadata_fn, device="cpu", cost_model=lambda metadata: metadata.cost,
            batch_sampler=build_dataset_batch_sampler(
                total_samples=2, micro_batch_size=2, global_batch_size=2, dp_world_size=1, dp_rank=0,
            ),
        )
        metadata_fn.assert_not_called()
        batch = next(loader)
        self.assertEqual(len(batch), 2)
        self.assertEqual(metadata_fn.call_count, 2)
        for call, sample in zip(metadata_fn.call_args_list, dataset.samples):
            self.assertIs(call.args[0], sample)
        self.assertEqual(dataset.payload_reads, 2)
