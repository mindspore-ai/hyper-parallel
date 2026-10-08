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
"""Regression tests for external Megatron text indexed datasets."""

import struct

import numpy as np
import pytest

from hyper_parallel.data.indexed.indexed_data_reader import IndexedDataReader
from hyper_parallel.data.tools.io import IndexedDataset
from hyper_parallel.data.tools.megatron_dataset import (
    adapt_megatron_dataset,
    inspect_megatron_dataset,
    load_megatron_dataset,
)
from tests.common.mark_utils import arg_mark


def _write_megatron_pair(prefix, *, include_modes=False):
    """Write a small standard Megatron index/bin pair without project builders."""
    lengths = np.asarray([2, 3, 1], dtype=np.int32)
    pointers = np.asarray([0, 4, 10], dtype=np.int64)
    documents = np.asarray([0, 2, 3], dtype=np.int64)
    values = np.asarray([10, 11, 20, 21, 22, 30], dtype=np.uint16)
    with open(str(prefix) + ".bin", "wb") as stream:
        stream.write(values.tobytes())
    with open(str(prefix) + ".idx", "wb") as stream:
        stream.write(b"MMIDIDX\x00\x00")
        stream.write(struct.pack("<Q", 1))
        stream.write(struct.pack("<B", 8))
        stream.write(struct.pack("<Q", len(lengths)))
        stream.write(struct.pack("<Q", len(documents)))
        stream.write(lengths.tobytes())
        stream.write(pointers.tobytes())
        stream.write(documents.tobytes())
        if include_modes:
            stream.write(np.asarray([0, 1, 0], dtype=np.int8).tobytes())


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_external_megatron_pair_is_read_by_both_hyperparallel_readers(tmp_path):
    """An externally encoded pair has identical sequence and document semantics."""
    prefix = tmp_path / "train_text_document"
    _write_megatron_pair(prefix)

    direct_reader = IndexedDataReader(str(prefix) + ".idx")
    dataset_reader = IndexedDataset(str(prefix) + ".bin", mmap=False)

    assert direct_reader.sequence_lengths.tolist() == [2, 3, 1]
    assert direct_reader.document_indices.tolist() == [0, 2, 3]
    assert direct_reader[1].tolist() == [20, 21, 22]
    assert dataset_reader.document_indices.tolist() == [0, 2, 3]
    assert dataset_reader[1].tolist() == [20, 21, 22]


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_megatron_validation_and_adaptation_preserve_files(tmp_path):
    """Validation reports metadata and adaptation copies an unchanged pair."""
    source = tmp_path / "source_text_document"
    target = tmp_path / "adapted" / "text_document"
    _write_megatron_pair(source, include_modes=True)

    info = inspect_megatron_dataset(str(source) + ".bin")
    assert info.dtype == np.dtype(np.uint16)
    assert info.sequence_count == 3
    assert info.document_count == 2
    assert info.token_count == 6
    assert info.has_sequence_modes

    copied = adapt_megatron_dataset(source, target)
    assert copied.path_prefix == str(target)
    assert (target.with_suffix(".idx")).read_bytes() == (source.with_suffix(".idx")).read_bytes()
    assert (target.with_suffix(".bin")).read_bytes() == (source.with_suffix(".bin")).read_bytes()
    assert load_megatron_dataset(target, mmap=False)[2].tolist() == [30]


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_megatron_validation_rejects_truncated_binary_payload(tmp_path):
    """A missing token is reported before the training iterator opens the pair."""
    prefix = tmp_path / "truncated_text_document"
    _write_megatron_pair(prefix)
    with open(str(prefix) + ".bin", "r+b") as stream:
        stream.truncate(10)

    with pytest.raises(ValueError, match="binary payload size mismatch"):
        inspect_megatron_dataset(prefix)
