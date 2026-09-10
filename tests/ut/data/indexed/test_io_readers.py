# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for indexed dataset reader resource cleanup."""

import gc
from typing import NoReturn, Type
import weakref

import numpy

from hyper_parallel.data.indexed.io import (
    IndexedDataset,
    _FileBinReader,
    _IndexReader,
    _IndexWriter,
    _MMapBinReader,
)


class _CapturingBinReader:
    """Capture read arguments without allocating the requested buffer."""

    def __init__(self) -> None:
        """Initialize captured read arguments."""
        self.args = None

    def read(self, dtype: Type[numpy.number], count: int, offset: int) -> NoReturn:
        """Record read arguments and stop before numpy.split."""
        self.args = (dtype, count, offset)
        raise RuntimeError("stop after read")


class _FakeIndex:
    """Minimal index object for IndexedDataset slice tests."""

    dtype = numpy.int32

    def __init__(self) -> None:
        """Initialize fake index metadata with overflowing int32 lengths."""
        self.sequence_lengths = numpy.array([2**30, 2**30], dtype=numpy.int32)
        self.sequence_pointers = numpy.array([0, 2**30 * 4], dtype=numpy.int64)
        self.sequence_modes = None

    def __len__(self) -> int:
        """Return fake sequence count."""
        return len(self.sequence_lengths)


def test_index_reader_close_releases_mmap_views(tmp_path):
    """_IndexReader.close should be safe and idempotent with live index views."""
    idx_path = tmp_path / "sample.idx"
    with _IndexWriter(str(idx_path), numpy.int32) as writer:
        writer.write([2, 3], None, [0, 2])

    reader = _IndexReader(str(idx_path), multimodal=False)
    assert reader.sequence_lengths.tolist() == [2, 3]

    reader.close()
    reader.close()


def test_index_reader_exposed_arrays_survive_reader_close(tmp_path):
    """Index arrays exposed through properties should not depend on the mmap lifetime."""
    idx_path = tmp_path / "sample.idx"
    with _IndexWriter(str(idx_path), numpy.int32) as writer:
        writer.write([2, 3], None, [0, 2])

    reader = _IndexReader(str(idx_path), multimodal=False)
    sequence_lengths = reader.sequence_lengths
    document_indices = reader.document_indices

    reader.close()

    assert sequence_lengths.tolist() == [2, 3]
    assert document_indices.tolist() == [0, 2]


def test_index_reader_getitem_does_not_keep_reader_alive(tmp_path):
    """_IndexReader.__getitem__ should not cache self at class scope."""
    idx_path = tmp_path / "sample.idx"
    with _IndexWriter(str(idx_path), numpy.int32) as writer:
        writer.write([2, 3], None, [0, 2])

    reader = _IndexReader(str(idx_path), multimodal=False)
    reader_ref = weakref.ref(reader)

    assert reader[0][1] == 2
    del reader
    gc.collect()

    assert reader_ref() is None


def test_mmap_bin_reader_close_closes_file_without_copying_reads(tmp_path):
    """_MMapBinReader.close should close the file while preserving zero-copy reads."""
    bin_path = tmp_path / "sample.bin"
    numpy.array([1, 2, 3], dtype=numpy.int32).tofile(bin_path)

    reader = _MMapBinReader(str(bin_path))
    array = reader.read(numpy.int32, 3, 0)
    assert array.tolist() == [1, 2, 3]
    assert array.base is not None

    reader.close()
    reader.close()
    assert array.tolist() == [1, 2, 3]
    assert reader._file is None  # pylint: disable=protected-access


def test_mmap_bin_reader_close_keeps_returned_view_alive_after_reader_delete(tmp_path):
    """_MMapBinReader.close should not invalidate live arrays returned from mmap."""
    bin_path = tmp_path / "sample.bin"
    numpy.array([1, 2, 3], dtype=numpy.int32).tofile(bin_path)

    reader = _MMapBinReader(str(bin_path))
    array = reader.read(numpy.int32, 3, 0)
    reader_ref = weakref.ref(reader)

    reader.close()
    del reader
    gc.collect()

    assert array.tolist() == [1, 2, 3]
    assert reader_ref() is None


def test_file_bin_reader_read_checks_short_reads(tmp_path):
    """_FileBinReader.read should reject truncated binary reads."""
    bin_path = tmp_path / "sample.bin"
    numpy.array([1, 2, 3], dtype=numpy.int32).tofile(bin_path)

    reader = _FileBinReader(str(bin_path))

    assert reader.read(numpy.int32, 3, 0).tolist() == [1, 2, 3]
    try:
        reader.read(numpy.int32, 4, 0)
    except ValueError as exc:
        assert "Short read" in str(exc)
    else:
        raise AssertionError("_FileBinReader.read should reject short reads")


def test_indexed_dataset_slice_uses_int64_token_count():
    """Slice reads should not overflow when int32 lengths exceed 2 Gi tokens."""
    dataset = object.__new__(IndexedDataset)
    dataset.index = _FakeIndex()
    dataset.bin_reader = _CapturingBinReader()
    dataset.multimodal = False

    try:
        dataset[:2]
    except RuntimeError as exc:
        assert str(exc) == "stop after read"
    else:
        raise AssertionError("test reader should stop before allocating the slice result")

    assert dataset.bin_reader.args == (numpy.int32, 2**31, 0)
