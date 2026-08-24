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
"""Read token sequences from paired ``.idx/.bin`` files."""

from __future__ import annotations

import os
import struct
import time
from functools import lru_cache
from typing import ClassVar

import numpy as np

from hyper_models.components.datasets.dataset_logging import get_dataset_logger
logger = get_dataset_logger(__name__)

_INDEX_HEADER = b"MMIDIDX\x00\x00"
_INDEX_VERSION = 1
_DTYPES = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float64,
    7: np.float32,
    8: np.uint16,
}


class _IndexReader:
    """Memory-map the compact sequence metadata stored in an index file."""

    def __init__(self, idx_path: str) -> None:
        """Parse and memory-map one index metadata file."""
        # Read the fixed-size header, which describes how the following index arrays are decoded.
        with open(idx_path, "rb") as stream:
            header = stream.read(len(_INDEX_HEADER))
            if header != _INDEX_HEADER:
                raise ValueError(f"Invalid indexed Dataset header in {idx_path!r}")

            version = struct.unpack("<Q", stream.read(8))[0]
            if version != _INDEX_VERSION:
                raise ValueError(f"Unsupported indexed Dataset version {version}; expected {_INDEX_VERSION}")

            dtype_code = struct.unpack("<B", stream.read(1))[0]
            self.dtype = np.dtype(_DTYPES[dtype_code])
            self.dtype_size = self.dtype.itemsize

            self.sequence_count = struct.unpack("<Q", stream.read(8))[0]
            self.document_count = struct.unpack("<Q", stream.read(8))[0]
            offset = stream.tell()

        # Map the three arrays stored after the header: token count per sequence, byte offset in .bin per sequence,
        # and sequence boundaries per document. The first two arrays have sequence_count entries, while the last
        # has document_count entries.
        self.bin_buffer_mmap = np.memmap(idx_path, mode="r", order="C")
        self.bin_buffer = memoryview(self.bin_buffer_mmap)
        self.sequence_lengths = np.frombuffer(
            self.bin_buffer, dtype=np.int32, count=self.sequence_count, offset=offset,
        )

        self.sequence_pointers = np.frombuffer(
            self.bin_buffer, dtype=np.int64, count=self.sequence_count, offset=offset + self.sequence_lengths.nbytes,
        )

        self.document_indices = np.frombuffer(
            self.bin_buffer, dtype=np.int64, count=self.document_count,
            offset=offset + self.sequence_lengths.nbytes + self.sequence_pointers.nbytes,
        )

        # Each document is a half-open sequence range; the final boundary must equal the sequence count.
        if self.document_indices.size == 0 or self.document_indices[-1] != self.sequence_count:
            raise ValueError("Indexed Dataset document boundaries do not match its sequence count")

    def __del__(self) -> None:
        """Close the index metadata mmap when it is no longer referenced."""
        bin_buffer_mmap = getattr(self, "bin_buffer_mmap", None)
        mmap_handle = getattr(bin_buffer_mmap, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()

    def __len__(self) -> int:
        """Return the number of indexed sequences."""
        return self.sequence_count

    @lru_cache(maxsize=8)
    def __getitem__(self, index: int | np.integer) -> tuple[np.int64, np.int32]:
        """Return the byte pointer and token length of one sequence."""
        return self.sequence_pointers[index], self.sequence_lengths[index]


class IndexedDataReader:
    """Read individual token sequences from a paired ``.idx/.bin`` Dataset.

    Args:
        path_prefix: Dataset path without the ``.idx`` or ``.bin`` suffix.
        mmap: Whether to memory-map the token payload.
        reuse_index: Whether to reuse the first loaded index metadata. This
            matches corpora whose shards share an identical index layout.
    """

    cached_index_reader: ClassVar[_IndexReader | None] = None

    def __init__(self, path_prefix: str, mmap: bool = True, reuse_index: bool = False) -> None:
        """Open the index metadata and token payload files."""
        self.path_prefix = ""
        self.mmap = False
        self.reuse_index = reuse_index
        self.index: _IndexReader | None = None
        self.bin_buffer_mmap: np.memmap | None = None
        self.bin_buffer: memoryview | None = None
        self.initialize(path_prefix, mmap)

    def initialize(self, path_prefix: str, mmap: bool) -> None:
        """Open the index metadata and token payload.

        Args:
            path_prefix: Dataset path without the ``.idx`` or ``.bin`` suffix.
            mmap: Whether to memory-map the token payload.
        """
        start_time = time.time()
        index_path = path_prefix + ".idx"
        data_path = path_prefix + ".bin"
        if not os.path.isfile(index_path) or not os.path.isfile(data_path):
            raise FileNotFoundError(f"Expected indexed Dataset files {index_path!r} and {data_path!r}")

        self.path_prefix = path_prefix
        self.mmap = mmap
        reused_cached_index = self.reuse_index and IndexedDataReader.cached_index_reader is not None
        if self.index is None:
            if reused_cached_index:
                self.index = IndexedDataReader.cached_index_reader
            else:
                self.index = _IndexReader(index_path)
                if self.reuse_index:
                    IndexedDataReader.cached_index_reader = self.index

        self.bin_buffer_mmap = None
        self.bin_buffer = None
        if mmap:
            self.bin_buffer_mmap = np.memmap(data_path, mode="r", order="C")
            self.bin_buffer = memoryview(self.bin_buffer_mmap)
        logger.debug(
            "Opened indexed Dataset: sequences=%d, documents=%d, dtype=%s, mmap=%s, prefix=%s, "
            "reused_index=%s, elapsed=%.4f seconds",
            len(self.index), self.index.document_indices.size - 1, self.index.dtype, mmap, path_prefix,
            reused_cached_index, time.time() - start_time,
        )

    def __getstate__(self) -> tuple[str, bool, bool]:
        """Serialize construction inputs instead of open mmap resources."""
        return self.path_prefix, self.mmap, self.reuse_index

    def __setstate__(self, state: tuple[str, bool, bool]) -> None:
        """Reopen index and data resources inside a DataLoader worker."""
        path_prefix, mmap, reuse_index = state
        self.__init__(path_prefix, mmap=mmap, reuse_index=reuse_index)

    def __del__(self) -> None:
        """Close the token mmap when this reader is released."""
        bin_buffer_mmap = getattr(self, "bin_buffer_mmap", None)
        mmap_handle = getattr(bin_buffer_mmap, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()

    def __len__(self) -> int:
        """Return the number of indexed token sequences."""
        return len(self._require_index())

    def _getitem_mmap(self, index: int | np.integer | slice) -> np.ndarray | list[np.ndarray]:
        """Return one sequence or a contiguous sequence slice from the mapped token payload."""
        index_reader = self._require_index()

        if isinstance(index, (int, np.integer)):
            sequence_pointer, sequence_length = index_reader[index]
            sequence = np.frombuffer(
                self.bin_buffer, dtype=index_reader.dtype, count=sequence_length, offset=sequence_pointer
            )
            return sequence

        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed Dataset must be contiguous")
            if start == stop:
                return []

            sequence_lengths = index_reader.sequence_lengths[index]
            sequence_offsets = np.cumsum(sequence_lengths)
            sequences = np.frombuffer(
                self.bin_buffer, dtype=index_reader.dtype, count=int(sequence_lengths.sum()),
                offset=int(index_reader.sequence_pointers[start]),
            )
            sequences = np.split(sequences, sequence_offsets[:-1])
            return sequences

        raise TypeError(f"Indexed Dataset indices must be integers or slices, got {type(index).__name__}")

    def _getitem_file(self, index: int | np.integer | slice) -> np.ndarray:
        """Return one sequence through direct file reading when mmap is disabled."""
        if isinstance(index, slice):
            raise NotImplementedError("Slicing is not implemented when mmap is disabled")

        if not isinstance(index, (int, np.integer)):
            raise TypeError(f"Indexed Dataset indices must be integers or slices, got {type(index).__name__}")

        return self.get(int(index))

    def __getitem__(self, index: int | np.integer | slice) -> np.ndarray | list[np.ndarray]:
        """Return one sequence, or a contiguous slice when mmap is enabled."""
        if self.bin_buffer is not None:
            return self._getitem_mmap(index)

        return self._getitem_file(index)

    def get(self, index: int, offset: int = 0, length: int | None = None) -> np.ndarray:
        """Read a contiguous token range from one sequence.

        Args:
            index: Sequence index.
            offset: First token offset in the sequence.
            length: Number of tokens to read, or all remaining tokens.

        Returns:
            A NumPy view of the requested tokens.
        """
        index_reader = self._require_index()
        sequence_pointer, sequence_length = index_reader[index]
        sequence_length = int(sequence_length)
        read_length = sequence_length - offset if length is None else length
        if offset < 0 or read_length < 0 or offset + read_length > sequence_length:
            raise ValueError(
                f"Token range [{offset}, {offset + read_length}) exceeds sequence length {sequence_length}"
            )

        byte_offset = int(sequence_pointer) + offset * index_reader.dtype.itemsize
        if self.bin_buffer:
            sequence = np.frombuffer(self.bin_buffer, dtype=index_reader.dtype, count=read_length, offset=byte_offset)
        else:
            sequence = np.empty(read_length, dtype=index_reader.dtype)
            with open(self.path_prefix + ".bin", mode="rb", buffering=0) as bin_buffer_file:
                bin_buffer_file.seek(byte_offset)
                bytes_read = bin_buffer_file.readinto(sequence)

            byte_length = read_length * index_reader.dtype.itemsize
            if bytes_read != byte_length:
                raise ValueError(f"Indexed Dataset read returned {bytes_read} bytes; expected {byte_length}")

        return sequence

    @property
    def sequence_lengths(self) -> np.ndarray:
        """Return the token length of every indexed sequence."""
        return self._require_index().sequence_lengths

    @property
    def document_indices(self) -> np.ndarray:
        """Return sequence indices that delimit documents."""
        return self._require_index().document_indices

    def get_document_indices(self) -> np.ndarray:
        """Return sequence indices that delimit documents."""
        return self.document_indices

    def set_document_indices(self, document_indices: np.ndarray) -> None:
        """Replace sequence indices that delimit documents."""
        self._require_index().document_indices = document_indices

    def _require_index(self) -> _IndexReader:
        """Return the initialized index reader."""
        if self.index is None:
            raise ValueError("Indexed Dataset metadata is not initialized")
        return self.index
