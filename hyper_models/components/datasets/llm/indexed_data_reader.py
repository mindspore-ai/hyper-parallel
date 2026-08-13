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

import os
import struct
from typing import ClassVar

import numpy as np


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

    def __init__(self, path: str) -> None:
        """Parse and memory-map one index metadata file."""
        with open(path, "rb") as stream:
            header = stream.read(len(_INDEX_HEADER))
            if header != _INDEX_HEADER:
                raise ValueError(f"Invalid indexed Dataset header in {path!r}")

            version = struct.unpack("<Q", stream.read(8))[0]
            if version != _INDEX_VERSION:
                raise ValueError(
                    f"Unsupported indexed Dataset version {version}; expected {_INDEX_VERSION}"
                )

            dtype_code = struct.unpack("<B", stream.read(1))[0]
            try:
                self.dtype = np.dtype(_DTYPES[dtype_code])
            except KeyError as error:
                raise ValueError(f"Unsupported indexed Dataset dtype code: {dtype_code}") from error

            self.sequence_count = struct.unpack("<Q", stream.read(8))[0]
            self.document_count = struct.unpack("<Q", stream.read(8))[0]
            metadata_offset = stream.tell()

        self._buffer = np.memmap(path, mode="r", order="C")
        self.sequence_lengths = np.frombuffer(
            self._buffer,
            dtype=np.int32,
            count=self.sequence_count,
            offset=metadata_offset,
        )
        pointer_offset = metadata_offset + self.sequence_lengths.nbytes
        self.sequence_pointers = np.frombuffer(
            self._buffer,
            dtype=np.int64,
            count=self.sequence_count,
            offset=pointer_offset,
        )
        document_offset = pointer_offset + self.sequence_pointers.nbytes
        self.document_indices = np.frombuffer(
            self._buffer,
            dtype=np.int64,
            count=self.document_count,
            offset=document_offset,
        )

        if self.document_indices.size == 0 or self.document_indices[-1] != self.sequence_count:
            raise ValueError("Indexed Dataset document boundaries do not match its sequence count")

    def __del__(self) -> None:
        """Close the index metadata mmap when it is no longer referenced."""
        buffer = getattr(self, "_buffer", None)
        mmap_handle = getattr(buffer, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()


class IndexedDataReader:
    """Read individual token sequences from a paired ``.idx/.bin`` Dataset.

    Args:
        path_prefix: Dataset path without the ``.idx`` or ``.bin`` suffix.
        mmap: Whether to memory-map the token payload.
        reuse_index: Whether to reuse the first loaded index metadata. This
            matches corpora whose shards share an identical index layout.
    """

    _cached_index: ClassVar[_IndexReader | None] = None

    def __init__(
        self,
        path_prefix: str,
        *,
        mmap: bool = True,
        reuse_index: bool = False,
    ) -> None:
        """Open the index metadata and token payload files."""
        self.path_prefix = path_prefix
        self.mmap = mmap
        self.reuse_index = reuse_index
        index_path = path_prefix + ".idx"
        data_path = path_prefix + ".bin"
        if not os.path.isfile(index_path) or not os.path.isfile(data_path):
            raise FileNotFoundError(
                f"Expected indexed Dataset files {index_path!r} and {data_path!r}"
            )

        if reuse_index and IndexedDataReader._cached_index is not None:
            self._index = IndexedDataReader._cached_index
        else:
            self._index = _IndexReader(index_path)
            if reuse_index:
                IndexedDataReader._cached_index = self._index

        if mmap:
            self._data = np.memmap(data_path, mode="r", order="C")
        else:
            self._data = None

    def __getstate__(self) -> tuple[str, bool, bool]:
        """Serialize construction inputs instead of open mmap resources."""
        return self.path_prefix, self.mmap, self.reuse_index

    def __setstate__(self, state: tuple[str, bool, bool]) -> None:
        """Reopen index and data resources inside a DataLoader worker."""
        path_prefix, mmap, reuse_index = state
        self.__init__(path_prefix, mmap=mmap, reuse_index=reuse_index)

    def __del__(self) -> None:
        """Close the token mmap when this reader is released."""
        data = getattr(self, "_data", None)
        mmap_handle = getattr(data, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()

    def __len__(self) -> int:
        """Return the number of indexed token sequences."""
        return int(self._index.sequence_count)

    def __getitem__(self, index: int) -> np.ndarray:
        """Return one complete indexed token sequence."""
        return self.get(index)

    def get(self, index: int, offset: int = 0, length: int | None = None) -> np.ndarray:
        """Read a contiguous token range from one sequence.

        Args:
            index: Sequence index.
            offset: First token offset in the sequence.
            length: Number of tokens to read, or all remaining tokens.

        Returns:
            A NumPy view of the requested tokens.
        """
        sequence_length = int(self._index.sequence_lengths[index])
        if length is None:
            length = sequence_length - offset
        if offset < 0 or length < 0 or offset + length > sequence_length:
            raise ValueError(
                f"Token range [{offset}, {offset + length}) exceeds sequence length {sequence_length}"
            )

        item_size = self._index.dtype.itemsize
        byte_offset = int(self._index.sequence_pointers[index]) + offset * item_size
        byte_length = length * item_size
        if self._data is not None:
            sequence = self._data[byte_offset:byte_offset + byte_length].view(self._index.dtype)
            return sequence

        sequence = np.empty(length, dtype=self._index.dtype)
        with open(self.path_prefix + ".bin", mode="rb", buffering=0) as data_file:
            data_file.seek(byte_offset)
            bytes_read = data_file.readinto(sequence)
        if bytes_read != byte_length:
            raise ValueError(
                f"Indexed Dataset read returned {bytes_read} bytes; expected {byte_length}"
            )
        return sequence

    @property
    def sequence_lengths(self) -> np.ndarray:
        """Return the token length of every indexed sequence."""
        return self._index.sequence_lengths
