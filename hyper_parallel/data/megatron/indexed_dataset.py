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
"""Megatron-LM ``.bin`` / ``.idx`` reader and writer.

Wire format (matches Megatron-LM ``mmap_indexed_dataset.py``):

``<prefix>.idx``:
- 9-byte magic ``b"MMIDIDX\\x00\\x00"``
- ``uint64`` version (=1)
- ``uint8`` dtype code (1=uint8, 2=int8, 3=int16, 4=int32, 5=int64,
  6=float64, 7=float32, 8=uint16)
- ``uint64`` sequence count ``N``
- ``uint64`` document count ``D``
- ``int32 [N]`` sequence lengths (in tokens)
- ``int64 [N]`` sequence pointers (byte offset into ``.bin``)
- ``int64 [D + 1]`` document indices (sequence-index boundaries; first is 0,
  last is ``N``)
- Optional ``int8 [N]`` sequence modes for multimodal datasets — ignored
  here (LM-only path)

``<prefix>.bin``: raw concatenation of all sequences in the declared dtype.

We expose:

- :func:`get_bin_path` / :func:`get_idx_path` — canonical suffixing logic.
- :class:`IndexedDataset` — memory-mapped ``.bin`` payload with in-memory
  ``.idx`` metadata; ``__getitem__`` returns ``np.ndarray``.
- :class:`IndexedDatasetBuilder` — sequential writer for tests, fixtures,
  and ad-hoc preprocessing.

Pure ``numpy`` — no Cython helpers — so this works wherever ``numpy`` runs.
The data path stays zero-copy: token slices view the memory-mapped ``.bin``;
the smaller ``.idx`` arrays are read into RAM once at open time.
"""
import logging
import os
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


# Megatron magic header — kept byte-identical so a ``.idx`` produced by
# Megatron-LM ``preprocess_data.py`` loads transparently here.
_MAGIC = b"MMIDIDX\x00\x00"
_VERSION = 1

_DTYPES: Dict[int, np.dtype] = {
    1: np.dtype(np.uint8),
    2: np.dtype(np.int8),
    3: np.dtype(np.int16),
    4: np.dtype(np.int32),
    5: np.dtype(np.int64),
    6: np.dtype(np.float64),
    7: np.dtype(np.float32),
    8: np.dtype(np.uint16),
}
_DTYPE_CODES: Dict[np.dtype, int] = {v: k for k, v in _DTYPES.items()}


def strip_suffix(prefix: str) -> str:
    """Return ``prefix`` with a trailing ``.bin``/``.idx`` removed (idempotent).

    The single source of truth for path-prefix normalisation — :func:`get_bin_path`,
    :func:`get_idx_path` and the dataset builder all route through here so the
    suffix rule never diverges across call sites.
    """
    return prefix[:-4] if prefix.endswith((".bin", ".idx")) else prefix


def get_bin_path(prefix: str) -> str:
    """Return ``<prefix>.bin`` (strips an already-present ``.bin``/``.idx``)."""
    return strip_suffix(prefix) + ".bin"


def get_idx_path(prefix: str) -> str:
    """Return ``<prefix>.idx``."""
    return strip_suffix(prefix) + ".idx"


def dtype_code(dtype: np.dtype) -> int:
    """Return the on-disk dtype code for ``dtype``.

    Raises:
        ValueError: When ``dtype`` is not part of the Megatron dtype table.
    """
    dtype = np.dtype(dtype)
    if dtype not in _DTYPE_CODES:
        raise ValueError(
            f"Unsupported Megatron indexed-dataset dtype: {dtype}. "
            f"Allowed: {sorted(d.name for d in _DTYPE_CODES)}"
        )
    return _DTYPE_CODES[dtype]


def code_dtype(code: int) -> np.dtype:
    """Inverse of :func:`dtype_code`."""
    try:
        return _DTYPES[code]
    except KeyError as exc:
        raise ValueError(f"Unknown Megatron dtype code: {code}") from exc


class _IndexReader:
    """In-memory metadata parsed from the ``.idx`` companion file.

    Holds the four arrays Megatron-LM stores: lengths, pointers,
    document indices and optional modes. The index payload is read into
    compact numpy arrays; the much larger ``.bin`` remains memory-mapped by
    :class:`IndexedDataset`.
    """

    def __init__(self, idx_path: str, *, multimodal: bool = False) -> None:
        if not os.path.isfile(idx_path):
            raise FileNotFoundError(f"Megatron index file not found: {idx_path}")
        with open(idx_path, "rb") as f:
            magic = f.read(len(_MAGIC))
            if magic != _MAGIC:
                raise ValueError(
                    f"Not a Megatron indexed dataset (magic mismatch): "
                    f"{idx_path}. Expected {_MAGIC!r}, got {magic!r}"
                )
            version = struct.unpack("<Q", f.read(8))[0]
            if version != _VERSION:
                raise ValueError(
                    f"Unsupported indexed-dataset version: {version} "
                    f"(only v{_VERSION} is recognised)"
                )
            code = struct.unpack("<B", f.read(1))[0]
            self.dtype = code_dtype(code)
            self.sequence_count = struct.unpack("<Q", f.read(8))[0]
            self.document_count = struct.unpack("<Q", f.read(8))[0]

            # The index payload is small (4-12 bytes per sequence) — read it
            # into RAM rather than memmap. Avoids the fragility of slicing a
            # memmap with offsetted ``np.frombuffer`` while still being fast.
            self.sequence_lengths = np.frombuffer(
                f.read(self.sequence_count * 4), dtype=np.int32,
            )
            self.sequence_pointers = np.frombuffer(
                f.read(self.sequence_count * 8), dtype=np.int64,
            )
            self.document_indices = np.frombuffer(
                f.read(self.document_count * 8), dtype=np.int64,
            )
            if multimodal:
                self.sequence_modes = np.frombuffer(
                    f.read(self.sequence_count), dtype=np.int8,
                )
            else:
                self.sequence_modes = None


class IndexedDataset:
    """Read-only Megatron ``.bin``/``.idx`` dataset.

    Args:
        path_prefix: Path without the ``.bin``/``.idx`` suffix. Either
            suffix is also accepted and stripped.
        mmap: When ``True`` (default), the ``.bin`` is memory-mapped; reads
            return zero-copy views. Set to ``False`` to load the entire
            ``.bin`` into RAM — useful only for sub-MB fixtures in tests.
        multimodal: Whether to parse the optional ``sequence_modes`` array
            in the ``.idx`` (LM training leaves this off).

    Example:
        >>> ds = IndexedDataset("/data/wiki")  # noqa: SKIP
        >>> tokens = ds[0]                     # noqa: SKIP
        >>> doc_0 = ds.get_document(0)         # noqa: SKIP

    Note:
        Equivalent to Megatron-LM's ``MMapIndexedDataset`` but rewritten
        for the HyperParallel coding conventions; the on-disk format is
        byte-compatible.
    """

    def __init__(self, path_prefix: str, *, mmap: bool = True, multimodal: bool = False) -> None:
        idx_path = get_idx_path(path_prefix)
        bin_path = get_bin_path(path_prefix)
        if not os.path.isfile(bin_path):
            raise FileNotFoundError(f"Megatron .bin not found: {bin_path}")
        self._index = _IndexReader(idx_path, multimodal=multimodal)
        if mmap:
            self._bin_buffer = np.memmap(bin_path, dtype=np.uint8, mode="r", order="C")
        else:
            with open(bin_path, "rb") as f:
                data = f.read()
            self._bin_buffer = np.frombuffer(data, dtype=np.uint8)
        self.path_prefix = path_prefix

    # ------------------------------------------------------------------
    # Sequence-level access
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return int(self._index.sequence_count)

    @property
    def dtype(self) -> np.dtype:
        """Token dtype (numpy)."""
        return self._index.dtype

    @property
    def sequence_lengths(self) -> np.ndarray:
        """int32[N] token-counts for each sequence."""
        return self._index.sequence_lengths

    @property
    def document_indices(self) -> np.ndarray:
        """int64[D+1] sequence-index boundaries marking documents."""
        return self._index.document_indices

    def get(self, idx: int, offset: int = 0, length: Optional[int] = None) -> np.ndarray:
        """Read a slice of sequence ``idx``.

        Args:
            idx: Sequence index in ``[0, len)``.
            offset: Token offset within the sequence.
            length: Tokens to read; ``None`` reads to the end.

        Returns:
            A ``np.ndarray`` of the configured dtype. The returned array is
            a view into the mmap when ``mmap=True``; callers must not write
            to it.
        """
        seq_len = int(self._index.sequence_lengths[idx])
        if length is None:
            length = seq_len - offset
        if offset < 0 or length < 0 or offset + length > seq_len:
            raise ValueError(
                f"Slice [{offset}:{offset + length}] out of bounds for "
                f"sequence {idx} of length {seq_len}"
            )
        item_size = self._index.dtype.itemsize
        byte_ptr = int(self._index.sequence_pointers[idx]) + offset * item_size
        # ``sequence_pointers`` are accumulated ``nbytes`` of items, so
        # ``byte_ptr`` is always a multiple of ``item_size`` and a typed
        # view is alignment-safe; the slice is a zero-copy view into the
        # mmap (or a plain ndarray slice in the RAM path).
        return self._bin_buffer[byte_ptr: byte_ptr + length * item_size].view(self._index.dtype)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.get(idx)

    # ------------------------------------------------------------------
    # Document-level access
    # ------------------------------------------------------------------

    @property
    def num_documents(self) -> int:
        """Document count (``len(document_indices) - 1``).

        On disk ``document_count`` is ``D + 1`` because Megatron stores the
        full boundary array including the trailing ``N`` terminator.
        """
        return int(self._index.document_count) - 1

    def get_document(self, doc_idx: int) -> np.ndarray:
        """Concatenate all sequences belonging to ``doc_idx``."""
        if doc_idx < 0 or doc_idx >= self.num_documents:
            raise IndexError(
                f"document index {doc_idx} out of range [0, {self.num_documents})"
            )
        start = int(self._index.document_indices[doc_idx])
        end = int(self._index.document_indices[doc_idx + 1])
        parts = [self.get(i) for i in range(start, end)]
        if not parts:
            return np.array([], dtype=self._index.dtype)
        return np.concatenate(parts)


class IndexedDatasetBuilder:
    """Write a ``.bin``/``.idx`` pair sequentially.

    Usage:
        builder = IndexedDatasetBuilder("/path/to/wiki", dtype=np.int32)
        builder.add_item(np.array([1, 2, 3, 4], dtype=np.int32))
        builder.end_document()
        builder.add_item(np.array([5, 6, 7], dtype=np.int32))
        builder.end_document()
        builder.finalize()

    Args:
        path_prefix: Output path without suffix.
        dtype: Token dtype; must be in the Megatron dtype table.

    Note:
        Designed for fixture generation and small-scale preprocessing.
        Streams tokens straight to the ``.bin`` and keeps lightweight
        Python lists for the index metadata until :meth:`finalize`.
    """

    def __init__(self, path_prefix: str, *, dtype: np.dtype = np.int32) -> None:
        self.path_prefix = path_prefix
        self.dtype = np.dtype(dtype)
        # Make sure the dtype is supported before opening any file.
        dtype_code(self.dtype)
        self._bin_path = get_bin_path(path_prefix)
        self._idx_path = get_idx_path(path_prefix)
        os.makedirs(os.path.dirname(self._bin_path) or ".", exist_ok=True)
        self._bin_file = open(self._bin_path, "wb")  # pylint: disable=R1732
        self._sequence_lengths: List[int] = []
        self._sequence_pointers: List[int] = []
        self._document_indices: List[int] = [0]
        self._byte_offset = 0

    def add_item(self, tokens: np.ndarray) -> None:
        """Append one sequence of tokens to the dataset."""
        tokens = np.asarray(tokens, dtype=self.dtype)
        self._bin_file.write(tokens.tobytes(order="C"))
        self._sequence_lengths.append(int(tokens.size))
        self._sequence_pointers.append(self._byte_offset)
        self._byte_offset += int(tokens.nbytes)

    def end_document(self) -> None:
        """Mark the current sequence boundary as a document boundary."""
        self._document_indices.append(len(self._sequence_lengths))

    def finalize(self) -> Tuple[str, str]:
        """Close ``.bin`` and write the companion ``.idx``.

        Returns:
            ``(idx_path, bin_path)`` for caller convenience.

        Raises:
            ValueError: When no document boundary was emitted.
        """
        self._bin_file.close()
        # ``end_document`` must have been called at least once OR the writer
        # is empty; an empty dataset still needs a 1-element boundary list.
        if self._document_indices[-1] != len(self._sequence_lengths):
            self.end_document()
        seq_lengths = np.asarray(self._sequence_lengths, dtype=np.int32)
        seq_pointers = np.asarray(self._sequence_pointers, dtype=np.int64)
        doc_indices = np.asarray(self._document_indices, dtype=np.int64)
        with open(self._idx_path, "wb") as f:
            f.write(_MAGIC)
            f.write(struct.pack("<Q", _VERSION))
            f.write(struct.pack("<B", dtype_code(self.dtype)))
            f.write(struct.pack("<Q", seq_lengths.size))
            f.write(struct.pack("<Q", doc_indices.size))
            f.write(seq_lengths.tobytes())
            f.write(seq_pointers.tobytes())
            f.write(doc_indices.tobytes())
        logger.info(
            "IndexedDatasetBuilder finalized: %d sequences, %d documents -> %s + %s",
            seq_lengths.size, doc_indices.size, self._bin_path, self._idx_path,
        )
        return self._idx_path, self._bin_path
