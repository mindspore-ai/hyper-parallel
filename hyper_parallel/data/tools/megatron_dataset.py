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
"""Validate and adapt Megatron-compatible ``.idx/.bin`` datasets.

Megatron text datasets are already consumable by HyperParallel. This module
provides a small, dependency-light compatibility boundary for checking an
external prefix and copying it to a training data directory when a workflow
needs a separate local name.
"""

from __future__ import annotations

import argparse
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from hyper_parallel.data.indexed.indexed_data_reader import IndexedDataReader

_INDEX_HEADER = b"MMIDIDX\x00\x00"
_INDEX_VERSION = 1
_DTYPES = {
    1: np.dtype(np.uint8),
    2: np.dtype(np.int8),
    3: np.dtype(np.int16),
    4: np.dtype(np.int32),
    5: np.dtype(np.int64),
    6: np.dtype(np.float64),
    7: np.dtype(np.float32),
    8: np.dtype(np.uint16),
}


@dataclass(frozen=True)
class MegatronDatasetInfo:
    """Metadata and file-size information for one indexed dataset prefix."""

    path_prefix: str
    dtype: np.dtype
    sequence_count: int
    document_count: int
    token_count: int
    index_bytes: int
    binary_bytes: int
    has_sequence_modes: bool


def normalize_prefix(path: str | Path) -> str:
    """Normalize a prefix or a path ending in ``.idx``/``.bin``."""
    value = str(path)
    if value.endswith((".idx", ".bin")):
        return value[:-4]
    return value


def _read_exact(stream, size: int, field: str) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ValueError(f"Truncated Megatron index while reading {field}")
    return value


def inspect_megatron_dataset(path: str | Path) -> MegatronDatasetInfo:
    """Validate a Megatron index/bin pair and return its metadata.

    The validator accepts the standard text layout and the optional trailing
    per-sequence mode array used by multimodal Megatron datasets. It checks
    boundaries, byte pointers, and the binary payload size before training.
    """
    path_prefix = normalize_prefix(path)
    idx_path = Path(path_prefix + ".idx")
    bin_path = Path(path_prefix + ".bin")
    if not idx_path.is_file() or not bin_path.is_file():
        raise FileNotFoundError(f"Expected Megatron files {idx_path!s} and {bin_path!s}")

    with idx_path.open("rb") as stream:
        if _read_exact(stream, len(_INDEX_HEADER), "header") != _INDEX_HEADER:
            raise ValueError(f"Invalid Megatron index header in {idx_path}")
        version = struct.unpack("<Q", _read_exact(stream, 8, "version"))[0]
        if version != _INDEX_VERSION:
            raise ValueError(f"Unsupported Megatron index version {version}; expected {_INDEX_VERSION}")
        dtype_code = struct.unpack("<B", _read_exact(stream, 1, "dtype"))[0]
        try:
            dtype = _DTYPES[dtype_code]
        except KeyError as error:
            raise ValueError(f"Unsupported Megatron dtype code {dtype_code}") from error
        sequence_count = struct.unpack("<Q", _read_exact(stream, 8, "sequence count"))[0]
        document_count = struct.unpack("<Q", _read_exact(stream, 8, "document count"))[0]
        payload_offset = stream.tell()

    if document_count == 0:
        raise ValueError("Megatron index must contain an initial document boundary")

    sequence_lengths = np.fromfile(
        idx_path, dtype=np.int32, count=sequence_count, offset=payload_offset,
    )
    pointer_offset = payload_offset + sequence_lengths.nbytes
    sequence_pointers = np.fromfile(idx_path, dtype=np.int64, count=sequence_count, offset=pointer_offset)
    document_offset = pointer_offset + sequence_pointers.nbytes
    document_indices = np.fromfile(idx_path, dtype=np.int64, count=document_count, offset=document_offset)
    if len(sequence_lengths) != sequence_count or len(sequence_pointers) != sequence_count:
        raise ValueError("Megatron index has truncated sequence metadata")
    if len(document_indices) != document_count:
        raise ValueError("Megatron index has truncated document metadata")
    if np.any(sequence_lengths < 0):
        raise ValueError("Megatron sequence lengths cannot be negative")
    if (
        document_indices[0] != 0
        or document_indices[-1] != sequence_count
        or np.any(np.diff(document_indices) < 0)
    ):
        raise ValueError("Megatron document boundaries are not monotonic or do not match sequence count")

    expected_pointers = np.cumsum(sequence_lengths, dtype=np.int64) - sequence_lengths.astype(np.int64)
    expected_pointers *= dtype.itemsize
    if not np.array_equal(sequence_pointers, expected_pointers):
        raise ValueError("Megatron sequence pointers do not match sequence lengths")

    metadata_end = document_offset + document_indices.nbytes
    trailing_bytes = idx_path.stat().st_size - metadata_end
    has_sequence_modes = trailing_bytes == sequence_count and sequence_count > 0
    if trailing_bytes not in (0, sequence_count):
        raise ValueError(
            f"Unexpected trailing bytes in Megatron index: {trailing_bytes}; "
            f"expected 0 or {sequence_count} sequence modes"
        )

    token_count = int(sequence_lengths.astype(np.int64).sum())
    binary_bytes = bin_path.stat().st_size
    expected_binary_bytes = token_count * dtype.itemsize
    if binary_bytes != expected_binary_bytes:
        raise ValueError(
            f"Megatron binary payload size mismatch: expected {expected_binary_bytes} bytes, got {binary_bytes}"
        )

    return MegatronDatasetInfo(
        path_prefix=path_prefix,
        dtype=dtype,
        sequence_count=int(sequence_count),
        document_count=int(document_count - 1),
        token_count=token_count,
        index_bytes=idx_path.stat().st_size,
        binary_bytes=binary_bytes,
        has_sequence_modes=has_sequence_modes,
    )


def load_megatron_dataset(path: str | Path, *, mmap: bool = True) -> IndexedDataReader:
    """Open a validated Megatron text dataset with the standard HyperParallel reader."""
    path_prefix = normalize_prefix(path)
    inspect_megatron_dataset(path_prefix)
    return IndexedDataReader(path_prefix, mmap=mmap)


def adapt_megatron_dataset(
    source: str | Path,
    output: str | Path,
    *,
    force: bool = False,
) -> MegatronDatasetInfo:
    """Validate and copy a Megatron pair to a new HyperParallel prefix."""
    info = inspect_megatron_dataset(source)
    source_prefix = Path(info.path_prefix).resolve()
    output_prefix = Path(normalize_prefix(output)).resolve()
    if source_prefix == output_prefix:
        raise ValueError("Source and output prefixes must be different")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_paths = (Path(str(output_prefix) + ".idx"), Path(str(output_prefix) + ".bin"))
    if not force and any(path.exists() for path in output_paths):
        raise FileExistsError(f"Output dataset already exists: {output_prefix}")
    shutil.copyfile(str(source_prefix) + ".idx", output_paths[0])
    shutil.copyfile(str(source_prefix) + ".bin", output_paths[1])
    return inspect_megatron_dataset(output_prefix)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate or adapt Megatron indexed datasets")
    parser.add_argument("--input-prefix", required=True, help="Megatron prefix or its .idx/.bin path")
    parser.add_argument("--output-prefix", help="Optional destination prefix for a validated copy")
    parser.add_argument("--force", action="store_true", help="Overwrite destination files")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the validation or copy command."""
    args = _parse_args(argv)
    info = (
        adapt_megatron_dataset(args.input_prefix, args.output_prefix, force=args.force)
        if args.output_prefix
        else inspect_megatron_dataset(args.input_prefix)
    )
    print(
        "validated prefix={path} dtype={dtype} sequences={sequences} documents={documents} "
        "tokens={tokens} sequence_modes={modes}".format(
            path=info.path_prefix,
            dtype=info.dtype,
            sequences=info.sequence_count,
            documents=info.document_count,
            tokens=info.token_count,
            modes=info.has_sequence_modes,
        )
    )


if __name__ == "__main__":
    main()
