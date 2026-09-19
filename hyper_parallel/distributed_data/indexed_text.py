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
"""Built-in constructor callbacks for unpacked indexed text samples."""
# This distributed-data package is intentionally PyTorch-only.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch  # pylint: disable=forbidden-backend-import

_IGNORE_INDEX = -100


def _as_1d_long_tensor(value: Any, field: str) -> torch.Tensor:
    """Normalize one source field to a CPU int64 vector."""
    try:
        tensor = torch.as_tensor(value, dtype=torch.int64)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Indexed source field {field!r} cannot be converted to an int64 tensor") from exc
    if tensor.ndim != 1 or tensor.numel() < 1:
        raise ValueError(f"Indexed source field {field!r} must be a non-empty one-dimensional sequence")
    return tensor


def pack_indexed_text_samples(samples: Sequence[Any], seq_len: int) -> dict[str, torch.Tensor]:
    """Pack one planned source-sample bin into a fixed-length text row.

    Args:
        samples: Source mappings containing pre-shifted ``input_ids`` and
            ``labels`` vectors.
        seq_len: Physical output length for one packed row.

    Returns:
        One padded row and its exact source-sample boundaries.
    """
    if not samples:
        raise ValueError("Indexed text packing requires at least one source sample")
    if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len < 1:
        raise ValueError(f"seq_len must be a positive integer, but got {seq_len!r}")

    input_parts = []
    label_parts = []
    sample_lengths = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise ValueError(f"Indexed text samples must be mappings, but got {type(sample)}")
        try:
            input_ids = _as_1d_long_tensor(sample["input_ids"], "input_ids")
            labels = _as_1d_long_tensor(sample["labels"], "labels")
        except KeyError as exc:
            raise ValueError(f"Indexed text sample is missing required field {exc.args[0]!r}") from exc
        if input_ids.shape != labels.shape:
            raise ValueError("Indexed text input_ids and labels must have identical shapes")
        input_parts.append(input_ids)
        label_parts.append(labels)
        sample_lengths.append(input_ids.numel())

    packed_length = sum(sample_lengths)
    if packed_length > seq_len:
        raise ValueError(f"Indexed text bin has {packed_length} tokens, exceeding seq_len={seq_len}")
    input_ids = torch.cat(input_parts)
    labels = torch.cat(label_parts)
    padding_length = seq_len - packed_length
    if padding_length:
        input_ids = torch.cat((input_ids, input_ids.new_zeros(padding_length)))
        labels = torch.cat((labels, labels.new_full((padding_length,), _IGNORE_INDEX)))

    sequence_ends = torch.tensor(sample_lengths, dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
    if padding_length:
        sequence_ends = torch.cat((sequence_ends, sequence_ends.new_tensor([seq_len])))
    cu_seq_lens = torch.cat((sequence_ends.new_zeros(1), sequence_ends))
    return {
        "input_ids": input_ids,
        "labels": labels,
        "cu_seq_lens": cu_seq_lens,
    }


def collate_indexed_text_sequences(packed_sequences: Sequence[Any]) -> dict[str, torch.Tensor]:
    """Combine fixed-length packed rows into one rank-local training batch.

    Args:
        packed_sequences: Outputs from :func:`pack_indexed_text_samples`.

    Returns:
        Batched token fields and flattened global sequence boundaries.
    """
    if not packed_sequences:
        raise ValueError("Indexed text collation requires at least one packed sequence")
    if any(not isinstance(sequence, Mapping) for sequence in packed_sequences):
        raise ValueError("Every indexed packed sequence must be a mapping")

    input_ids = torch.stack([sequence["input_ids"] for sequence in packed_sequences])
    labels = torch.stack([sequence["labels"] for sequence in packed_sequences])
    if input_ids.shape != labels.shape or input_ids.ndim != 2:
        raise ValueError("Indexed packed input_ids and labels must form equal two-dimensional batches")

    seq_len = input_ids.shape[1]
    boundary_parts = [input_ids.new_zeros(1, dtype=torch.int32)]
    for row_index, sequence in enumerate(packed_sequences):
        cu_seq_lens = sequence["cu_seq_lens"]
        if not isinstance(cu_seq_lens, torch.Tensor) or cu_seq_lens.dtype != torch.int32:
            raise ValueError("Indexed packed cu_seq_lens must be an int32 tensor")
        if cu_seq_lens.ndim != 1 or cu_seq_lens.numel() < 2 or int(cu_seq_lens[0]) != 0:
            raise ValueError("Indexed packed cu_seq_lens must be a leading-zero boundary vector")
        if int(cu_seq_lens[-1]) != seq_len:
            raise ValueError("Indexed packed cu_seq_lens must cover the complete padded row")
        boundary_parts.append(cu_seq_lens[1:] + row_index * seq_len)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "cu_seq_lens": torch.cat(boundary_parts),
    }


__all__ = ["collate_indexed_text_sequences", "pack_indexed_text_samples"]
