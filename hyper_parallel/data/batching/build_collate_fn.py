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
"""Build final batch collators for Indexed and Online sources."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from torch.utils.data import default_collate

from hyper_parallel.data.constants import (
    IGNORE_INDEX,
    OMNI_CONCAT_FIELDS,
    OMNI_CP_PAD_VALUES,
    OMNI_CP_TOKEN_FIELDS,
)
from hyper_parallel.data.dataset_logging import get_dataset_logger
from hyper_parallel.data.batching.packing import SamplePacker

logger = get_dataset_logger(__name__)


def _get_token_alignment_size(mesh_context: Any | None) -> int:
    """Return TP X CP, the required final batch token alignment."""

    if mesh_context is None:
        return 1

    cp_size = int(getattr(mesh_context, "cp_size", 1))
    tp_size = int(getattr(mesh_context, "tp_size", 1))

    if cp_size <= 0 or tp_size <= 0:
        raise ValueError("mesh_context tp_size and cp_size must be positive")

    alignment_size = cp_size * tp_size
    return alignment_size


def _validate_padding_config(alignment_size: int, pad_to_length: int | None) -> None:
    """Check the shared text/Omni padding contract once at construction."""

    if alignment_size <= 0:
        raise ValueError("alignment_size must be positive")

    if pad_to_length is None:
        return

    if isinstance(pad_to_length, bool) or not isinstance(pad_to_length, int) or pad_to_length <= 0:
        raise ValueError("pad_to_length must be a positive integer or None")

    if pad_to_length % alignment_size:
        raise ValueError(
            f"pad_to_length={pad_to_length} must be divisible by TP * CP alignment_size={alignment_size}"
        )


def _pad_packed_token_field(
        name: str,
        values: Sequence[torch.Tensor],
        packed_samples: Sequence[Mapping[str, Any]],
        sequence_lengths: Sequence[int],
        target_length: int,
) -> torch.Tensor:
    """Validate one token field and pad every packed row to the target length."""
    if len(values) != len(packed_samples) or any(value.shape != sample["input_ids"].shape
                                                for sample, value in zip(packed_samples, values)):
        raise ValueError(f"Packed token field {name!r} must match input_ids in every packed sample")

    output = values[0].new_full((len(packed_samples), target_length), OMNI_CP_PAD_VALUES.get(name, 0))

    for batch_index, (value, length) in enumerate(zip(values, sequence_lengths)):
        output[batch_index, :length] = value

    return output


def _collate_packed_boundaries(
        packed_samples: Sequence[Mapping[str, Any]],
        sequence_lengths: Sequence[int],
        target_length: int,
) -> torch.Tensor:
    """Flatten document coordinates and isolate each packed row's padding tail."""
    reference = packed_samples[0]["input_ids"]
    boundary_parts = [reference.new_zeros(1, dtype=torch.int32)]

    for batch_index, (sample, length) in enumerate(zip(packed_samples, sequence_lengths)):
        boundaries = sample.get("cu_seq_lens")

        if boundaries is None:
            boundaries = reference.new_tensor([0, length], dtype=torch.int32)

        boundaries = boundaries.reshape(-1).to(torch.int32)

        if (boundaries.numel() < 2 or int(boundaries[0]) != 0 or int(boundaries[-1]) != length
                or bool(torch.any(boundaries[1:] <= boundaries[:-1]))):
            raise ValueError("cu_seq_lens must cover each complete packed row with increasing boundaries")

        boundary_parts.append(boundaries[1:] + batch_index * target_length)

        if length < target_length:
            boundary_parts.append(boundaries.new_tensor([(batch_index + 1) * target_length]))

    return torch.cat(boundary_parts)


def _collate_packed_samples(
        packed_samples: Sequence[Mapping[str, Any]],
        alignment_size: int,
        pad_to_length: int | None,
        max_seq_len: int | None,
) -> dict[str, Any]:
    """Pad packed samples into [B, T] token tensors and collate their metadata.

    Args:
        packed_samples: B packed sample mappings, each with one-dimensional
            input_ids [L], matching token fields, and optional document/media metadata.
        alignment_size: Required TP * CP sequence-length alignment.
        pad_to_length: Fixed sequence length, or None to align the longest sample.
        max_seq_len: Upper bound on the padded sequence length.

    Returns:
        Padded token tensors with combined document boundaries and media fields.

    Example:
        B=2, pad_to_length=4096: packed samples [1800+2000], [500+3000]
        become input_ids/labels [2, 4096], with 296/596 padding tokens (0/-100).
        cu_seq_lens = [0, 1800, 3800, 4096, 4596, 7596, 8192].
    """

    if any(sample["input_ids"].ndim != 1 or sample["input_ids"].shape[-1] == 0
           for sample in packed_samples):
        raise ValueError("Packed samples must contain nonempty one-dimensional input_ids")

    sequence_lengths = [sample["input_ids"].shape[-1] for sample in packed_samples]
    longest_length = max(sequence_lengths)
    target_length = pad_to_length

    if target_length is None:
        target_length = longest_length + (-longest_length) % alignment_size

    if longest_length > target_length:
        raise ValueError(f"Packed sequence length {longest_length} exceeds pad_to_length={target_length}")

    if max_seq_len is not None and target_length > max_seq_len:
        raise ValueError(f"Batch padding length {target_length} exceeds max_seq_len={max_seq_len}")

    # Token fields become [B, T]; media retain their modality-specific leading dimension.
    batch = {}
    fields = set().union(*(sample.keys() for sample in packed_samples)) - {"cu_seq_lens", "packing_length"}
    image_fields = {"image_patch_lengths", "image_vit_grid_hw", "image_llm_grid_hw", "image_token_starts"}

    for name in fields:
        values = [sample[name] for sample in packed_samples if name in sample]

        if name in OMNI_CP_TOKEN_FIELDS:
            batch[name] = _pad_packed_token_field(name, values, packed_samples, sequence_lengths, target_length)

        elif name in OMNI_CONCAT_FIELDS or name in image_fields:
            batch[name] = torch.cat(values, dim=0)
        else:
            batch[name] = default_collate(values)

    batch["cu_seq_lens"] = _collate_packed_boundaries(packed_samples, sequence_lengths, target_length)

    if "image_patch_lengths" in fields:
        batch["image_batch_indices"] = torch.cat([
            sample["image_patch_lengths"].new_full((sample["image_patch_lengths"].numel(),), batch_index)
            for batch_index, sample in enumerate(packed_samples) if "image_patch_lengths" in sample
        ])

    return batch


@dataclass
class TextPackingCollator:
    """Pack unpadded Online text samples and emit ``cu_seq_lens``.

    Only ``input_ids`` and pre-shifted ``labels`` are packed here. Unified
    ``get_batch`` constructs loss, position, mask, and CP runtime fields.
    """

    alignment_size: int = 1
    pad_to_length: int | None = None
    max_seq_len: int | None = None

    def __post_init__(self) -> None:
        """Validate fixed padding before any samples are read."""
        _validate_padding_config(self.alignment_size, self.pad_to_length)
        _validate_padding_config(self.alignment_size, self.max_seq_len)

    def __call__(self, model_samples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Pad independently packed rows to one common ``[B, T]`` shape.

        Args:
            model_samples: Packed rows with cu_seq_lens. A flat group without
                boundaries is accepted as one row for direct callers.

        Returns:
            Packed ``input_ids`` and ``labels`` plus int32 ``cu_seq_lens``.

        Raises:
            ValueError: If no samples are provided.
        """

        if not model_samples:
            raise ValueError("model_samples must contain at least one Online sample")

        if not any("cu_seq_lens" in sample for sample in model_samples):
            model_samples = [SamplePacker().pack_selected_samples(model_samples)]

        padded_batch = _collate_packed_samples(
            model_samples, self.alignment_size, self.pad_to_length, self.max_seq_len,
        )
        return padded_batch


@dataclass
class OmniCollator:
    """Collate Omni samples, then pad token fields to a fixed length or TP × CP."""

    alignment_size: int = 1
    pad_to_length: int | None = None
    max_seq_len: int | None = None

    def __post_init__(self) -> None:
        """Validate fixed padding before any samples are read."""
        _validate_padding_config(self.alignment_size, self.pad_to_length)
        _validate_padding_config(self.alignment_size, self.max_seq_len)

    def __call__(self, model_samples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Stack token fields, concatenate media, and pad only the final batch.

        Example:
            One packed sample of length 18, TP=4, CP=2 produces [1, 24]
            token fields. Image tensors and global image starts are unchanged.
        """

        if not model_samples:
            raise ValueError("model_samples must contain at least one Omni sample")

        aligned_batch = _collate_packed_samples(
            model_samples, self.alignment_size, self.pad_to_length, self.max_seq_len,
        )

        if logger.isEnabledFor(logging.DEBUG) and aligned_batch["input_ids"].device.type == "cpu":
            self._log_batch(aligned_batch, sum(sample["input_ids"].numel() for sample in model_samples))

        return aligned_batch

    def _log_batch(self, batch: Mapping[str, Any], real_token_count: int) -> None:
        """Inspect CPU batches before transport without synchronizing accelerator streams."""
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        boundaries = batch.get("cu_seq_lens")
        boundaries = [] if boundaries is None else boundaries.reshape(-1).tolist()
        padding = input_ids.numel() - real_token_count
        supervised = int((labels >= 0).sum())
        lengths = [end - start for start, end in zip(boundaries[:-1], boundaries[1:])]
        logger.debug(
            "Omni collate: input_shape=%s label_shape=%s real_tokens=%d padding_tokens=%d "
            "alignment=%d cu_seq_lens=%s segment_lengths=%s supervised_tokens=%d",
            tuple(input_ids.shape), tuple(labels.shape), real_token_count, padding, self.alignment_size,
            boundaries, lengths, supervised, enabled=True,
        )

        if not supervised:
            logger.warning("Omni collate batch contains no supervised tokens")


def build_default_collate_fn() -> Callable[[list[Any]], Any]:
    """Build ordinary PyTorch collation for already prepared samples.

    Returns:
        PyTorch default collation.
    """
    collate_fn = default_collate

    return collate_fn


# Indexed samples are already prepared and use the same ordinary collation.
build_indexed_collate_fn = build_default_collate_fn


def _sft_row_boundaries(boundaries: Any, sequence_length: int) -> list[int]:
    """Remove boundary sentinels and validate one complete SFT row."""
    row = torch.as_tensor(boundaries)

    if row.ndim != 1 or row.dtype not in (torch.int32, torch.int64):
        raise ValueError("Each cu_seq_lens row must be a one-dimensional integer array")

    row = row[row != -1]

    if (row.numel() < 2 or row[0] != 0 or row[-1] != sequence_length
            or torch.any(row[1:] <= row[:-1])):
        raise ValueError("Document boundaries must strictly increase from zero to row length")

    return row.tolist()


def _collate_sft_samples(batch: list[Any], *, alignment_size: int) -> Any:
    """Stack SFT rows and retain per-row boundaries without sentinel padding.

    Records from SFTFromMRDataset have 1D input_ids/labels; already grouped
    records may have [rows, S]. Variable-length rows are padded, never truncated.
    """

    if not isinstance(batch[0], Mapping) or "cu_seq_lens" not in batch[0]:
        return default_collate(batch)

    token_rows, label_rows, boundary_rows = [], [], []

    for sample in batch:
        tokens, labels = torch.as_tensor(sample["input_ids"]), torch.as_tensor(sample["labels"])

        if tokens.ndim not in (1, 2) or tokens.shape != labels.shape or not tokens.numel():
            raise ValueError("SFT tokens and labels must have matching nonempty [S] or [rows, S] shapes")

        boundaries = sample["cu_seq_lens"]

        if tokens.ndim == 1:
            tokens, labels, boundaries = tokens.unsqueeze(0), labels.unsqueeze(0), [boundaries]

        if len(boundaries) != len(tokens):
            raise ValueError("cu_seq_lens must have one boundary array per token row")

        for row in boundaries:
            boundary_rows.append(_sft_row_boundaries(row, tokens.shape[-1]))
        token_rows.extend(tokens.unbind(0))
        label_rows.extend(labels.unbind(0))

    sequence_length = max(row.numel() for row in token_rows)
    sequence_length += (-sequence_length) % alignment_size
    packed_keys = {"input_ids", "labels", "cu_seq_lens"}
    result = default_collate([
        {key: value for key, value in sample.items() if key not in packed_keys} for sample in batch
    ])

    for key, rows, value in (("input_ids", token_rows, 0), ("labels", label_rows, IGNORE_INDEX)):
        result[key] = torch.stack([
            torch.nn.functional.pad(row, (0, sequence_length - row.numel()), value=value) for row in rows
        ])

    for row in boundary_rows:
        if row[-1] < sequence_length:
            # Isolate ignored tail padding from every real document.
            row.append(sequence_length)

    result["cu_seq_lens"] = boundary_rows
    return result


def build_sft_collate_fn(mesh_context: Any | None = None) -> Callable[[list[Any]], Any]:
    """Build SFT collation preserving [B, S] and each row's document boundaries.

    Args:
        mesh_context: Optional topology for TP x CP sequence-length alignment.

    Returns:
        A CPU collator accepting 1D records or already grouped 2D records.
        Ordinary records without cu_seq_lens use default collation.
    """
    return partial(_collate_sft_samples, alignment_size=_get_token_alignment_size(mesh_context))


def _get_pad_length(
        pad_to_length: bool,
        max_seq_len: int | None = None,
) -> int | None:
    """Use the shared packing limit for fixed padding, or align actual lengths."""

    if not isinstance(pad_to_length, bool):
        raise ValueError("pad_to_length must be a bool")

    if not pad_to_length:
        return None

    if max_seq_len is None:
        raise ValueError("Fixed batch padding requires max_seq_len")

    return max_seq_len


def build_omni_collate_fn(
        mesh_context: Any | None = None,
        pad_to_length: bool = True,
        max_seq_len: int | None = None,
) -> OmniCollator:
    """Build Omni collation with optional fixed padding after packing.

    Args:
        mesh_context: Runtime TP/CP topology injected by the Trainer.
        pad_to_length: True pads every row to max_seq_len; False pads to the
            aligned maximum actual row length. Defaults to True.
        max_seq_len: Maximum packed row length supplied by the Trainer.

    Returns:
        A collator that pads token lengths and preserves modality metadata.

    Raises:
        ValueError: If pad_to_length is not a bool, fixed padding has no max_seq_len,
            or the length is invalid or not divisible by TP × CP.
    """
    alignment_size = _get_token_alignment_size(mesh_context)
    collate_fn = OmniCollator(alignment_size=alignment_size, max_seq_len=max_seq_len,
                             pad_to_length=_get_pad_length(pad_to_length, max_seq_len))

    return collate_fn


def build_online_text_collate_fn(
        mesh_context: Any | None = None,
        pad_to_length: bool = True,
        max_seq_len: int | None = None,
) -> TextPackingCollator:
    """Build Online text collation shared by fixed N and dynamic K batching.

    The loader independently packs B rows. Collation pads them to max_seq_len
    by default, preserving document boundaries and isolating every padding tail.
    Dynamic padding uses the longest actual row rounded to ``cp_size * tp_size``.

    Args:
        mesh_context: Runtime TP/CP topology injected by the Trainer.
        pad_to_length: True pads every row to max_seq_len; False pads to the
            aligned maximum actual row length. Defaults to True.
        max_seq_len: Maximum packed row length supplied by the Trainer.

    Returns:
        A collator producing one forward-backward batch.

    Raises:
        ValueError: If pad_to_length is not a bool, fixed padding has no max_seq_len,
            or the length is invalid or not divisible by TP × CP.
    """
    alignment_size = _get_token_alignment_size(mesh_context)
    collate_fn = TextPackingCollator(alignment_size=alignment_size, max_seq_len=max_seq_len,
                                    pad_to_length=_get_pad_length(pad_to_length, max_seq_len))

    return collate_fn
