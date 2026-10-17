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
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch.utils.data import default_collate  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.constants import (
    IGNORE_INDEX,
    OMNI_CONCAT_FIELDS,
    OMNI_CP_PAD_VALUES,
    OMNI_CP_TOKEN_FIELDS,
)
from hyper_parallel.data.dataset_logging import get_dataset_logger

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


def _pad_batch_tokens(
        batch: Mapping[str, Any],
        alignment_size: int,
        token_pad_values: Mapping[str, int],
        pad_to_length: int | None = None,
) -> dict[str, Any]:
    """Pad to a fixed length or alignment while preserving real boundaries and media.

    Example:
        Length 18, TP=4, CP=2: token fields grow to 24, labels pad with
        -100, and cu_seq_lens [0, 10, 18] becomes [0, 10, 18, 24].
        With pad_to_length=32, the same boundaries become [0, 10, 18, 32].
    """
    padded_batch = dict(batch)
    input_ids = padded_batch["input_ids"]
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
        raise ValueError("Batch alignment requires input_ids with shape [batch, sequence]")

    sequence_length = input_ids.shape[-1]
    if pad_to_length is None:
        pad_length = (-sequence_length) % alignment_size
    else:
        if sequence_length > pad_to_length:
            raise ValueError(
                f"Packed sequence length {sequence_length} exceeds pad_to_length={pad_to_length}; "
                "increase pad_to_length or reduce the packing token budget and sample lengths"
            )

        pad_length = pad_to_length - sequence_length

    if not pad_length:
        return padded_batch

    for field, pad_value in token_pad_values.items():
        if field not in padded_batch:
            continue

        value = padded_batch[field]
        if not torch.is_tensor(value) or value.shape != input_ids.shape:
            raise ValueError(f"Batch token field {field!r} must match input_ids shape")

        padded_batch[field] = torch.nn.functional.pad(value, (0, pad_length), value=pad_value)

    if "cu_seq_lens" in padded_batch:
        boundaries = padded_batch["cu_seq_lens"].reshape(-1)
        if (input_ids.shape[0] != 1 or boundaries.numel() < 2
                or int(boundaries[0]) != 0 or int(boundaries[-1]) != sequence_length):
            raise ValueError("cu_seq_lens must cover one complete packed input row")

        # A separate ignored sequence keeps attention metadata consistent with
        # the physical tail without changing any real sample boundary.
        padded_batch["cu_seq_lens"] = torch.cat((boundaries, boundaries[-1:] + pad_length))

    return padded_batch


@dataclass
class TextPackingCollator:
    """Pack unpadded Online text samples and emit ``cu_seq_lens``.

    Only ``input_ids`` and pre-shifted ``labels`` are packed here. Unified
    ``get_batch`` constructs loss, position, mask, and CP runtime fields.
    """

    alignment_size: int = 1
    pad_to_length: int | None = None

    def __post_init__(self) -> None:
        """Validate fixed padding before any samples are read."""
        _validate_padding_config(self.alignment_size, self.pad_to_length)

    def __call__(self, model_samples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Pack samples into one ``[1, packed_length]`` forward-backward batch.

        Args:
            model_samples: Online samples selected for one forward-backward step.

        Returns:
            Packed ``input_ids`` and ``labels`` plus int32 ``cu_seq_lens``.

        Raises:
            ValueError: If no samples are provided.
        """
        if not model_samples:
            raise ValueError("model_samples must contain at least one Online sample")

        packed_batch = {}
        for field in ("input_ids", "labels"):
            values = [model_sample[field] for model_sample in model_samples]
            packed_batch[field] = torch.cat(values, dim=-1).unsqueeze(0)

        seq_lens = model_samples[0]["input_ids"].new_tensor(
            [model_sample["input_ids"].shape[-1] for model_sample in model_samples]
        )
        zero = seq_lens.new_zeros(1)
        seq_ends = seq_lens.cumsum(dim=0)
        cu_seq_lens = torch.cat((zero, seq_ends))
        cu_seq_lens = cu_seq_lens.to(torch.int32)
        packed_batch["cu_seq_lens"] = cu_seq_lens

        aligned_batch = _pad_batch_tokens(
            packed_batch, self.alignment_size, {"input_ids": 0, "labels": IGNORE_INDEX}, self.pad_to_length,
        )
        return aligned_batch


@dataclass
class OmniCollator:
    """Collate Omni samples, then pad token fields to a fixed length or TP × CP."""

    alignment_size: int = 1
    pad_to_length: int | None = None

    def __post_init__(self) -> None:
        """Validate fixed padding before any samples are read."""
        _validate_padding_config(self.alignment_size, self.pad_to_length)

    def __call__(self, model_samples: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        """Stack token fields, concatenate media, and pad only the final batch.

        Example:
            One packed sample of length 18, TP=4, CP=2 produces [1, 24]
            token fields. Image tensors and global image starts are unchanged.
        """
        if not model_samples:
            raise ValueError("model_samples must contain at least one Omni sample")

        sample_fields = set()
        for model_sample in model_samples:
            sample_fields.update(model_sample)

        batch = {}
        for field in sample_fields:
            values = []
            for model_sample in model_samples:
                if field in model_sample:
                    values.append(model_sample[field])

            if field in OMNI_CONCAT_FIELDS:
                batch[field] = torch.cat(values, dim=0)
            else:
                batch[field] = default_collate(values)

        token_pad_values = {field: OMNI_CP_PAD_VALUES.get(field, 0) for field in OMNI_CP_TOKEN_FIELDS}
        aligned_batch = _pad_batch_tokens(batch, self.alignment_size, token_pad_values, self.pad_to_length)
        if logger.isEnabledFor(logging.DEBUG) and batch["input_ids"].device.type == "cpu":
            self._log_batch(aligned_batch, batch["input_ids"].shape[-1])
        return aligned_batch


    def _log_batch(self, batch: Mapping[str, Any], real_length: int) -> None:
        """Inspect CPU batches before transport without synchronizing accelerator streams."""
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        boundaries = batch.get("cu_seq_lens")
        boundaries = [] if boundaries is None else boundaries.reshape(-1).tolist()
        sequence_length = input_ids.shape[-1]
        issues = []
        if labels.shape != input_ids.shape:
            issues.append("labels shape differs from input_ids")
        if sequence_length == 0:
            issues.append("empty input_ids")
        if boundaries and (boundaries[0] != 0 or boundaries[-1] != sequence_length
                           or any(end <= start for start, end in zip(boundaries[:-1], boundaries[1:]))):
            issues.append("cu_seq_lens do not cover strictly increasing packed intervals")
        padding = sequence_length - real_length
        supervised = int((labels >= 0).sum())
        if not supervised:
            issues.append("no supervised tokens")
        if padding and bool(torch.any(labels[..., real_length:] != IGNORE_INDEX)):
            issues.append("padding tail contains supervised labels")
        lengths = [end - start for start, end in zip(boundaries[:-1], boundaries[1:])]
        logger.debug(
            "Omni collate: input_shape=%s label_shape=%s real_tokens=%d padding_tokens=%d "
            "alignment=%d cu_seq_lens=%s segment_lengths=%s supervised_tokens=%d issues=%s",
            tuple(input_ids.shape), tuple(labels.shape), real_length, padding, self.alignment_size,
            boundaries, lengths, supervised, issues, enabled=True,
        )
        if issues:
            logger.warning("Omni collate data anomalies: %s", issues)


def build_default_collate_fn() -> Callable[[list[Any]], Any]:
    """Build ordinary PyTorch collation for already prepared samples.

    Returns:
        PyTorch default collation.
    """
    collate_fn = default_collate

    return collate_fn


# Indexed samples are already prepared and use the same ordinary collation.
build_indexed_collate_fn = build_default_collate_fn


def build_omni_collate_fn(
        mesh_context: Any | None = None,
        pad_to_length: int | None = None,
) -> OmniCollator:
    """Build Omni collation with optional fixed padding after packing.

    Args:
        mesh_context: Runtime TP/CP topology injected by the Trainer.
        pad_to_length: Exact global token length, divisible by TP × CP. None
            pads only for alignment. Longer packed sequences raise ValueError.

    Returns:
        A collator that pads token lengths and preserves modality metadata.

    Raises:
        ValueError: If pad_to_length is not a positive aligned integer or None.
    """
    alignment_size = _get_token_alignment_size(mesh_context)
    collate_fn = OmniCollator(alignment_size=alignment_size, pad_to_length=pad_to_length)
    return collate_fn


def build_online_text_collate_fn(
        mesh_context: Any | None = None,
        pad_to_length: int | None = None,
) -> TextPackingCollator:
    """Build Online text collation shared by fixed N and dynamic K batching.

    Packing concatenates ``input_ids`` and ``labels`` and emits ``cu_seq_lens``.
    Tail padding defaults to alignment from the runtime topology. A fixed
    pad_to_length can instead specify the exact global packed token length.
    The unified alignment size is ``cp_size * tp_size``, derived directly
    from the topology without a separate SP switch. Real sample boundaries
    remain unchanged and one synthetic final boundary covers the physical
    padding segment.

    Args:
        mesh_context: Runtime TP/CP topology injected by the Trainer.
        pad_to_length: Exact global token length, divisible by TP × CP. None
            pads only for alignment. Longer packed sequences raise ValueError.

    Returns:
        A collator producing one forward-backward batch.

    Raises:
        ValueError: If pad_to_length is not a positive aligned integer or None.
    """
    alignment_size = _get_token_alignment_size(mesh_context)
    collate_fn = TextPackingCollator(alignment_size=alignment_size, pad_to_length=pad_to_length)

    return collate_fn
