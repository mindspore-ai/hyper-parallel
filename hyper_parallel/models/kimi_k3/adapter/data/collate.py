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
"""K3 collation that preserves variable image counts and final token offsets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from math import lcm

# The project is Torch-only; the legacy platform lint does not apply.
import torch  # pylint: disable=forbidden-backend-import
import torch.nn.functional as F  # pylint: disable=forbidden-backend-import

from hyper_parallel.data.constants import IGNORE_INDEX, PACKING_METADATA_FIELDS
from hyper_parallel.models.kimi_k3.adapter.data.transform_fn import validate_media_grid

_TOKEN_PAD_VALUES = {
    "input_ids": 0,
    "labels": IGNORE_INDEX,
    "attention_mask": 0,
    "position_ids": 0,
    "media_token_mask": False,
}
_MEDIA_FIELDS = {"grid_thws", "pixel_values", "media_spans", "media_patch_offsets"}


def validate_kimi_k3_mesh(mesh_context: Any | None) -> None:
    """Validate the image data topology before constructing a DataLoader.

    Args:
        mesh_context: Framework mesh metadata, or None for local execution.

    Raises:
        ValueError: TP, PP or sequence parallelism is enabled for media batches.
    """
    if mesh_context is None:
        return
    for dimension in ("tp", "pp"):
        if int(getattr(mesh_context, f"{dimension}_size", 1)) != 1:
            raise ValueError("K3 media pretraining data requires TP=PP=1")
    if bool(getattr(mesh_context, "sequence_parallel", False)):
        raise ValueError("K3 image pretraining data does not support sequence parallelism yet")


class KimiK3Collator:
    """Pad independent samples and concatenate images in sample/media order.

    This callable supports different image counts, including text-only samples,
    in each row of a ``FixedBatchDataLoader`` / ``OmniParallelBatch`` batch.
    Optional packing retains one explicit boundary per input document.
    A two-token alignment prevents that runtime from padding only its registered
    token fields while leaving model-owned ``media_token_mask`` shorter.
    Single-document loader metadata is validated before optional packing.
    Pre-shifted targets include cross-rank next-token targets and mask document starts.
    """

    def __init__(self, *, pad_token_id: int = 0, pad_to_multiple_of: int = 2,
                 packed: bool = False, pre_shift_labels: bool = False, max_seq_len: int | None = None) -> None:
        """Set the decoder padding ID and even sequence alignment."""
        if (isinstance(pad_to_multiple_of, bool) or not isinstance(pad_to_multiple_of, int)
                or pad_to_multiple_of < 2 or pad_to_multiple_of % 2):
            raise ValueError("pad_to_multiple_of must be a positive multiple of 2 for OmniParallelBatch")
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.packed = packed
        self.pre_shift_labels = pre_shift_labels or packed
        self.max_seq_len = max_seq_len

    @staticmethod
    def _validate_sample(sample: Mapping[str, Any]) -> None:
        """Fail before padding if image correspondence or token ownership is lost."""
        expected_fields = set(_TOKEN_PAD_VALUES) | _MEDIA_FIELDS
        missing = expected_fields - set(sample)
        extra = set(sample) - expected_fields - PACKING_METADATA_FIELDS - {"media_metadata"}
        if missing or extra:
            raise ValueError(f"K3 encoded sample fields differ: missing={sorted(missing)}, extra={sorted(extra)}")
        token_ids = sample["input_ids"]
        if token_ids.ndim != 1 or token_ids.numel() == 0:
            raise ValueError("K3 collator expects independent nonempty [S] token sequences")
        KimiK3Collator._validate_loader_metadata(sample, len(token_ids))
        for field in _TOKEN_PAD_VALUES:
            if sample[field].shape != token_ids.shape:
                raise ValueError(f"K3 token field {field!r} must match input_ids shape")
        if not bool((sample["attention_mask"] == 1).all()):
            raise ValueError("K3 samples must be unpadded before collation")
        if sample["media_token_mask"].dtype != torch.bool:
            raise ValueError("media_token_mask must be boolean")
        KimiK3Collator._validate_media_sample(sample, token_ids)
        KimiK3Collator._validate_patch_payload(sample)

    @staticmethod
    def _validate_media_sample(sample: Mapping[str, Any], token_ids: torch.Tensor) -> None:
        """Check span ownership and the visual label mask before padding."""
        counts = validate_media_grid(sample["grid_thws"])
        KimiK3Collator._validate_media_metadata(sample)
        spans = sample["media_spans"]
        if spans.shape != (len(counts), 3) or spans.dtype not in (torch.int32, torch.int64):
            raise ValueError("media_spans must have integer shape [media_count, 3]")
        expected_mask = torch.zeros_like(sample["media_token_mask"])
        previous_end = 0
        for (sample_index, start, end), count in zip(spans.tolist(), counts.tolist()):
            if sample_index != 0 or start < previous_end or end > len(token_ids) or end - start != count:
                raise ValueError("media_spans must name ordered, non-overlapping image slots in the final sequence")
            expected_mask[start:end] = True
            previous_end = end
        if not torch.equal(expected_mask, sample["media_token_mask"]):
            raise ValueError("media_spans and media_token_mask must identify the same feature slots")
        if bool((sample["labels"][expected_mask] != IGNORE_INDEX).any()):
            raise ValueError("Visual feature slots must be excluded from language-model labels")

    @staticmethod
    def _validate_patch_payload(sample: Mapping[str, Any]) -> None:
        """Require offsets and RGB patch rows to describe the same media."""
        patch_counts = sample["grid_thws"].prod(dim=-1)
        expected_offsets = torch.cat((patch_counts.new_zeros(1), patch_counts.cumsum(0)))
        if not torch.equal(expected_offsets, sample["media_patch_offsets"]):
            raise ValueError("media_patch_offsets must follow grid_thws in processor order")
        pixels = sample["pixel_values"]
        if (pixels.ndim != 4 or pixels.shape[0] != int(expected_offsets[-1])
                or pixels.shape[1] != 3 or pixels.shape[2] != pixels.shape[3]):
            raise ValueError("pixel_values must match the image grids and contain square RGB patches")

    @staticmethod
    def _validate_media_metadata(sample: Mapping[str, Any]) -> None:
        """Check optional provenance against the temporal groups without moving it to a device."""
        metadata = sample.get("media_metadata")
        if metadata is None:
            return
        if len(metadata) != len(sample["grid_thws"]):
            raise ValueError("media_metadata must contain one record per media grid")
        for item, grid in zip(metadata, sample["grid_thws"].tolist()):
            if not isinstance(item, Mapping) or item.get("sample_index") != 0 or item.get("frame_count") != grid[0]:
                raise ValueError("media_metadata must preserve source sample and temporal frame count")
            for field in ("requested_timestamps", "frame_timestamps"):
                if field in item and len(item[field]) != grid[0]:
                    raise ValueError("media timestamps must contain one entry per sampled frame")

    def _collate_media_metadata(self, samples: Sequence[Mapping[str, Any]]) -> tuple[Any, ...] | None:
        """Keep source-document identity while media spans move into packed coordinates."""
        if not any("media_metadata" in sample for sample in samples):
            return None
        metadata = []
        for index, sample in enumerate(samples):
            items = sample.get("media_metadata")
            if items is None:
                items = [None] * len(sample["grid_thws"])
            for item in items:
                metadata.append(None if item is None else dict(
                    item, sample_index=0 if self.packed else index, document_index=index,
                ))
        return tuple(metadata)

    @staticmethod
    def _validate_loader_metadata(sample: Mapping[str, Any], sequence_length: int) -> None:
        """Consume only the loader's description of one complete document."""
        if "cu_seq_lens" in sample:
            boundaries = torch.as_tensor(sample["cu_seq_lens"])
            if boundaries.dtype not in (torch.int32, torch.int64) or boundaries.ndim != 1:
                raise ValueError("K3 cu_seq_lens metadata must be a one-dimensional integer tensor")
            if boundaries.tolist() != [0, sequence_length]:
                raise ValueError(
                    "K3 input samples must each contain one complete document; enable packed=True "
                    "on the K3 collator to combine documents with isolated KDA state"
                )
        if "packing_length" in sample:
            packing_length = torch.as_tensor(sample["packing_length"])
            if (packing_length.ndim != 0 or packing_length.dtype not in (torch.int32, torch.int64)
                    or int(packing_length) != sequence_length):
                raise ValueError("K3 packing_length metadata must equal the final unpadded sequence length")

    def _collate_tokens(self, samples: Sequence[Mapping[str, Any]], used_length: int,
                        target_length: int) -> dict[str, torch.Tensor]:
        """Pad or concatenate all token fields with the same global geometry."""
        batch = {}
        for field, pad_value in _TOKEN_PAD_VALUES.items():
            if field == "input_ids":
                pad_value = self.pad_token_id
            if self.packed:
                values = torch.cat([sample[field] for sample in samples])
                batch[field] = F.pad(values, (0, target_length - used_length), value=pad_value).unsqueeze(0)
            else:
                batch[field] = torch.stack([
                    F.pad(sample[field], (0, target_length - len(sample[field])), value=pad_value)
                    for sample in samples
                ])
        return batch

    def _collate_spans(self, samples: Sequence[Mapping[str, Any]], lengths: Sequence[int],
                       labels: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        """Offset media ownership with documents and mask each new document start."""
        batch_spans, offset, boundaries = [], 0, [0]
        for sample_index, (sample, length) in enumerate(zip(samples, lengths)):
            spans = sample["media_spans"].clone()
            spans[:, 0] = 0 if self.packed else sample_index
            if self.packed:
                spans[:, 1:] += offset
                labels[0, offset] = IGNORE_INDEX
                offset += length
                boundaries.append(offset)
            batch_spans.append(spans)
        return torch.cat(batch_spans, dim=0), boundaries

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Collate complete documents, then globally shift targets before CP slicing.

        Args:
            samples: Unpadded independent documents with expanded media placeholders.

        Returns:
            Independent rows or a flattened packed row with global media coordinates.
        """
        if not samples:
            raise ValueError("K3 collator requires at least one sample")
        for sample in samples:
            self._validate_sample(sample)
        lengths = [len(sample["input_ids"]) for sample in samples]
        used_length = sum(lengths) if self.packed else max(lengths)
        target_length = ((used_length + self.pad_to_multiple_of - 1)
                         // self.pad_to_multiple_of * self.pad_to_multiple_of)
        if self.max_seq_len is not None and target_length > self.max_seq_len:
            raise ValueError("K3 media packing exceeds max_seq_len; documents cannot be truncated")
        batch = self._collate_tokens(samples, used_length, target_length)
        batch["grid_thws"] = torch.cat([sample["grid_thws"] for sample in samples], dim=0)
        batch["pixel_values"] = torch.cat([sample["pixel_values"] for sample in samples], dim=0)
        batch["media_spans"], boundaries = self._collate_spans(samples, lengths, batch["labels"])
        patch_counts = batch["grid_thws"].prod(dim=-1)
        batch["media_patch_offsets"] = torch.cat((patch_counts.new_zeros(1), patch_counts.cumsum(0)))
        if self.packed:
            if boundaries[-1] < target_length:
                boundaries.append(target_length)
            batch["cu_seq_lens"] = torch.tensor(boundaries, dtype=torch.int32)
        metadata = self._collate_media_metadata(samples)
        if metadata is not None:
            batch["media_metadata"] = metadata
        if self.pre_shift_labels:
            batch["labels"] = F.pad(batch["labels"][:, 1:], (0, 1), value=IGNORE_INDEX)
            batch["labels_are_shifted"] = torch.tensor(True)
        return batch


def build_kimi_k3_collate_fn(
        *, mesh_context: Any | None = None, pad_token_id: int = 0, pad_to_multiple_of: int = 2,
        packed: bool = False, pre_shift_labels: bool | None = None, max_seq_len: int | None = None,
) -> KimiK3Collator:
    """Build media collation with alignment and target ownership known before CP.

    Args:
        mesh_context: Decoder topology; TP and PP remain one.
        pad_token_id: Token for right padding.
        pad_to_multiple_of: Additional requested sequence alignment.
        packed: Concatenate complete documents into one isolated row.
        pre_shift_labels: Shift before CP; automatically enabled for CP and packing.
        max_seq_len: Optional physical packed sequence limit.

    Returns:
        Collator preserving global media spans and document boundaries.
    """
    validate_kimi_k3_mesh(mesh_context)
    cp_size = int(getattr(mesh_context, "cp_size", 1))
    required_shift = cp_size > 1 or packed
    if pre_shift_labels is False and required_shift:
        raise ValueError("K3 media CP and packing require globally pre-shifted labels")
    return KimiK3Collator(pad_token_id=pad_token_id,
                          pad_to_multiple_of=lcm(pad_to_multiple_of, 2 * cp_size), packed=packed,
                          pre_shift_labels=required_shift if pre_shift_labels is None else pre_shift_labels,
                          max_seq_len=max_seq_len)
