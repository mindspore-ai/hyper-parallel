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
"""Build DeepSeek-V4.1 model inputs from generic runtime batches."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import logging
from dataclasses import dataclass
from math import lcm
from typing import Any

import torch

from hyper_parallel.components.modules.attention.compressed_sparse_attention2.document_layout import (
    GraphPackedSequence,
    SharedCompressedPackedSequence,
)
from hyper_parallel.data.dataset_logging import get_dataset_logger
from hyper_parallel.data.batching.build_collate_fn import build_omni_collate_fn
from hyper_parallel.data.batching.runtime_input import (
    RuntimeInputAdapter,
    RuntimeInputContext,
)


class DeepseekV41DatasetAdapter(RuntimeInputAdapter):
    """Build V4.1 packed attention and local image insertion inputs."""

    def __init__(
            self,
            *,
            include_position_ids: bool = True,
            include_image_sequence_start: bool = True,
            compress_ratios: "list[int] | None" = None,
    ) -> None:
        """Configure fields already supplied by the selected batch runtime.

        Args:
            include_position_ids: Build CP-global position IDs. Omni batches
                require this; ``TextParallelBatch`` already owns them.
            include_image_sequence_start: Build the local decoder image insertion
                offset. Text-only batches do not need this field.
            compress_ratios: Every active model compression ratio (for example
                ``[1, 2]``). The graph boundary prepares per-ratio compressed-key
                metadata and validates document alignment from this list; eager
                never reads it. The set must cover every nonzero
                ``v41_compress_ratios`` entry of the model configuration.
        """
        self.include_position_ids = include_position_ids
        self.include_image_sequence_start = include_image_sequence_start
        for ratio in compress_ratios or ():
            if isinstance(ratio, bool) or not isinstance(ratio, int):
                raise ValueError(f"compress_ratios entries must be ints, got {ratio!r}")
            if ratio < 0:
                raise ValueError(f"compress_ratios entries must be nonnegative, got {ratio}")
        self.compress_ratios = tuple(int(ratio) for ratio in compress_ratios or ())

    def prepare_graph_inputs(self, model_inputs: Mapping[str, Any]) -> dict[str, Any]:
        """Build an isolated packing representation with compressed-key metadata.

        This seam is the graph-mode twin of eager's in-forward preparation: the
        per-ratio compressed-key minima are rebuilt from every batch's real
        boundaries here, outside capture, and enter the graph as pytree tensors
        that the in-graph attention consumes exactly like eager does.

        Eager FSDP also rebuilds pytrees, so its inputs must retain real document
        boundaries. Only the compiler receives this prepared metadata copy.
        """
        inputs = dict(model_inputs)
        source = inputs["packed_seq_params"]
        device = inputs["input_ids"].device
        packed = GraphPackedSequence(
            source.cu_seq_lens, source.local_query_start,
            source.local_query_length, source.global_sequence_length,
        )
        ratios = tuple(sorted({ratio for ratio in self.compress_ratios if ratio > 0}))
        if ratios:
            packed = packed.prepare(device, ratios)
            alignment = lcm(*ratios)
            misaligned = [boundary for boundary in packed.document_boundaries() if boundary % alignment]
            if misaligned:
                raise ValueError(
                    "graph-mode DSA compression requires every packed document boundary "
                    f"to be a multiple of {alignment} (lcm of compress ratios {list(ratios)}); "
                    f"misaligned boundaries: {misaligned[:8]}. Pad documents at the packing "
                    "boundary (per-doc alignment) so eager and the graph share one shape policy"
                )
        else:
            packed = packed.prepare(device, ())
        object.__setattr__(
            packed, "_packing_alignment", lcm(*ratios) if ratios else 0
        )  # pylint: disable=protected-access
        packed.prepare_packing_geometry(device)
        inputs["packed_seq_params"] = packed
        return inputs

    def runtime_input_fields(self) -> tuple[str, ...]:
        """Declare the stable set of model inputs produced by this instance."""
        fields = ["packed_seq_params"]
        if self.include_position_ids:
            fields.append("position_ids")
        if self.include_image_sequence_start:
            fields.append("image_sequence_start")
        return tuple(fields)

    def build_runtime_inputs(
            self,
            *,
            batch: Mapping[str, Any],
            context: RuntimeInputContext,
    ) -> Mapping[str, Any]:
        """Build packed attention metadata and Omni image coordinates."""
        local_input_shape = context.local_input_shape
        if len(local_input_shape) != 2 or int(local_input_shape[0]) != 1:
            raise ValueError(
                "DeepSeek-V4.1 compact packing requires local input shape [1, sequence], "
                f"got {tuple(local_input_shape)}"
            )
        if "cu_seq_lens" not in batch:
            raise ValueError("DeepSeek-V4.1 batching requires packed cu_seq_lens")

        local_sequence_length = int(local_input_shape[1])
        cp_rank = context.parallel_ranks.get("cp", 0)
        cp_size = context.parallel_sizes.get("cp", 1)
        runtime_inputs = {
            "packed_seq_params": SharedCompressedPackedSequence(
                cu_seq_lens=batch["cu_seq_lens"],
                local_query_start=cp_rank * local_sequence_length,
                local_query_length=local_sequence_length,
                global_sequence_length=cp_size * local_sequence_length,
            )
        }
        cp_start = runtime_inputs["packed_seq_params"].local_query_start
        if self.include_position_ids:
            input_ids = batch["input_ids"]
            position_ids = torch.arange(
                cp_start,
                cp_start + local_sequence_length,
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0)
            runtime_inputs["position_ids"] = position_ids
        if self.include_image_sequence_start:
            tp_size = context.parallel_sizes.get("tp", 1)
            if context.options.get("sequence_parallel", False):
                if local_sequence_length % tp_size:
                    raise ValueError("CP-local image sequence must divide evenly across sequence-parallel TP ranks")
                tp_rank = context.parallel_ranks.get("tp", 0)
                runtime_inputs["image_sequence_start"] = (
                    cp_start + tp_rank * (local_sequence_length // tp_size)
                )
            else:
                runtime_inputs["image_sequence_start"] = cp_start
        return runtime_inputs


logger = get_dataset_logger("hyper_parallel.data.deepseek_v41")


def prepare_deepseek_v41_media_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Convert packing metadata into the V4.1 model batch contract.

    Args:
        batch: Packed token fields and optional image metadata.
    """
    encoded_batch = dict(batch)
    if "pixel_values" not in encoded_batch:
        return encoded_batch

    required_fields = (
        "image_patch_lengths",
        "image_vit_grid_hw",
        "image_llm_grid_hw",
        "image_token_starts",
    )
    missing_fields = []
    for field in required_fields:
        if field not in encoded_batch:
            missing_fields.append(field)
    if missing_fields:
        raise ValueError(
            "DeepSeek-V4.1 image batch is missing fields: "
            + ", ".join(missing_fields)
        )

    patch_lengths = encoded_batch.pop("image_patch_lengths")
    # [images] belongs to one packed row; [batch, images] retains its row ownership.
    if patch_lengths.ndim == 1:
        image_batch_indices = patch_lengths.new_zeros(patch_lengths.numel())
    elif patch_lengths.ndim == 2:
        batch_size, images_per_batch = patch_lengths.shape
        image_batch_indices = torch.arange(
            batch_size, dtype=patch_lengths.dtype, device=patch_lengths.device,
        ).repeat_interleave(images_per_batch)
    else:
        raise ValueError("image_patch_lengths must have shape [images] or [batch, images]")

    flat_patch_lengths = patch_lengths.reshape(-1)
    zero = flat_patch_lengths.new_zeros(1)
    encoded_batch["image_patch_offsets"] = torch.cat(
        (zero, flat_patch_lengths.cumsum(dim=0))
    )
    encoded_batch["image_batch_indices"] = image_batch_indices
    encoded_batch["image_vit_grid_hw"] = encoded_batch["image_vit_grid_hw"].reshape(-1, 2)
    encoded_batch["image_llm_grid_hw"] = encoded_batch["image_llm_grid_hw"].reshape(-1, 2)
    encoded_batch["image_token_starts"] = encoded_batch["image_token_starts"].reshape(-1)
    if logger.isEnabledFor(logging.DEBUG) and encoded_batch["input_ids"].device.type == "cpu":
        starts = encoded_batch["image_token_starts"].tolist()
        grids = encoded_batch["image_llm_grid_hw"].tolist()
        # Native image spans contain start/end tokens and one newline per grid row.
        ends = [start + height * (width + 1) + 2 for start, (height, width) in zip(starts, grids)]
        rows = image_batch_indices.tolist()
        sequence_length = encoded_batch["input_ids"].shape[-1]
        offsets = encoded_batch["image_patch_offsets"].tolist()
        valid_spans = (len(starts) == len(grids) == len(rows)
                       and all(0 <= start < end <= sequence_length for start, end in zip(starts, ends))
                       and all(0 <= row < encoded_batch["input_ids"].shape[0] for row in rows))
        valid_patches = (all(length > 0 for length in flat_patch_lengths.tolist())
                         and offsets[-1] == encoded_batch["pixel_values"].shape[0])
        logger.debug(
            "DeepSeek image batch: rows=%s token_starts=%s token_ends=%s llm_grids=%s "
            "patch_offsets=%s pixel_shape=%s valid_spans=%s valid_patches=%s",
            rows, starts, ends, grids, offsets, tuple(encoded_batch["pixel_values"].shape),
            valid_spans, valid_patches, enabled=True,
        )
        if not valid_spans or not valid_patches:
            logger.warning("DeepSeek image batch has inconsistent token spans or patch metadata")
    return encoded_batch


@dataclass
class DeepseekV41OmniCollator:
    """Attach single-conversation document boundaries before CP transport."""

    collate: Callable

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Keep media collation unchanged and describe real plus alignment-tail intervals."""
        batch = dict(self.collate(samples))
        if batch.get("cu_seq_lens") is None:
            if len(samples) != 1 or batch["input_ids"].shape[0] != 1:
                raise ValueError("V4.1 unpacked multimodal collation requires one conversation")
            length, physical = samples[0]["input_ids"].numel(), batch["input_ids"].shape[1]
            boundaries = [0, length] + ([physical] if physical > length else [])
            batch["cu_seq_lens"] = torch.tensor(boundaries, dtype=torch.int32)
        return prepare_deepseek_v41_media_batch(batch)


def build_deepseek_v41_omni_collate_fn(
    mesh_context: Any = None, pad_to_length: int | None = None,
) -> DeepseekV41OmniCollator:
    """Combine standard media collation with V4.1's document metadata contract."""
    return DeepseekV41OmniCollator(build_omni_collate_fn(mesh_context=mesh_context, pad_to_length=pad_to_length))


__all__ = ["DeepseekV41DatasetAdapter", "build_deepseek_v41_omni_collate_fn"]
