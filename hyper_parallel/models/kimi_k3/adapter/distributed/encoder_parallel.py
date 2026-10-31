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
"""MoonViT sequence shards and differentiable media-to-decoder routing."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

# This model adapter uses the Torch-only encoder runtime.
import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.activation_memory.recompute_state import get_recompute_state
from hyper_parallel.distributed.context_parallel import ulysses_head_to_seq, ulysses_seq_to_head
from hyper_parallel.distributed.encoder_parallel.attention import current_encoder_attention_context
from hyper_parallel.distributed.encoder_parallel.contracts import (
    EncoderLocalInputs, EncoderParallelAdapterSpec, EncoderSegment, EncoderWorkItem,
)
from hyper_parallel.models.kimi_k3.vision import (
    MoonViTEncoderLayer, apply_spatial_rope, spatial_rope_angles, temporal_spatial_merge, validate_grid,
)
from hyper_parallel.models.replacement import module_replacement

_MEDIA_FIELDS = ("pixel_values", "grid_thws", "media_spans", "media_patch_offsets")


def build_kimi_k3_work_items(batch: Mapping[str, Any], *, max_frames: int = 4) -> tuple[EncoderWorkItem, ...]:
    """Translate complete media records into encoder jobs without localizing spans.

    Args:
        batch: CP-local tokens with replicated global media metadata and patches.
        max_frames: Maximum temporal grid supported by the vision position table.

    Returns:
        Jobs in original media order, including their global decoder intervals.
    """
    pixels, grid, spans, offsets = (batch[name] for name in _MEDIA_FIELDS)
    grids = validate_grid(grid, max_frames=max_frames)
    if spans.shape != (len(grids), 3) or spans.dtype not in (torch.int32, torch.int64):
        raise ValueError("media_spans must be integer [media_count,3]")
    expected = grid.prod(-1)
    expected = torch.cat((expected.new_zeros(1), expected.cumsum(0)))
    if offsets.shape != expected.shape or not torch.equal(offsets.to(expected.device), expected):
        raise ValueError("media_patch_offsets must partition the complete grid payload")
    if pixels.ndim != 4 or pixels.shape[0] != int(expected[-1]):
        raise ValueError("pixel_values must contain the complete replicated media payload")
    jobs = []
    previous = (-1, -1)
    for index, ((frames, height, width), (batch_index, start, stop)) in enumerate(zip(grids, spans.tolist())):
        rows = height * width // 4
        if not 0 <= batch_index < batch["input_ids"].shape[0] or start < 0 or stop - start != rows:
            raise ValueError("media span must name exactly the spatially merged feature rows")
        if (batch_index, start) < previous:
            raise ValueError("media spans must be ordered and disjoint within each sample")
        previous = (batch_index, stop)
        jobs.append(EncoderWorkItem(
            item_id=index, batch_index=batch_index, sequence_length=frames * height * width,
            output_rows=rows, decoder_token_start=start, decoder_token_length=rows,
            payload_start=int(expected[index]), payload_stop=int(expected[index + 1]),
            metadata={"grid": (frames, height, width)},
        ))
    return tuple(jobs)


class KimiK3VisionBridge:
    """Bind the complete native tower and projector to public encoder execution."""

    input_fields = _MEDIA_FIELDS
    payload_field = "pixel_values"

    def __init__(self, model: nn.Module) -> None:
        """Retain the original parameter-owning modules and their FSDP wrappers."""
        self.vision = model.vision_tower
        self.projector = model.mm_projector
        self.config = model.config.vision_config

    @property
    def empty_payload_shape(self) -> tuple[int, int, int]:
        """Describe a patch without allocating model-device storage."""
        return (3, self.vision.patch_size, self.vision.patch_size)

    @property
    def head_counts(self) -> dict[str, int]:
        """Declare equal Q/K/V head counts for encoder Ulysses."""
        return dict.fromkeys(("query", "key", "value"), self.config.vt_num_attention_heads)

    def validate_training(self, options: Mapping[str, Any]) -> None:
        """Check supported attention and loss modes before collective group creation.

        Args:
            options: Resolved Trainer execution settings.
        """
        if self.config.attention_backend != "eager":
            raise ValueError("K3 encoder parallel currently requires vision attention_backend=eager")
        if options.get("loss_parallel"):
            raise ValueError("K3 encoder parallel requires loss_parallel=false")

    def parse_images(self, batch: Mapping[str, Any]) -> tuple[EncoderWorkItem, ...]:
        """Read image or sampled-video grids from a complete media bundle.

        Args:
            batch: CP-local token fields and replicated media tensors.

        Returns:
            Global media jobs, including an empty tuple for text-only samples.
        """
        items = build_kimi_k3_work_items(batch, max_frames=self.vision.max_frames)
        if tuple(batch["pixel_values"].shape[1:]) != self.empty_payload_shape:
            raise ValueError(f"pixel_values patches must have shape {self.empty_payload_shape}")
        return items

    def pack_payload(self, batch: Mapping[str, Any], segments: Sequence[EncoderSegment],
                     *, device: torch.device | str = "cpu") -> torch.Tensor:
        """Pack only this owner's patch intervals, including padding and dummy rows.

        Args:
            batch: Complete native media fields retained on the CPU by transport.
            segments: Ordered image/video fragments assigned to one encoder owner.
            device: Output device; CPU for the standard batch transport.

        Returns:
            Packed patches in segment order without a complete device image copy.
        """
        pixels = batch.get("pixel_values")
        dtype = pixels.dtype if pixels is not None else torch.float32
        count = sum(segment.local_padded_length for segment in segments)
        patches = torch.zeros((count, *self.empty_payload_shape), dtype=dtype, device=device)
        for segment in segments:
            if segment.is_dummy or not segment.local_valid_length:
                continue
            if pixels is None:
                raise ValueError("real media segment requires pixel_values")
            start = segment.item.payload_start + segment.local_start
            length = segment.local_valid_length
            patches[segment.packed_offset:segment.packed_offset + length].copy_(pixels[start:start + length])
        return patches

    def _local_inputs(self, batch: Mapping[str, Any] | EncoderLocalInputs, segments: Sequence[EncoderSegment]
                      ) -> tuple[torch.Tensor, tuple[tuple[int, int, int], ...], torch.Tensor]:
        if isinstance(batch, EncoderLocalInputs):
            patches = batch.patches
        else:
            pixels = batch.get("pixel_values")
            device = pixels.device if pixels is not None else self.vision.patch_embed.proj.weight.device
            patches = self.pack_payload(batch, segments, device=device)
        positions, grids = [], []
        offset = 0
        for segment in segments:
            grid = (1, 2, 2) if segment.is_dummy else segment.item.metadata["grid"]
            grids.append(grid)
            indices = torch.arange(segment.local_start, segment.local_start + segment.local_padded_length,
                                   device=patches.device)
            indices = torch.where(indices < math.prod(grid), indices, 0) + offset
            positions.append(indices)
            offset += math.prod(grid)
        return patches, tuple(grids), torch.cat(positions)

    def encode_local(self, batch: Mapping[str, Any] | EncoderLocalInputs,
                     segments: Sequence[EncoderSegment]) -> torch.Tensor:
        """Run every vision FSDP unit once on owned patches, including empty owners.

        Args:
            batch: Prepared local device patches, or a native bundle for direct callers.
            segments: Padded media fragments assigned to this encoder rank.

        Returns:
            Local encoded patch rows in segment order.
        """
        patches, grids, indices = self._local_inputs(batch, segments)
        hidden = self.vision.patch_embed(patches, grids, indices)
        angles = spatial_rope_angles(grids, self.vision.encoder.head_dim, hidden.device).index_select(0, indices)
        lengths = tuple(segment.local_padded_length for segment in segments)
        for block in self.vision.encoder.blocks:
            hidden = block(hidden, lengths, angles)
        return self.vision.encoder.final_layernorm(hidden)

    def project(self, hidden: torch.Tensor, item: EncoderWorkItem | None) -> torch.Tensor:
        """Pool complete frames/spatial groups and invoke the original projector.

        Args:
            hidden: Gathered complete media patch rows, or dummy local rows.
            item: Geometry of a real media record, or None for synchronization work.

        Returns:
            Projected decoder feature rows, with a gradient edge on every rank.
        """
        return self.project_many((hidden,), (item,))[0]

    def project_many(self, hidden_states: Sequence[torch.Tensor], items: Sequence[EncoderWorkItem | None]
                     ) -> tuple[torch.Tensor, ...]:
        """Project every real/dummy media slot in a single FSDP module invocation.

        Args:
            hidden_states: Complete patch rows for each local projection slot.
            items: Real media geometry or None for synchronized dummy work.

        Returns:
            Projected tensors in slot order, including connected dummy outputs.
        """
        pooled = []
        for hidden, item in zip(hidden_states, items):
            if item is None:
                hidden = hidden[:1].expand(4, -1) * 0.0
                grid = (1, 2, 2)
            else:
                grid = item.metadata["grid"]
            pooled.extend(temporal_spatial_merge(hidden, (grid,)))
        return self.projector(pooled)


class KimiK3EncoderParallelLayer(MoonViTEncoderLayer):
    """Retain a native MoonViT block while distributing only its attention rows."""

    @classmethod
    def from_source(cls, source: MoonViTEncoderLayer, module_fqn: str) -> "KimiK3EncoderParallelLayer":
        """Preserve all existing modules, parameters and checkpoint names.

        Args:
            source: Executable native vision block.
            module_fqn: Exact execution-audit name.

        Returns:
            A structure-preserving encoder attention replacement.
        """
        result = copy.copy(source)
        result.__class__ = cls
        result.module_fqn = module_fqn
        return result

    def forward(self, hidden_states: torch.Tensor, lengths: tuple[int, ...], angles: torch.Tensor) -> torch.Tensor:
        """Exchange token/head shards for each independent image or video segment.

        Args:
            hidden_states: Local padded patch states.
            lengths: Native media lengths outside an active encoder plan.
            angles: Global-position RoPE angles for the local rows.

        Returns:
            Updated local hidden rows with native block mathematics.
        """
        context = current_encoder_attention_context()
        recompute_state = get_recompute_state()
        if recompute_state is not None:
            # The dynamic encoder scope ends after forward. Preserve its plan
            # per checkpoint invocation so outstanding graphs replay independently.
            context = recompute_state.get_resource(
                ("kimi_k3.encoder_attention", self.module_fqn), lambda: context,
            )
        if context is None:
            return super().forward(hidden_states, lengths, angles)
        if recompute_state is None or not recompute_state.is_recomputing:
            context.record(self.module_fqn)
        projected = self.wqkv(self.norm0(hidden_states))
        query, key, value = projected.reshape(-1, 3, self.num_heads, self.head_dim).unbind(1)
        query, key = (apply_spatial_rope(part, angles, self.rope_backend) for part in (query, key))
        context.validate_runtime_head_counts(module_fqn=self.module_fqn, query_heads=self.num_heads,
                                             key_heads=self.num_heads, value_heads=self.num_heads)
        outputs = []
        for segment in context.segments:
            start, stop = segment.packed_offset, segment.packed_offset + segment.local_padded_length
            q, k, v = (ulysses_seq_to_head(part[start:stop], seq_dim=0, head_dim=1, cp_mesh=context.cp_mesh)
                       for part in (query, key, value))
            scores = q.transpose(0, 1) @ k.transpose(0, 1).transpose(-1, -2) / math.sqrt(self.head_dim)
            if not segment.is_dummy:
                valid = torch.arange(k.shape[0], device=k.device) < segment.item.sequence_length
                scores = scores.masked_fill(~valid[None, None, :], float("-inf"))
            probabilities = scores.softmax(-1, dtype=torch.float32).to(query.dtype)
            output = (probabilities @ v.transpose(0, 1)).transpose(0, 1)
            outputs.append(ulysses_head_to_seq(output, seq_dim=0, head_dim=1, cp_mesh=context.cp_mesh))
        output = torch.cat(outputs).reshape(hidden_states.shape[0], -1)
        hidden_states = hidden_states + self.wo(output)
        return hidden_states + self.mlp(self.norm1(hidden_states))


@module_replacement
def replace_vision_encoder_layer(*, module: MoonViTEncoderLayer,
                                  module_fqn: str, context: Any) -> KimiK3EncoderParallelLayer:
    """Select distributed MoonViT attention through the normal replacement plan.

    Args:
        module: Native encoder block selected by the recipe.
        module_fqn: Final source path retained for execution auditing.
        context: Generic replacement context, unused by the native block semantics.

    Returns:
        Block preserving every source parameter object.
    """
    del context
    return KimiK3EncoderParallelLayer.from_source(module, module_fqn)


def get_kimi_k3_encoder_parallel_spec() -> EncoderParallelAdapterSpec:
    """Declare K3-owned media parsing over the shared encoder runtime.

    Returns:
        Vision/projector isolation, exact block pattern and native bridge binding.
    """
    return EncoderParallelAdapterSpec(
        isolated_subtrees=("vision_tower", "mm_projector"),
        attention_pattern="vision_tower.encoder.blocks.*",
        replacement_attention_type=KimiK3EncoderParallelLayer,
        bind=KimiK3VisionBridge,
        source_attention_type=MoonViTEncoderLayer,
    )
