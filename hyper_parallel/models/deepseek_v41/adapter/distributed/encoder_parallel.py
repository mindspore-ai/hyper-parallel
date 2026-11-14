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
"""DeepSeek-V4.1 image metadata, ViT bindings, and attention replacement."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from hyper_parallel.core.activation_memory.wrapper import CheckpointWrapper
from hyper_parallel.distributed.context_parallel import ulysses_head_to_seq, ulysses_seq_to_head
from hyper_parallel.distributed.encoder_parallel.attention import (
    current_encoder_attention_context,
    encoder_attention_checkpoint_contexts,
)
from hyper_parallel.distributed.encoder_parallel.contracts import (
    EncoderParallelAdapterSpec,
    EncoderLocalInputs,
    EncoderSegment,
    EncoderWorkItem,
)
from hyper_parallel.models.deepseek_v41.vision import (
    DeepseekV41VisionAttention,
    _apply_vision_rotary,
    _vision_attention,
    _vision_cos_sin,
)
from hyper_parallel.models.replacement import module_replacement


_IMAGE_FIELDS = (
    "pixel_values", "image_patch_offsets", "image_vit_grid_hw", "image_llm_grid_hw",
    "image_batch_indices", "image_token_starts",
)


def _bundle_shapes_valid(pixels: torch.Tensor, offsets: torch.Tensor,
                         vit_hw: torch.Tensor, llm_hw: torch.Tensor,
                         indices: torch.Tensor, starts: torch.Tensor, count: int) -> bool:
    """Check tensor ranks and image counts before reading individual values."""
    return (pixels.ndim == 4 and count > 0 and vit_hw.shape == (count, 2)
            and llm_hw.shape == (count, 2) and offsets.numel() == count + 1
            and indices.numel() == count and starts.numel() == count)


def _offsets_valid(values: Sequence[int], patch_count: int) -> bool:
    """Check image patch offsets cover a single monotonic payload."""
    return (values[0] == 0 and values[-1] == patch_count
            and all(left <= right for left, right in zip(values, values[1:])))


def _invalid_image_items(vit_hw: torch.Tensor, llm_hw: torch.Tensor,
                         indices: torch.Tensor, starts: torch.Tensor,
                         offsets: Sequence[int]) -> list[int]:
    """Locate image records whose grid or decoder metadata is invalid."""
    invalid = []
    for index in range(vit_hw.shape[0]):
        vit_h, vit_w = (int(value) for value in vit_hw[index])
        llm_h, llm_w = (int(value) for value in llm_hw[index])
        if (min(vit_h, vit_w, llm_h, llm_w) <= 0
                or offsets[index + 1] - offsets[index] != vit_h * vit_w
                or int(indices[index]) != 0 or int(starts[index]) < 0):
            invalid.append(index)
    return invalid


def _validate_image_bundle(batch: Mapping[str, Any]) -> int:
    """Validate image metadata tensors and return the image count."""
    non_tensors = [name for name in _IMAGE_FIELDS if not torch.is_tensor(batch[name])]
    if non_tensors:
        raise ValueError(f"image metadata must be tensors; non_tensors={non_tensors}")
    pixels = batch["pixel_values"]
    offsets = batch["image_patch_offsets"].reshape(-1)
    vit_hw = batch["image_vit_grid_hw"]
    llm_hw = batch["image_llm_grid_hw"]
    indices = batch["image_batch_indices"].reshape(-1)
    starts = batch["image_token_starts"].reshape(-1)
    count = int(vit_hw.shape[0]) if vit_hw.ndim == 2 else -1
    shapes_valid = _bundle_shapes_valid(pixels, offsets, vit_hw, llm_hw, indices, starts, count)
    offset_values = [int(value) for value in offsets] if shapes_valid else []
    offsets_valid = shapes_valid and _offsets_valid(offset_values, pixels.shape[0])
    invalid = _invalid_image_items(vit_hw, llm_hw, indices, starts, offset_values) if offsets_valid else []
    if not shapes_valid or not offsets_valid or invalid:
        raise ValueError(
            "invalid DeepSeek image metadata bundle; "
            f"valid_shapes={shapes_valid}; valid_offsets={offsets_valid}; invalid_items={invalid}"
        )
    return count


def build_deepseek_v41_work_items(batch: Mapping[str, Any]) -> tuple[EncoderWorkItem, ...]:
    """Convert the native image metadata bundle into model-neutral work.

    Args:
        batch: DeepSeek Omni batch with native image fields.
    """
    input_ids = batch.get("input_ids")
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            "DeepSeek-V4.1 encoder parallel v1 requires micro_batch_size=1; "
            f"input_shape={getattr(input_ids, 'shape', None)}"
        )
    present = [batch.get(name) is not None for name in _IMAGE_FIELDS]
    if not any(present):
        return ()
    if not all(present):
        missing = [name for name in _IMAGE_FIELDS if batch.get(name) is None]
        raise ValueError(f"incomplete DeepSeek image metadata bundle: missing={missing}")
    if batch.get("token_types") is None:
        raise ValueError("DeepSeek image inputs require token_types")
    count = _validate_image_bundle(batch)
    items = []
    for index in range(count):
        vit_h, vit_w = (int(value) for value in batch["image_vit_grid_hw"][index])
        llm_h, llm_w = (int(value) for value in batch["image_llm_grid_hw"][index])
        items.append(EncoderWorkItem(
            item_id=index, batch_index=int(batch["image_batch_indices"][index]),
            sequence_length=vit_h * vit_w, output_rows=llm_h * llm_w,
            decoder_token_start=int(batch["image_token_starts"][index]),
            decoder_token_length=llm_h * (llm_w + 1) + 2,
            payload_start=int(batch["image_patch_offsets"][index]),
            payload_stop=int(batch["image_patch_offsets"][index + 1]),
            metadata={"vit_hw": (vit_h, vit_w), "llm_hw": (llm_h, llm_w)},
        ))
    return tuple(items)


class DeepseekV41VisionBridge:
    """Bind native DeepSeek image operators to one constructed model."""

    input_fields = _IMAGE_FIELDS
    payload_field = "pixel_values"

    def __init__(self, model: nn.Module) -> None:
        """Keep direct references to the existing ViT and aligner modules."""
        backbone = getattr(model, "model", model)
        if backbone.vision is None or backbone.aligner is None:
            raise ValueError("DeepSeek vision and aligner must be enabled")
        self.vision = backbone.vision
        self.aligner = backbone.aligner

    @property
    def empty_payload_shape(self) -> tuple[int, int, int]:
        """Describe a dummy patch without allocating model-device storage."""
        return (3, self.vision.patch_size, self.vision.patch_size)

    @property
    def head_counts(self) -> dict[str, int]:
        """Declare the native Q/K/V head count for Ulysses validation."""
        return {name: self.vision.num_attention_heads for name in ("query", "key", "value")}

    def validate_training(self, options: Mapping[str, Any]) -> None:
        """Reject model-specific execution modes before deriving process groups.

        Args:
            options: Trainer microbatch and loss-layout settings.
        """
        if options.get("micro_batch_size") != 1:
            raise ValueError(
                "encoder adapter does not support the requested micro batch; "
                f"micro_batch_size={options.get('micro_batch_size')}; supported=(1,); "
                "config_path=training.micro_batch_size"
            )
        if options.get("loss_parallel"):
            raise NotImplementedError(
                "unsupported encoder loss layout; config_path=accelerator.loss_parallel"
            )
        supported = {id(block) for block in self.vision.blocks
                     if isinstance(block, CheckpointWrapper)
                     and block.checkpoint_kwargs.get("context_fn") is encoder_attention_checkpoint_contexts
                     and block.checkpoint_kwargs.get("early_stop") is False}
        forbidden = [name for name, module in self.vision.named_modules()
                     if isinstance(module, CheckpointWrapper) and id(module) not in supported]
        forbidden.extend(name for name, module in self.aligner.named_modules()
                         if isinstance(module, CheckpointWrapper))
        if forbidden:
            raise RuntimeError(f"encoder requires context-preserving full vision-block checkpoints: {forbidden}")

    def parse_images(self, batch: Mapping[str, Any]) -> tuple[EncoderWorkItem, ...]:
        """Validate native image metadata and expose complete image jobs.

        Args:
            batch: Native DeepSeek image fields, normally on the CPU before transport.

        Returns:
            Validated image jobs in their original order.
        """
        items = build_deepseek_v41_work_items(batch)
        if items and tuple(batch["pixel_values"].shape[1:]) != self.empty_payload_shape:
            raise ValueError(f"pixel_values patches must have shape {self.empty_payload_shape}")
        return items

    def pack_payload(self, batch: Mapping[str, Any], segments: Sequence[EncoderSegment],
                     *, device: torch.device | str = "cpu") -> torch.Tensor:
        """Copy only the selected patch intervals and zero padding into one buffer.

        Args:
            batch: Native image bundle retained on its source device.
            segments: Ordered image intervals assigned to the destination rank.
            device: Device of the packed output; CPU for training input transport.

        Returns:
            Packed patches with explicit zero padding and dummy rows.
        """
        pixels = batch.get("pixel_values")
        dtype = pixels.dtype if pixels is not None else torch.float32
        count = sum(segment.local_padded_length for segment in segments)
        packed = torch.zeros((count, *self.empty_payload_shape), dtype=dtype, device=device)
        for segment in segments:
            if segment.is_dummy or not segment.local_valid_length:
                continue
            if pixels is None:
                raise ValueError("real image segment requires pixel_values")
            start = segment.item.payload_start + segment.local_start
            length = segment.local_valid_length
            packed[segment.packed_offset:segment.packed_offset + length].copy_(pixels[start:start + length])
        return packed

    def _pack_local_inputs(self, batch: Mapping[str, Any] | EncoderLocalInputs, segments: Sequence[EncoderSegment]
                           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack image shards and identity RoPE for padded positions."""
        vision = self.vision
        if isinstance(batch, EncoderLocalInputs):
            patches = batch.patches
        else:
            pixels = batch.get("pixel_values")
            device = pixels.device if pixels is not None else vision.patch_embed.proj.weight.device
            patches = self.pack_payload(batch, segments, device=device)
        cosine_parts = []
        sine_parts = []
        for segment in segments:
            if segment.is_dummy:
                cosine = torch.ones((segment.local_padded_length, 1, vision.rope_dim),
                                    dtype=torch.float32, device=patches.device)
                sine = torch.zeros_like(cosine)
            else:
                item = segment.item
                pad_count = segment.local_padded_length - segment.local_valid_length
                cosine, sine = _vision_cos_sin(
                    *item.metadata["vit_hw"], vision.rope_dim, vision.rope_theta, patches.device,
                    patch_start=segment.local_start, patch_count=segment.local_valid_length,
                )
                if pad_count:
                    cosine = torch.cat((cosine, torch.ones((pad_count, 1, vision.rope_dim),
                                                           device=cosine.device, dtype=cosine.dtype)))
                    sine = torch.cat((sine, torch.zeros((pad_count, 1, vision.rope_dim),
                                                        device=sine.device, dtype=sine.dtype)))
            cosine_parts.append(cosine)
            sine_parts.append(sine)
        return patches, torch.cat(cosine_parts), torch.cat(sine_parts)

    def encode_local(self, batch: Mapping[str, Any] | EncoderLocalInputs,
                     segments: Sequence[EncoderSegment]) -> torch.Tensor:
        """Run native patch embedding, ViT blocks and final norm on packed shards.

        Args:
            batch: Prepared local device patches, or a native bundle for direct callers.
            segments: This Encoder-CP rank's padded image shards.

        Returns:
            Local packed ViT hidden rows.
        """
        patches, cosine, sine = self._pack_local_inputs(batch, segments)
        if patches.shape[0] != sum(segment.local_padded_length for segment in segments):
            raise ValueError("packed patch count does not match segment table")
        hidden = self.vision.patch_embed(patches)
        for block in self.vision.blocks:
            hidden = block(hidden, cosine, sine)
        return self.vision.norm(hidden)

    def project(self, hidden: torch.Tensor, item: EncoderWorkItem | None) -> torch.Tensor:
        """Run real aligner on ECP rank zero or a graph-connected dummy elsewhere.

        Args:
            hidden: Complete ViT rows for a real image, or local rows for a dummy.
            item: Real image geometry, or None for a zero-valued dummy call.

        Returns:
            Native aligner output with an autograd path to the ViT.
        """
        if item is None:
            ratio = self.aligner.downsample_ratio
            hidden = hidden[:1].expand(ratio * ratio, -1) * 0.0
            height, width = ratio, ratio
        else:
            height, width = item.metadata["vit_hw"]
        return self.aligner(hidden, height, width)


class DeepseekV41EncoderParallelAttention(DeepseekV41VisionAttention):
    """Structure-preserving attention with an active segmented Ulysses path."""

    @classmethod
    def from_source(cls, source: DeepseekV41VisionAttention,
                    *, module_fqn: str) -> "DeepseekV41EncoderParallelAttention":
        """Retain source projections, parameter names, dtype, and training state.

        Args:
            source: Native visual attention module to replace.
            module_fqn: Final path used by execution diagnostics.
        """
        result = cls(source.num_attention_heads * source.head_dim, source.num_attention_heads)
        result.wqkv = source.wqkv
        result.wo = source.wo
        result.module_fqn = module_fqn
        result.train(source.training)
        return result

    def forward(self, hidden_states: torch.Tensor, cosine: torch.Tensor,
                sine: torch.Tensor) -> torch.Tensor:
        """Use native full attention outside an active encoder execution.

        Args:
            hidden_states: Local image patch states.
            cosine: Rotary cosine coefficients for local patches.
            sine: Rotary sine coefficients for local patches.
        """
        context = current_encoder_attention_context()
        if context is None:
            return super().forward(hidden_states, cosine, sine)
        context.record(self.module_fqn)
        count = hidden_states.shape[0]
        query, key, value = (tensor.view(count, self.num_attention_heads, self.head_dim)
                             for tensor in self.wqkv(hidden_states).chunk(3, dim=-1))
        query = _apply_vision_rotary(query, cosine, sine)
        key = _apply_vision_rotary(key, cosine, sine)
        context.validate_runtime_head_counts(
            module_fqn=self.module_fqn, query_heads=query.shape[1],
            key_heads=key.shape[1], value_heads=value.shape[1],
        )
        outputs = []
        for segment in context.segments:
            start = segment.packed_offset
            stop = start + segment.local_padded_length
            q = ulysses_seq_to_head(query[start:stop], seq_dim=0, head_dim=1, cp_mesh=context.cp_mesh)
            k = ulysses_seq_to_head(key[start:stop], seq_dim=0, head_dim=1, cp_mesh=context.cp_mesh)
            v = ulysses_seq_to_head(value[start:stop], seq_dim=0, head_dim=1, cp_mesh=context.cp_mesh)
            global_length = segment.local_padded_length * context.cp_mesh.size()
            key_length = global_length if segment.is_dummy else segment.item.sequence_length
            attended = _vision_attention(self, q, k, v, key_length=key_length)
            outputs.append(ulysses_head_to_seq(attended, seq_dim=0, head_dim=1, cp_mesh=context.cp_mesh))
        return self.wo(torch.cat(outputs, dim=0).reshape(count, -1))


@module_replacement
def replace_vision_attention(*, module: DeepseekV41VisionAttention,
                             module_fqn: str, context: Any) -> DeepseekV41EncoderParallelAttention:
    """Install the visual attention replacement independent of runtime mode."""
    del context
    return DeepseekV41EncoderParallelAttention.from_source(module, module_fqn=module_fqn)


def get_deepseek_v41_encoder_parallel_spec() -> EncoderParallelAdapterSpec:
    """Declare model-owned visual boundaries and the runtime bridge."""
    return EncoderParallelAdapterSpec(
        isolated_subtrees=("model.vision", "model.aligner"),
        attention_pattern="model.vision.blocks.*.attn",
        replacement_attention_type=DeepseekV41EncoderParallelAttention,
        bind=DeepseekV41VisionBridge,
    )
