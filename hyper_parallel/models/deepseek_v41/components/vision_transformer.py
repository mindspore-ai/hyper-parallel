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
"""Training-capable DeepSeek-V4.1 native vision tower and 3x3 aligner."""

from __future__ import annotations

from copy import copy
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional

from hyper_parallel.components.modules.common.rms_norm import NativeRMSNormModule
from hyper_parallel.components.modules.common.spatial_merge_projector import SpatialMergeProjectorModule
from hyper_parallel.components.modules.common.swiglu_mlp import W1W2SwiGLUMLPModule
from hyper_parallel.components.modules.module_family import ModuleFamily, build_family, register_module_family
from hyper_parallel.core.activation_memory.wrapper import CheckpointWrapper
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.distributed.context_parallel import ulysses_head_to_seq, ulysses_seq_to_head
from hyper_parallel.distributed.encoder_parallel.attention import (
    current_encoder_attention_context,
    encoder_attention_checkpoint_contexts,
)
from hyper_parallel.distributed.encoder_parallel.contracts import (
    EncoderLocalInputs,
    EncoderParallelAdapterSpec,
    EncoderSegment,
    EncoderWorkItem,
)
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.replacement import module_replacement


def encode_image_features(vision: nn.Module, aligner: nn.Module, pixel_values: torch.Tensor,
                          image_patch_offsets: torch.Tensor, image_vit_grid_hw: torch.Tensor,
                          image_llm_grid_hw: torch.Tensor, image_batch_indices: torch.Tensor,
                          image_token_starts: torch.Tensor) -> list[tuple[int, int, int, int, torch.Tensor]]:
    """Encode each image independently and retain its global decoder placement.

    Args:
        vision: Native ViT.
        aligner: Projector into decoder hidden width.
        pixel_values: Concatenated image patches.
        image_patch_offsets: Patch boundaries for each image.
        image_vit_grid_hw: ViT grid heights and widths.
        image_llm_grid_hw: Decoder grid heights and widths.
        image_batch_indices: Sample index of each image.
        image_token_starts: First decoder token of each image span.
    """
    if vision is None or aligner is None:
        raise ValueError("received image inputs but V4.1 vision is disabled")
    image_count = image_vit_grid_hw.shape[0]
    required_shapes = {
        "image_patch_offsets": image_patch_offsets.numel() == image_count + 1,
        "image_llm_grid_hw": image_llm_grid_hw.shape == image_vit_grid_hw.shape,
        "image_batch_indices": image_batch_indices.numel() == image_count,
        "image_token_starts": image_token_starts.numel() == image_count,
    }
    invalid = [name for name, valid in required_shapes.items() if not valid]
    if invalid:
        raise ValueError(f"inconsistent V4.1 image metadata: {', '.join(invalid)}")
    if pixel_values.ndim != 4:
        raise ValueError("pixel_values must have shape [total_patches, 3, patch, patch]")
    image_records = []
    for image_index in range(image_count):
        patch_start = int(image_patch_offsets[image_index])
        patch_end = int(image_patch_offsets[image_index + 1])
        vit_height, vit_width = (int(value) for value in image_vit_grid_hw[image_index])
        llm_height, llm_width = (int(value) for value in image_llm_grid_hw[image_index])
        batch_index = int(image_batch_indices[image_index])
        token_start = int(image_token_starts[image_index])
        vision_features = vision(pixel_values[patch_start:patch_end], vit_height, vit_width)
        image_features = aligner(vision_features, vit_height, vit_width)
        if image_features.shape[0] != llm_height * llm_width:
            raise ValueError(
                f"aligner/image-token count mismatch for image {image_index}: "
                f"{image_features.shape[0]} versus {llm_height * llm_width}"
            )
        image_records.append((batch_index, token_start, llm_height, llm_width, image_features))
    return image_records


def _vision_cos_sin(
        grid_height: int,
        grid_width: int,
        rope_dim: int,
        rope_theta: float,
        device: torch.device,
        *,
        patch_start: int = 0,
        patch_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build DeepSeek's flattened 2D RoPE coefficients for one image grid."""
    inverse_frequency = 1.0 / (
        rope_theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim)
    )
    count = grid_height * grid_width if patch_count is None else patch_count
    positions = torch.arange(patch_start, patch_start + count, device=device)
    frequencies = torch.stack((positions // grid_width, positions % grid_width), dim=-1).unsqueeze(-1).float()
    frequencies = (frequencies * inverse_frequency).flatten(1)
    return frequencies.cos().unsqueeze(1), frequencies.sin().unsqueeze(1)


def _apply_vision_rotary(
        tensor: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
) -> torch.Tensor:
    """Apply the released split-half 2D RoPE convention to vision Q or K."""
    data_type = tensor.dtype
    first_half, second_half = tensor.float().chunk(2, dim=-1)
    return torch.cat((first_half * cosine - second_half * sine, second_half * cosine + first_half * sine), dim=-1).to(
        data_type
    )


def _npu_vision_attention(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
) -> torch.Tensor:
    """Run bidirectional fused attention on canonical BNSD tensors."""
    # The NPU extension is optional for CPU/GPU vision and checkpoint tests.
    from hyper_parallel.components.functional.npu_fusion_attention import (  # pylint: disable=C0415
        npu_fusion_attention_forward,
    )

    output, _ = npu_fusion_attention_forward(
        module, query, key, value, attention_mask=None,
        dropout=0.0, scaling=query.shape[-1] ** -0.5,
        is_causal=False, sparse_mode=0,
        pre_tokens=2147483647, next_tokens=2147483647,
    )
    return output


def _vision_attention(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_length: int | None = None,
) -> torch.Tensor:
    """Attend one image without materializing its padding mask.

    Args:
        module: Owning vision attention module.
        query: Query rows in sequence/head/dimension order, including padding.
        key: Key rows in the same order.
        value: Value rows in the same order.
        key_length: Valid key/value prefix length, or None for a complete image.

    Returns:
        Attended rows in sequence/head/dimension order.
    """
    if key_length is not None:
        key = key[:key_length]
        value = value[:key_length]
    heads, head_dim = query.shape[1:]
    # Canonical singleton strides are required by native attention entry points.
    query = query.transpose(0, 1).contiguous().view(1, heads, query.shape[0], head_dim)
    key = key.transpose(0, 1).contiguous().view(1, heads, key.shape[0], head_dim)
    value = value.transpose(0, 1).contiguous().view(1, heads, value.shape[0], head_dim)
    if query.device.type == "npu":
        return _npu_vision_attention(module, query, key, value).squeeze(0)
    output = functional.scaled_dot_product_attention(  # pylint: disable=not-callable
        query, key, value, is_causal=False,
    )
    return output.squeeze(0).transpose(0, 1)


class DeepseekV41VisionRMSNorm(NativeRMSNormModule):
    """Compatibility constructor for the ordinary FP32-weight vision norm."""

    def __init__(self, hidden_size: int, eps: float = 1.0e-6) -> None:
        """Declare the vision scale and final cast policy explicitly."""
        super().__init__(config=SimpleNamespace(hidden_size=hidden_size, eps=eps),
                         cast_order="after_weight", fp32_weight=True)


class DeepseekV41VisionPatchEmbed(nn.Module):
    """Project flattened RGB patches into the vision hidden space."""

    def __init__(self, patch_size: int, hidden_size: int) -> None:
        """Create the released linear patch projection."""
        super().__init__()
        self.proj = nn.Linear(3 * patch_size ** 2, hidden_size)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Embed ``[patches, 3, patch, patch]`` RGB patches."""
        if patches.ndim != 4 or patches.shape[1] != 3:
            raise ValueError(f"pixel patches must have shape [N, 3, P, P], got {tuple(patches.shape)}")
        return self.proj(patches.flatten(1).to(self.proj.weight.dtype))


class DeepseekV41VisionAttention(nn.Module):
    """Bidirectional full-image attention with DeepSeek 2D RoPE."""

    def __init__(self, hidden_size: int, num_attention_heads: int) -> None:
        """Create fused QKV and output projections.

        Args:
            hidden_size: Vision hidden dimension.
            num_attention_heads: Number of equal-size attention heads.
        """
        super().__init__()
        if hidden_size % num_attention_heads:
            raise ValueError("vision hidden_size must divide evenly into num_attention_heads")
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        # Keep the released checkpoint names (wqkv / wo) stable for future
        # full-checkpoint loading.
        self.wqkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.wo = nn.Linear(hidden_size, hidden_size)

    def forward(
            self,
            hidden_states: torch.Tensor,
            cosine: torch.Tensor,
            sine: torch.Tensor,
    ) -> torch.Tensor:
        """Run full bidirectional attention for one image's patch sequence."""
        token_count = hidden_states.shape[0]
        query, key, value = (
            value.view(token_count, self.num_attention_heads, self.head_dim)
            for value in self.wqkv(hidden_states).chunk(3, dim=-1)
        )
        query = _apply_vision_rotary(query, cosine, sine)
        key = _apply_vision_rotary(key, cosine, sine)
        attention_output = _vision_attention(self, query, key, value)
        return self.wo(attention_output.reshape(token_count, -1))


class DeepseekV41VisionBlock(nn.Module):
    """One pre-normalized full-attention vision Transformer block."""

    def __init__(self, hidden_size: int, num_attention_heads: int, intermediate_size: int) -> None:
        """Create a released V4.1 vision block."""
        super().__init__()
        self.norm1 = NativeRMSNormModule(config=SimpleNamespace(hidden_size=hidden_size, eps=1e-6),
                                   cast_order="after_weight", fp32_weight=True)
        self.attn = DeepseekV41VisionAttention(hidden_size, num_attention_heads)
        self.norm2 = NativeRMSNormModule(config=SimpleNamespace(hidden_size=hidden_size, eps=1e-6),
                                   cast_order="after_weight", fp32_weight=True)
        self.mlp = W1W2SwiGLUMLPModule(config=SimpleNamespace(
            hidden_size=hidden_size, intermediate_size=intermediate_size, bias=False))

    def forward(
            self,
            hidden_states: torch.Tensor,
            cosine: torch.Tensor,
            sine: torch.Tensor,
    ) -> torch.Tensor:
        """Apply attention then MLP residual updates."""
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cosine, sine)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class DeepseekV41VisionTower(nn.Module):
    """DeepSeek V4.1 ViT over one independently processed image grid."""

    def __init__(self, config: Any = None, *, module: nn.Module | None = None,
                 module_fqn: str = "", context: Any = None) -> None:
        """Construct the complete vision family or retain a supported source's state."""
        super().__init__()
        del context
        if (config is None) == (module is None):
            raise ValueError("VisionTransformer requires exactly one of config or module")
        self.config = config if module is None else getattr(module, "config", None)
        prefix = module_fqn or "model.vision"
        if module is not None:
            for name in ("hidden_size", "patch_size", "num_attention_heads", "rope_theta", "rope_dim",
                         "gradient_checkpointing"):
                setattr(self, name, getattr(module, name))
            self.patch_embed, self.norm = module.patch_embed, module.norm
            blocks = []
            for index, source in enumerate(module.blocks):
                block = copy(source)
                block._modules = source._modules.copy()
                block.attn = DeepseekV41EncoderParallelAttention.from_source(
                    source.attn, module_fqn=f"{prefix}.blocks.{index}.attn",
                )
                blocks.append(block)
            self.blocks = nn.ModuleList(blocks)
            self.training = module.training
            return
        self.gradient_checkpointing = False
        self.hidden_size = int(config.v41_vision_hidden_size)
        self.patch_size = int(config.v41_vision_patch_size)
        self.num_attention_heads = int(config.v41_vision_num_attention_heads)
        self.rope_theta = float(config.v41_vision_rope_theta)
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("V4.1 vision hidden_size must be divisible by its attention head count")
        self.rope_dim = self.hidden_size // self.num_attention_heads // 2
        if self.rope_dim <= 0 or self.rope_dim % 2:
            raise ValueError("V4.1 vision RoPE dimension must be a positive even number")
        self.patch_embed = DeepseekV41VisionPatchEmbed(self.patch_size, self.hidden_size)
        self.blocks = nn.ModuleList([
            DeepseekV41VisionBlock(
                self.hidden_size,
                self.num_attention_heads,
                int(config.v41_vision_intermediate_size),
            )
            for _ in range(int(config.v41_vision_num_hidden_layers))
        ])
        self.norm = NativeRMSNormModule(config=SimpleNamespace(hidden_size=self.hidden_size, eps=1e-6),
                                  cast_order="after_weight", fp32_weight=True)
        for index, block in enumerate(self.blocks):
            block.attn = DeepseekV41EncoderParallelAttention.from_source(
                block.attn, module_fqn=f"{prefix}.blocks.{index}.attn",
            )

    def forward(self, patches: torch.Tensor, grid_height: int, grid_width: int) -> torch.Tensor:
        """Encode one image's raster-order ViT patches.

        Args:
            patches: Normalized RGB patches in raster order.
            grid_height: Number of ViT patch rows.
            grid_width: Number of ViT patch columns.

        Returns:
            Vision features with one row per input patch.
        """
        expected_patches = grid_height * grid_width
        if expected_patches <= 0 or patches.shape[0] != expected_patches:
            raise ValueError(
                "vision patch count must equal grid_height * grid_width; "
                f"got patches={patches.shape[0]}, grid=({grid_height}, {grid_width})"
            )
        hidden_states = self.patch_embed(patches)
        cosine, sine = _vision_cos_sin(grid_height, grid_width, self.rope_dim, self.rope_theta, hidden_states.device)
        for block in self.blocks:
            hidden_states = block(hidden_states, cosine, sine)
        return self.norm(hidden_states)


class DeepseekV41VisionAligner(SpatialMergeProjectorModule):
    """Pixel-unshuffle 3x3 (or configured) ViT features into LLM hidden rows."""

    def __init__(self, config: Any) -> None:
        """Create the released two-layer aligner."""
        super().__init__(config=SimpleNamespace(
            input_size=int(config.v41_vision_hidden_size), hidden_size=int(config.hidden_size),
            output_size=int(config.hidden_size), downsample_ratio=int(config.v41_vision_downsample_ratio), bias=True,
        ))


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
        result = cls.__new__(cls)
        # Adopt existing parameters without running an allocating source constructor.
        nn.Module.__init__(result)  # pylint: disable=unnecessary-dunder-call
        result.num_attention_heads, result.head_dim = source.num_attention_heads, source.head_dim
        result.wqkv, result.wo = source.wqkv, source.wo
        result.module_fqn = module_fqn
        result.training = source.training
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


def _vision_sharding(*, module: nn.Module, module_fqn: str, member: Any, context: Any) -> ModuleShardingSpec:
    """Own complete visual parameters; image token/head transport stays in the encoder resource view."""
    del module_fqn, member
    placements = {axis: Replicate() for axis in context.mesh_dim_names if axis in context.axis_sizes}
    return ModuleShardingSpec(params={name: dict(placements) for name, _ in module.named_parameters()},
                              in_src={}, in_dst={}, is_boundary=False)


@module_replacement
def VisionTransformer(
        *, config: Any = None, module: nn.Module | None = None,
        module_fqn: str = "", context: Any = None, mesh: Any = None,
) -> nn.Module:
    """Build the complete V4.1 raster-grid vision family, including owned encoder CP execution."""
    return build_family(VisionTransformer, config=config, module=module,
                        module_fqn=module_fqn, context=context, mesh=mesh)


register_module_family(ModuleFamily(
    name="deepseek_v41_vision_transformer", baseline=VisionTransformer,
    variants={"local": "hyper_parallel.models.deepseek_v41.components.vision_transformer.DeepseekV41VisionTower"},
    default_variant="local", sharding=_vision_sharding,
))
