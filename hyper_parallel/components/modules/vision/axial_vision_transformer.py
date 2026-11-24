# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Packed axial vision Transformer with learned spatial positions and optional deepstack outputs.

The source boundary covers patch projection through final spatial merging.
Equations and checkpoint layout follow the native Transformers Qwen3 vision implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils.generic import is_flash_attention_requested
from transformers.vision_utils import (
    get_vision_attention_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

from hyper_parallel.components.modules.module_family import ModuleFamily, build_family, register_module_family
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.core.fully_shard.api import HSDPModule
from hyper_parallel.distributed.context_parallel.collectives import ulysses_head_to_seq, ulysses_seq_to_head
from hyper_parallel.distributed.encoder_parallel.attention import current_encoder_attention_context
from hyper_parallel.distributed.encoder_parallel.contracts import (
    EncoderLocalInputs,
    EncoderSegment,
    EncoderWorkItem,
)
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.materialization import register_rebuildable_buffer
from hyper_parallel.models.replacement import module_replacement


@dataclass
class AxialVisionOutput(BaseModelOutputWithPooling):
    """Complete encoded patches, merged language-width features and optional deepstack features."""

    deepstack_features: list[torch.Tensor] | None = None


class VisionPatchEmbedding(nn.Module):
    """Project already ordered temporal/spatial patches into the encoder width."""

    def __init__(self, config: Any, module: nn.Module | None = None) -> None:
        """Adopt a Conv3d or allocate the canonical nonoverlapping patch projection."""
        super().__init__()
        self.patch_size, self.temporal_patch_size = config.patch_size, config.temporal_patch_size
        self.in_channels, self.embed_dim = config.in_channels, config.hidden_size
        kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = (
            module.proj
            if module is not None
            else nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel, stride=kernel, bias=True)
        )
        if module is not None:
            self.training = module.training

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Consume flattened processor patches without changing their merge order."""
        patches = hidden_states.reshape(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        return self.proj(patches.to(self.proj.weight.dtype)).reshape(-1, self.embed_dim)


class VisionMLP(nn.Module):
    """Bias-enabled two-projection vision MLP with the declared source activation."""

    def __init__(self, config: Any, module: nn.Module | None = None) -> None:
        """Preserve separate source matrices and the source GELU approximation."""
        super().__init__()
        self.linear_fc1 = (
            module.linear_fc1
            if module is not None
            else nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        )
        self.linear_fc2 = (
            module.linear_fc2
            if module is not None
            else nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        )
        self.act_fn = module.act_fn if module is not None else ACT2FN[config.hidden_act]
        if module is not None:
            self.training = module.training

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply activation between the independent input and output matrices."""
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))


class VisionPatchMerger(nn.Module):
    """Normalize patches before or after spatial shuffle and project to language width."""

    def __init__(self, config: Any, module: nn.Module | None = None, *, postshuffle_norm: bool = False) -> None:
        """Keep regular and deepstack merger normalization positions distinct."""
        super().__init__()
        self.hidden_size = config.hidden_size * config.spatial_merge_size**2
        self.use_postshuffle_norm = module.use_postshuffle_norm if module is not None else postshuffle_norm
        self.norm = (
            module.norm
            if module is not None
            else nn.LayerNorm(self.hidden_size if self.use_postshuffle_norm else config.hidden_size, eps=1e-6)
        )
        self.linear_fc1 = module.linear_fc1 if module is not None else nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = module.act_fn if module is not None else nn.GELU()
        self.linear_fc2 = (
            module.linear_fc2 if module is not None else nn.Linear(self.hidden_size, config.out_hidden_size)
        )
        if module is not None:
            self.training = module.training

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Merge one already adjacent spatial group into a language-width feature."""
        values = hidden_states.reshape(-1, self.hidden_size) if self.use_postshuffle_norm else hidden_states
        values = self.norm(values).reshape(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(values)))


class AxialVisionRotaryEmbedding(nn.Module):
    """Own reconstructible two-axis RoPE frequencies for complete vision attention heads."""

    def __init__(self, config: Any, module: nn.Module | None = None) -> None:
        """Construct the frequency state on CPU, including when the parameter tree is meta."""
        super().__init__()
        self.config = config
        self.rope_type = config.rope_parameters["rope_type"]
        if self.rope_type != "axial":
            raise ValueError("Axial vision Transformer requires axial RoPE")
        frequency, self.attention_scaling = self.compute_axial_rope_parameters(config)
        target = torch.empty(0).device
        for name in ("inv_freq", "original_inv_freq"):
            register_rebuildable_buffer(self, name, value=frequency.clone(), persistent=False)
            value = getattr(module, name, None) if module is not None else None
            setattr(self, name, frequency.clone().to(target) if value is None else value)
        if module is not None:
            self.training = module.training

    @staticmethod

    def compute_axial_rope_parameters(config: Any, device: Any = None, **kwargs: Any) -> tuple[torch.Tensor, float]:
        """Use half the attention head for each spatial axis, then duplicate split-half rotations."""
        del kwargs
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_heads
        spatial_dim = head_dim // 2
        base = config.rope_parameters["rope_theta"]
        frequencies = 1.0 / (base ** (torch.arange(0, spatial_dim, 2, device="cpu").float() / spatial_dim))
        return frequencies.to(device), 1.0

    @torch.no_grad()

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return axial cosine/sine in FP32 with the source spatial channel arrangement."""
        angles = position_ids.to(hidden_states.device).float()[..., None] * self.inv_freq.float()
        cosine, sine = angles.cos() * self.attention_scaling, angles.sin() * self.attention_scaling
        outputs = []
        for value in (cosine, sine):
            spatial = torch.cat((value[:, 0], value[:, 1]), dim=-1)
            outputs.append(torch.cat((spatial, spatial), dim=-1))
        return tuple(outputs)


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _eager_vision_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    scaling: float,
    **kwargs: Any,
) -> tuple:
    """Reference noncausal vision attention with FP32 softmax and source rounding."""
    del module, kwargs
    scores = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask
    probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(probabilities, value).transpose(1, 2).contiguous(), probabilities


class VisionAttention(nn.Module):
    """Own packed QKV projection, axial rotations, frame visibility and output projection."""

    def __init__(self, config: Any, module: nn.Module | None = None) -> None:
        """Preserve the existing packed QKV matrix and its optimizer boundary."""
        super().__init__()
        self.config, self.dim, self.num_heads = config, config.hidden_size, config.num_heads
        self.head_dim, self.num_key_value_groups = self.dim // self.num_heads, 1
        self.scaling, self.attention_dropout, self.is_causal = self.head_dim**-0.5, 0.0, False
        self.qkv = module.qkv if module is not None else nn.Linear(self.dim, 3 * self.dim, bias=True)
        self.proj = module.proj if module is not None else nn.Linear(self.dim, self.dim, bias=True)
        if module is not None:
            self.training = module.training

    def project_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project the canonical packed matrix into token/head operands."""
        tokens = hidden_states.shape[0]
        return self.qkv(hidden_states).reshape(tokens, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        max_seqlen: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Execute independently visible frames while preserving caller-supplied media boundaries."""
        tokens = hidden_states.shape[0]
        query, key, value = self.project_qkv(hidden_states)
        cosine, sine = (item.unsqueeze(-2).float() for item in position_embeddings)
        query_dtype, key_dtype = query.dtype, key.dtype
        query, key = query.float(), key.float()
        query = (query * cosine + _rotate_half(query) * sine).to(query_dtype)
        key = (key * cosine + _rotate_half(key) * sine).to(key_dtype)
        query, key, value = (item.transpose(0, 1).unsqueeze(0) for item in (query, key, value))
        interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, _eager_vision_attention)
        if is_flash_attention_requested(self.config):
            output, _ = interface(
                self,
                query,
                key,
                value,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            lengths = kwargs.pop("vision_attention_lengths", None)
            if lengths is None:
                if cu_seqlens.device.type != "cpu":
                    raise ValueError("Non-flash vision attention requires precomputed host vision_attention_lengths")
                lengths = tuple((cu_seqlens[1:] - cu_seqlens[:-1]).tolist())
            splits = [torch.split(item, lengths, dim=2) for item in (query, key, value)]
            output = torch.cat(
                [
                    interface(
                        self, q, k, v, attention_mask=None, scaling=self.scaling, dropout=0.0, is_causal=False, **kwargs
                    )[0]
                    for q, k, v in zip(*splits)
                ],
                dim=1,
            )
        return self.proj(output.reshape(tokens, self.dim).contiguous())


class VisionBlock(nn.Module):
    """Pre-normalized attention and MLP residual block with source-compatible parameter names."""

    def __init__(self, config: Any, module: nn.Module | None = None) -> None:
        """Adopt primitive children and replace full block mathematics through ordinary module calls."""
        super().__init__()
        self.norm1 = module.norm1 if module is not None else nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = module.norm2 if module is not None else nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = VisionAttention(config, None if module is None else module.attn)
        self.mlp = VisionMLP(config, None if module is None else module.mlp)
        if module is not None:
            self.training = module.training

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Keep attention and MLP residual ordering identical to the source encoder."""
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens, position_embeddings, **kwargs)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class AxialVisionTransformerModule(nn.Module):
    """Complete public vision encoder, including patch merger and optional deepstack branches."""

    def __init__(
        self, *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "", context: Any = None
    ) -> None:
        """Allocate from a vision config or adopt the complete supported source state boundary."""
        super().__init__()
        del context
        attention_prefix = module_fqn or "model.visual"
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        self.config = config if module is None else module.config
        config = self.config
        self.spatial_merge_size, self.patch_size = config.spatial_merge_size, config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        self.interpolation_align_corners, self.interpolation_mode = True, "bilinear"
        self._validate_vision_source(config, module)

        def child(name: str) -> nn.Module | None:
            return None if module is None else getattr(module, name)

        self.patch_embed = VisionPatchEmbedding(config, child("patch_embed"))
        self.pos_embed = (
            nn.Embedding(config.num_position_embeddings, config.hidden_size) if module is None else module.pos_embed
        )
        self.rotary_pos_emb = AxialVisionRotaryEmbedding(config, child("rotary_pos_emb"))
        self.blocks = nn.ModuleList(
            [VisionBlock(config, None if module is None else module.blocks[index]) for index in range(config.depth)]
        )
        for index, block in enumerate(self.blocks):
            block.attn = CPVisionAttention(block.attn, f"{attention_prefix}.blocks.{index}.attn")
        self.merger = VisionPatchMerger(config, child("merger"))
        if hasattr(module, "deepstack_merger_list") if module is not None else bool(self.deepstack_visual_indexes):
            self.deepstack_merger_list = nn.ModuleList(
                [
                    VisionPatchMerger(
                        config, None if module is None else module.deepstack_merger_list[index], postshuffle_norm=True
                    )
                    for index in range(len(self.deepstack_visual_indexes))
                ]
            )
        self.gradient_checkpointing = False
        if module is not None:
            self.training = module.training

    def _validate_vision_source(self, config: Any, module: nn.Module | None) -> None:
        """Validate complete vision state and deepstack layer ownership before child adoption."""
        if (
            config.hidden_size % config.num_heads
            or config.hidden_size // config.num_heads % 4
            or self.num_grid_per_side**2 != config.num_position_embeddings
        ):
            raise ValueError(
                "Axial vision requires complete heads divisible by four and a square learned position grid"
            )
        self.deepstack_visual_indexes = tuple(getattr(config, "deepstack_visual_indexes", ()))
        if len(set(self.deepstack_visual_indexes)) != len(self.deepstack_visual_indexes) or any(
            index < 0 or index >= config.depth for index in self.deepstack_visual_indexes
        ):
            raise ValueError("deepstack indices must be distinct actual vision layers")
        if module is not None:
            allowed = {"patch_embed", "pos_embed", "rotary_pos_emb", "blocks", "merger", "deepstack_merger_list"}
            if set(dict(module.named_children())) - allowed or len(module.blocks) != config.depth:
                raise ValueError("Vision source state boundary does not match the complete axial Transformer contract")
            if module.interpolation_mode != "bilinear" or not module.interpolation_align_corners:
                raise ValueError("Vision source requires a supported learned-position interpolation policy")



    @property

    def dtype(self) -> torch.dtype:
        """Expose the source vision dtype used by the model's media-input conversion."""
        return self.patch_embed.proj.weight.dtype

    @property

    def device(self) -> torch.device:
        """Expose the device that owns the first patch projection."""
        return self.patch_embed.proj.weight.device

    def make_transforms(self) -> list:
        """All source parameter names, packed logical matrices and tensor layouts are retained."""
        return []

    def _device_interpolation(
        self,
        grid_thw: torch.Tensor,
        positions: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep geometry on the host but evaluate interpolation arithmetic on the source compute device.

        Device division can round differently from CPU division. Reusing host
        weights changes BF16 features at rounding boundaries, so only integer
        geometry crosses the host/device boundary.
        """
        sizes = torch.repeat_interleave(grid_thw[:, 1:3], grid_thw.prod(-1), dim=0).to(device)
        coordinates = positions.to(device).float() * (self.num_grid_per_side - 1) / (sizes - 1).clamp(min=1)
        floor = coordinates.floor()
        offsets = torch.arange(2, device=device)
        taps = (floor.long()[..., None] + offsets).clamp(0, self.num_grid_per_side - 1)
        weights = (1 - (coordinates[..., None] - floor[..., None] - offsets).abs()).clamp(min=0)
        indices = (taps[:, 0, :, None] * self.num_grid_per_side + taps[:, 1, None, :]).reshape(-1, 4)
        weights = (weights[:, 0, :, None] * weights[:, 1, None, :]).reshape(-1, 4)
        return indices, weights

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Any) -> AxialVisionOutput:
        """Encode complete media patches and emit merged and deepstack feature protocols directly."""
        geometry = kwargs.pop("vision_geometry", None)
        if geometry is not None:
            geometry = dict(geometry)
            host_grid = geometry.pop("grid_thw")
            if host_grid.device.type != "cpu" or host_grid.shape != grid_thw.shape:
                raise ValueError("Precomputed vision geometry requires matching CPU grid dimensions")
            if set(geometry).intersection(kwargs):
                raise ValueError("Precomputed vision geometry conflicts with explicit vision inputs")
            grid_thw = host_grid
            kwargs.update(geometry)
        if grid_thw.device.type != "cpu":
            required = {"position_ids", "interp_indices", "interp_weights", "cu_seqlens"}
            required.add("max_seqlen" if is_flash_attention_requested(self.config) else "vision_attention_lengths")
            missing = sorted(name for name in required if kwargs.get(name) is None)
            if missing:
                raise ValueError(f"Device-resident vision grids require precomputed geometry: {missing}")
        precomputed_interpolation = (
            kwargs.get("interp_indices") is not None and kwargs.get("interp_weights") is not None
        )
        indices, weights = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side=self.num_grid_per_side,
            mode=self.interpolation_mode,
            align_corners=self.interpolation_align_corners,
            spatial_merge_size=self.spatial_merge_size,
            kwargs=kwargs,
        )
        positions = get_vision_position_ids(grid_thw, self.spatial_merge_size, kwargs=kwargs)
        cumulative, maximum = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)
        if (
            grid_thw.device.type == "cpu"
            and hidden_states.device.type != "cpu"
            and (geometry is not None or not precomputed_interpolation)
        ):
            indices, weights = self._device_interpolation(grid_thw, positions, hidden_states.device)
        hidden_states = self.patch_embed(hidden_states)
        spatial = (self.pos_embed(indices.to(hidden_states.device)) * weights.to(hidden_states.device)[:, :, None]).sum(
            1
        )
        hidden_states = hidden_states + spatial.to(hidden_states.dtype)
        embeddings = self.rotary_pos_emb(hidden_states, positions)
        deepstack = []
        for index, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, cumulative, embeddings, max_seqlen=maximum, **kwargs)
            if index in self.deepstack_visual_indexes:
                deepstack.append(self.deepstack_merger_list[self.deepstack_visual_indexes.index(index)](hidden_states))
        return AxialVisionOutput(
            last_hidden_state=hidden_states,
            pooler_output=self.merger(hidden_states),
            deepstack_features=deepstack if hasattr(self, "deepstack_merger_list") else None,
        )


def replicated_sharding(*, module: nn.Module, module_fqn: str, member: Any, context: Any) -> ModuleShardingSpec:
    """Own the local member's complete parameter tree without inferring projection names."""
    del module_fqn, member
    placements = {axis: Replicate() for axis in context.mesh_dim_names if axis in context.axis_sizes}
    return ModuleShardingSpec(
        params={name: dict(placements) for name, _ in module.named_parameters()},
        in_src={},
        in_dst={},
        out_src=None,
        out_dst=None,
        is_boundary=True,
    )


@module_replacement
def AxialVisionTransformer(  # pylint: disable=invalid-name
    *,
    config: Any = None,
    module: nn.Module | None = None,
    module_fqn: str = "",
    context: Any = None,
    mesh: Any = None,
) -> nn.Module:
    """Construct the complete axial vision encoder family without an external feature bridge."""
    return build_family(
        AxialVisionTransformer, config=config, module=module, module_fqn=module_fqn, context=context, mesh=mesh
    )


register_module_family(
    ModuleFamily(
        name="axial_vision_transformer",
        baseline=AxialVisionTransformer,
        variants={"local": (
            "hyper_parallel.components.modules.vision.axial_vision_transformer."
            "AxialVisionTransformerModule"
        )},
        default_variant="local",
        sharding=replicated_sharding,
    )
)


class CPVisionAttention(VisionAttention):
    """Keep native parameter names; exchange sequence/head axes only when active."""

    def __init__(self, source: nn.Module, module_fqn: str) -> None:
        """Retain the Torch/HF module state and its inactive forward implementation."""
        super().__init__(source.config, module=source)
        self.module_fqn = module_fqn

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor | None = None,
                position_embeddings: tuple | None = None, **kwargs: Any) -> torch.Tensor:
        """Apply independent, padding-masked image attention over Encoder-CP."""
        context = current_encoder_attention_context()
        if context is None:
            return super().forward(hidden_states, cu_seqlens=cu_seqlens,
                                       position_embeddings=position_embeddings, **kwargs)
        context.record(self.module_fqn)
        context.validate_runtime_head_counts(module_fqn=self.module_fqn, query_heads=self.num_heads,
                                             key_heads=self.num_heads, value_heads=self.num_heads)
        query, key, value = self.qkv(hidden_states).reshape(-1, 3, self.num_heads, self.head_dim).unbind(1)
        cosine, sine = (entry.float().unsqueeze(-2) for entry in position_embeddings)

        def rotate(tensor: torch.Tensor) -> torch.Tensor:
            """Apply Qwen's FP32 vision rotary calculation and restore Q/K dtype."""
            left, right = tensor.float().chunk(2, dim=-1)
            return (tensor.float() * cosine + torch.cat((-right, left), dim=-1) * sine).to(tensor.dtype)

        query, key = rotate(query), rotate(key)
        outputs = []
        for segment in context.segments:
            start, length = segment.packed_offset, segment.local_padded_length
            parts = [tensor[start:start + length] for tensor in (query, key, value)]
            if context.cp_mesh.size() > 1:
                parts = [ulysses_seq_to_head(tensor, seq_dim=0, head_dim=1, cp_mesh=context.cp_mesh)
                         for tensor in parts]
            valid = parts[0].shape[0] if segment.is_dummy else segment.item.sequence_length
            mask = torch.arange(parts[0].shape[0], device=hidden_states.device) < valid
            result = F.scaled_dot_product_attention(  # pylint: disable=not-callable
                *(tensor.transpose(0, 1).unsqueeze(0) for tensor in parts),
                attn_mask=mask.reshape(1, 1, 1, -1), dropout_p=0.0, is_causal=False, scale=self.scaling,
            ).squeeze(0).transpose(0, 1).contiguous()
            if context.cp_mesh.size() > 1:
                result = ulysses_head_to_seq(result, seq_dim=0, head_dim=1, cp_mesh=context.cp_mesh)
            outputs.append(result.reshape(length, -1))
        return self.proj(torch.cat(outputs, dim=0))


class AxialVisionEncoderRuntime:
    """Bind the native patch embedding, vision blocks and spatial merger."""

    input_fields = ("pixel_values", "image_grid_thw")

    payload_field = "pixel_values"

    def __init__(self, vision: nn.Module, image_token_id: int) -> None:
        """Borrow the complete encoder and dataset token identity without retaining a parent model."""
        self.vision, self.image_token_id = vision, image_token_id
        self.merge_size = vision.spatial_merge_size

    @property

    def empty_payload_shape(self) -> tuple[int]:
        """Describe one flattened temporal/spatial image patch."""
        config = self.vision.config
        return (config.in_channels * config.temporal_patch_size * config.patch_size**2,)

    @property

    def head_counts(self) -> dict[str, int]:
        """Return native visual head counts, independent of decoder TP."""
        return dict.fromkeys(("query", "key", "value"), self.vision.config.num_heads)

    def validate_training(self, options: Mapping[str, Any]) -> None:
        """Reject modes whose HF feature insertion contract is not yet adapted."""
        if isinstance(self.vision, HSDPModule):
            raise NotImplementedError("Wrap visual children, not model.visual/model as standalone FSDP units")
        if not isinstance(self.vision, AxialVisionTransformerModule):
            raise TypeError("Encoder parallel execution requires the complete AxialVisionTransformer family")
        if self.vision.deepstack_visual_indexes:
            raise NotImplementedError("Encoder feature routing does not yet carry deepstack outputs")
        if options.get("micro_batch_size") != 1:
            raise ValueError("Qwen encoder parallel currently requires micro_batch_size=1")
        if options.get("loss_parallel"):
            raise NotImplementedError("Qwen image insertion requires loss_parallel=false")
        if getattr(self.vision, "gradient_checkpointing", False):
            raise NotImplementedError("Qwen encoder parallel does not support vision gradient checkpointing")

    def parse_images(self, batch: Mapping[str, Any]) -> tuple[EncoderWorkItem, ...]:
        """Validate image/token correspondence before launching visual collectives."""
        tokens = batch.get("vlm_global_input_ids", batch["input_ids"])
        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError("Qwen encoder parallel requires a single unpacked sample")
        self._validate_image_batch(batch)
        positions = (tokens[0] == self.image_token_id).nonzero().flatten().tolist()
        pixels, grids = (batch.get(name) for name in self.input_fields)
        if pixels is None and grids is None:
            if positions:
                raise ValueError("Image tokens are present without image tensors")
            return ()
        self._validate_image_payload(pixels, grids)
        items, offset, token_offset = [], 0, 0
        for index, (temporal, height, width) in enumerate(grids.tolist()):
            if temporal != 1 or min(height, width) <= 0 or height % self.merge_size or width % self.merge_size:
                raise ValueError("Images require T=1 and positive H/W divisible by spatial_merge_size")
            length = height * width
            rows = length // self.merge_size ** 2
            span = positions[token_offset:token_offset + rows]
            if len(span) != rows or span != list(range(span[0], span[0] + rows)):
                raise ValueError("Image token span is truncated or does not match image_grid_thw")
            items.append(EncoderWorkItem(index, 0, length, rows, span[0], rows, offset, offset + length,
                                         {"grid": (temporal, height, width)}))
            offset += length
            token_offset += rows
        if offset != pixels.shape[0] or token_offset != len(positions):
            raise ValueError("Image payload/token count does not match image_grid_thw")
        return tuple(items)

    @staticmethod

    def _validate_image_batch(batch: Mapping[str, Any]) -> None:
        """Reject unsupported packed samples and video inputs."""
        boundaries = batch.get("cu_seq_lens")
        if boundaries is not None and boundaries.numel() > 2:
            raise NotImplementedError("Qwen encoder parallel does not support multi-sample packing")
        if batch.get("pixel_values_videos") is not None or batch.get("video_grid_thw") is not None:
            raise NotImplementedError("Qwen encoder parallel currently supports images, not videos")

    def _validate_image_payload(self, pixels: torch.Tensor, grids: torch.Tensor) -> None:
        """Check flattened image patches and integer grid metadata."""
        if not torch.is_tensor(pixels) or not torch.is_tensor(grids) or pixels.ndim != 2:
            raise ValueError("Qwen image tensors must contain flattened patches and grid_thw")
        if pixels.shape[1:] != self.empty_payload_shape or not pixels.is_floating_point():
            raise ValueError("pixel_values must contain floating patches with the encoder's flattened patch width")
        if grids.ndim != 2 or grids.shape[1] != 3 or grids.dtype not in (torch.int32, torch.int64):
            raise ValueError("image_grid_thw must be an integer [images, 3] tensor")

    def pack_payload(self, batch: Mapping[str, Any], segments: Sequence[EncoderSegment],
                     *, device: torch.device | str = "cpu") -> torch.Tensor:
        """Materialize only assigned patch intervals, keeping full images on their source."""
        pixels = batch.get(self.payload_field)
        dtype = pixels.dtype if pixels is not None else torch.float32
        result = torch.zeros((sum(item.local_padded_length for item in segments), *self.empty_payload_shape),
                             device=device, dtype=dtype)
        for segment in segments:
            if segment.is_dummy or not segment.local_valid_length:
                continue
            if pixels is None:
                raise ValueError("Real axial image segments require pixel_values")
            source = segment.item.payload_start + segment.local_start
            result[segment.packed_offset:segment.packed_offset + segment.local_valid_length].copy_(
                pixels[source:source + segment.local_valid_length],
            )
        return result

    def encode_local(self, batch: Mapping[str, Any] | EncoderLocalInputs,
                     segments: tuple[EncoderSegment, ...]) -> torch.Tensor:
        """Compute owned image shards with source-equivalent device interpolation and axial RoPE."""
        device, dtype = self.vision.device, self.vision.dtype
        patches = (batch.patches if isinstance(batch, EncoderLocalInputs) else self.pack_payload(batch, segments))
        indices, weights, positions = [], [], []
        for segment in segments:
            length, valid = segment.local_padded_length, segment.local_valid_length
            index = torch.zeros(length, 4, device=device, dtype=torch.long)
            weight = torch.zeros(length, 4, device=device)
            position = torch.zeros(length, 2, device=device, dtype=torch.long)
            if not segment.is_dummy:
                grid = torch.tensor([segment.item.metadata["grid"]], dtype=torch.long)
                full_position = get_vision_position_ids(grid, self.merge_size)
                full_index, full_weight = self.vision._device_interpolation(grid, full_position, device)
                start = segment.local_start
                index[:valid] = full_index[start:start + valid]
                weight[:valid] = full_weight[start:start + valid]
                position[:valid] = full_position[start:start + valid].to(device)
            indices.append(index)
            weights.append(weight)
            positions.append(position)
        indices, weights, positions = torch.cat(indices), torch.cat(weights), torch.cat(positions)
        hidden = self.vision.patch_embed(patches.to(device=device, dtype=dtype))
        hidden = hidden + (self.vision.pos_embed(indices) * weights[..., None]).sum(1).to(hidden.dtype)
        embeddings = self.vision.rotary_pos_emb(hidden, positions)
        lengths = tuple(segment.local_padded_length for segment in segments)
        cumulative = torch.tensor((0, *lengths), dtype=torch.int32).cumsum(0, dtype=torch.int32)
        for block in self.vision.blocks:
            hidden = block(hidden, cumulative, embeddings, vision_attention_lengths=lengths)
        return hidden

    def project(self, hidden: torch.Tensor, item: EncoderWorkItem | None) -> torch.Tensor:
        """Apply the owned spatial merger after rebuilding complete image features."""
        if item is None:
            hidden = hidden[:1].expand(self.merge_size**2, -1) * 0.0
        return self.vision.merger(hidden)


@module_replacement
def replace_axial_vision_attention(*, module: nn.Module, module_fqn: str, context: Any) -> nn.Module:
    """Resolve historical leaf declarations to the same class-owned encoder computation."""
    del context
    return CPVisionAttention(module, module_fqn)
