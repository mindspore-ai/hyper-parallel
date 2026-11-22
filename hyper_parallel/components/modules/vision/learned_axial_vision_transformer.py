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
"""Learned spatial/time positions, separate Q/K/V vision blocks and temporal-spatial patch pooling.

The source-compatible equations follow Transformers' Apache-2.0 Kimi K2.5 vision implementation.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils.generic import get_max_seqlen, is_flash_attention_requested
from transformers.vision_utils import (
    _interpolation_axis_taps_weights,
    get_vision_attention_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

from hyper_parallel.components.modules.module_family import ModuleFamily, build_family, register_module_family
from hyper_parallel.components.modules.vision.axial_vision_transformer import VisionAttention, _eager_vision_attention
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec
from hyper_parallel.models.materialization import register_rebuildable_buffer
from hyper_parallel.models.replacement import module_replacement


def frame_indices(grid: torch.Tensor, kwargs: dict) -> torch.Tensor:
    """Resolve temporal table rows on the data side or from an explicitly host-resident grid."""
    supplied = kwargs.pop("frame_index", None)
    if supplied is not None:
        return supplied
    if grid.device.type != "cpu":
        raise ValueError("Device vision grids require precomputed frame_index")
    return torch.cat([(torch.arange(t) + int(t > 1)).repeat_interleave(h * w) for t, h, w in grid.tolist()])


def temporal_merge_indices(grid: torch.Tensor, kernel: tuple[int, int], kwargs: dict) -> torch.Tensor:
    """Resolve a gather with temporal reduction and preserved spatial patch groups."""
    supplied = kwargs.pop("temporal_merge_index", None)
    if supplied is not None:
        return supplied
    if grid.device.type != "cpu":
        raise ValueError("Device vision grids require precomputed temporal_merge_index")
    offset, groups = 0, []
    times = grid[:, 0].tolist()
    if not times or len(set(times)) != 1:
        raise ValueError("A temporal merge batch must use a common positive clip length")
    for time, height, width in grid.tolist():
        rows = torch.arange(offset, offset + time * height * width).reshape(
            time, height // kernel[0], kernel[0], width // kernel[1], kernel[1]
        )
        groups.append(rows.permute(1, 3, 0, 2, 4).reshape(-1, time, kernel[0] * kernel[1]))
        offset += time * height * width
    return torch.cat(groups)


class LearnedPositionEmbedding(nn.Module):
    """Own the learned bicubic spatial table and a deterministic sinusoidal time table."""

    def __init__(self, config: Any, module: nn.Module | None = None) -> None:
        """Keep source state names and register time-state reconstruction before materialization."""
        super().__init__()
        self.dim, self.num_frames = config.hidden_size, config.pos_emb_time
        self.num_grid_per_side = config.pos_emb_height
        if config.pos_emb_height != config.pos_emb_width:
            raise ValueError("Learned axial spatial interpolation requires a square source grid")
        self.interpolation_align_corners, self.interpolation_mode = False, "bicubic"
        self.position_embeddings = (
            module.position_embeddings
            if module is not None
            else nn.Parameter(torch.zeros(config.pos_emb_height, config.pos_emb_width, config.hidden_size))
        )
        time_table = self.compute_pos_embed() if module is None else module.time_position_embeddings
        self.register_buffer("time_position_embeddings", time_table, persistent=False)
        register_rebuildable_buffer(
            self,
            "time_position_embeddings",
            factory=lambda ctx: self.compute_pos_embed().to(self.time_position_embeddings.dtype),
        )
        if module is not None:
            self.training = module.training

    def compute_pos_embed(self) -> torch.Tensor:
        """Build time rows on CPU independently of an outer meta/default-device context."""
        with torch.device("cpu"):
            frequencies = 1.0 / (10000 ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
            angles = torch.outer(torch.arange(self.num_frames, dtype=torch.float32), frequencies)
            values = torch.cat((angles.sin(), angles.cos()), dim=1)
            return torch.cat((values.new_zeros(1, self.dim), values))

    def reset_parameters(self) -> None:
        """Initialize the family-owned learned position table."""
        nn.init.zeros_(self.position_embeddings)

    def make_transforms(self) -> list:
        """Learned position keys and representation are unchanged."""
        return []

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Add bicubic spatial interpolation and image/video-specific temporal rows."""
        indices, weights = get_vision_interpolation_indices_and_weights(
            grid_thw, self.num_grid_per_side, mode="bicubic", align_corners=False, kwargs=kwargs
        )
        table = self.position_embeddings.flatten(0, 1)
        spatial = F.embedding_bag(indices, table, per_sample_weights=weights.to(table.dtype), mode="sum")
        temporal = self.time_position_embeddings[frame_indices(grid_thw, kwargs)]
        return hidden_states + (spatial + temporal).to(hidden_states.dtype)


class LearnedPatchEmbedding(nn.Module):
    """Project individual image patches and add complete learned spatial/time positions."""

    def __init__(self, config: Any, module: nn.Module | None = None) -> None:
        """Adopt source primitive convolution parameters without a source forward callback."""
        super().__init__()
        patch = config.patch_size
        patch = (patch, patch) if isinstance(patch, int) else tuple(patch)
        self.proj = module.proj if module is not None else nn.Conv2d(3, config.hidden_size, patch, stride=patch)
        self.pos_emb = LearnedPositionEmbedding(config, None if module is None else module.pos_emb)
        if module is not None:
            self.training = module.training

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Keep the processor's patch order and the source convolution rounding boundary."""
        hidden_states = self.proj(pixel_values).view(pixel_values.size(0), -1)
        return self.pos_emb(hidden_states, grid_thw, **kwargs)


class LearnedAxialRotaryEmbedding(nn.Module):
    """Own the width/height interleaved rotary frequency layout and both derived buffers."""

    def __init__(self, config: Any, module: nn.Module | None = None) -> None:
        """Declare deterministic CPU reconstruction independent of the source model class."""
        super().__init__()
        self.config = config
        self.rope_type, self.attention_scaling = "axial", 1.0
        if config.rope_parameters["rope_type"] != "axial":
            raise ValueError("Learned axial vision requires axial rotary coordinates")
        frequencies, _ = self.compute_axial_rope_parameters(config)
        for name in ("inv_freq", "original_inv_freq"):
            value = getattr(module, name) if module is not None else frequencies.clone()
            self.register_buffer(name, value, persistent=False)
            register_rebuildable_buffer(
                self,
                name,
                factory=lambda ctx, key=name: self.compute_axial_rope_parameters(config)[0].to(
                    getattr(self, key).dtype
                ),
            )

    @staticmethod

    def compute_axial_rope_parameters(config: Any, device: Any = None, **kwargs: Any) -> tuple[torch.Tensor, float]:
        """Recreate the source half-head axial frequencies."""
        del kwargs
        dim = (getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads) // 2
        with torch.device("cpu"):
            values = 1.0 / (config.rope_parameters["rope_theta"] ** (torch.arange(0, dim, 2).float() / dim))
        return values.to(device), 1.0

    @torch.no_grad()

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compose width then height coordinates twice across the full attention head."""
        angles = position_ids[..., None].float() * self.inv_freq.to(hidden_states.device, torch.float32)
        result = []
        for value in (angles.cos(), angles.sin()):
            axes = value.transpose(1, 2).flip(-1).flatten(1)
            result.append(torch.cat((axes, axes), -1))
        return tuple(result)


def _apply_rotary(
    query: torch.Tensor, key: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the source's four-dimensional cast nodes before both axial arithmetic branches."""
    query_dtype, key_dtype = query.dtype, key.dtype
    query, key = query.float(), key.float()
    cosine, sine = cosine.unsqueeze(-2).float(), sine.unsqueeze(-2).float()
    return (
        (
            query * cosine
            + torch.cat((-query[..., query.shape[-1] // 2 :], query[..., : query.shape[-1] // 2]), dim=-1) * sine
        ).to(query_dtype),
        (key * cosine + torch.cat((-key[..., key.shape[-1] // 2 :], key[..., : key.shape[-1] // 2]), dim=-1) * sine).to(
            key_dtype
        ),
    )


class SeparateVisionAttention(VisionAttention):
    """Use common axial attention with independent Q/K/V source optimizer matrices."""

    def __init__(self, config: Any, module: nn.Module | None = None) -> None:
        """Retain separate projection identities and their bias vectors."""
        nn.Module.__init__(self)  # pylint: disable=non-parent-init-called
        self.config, self.dim, self.num_heads = config, config.hidden_size, config.num_attention_heads
        self.head_dim, self.num_key_value_groups = self.dim // self.num_heads, 1
        self.scaling, self.attention_dropout, self.is_causal = self.head_dim**-0.5, 0.0, False
        self.proj = module.proj if module is not None else nn.Linear(self.dim, self.dim)
        for name in ("q_proj", "k_proj", "v_proj"):
            setattr(self, name, getattr(module, name) if module is not None else nn.Linear(self.dim, self.dim))
        if module is not None:
            self.training = module.training

    def project_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project separate source matrices into common token/head operands."""
        return tuple(
            projection(hidden_states).reshape(-1, self.num_heads, self.head_dim)
            for projection in (self.q_proj, self.k_proj, self.v_proj)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        max_seqlen: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Preserve source four-dimensional projection and rotary autograd rounding boundaries."""
        seq_length = hidden_states.shape[0]

        query_states = self.q_proj(hidden_states).reshape(1, seq_length, -1, self.head_dim)
        key_states = self.k_proj(hidden_states).reshape(1, seq_length, -1, self.head_dim)
        value_states = self.v_proj(hidden_states).reshape(1, seq_length, -1, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary(query_states, key_states, cos, sin)

        query_states = query_states.transpose(2, 1)
        key_states = key_states.transpose(2, 1)
        value_states = value_states.transpose(2, 1)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, _eager_vision_attention
        )

        if is_flash_attention_requested(self.config):
            # Flash Attention: Use cu_seqlens for variable length attention
            max_seqlen = get_max_seqlen(cu_seqlens, self.config, kwargs={"max_seqlen": max_seqlen})
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = kwargs.pop("vision_attention_lengths", None)
            if lengths is None:
                if cu_seqlens.device.type != "cpu":
                    raise ValueError("Non-flash vision requires precomputed host vision_attention_lengths")
                lengths = tuple((cu_seqlens[1:] - cu_seqlens[:-1]).tolist())
            splits = [torch.split(tensor, lengths, dim=2) for tensor in (query_states, key_states, value_states)]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class LearnedVisionMLP(nn.Module):
    """Bias-enabled vision MLP preserving the source activation approximation and state names."""

    def __init__(self, config: Any, module: nn.Module | None = None) -> None:
        """Allocate or adopt separate source matrices."""
        super().__init__()
        self.fc1 = module.fc1 if module is not None else nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = module.act if module is not None else ACT2FN[config.hidden_act]
        self.fc2 = module.fc2 if module is not None else nn.Linear(config.intermediate_size, config.hidden_size)
        if module is not None:
            self.training = module.training

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Evaluate the configured vision activation between two independent matrices."""
        return self.fc2(self.act(self.fc1(hidden_states)))


class LearnedVisionBlock(nn.Module):
    """Complete pre-normalized separate-QKV vision residual block."""

    def __init__(self, config: Any, module: nn.Module | None = None) -> None:
        """Preserve source layer normalization epsilon and child registration order."""
        super().__init__()
        self.norm1 = module.norm1 if module is not None else nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.norm2 = module.norm2 if module is not None else nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.attn = SeparateVisionAttention(config, None if module is None else module.attn)
        self.mlp = LearnedVisionMLP(config, None if module is None else module.mlp)
        if module is not None:
            self.training = module.training

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Evaluate attention then MLP residuals using the complete frame geometry."""
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), **kwargs)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class LearnedAxialVisionTransformerModule(nn.Module):
    """Complete learned-position vision encoder with temporal pooling and spatial patch grouping."""

    def __init__(
        self, *, config: Any = None, module: nn.Module | None = None, module_fqn: str = "", context: Any = None
    ) -> None:
        """Build full vision mathematics from a config or adopt primitive source state."""
        super().__init__()
        del module_fqn, context
        if (config is None) == (module is None):
            raise ValueError("provide exactly one of config or module")
        self.config = config if module is None else module.config
        config = self.config
        self.merge_kernel_size = tuple(config.merge_kernel_size)
        if config.hidden_size % config.num_attention_heads or config.hidden_size // config.num_attention_heads % 4:
            raise ValueError("Learned axial vision requires complete attention heads divisible by four")
        if module is not None and set(dict(module.named_children())) != {
            "patch_embed",
            "rotary_emb",
            "layers",
            "final_layernorm",
        }:
            raise ValueError("Unsupported learned axial vision source boundary")
        self.patch_embed = LearnedPatchEmbedding(config, None if module is None else module.patch_embed)
        self.rotary_emb = LearnedAxialRotaryEmbedding(config, None if module is None else module.rotary_emb)
        self.layers = nn.ModuleList(
            [
                LearnedVisionBlock(config, None if module is None else module.layers[index])
                for index in range(config.num_hidden_layers)
            ]
        )
        self.final_layernorm = (
            module.final_layernorm if module is not None else nn.LayerNorm(config.hidden_size, eps=1e-5)
        )
        if module is not None:
            self.training = module.training

    @property

    def dtype(self) -> torch.dtype:
        """Expose the source vision boundary's parameter dtype."""
        return self.patch_embed.proj.weight.dtype

    def make_transforms(self) -> list:
        """Preserve separate source matrices and the entire checkpoint key layout."""
        return []

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, **kwargs: Any) -> BaseModelOutputWithPooling:
        """Encode complete media and return source-compatible encoded and merged features."""
        geometry = kwargs.pop("vision_geometry", None)
        if geometry is not None:
            geometry = dict(geometry)
            host_grid = geometry.pop("grid_thw")
            if host_grid.device.type != "cpu" or host_grid.shape != grid_thw.shape:
                raise ValueError("Learned vision geometry requires matching CPU grid dimensions")
            if set(geometry).intersection(kwargs):
                raise ValueError("Learned vision geometry conflicts with explicit vision inputs")
            grid_thw = host_grid
            kwargs.update({name: value.to(pixel_values.device) if torch.is_tensor(value) else value
                           for name, value in geometry.items()})
            if pixel_values.device.type != "cpu":
                # Host geometry fixes output sizes; interpolation arithmetic must
                # run on the compute device to retain the source rounding order.
                positions = kwargs["position_ids"]
                sizes = torch.repeat_interleave(host_grid[:, 1:3], host_grid.prod(-1), dim=0).to(pixel_values.device)
                side = self.patch_embed.pos_emb.num_grid_per_side
                h_taps, h_weights = _interpolation_axis_taps_weights(positions[:, 0], sizes[:, 0],
                                                                    side, "bicubic", False)
                w_taps, w_weights = _interpolation_axis_taps_weights(positions[:, 1], sizes[:, 1],
                                                                    side, "bicubic", False)
                kwargs["interp_indices"] = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 16)
                kwargs["interp_weights"] = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, 16)
        if grid_thw.device.type != "cpu":
            required = {
                "position_ids",
                "interp_indices",
                "interp_weights",
                "frame_index",
                "cu_seqlens",
                "temporal_merge_index",
            }
            required.add("max_seqlen" if is_flash_attention_requested(self.config) else "vision_attention_lengths")
            if any(kwargs.get(name) is None for name in required):
                raise ValueError(
                    "Device vision grids require complete precomputed spatial/time/attention/merge geometry"
                )
        hidden = self.patch_embed(pixel_values, grid_thw, **kwargs)
        positions = get_vision_position_ids(grid_thw, spatial_merge_size=1, kwargs=kwargs)
        embeddings = self.rotary_emb(hidden, positions)
        offsets, maximum = get_vision_attention_seqlens(grid_thw, self.config, merge_temporal=True, kwargs=kwargs)
        for layer in self.layers:
            hidden = layer(hidden, cu_seqlens=offsets, max_seqlen=maximum, position_embeddings=embeddings, **kwargs)
        hidden = self.final_layernorm(hidden)
        indices = temporal_merge_indices(grid_thw, self.merge_kernel_size, kwargs)
        return BaseModelOutputWithPooling(last_hidden_state=hidden, pooler_output=hidden[indices].mean(dim=1))


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
def LearnedAxialVisionTransformer(  # pylint: disable=invalid-name
    *,
    config: Any = None,
    module: nn.Module | None = None,
    module_fqn: str = "",
    context: Any = None,
    mesh: Any = None,
) -> nn.Module:
    """Construct complete learned spatial/time vision through the module-family mechanism."""
    return build_family(
        LearnedAxialVisionTransformer, config=config, module=module, module_fqn=module_fqn, context=context, mesh=mesh
    )


register_module_family(
    ModuleFamily(
        name="learned_axial_vision_transformer",
        baseline=LearnedAxialVisionTransformer,
        variants={
            "local": ("hyper_parallel.components.modules.vision.learned_axial_vision_transformer."
                      "LearnedAxialVisionTransformerModule")
        },
        default_variant="local",
        sharding=replicated_sharding,
    )
)
