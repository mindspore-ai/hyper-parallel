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
"""Implementation for Qwen3-VL-MoE vision tower parity checks."""

import torch
from torch.nn import functional as F

from hyper_parallel.models.qwen3_vl_vision import (
    Qwen3VLMoeVisionConfig,
    Qwen3VLMoeVisionModel,
)


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states_1 = hidden_states[..., : hidden_states.shape[-1] // 2]
    hidden_states_2 = hidden_states[..., hidden_states.shape[-1] // 2:]
    return torch.cat((-hidden_states_2, hidden_states_1), dim=-1)


def _apply_rotary_pos_emb(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding with the legacy eager attention semantics."""
    orig_query_dtype = query_states.dtype
    orig_key_dtype = key_states.dtype
    query_states = query_states.float()
    key_states = key_states.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    query_embed = (query_states * cos) + (_rotate_half(query_states) * sin)
    key_embed = (key_states * cos) + (_rotate_half(key_states) * sin)
    return query_embed.to(orig_query_dtype), key_embed.to(orig_key_dtype)


def _legacy_patch_embed(patch_embed, hidden_states: torch.Tensor) -> torch.Tensor:
    """Run the pre-change Conv3d patch projection semantics."""
    hidden_states = hidden_states.view(
        -1,
        patch_embed.in_channels,
        patch_embed.temporal_patch_size,
        patch_embed.patch_size,
        patch_embed.patch_size,
    )
    hidden_states = hidden_states.to(dtype=patch_embed.proj.weight.dtype)
    hidden_states = patch_embed.proj(hidden_states)
    return hidden_states.view(-1, patch_embed.embed_dim)


def _legacy_attention(
    attention,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Run the pre-change eager attention semantics for one visual block."""
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        attention.qkv(hidden_states)
        .reshape(seq_length, 3, attention.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]

    attn_outputs = []
    for query_chunk, key_chunk, value_chunk in zip(
        torch.split(query_states, lengths.tolist(), dim=2),
        torch.split(key_states, lengths.tolist(), dim=2),
        torch.split(value_states, lengths.tolist(), dim=2),
    ):
        attn_weights = torch.matmul(
            query_chunk,
            key_chunk.transpose(2, 3),
        ) * attention.scaling
        attn_weights = torch.softmax(attn_weights.float(), dim=-1).to(query_chunk.dtype)
        attn_weights = F.dropout(
            attn_weights,
            p=0.0 if not attention.training else attention.attention_dropout,
            training=attention.training,
        )
        attn_outputs.append(torch.matmul(attn_weights, value_chunk).transpose(1, 2).contiguous())

    attn_output = torch.cat(attn_outputs, dim=1)
    return attention.proj(attn_output.reshape(seq_length, -1).contiguous())


def _legacy_vision_forward(
    model: Qwen3VLMoeVisionModel,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Run the pre-change vision forward semantics used as the parity baseline."""
    hidden_states = _legacy_patch_embed(model.patch_embed, hidden_states)
    hidden_states = hidden_states + model.fast_pos_embed_interpolate(grid_thw)

    rotary_pos_emb = model.rot_pos_emb(grid_thw)
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2],
        grid_thw[:, 0],
    ).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    deepstack_features = []
    for layer_idx, block in enumerate(model.blocks):
        hidden_states = hidden_states + _legacy_attention(
            block.attn,
            block.norm1(hidden_states),
            cu_seqlens,
            position_embeddings,
        )
        hidden_states = hidden_states + block.mlp(block.norm2(hidden_states))
        if layer_idx in model.deepstack_visual_indexes:
            merge_idx = model.deepstack_visual_indexes.index(layer_idx)
            deepstack_features.append(model.deepstack_merger_list[merge_idx](hidden_states))

    return hidden_states, model.merger(hidden_states), deepstack_features


def _tiny_vision_config() -> Qwen3VLMoeVisionConfig:
    """Build a small deterministic vision config for CPU parity checks."""
    return Qwen3VLMoeVisionConfig(
        depth=2,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=1,
        out_hidden_size=8,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0],
        _attn_implementation="eager",
    )


def test_qwen3_vl_moe_vision_forward_matches_legacy_eager_path():
    """
    Feature: Qwen3-VL-MoE visual tower precision regression.
    Description: Compare current vision forward with pre-change Conv3d/eager-attention semantics.
    Expectation: Outputs keep float32 parity.
    """
    torch.manual_seed(2026)
    model = Qwen3VLMoeVisionModel(_tiny_vision_config())
    model.eval()

    grid_thw = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.long)
    hidden_size = (
        model.patch_embed.in_channels
        * model.patch_embed.temporal_patch_size
        * model.patch_embed.patch_size
        * model.patch_embed.patch_size
    )
    hidden_states = torch.randn(8, hidden_size, dtype=torch.float32)

    actual = model(hidden_states, grid_thw)
    expected_last, expected_pooler, expected_deepstack = _legacy_vision_forward(
        model,
        hidden_states,
        grid_thw,
    )

    torch.testing.assert_close(actual.last_hidden_state, expected_last, rtol=1.0e-6, atol=1.0e-6)
    torch.testing.assert_close(actual.pooler_output, expected_pooler, rtol=1.0e-6, atol=1.0e-6)
    assert len(actual.deepstack_features) == len(expected_deepstack)
    for actual_feature, expected_feature in zip(actual.deepstack_features, expected_deepstack):
        torch.testing.assert_close(actual_feature, expected_feature, rtol=1.0e-6, atol=1.0e-6)
