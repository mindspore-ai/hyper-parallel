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
"""Context-parallel input and vision-boundary adapters for Qwen3-VL-MoE."""

from __future__ import annotations

import functools
from typing import Any

import torch

from hyper_parallel.components.functional.npu_fusion_attention import npu_fusion_attention_forward
from hyper_parallel.distributed._builder.forward_rewriter import _ForwardRewriteRequest
from hyper_parallel.distributed.context_parallel.attention import _cp_offset_causal_mask
from hyper_parallel.distributed.context_parallel.collectives import (
    async_ulysses_seq_to_head_launch,
    ulysses_head_to_seq,
)
from hyper_parallel.distributed.recipe_spec import inner_wrapper

_SEQUENCE_DIM = 2
_HEAD_DIM = 1
_COMPRESSED_CAUSAL_MASK_SIZE = 2048
_COMPRESSED_CAUSAL_MASKS: dict[torch.device, torch.Tensor] = {}


def _require_text_attention(module: Any) -> None:
    """Validate the Qwen3-VL-MoE text-attention contract."""
    required = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "q_norm",
        "k_norm",
        "head_dim",
        "scaling",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise TypeError(
            "Qwen3-VL-MoE async CP requires a text attention module with "
            f"attributes {required}; missing {missing} on {type(module).__name__}"
        )


def _prepare_attention_mask(
    attention_mask: torch.Tensor | None,
    query: torch.Tensor,
    key: torch.Tensor,
) -> torch.Tensor | None:
    """Normalize the global Qwen3-VL-MoE mask after Ulysses redistribution."""
    q_len = query.shape[_SEQUENCE_DIM]
    kv_len = key.shape[_SEQUENCE_DIM]
    if attention_mask is None:
        return None
    if attention_mask.shape[-1] != kv_len:
        raise ValueError(
            "Qwen3-VL-MoE CP attention_mask must cover the global KV sequence: "
            f"mask kv length={attention_mask.shape[-1]}, expected {kv_len}"
        )
    if attention_mask.ndim == 2:
        return attention_mask.to(device=query.device, dtype=torch.bool)
    if attention_mask.ndim >= 2 and attention_mask.shape[-2] != q_len:
        if attention_mask.shape[-2] < q_len:
            raise ValueError(
                "Qwen3-VL-MoE CP attention_mask does not cover the global query sequence"
            )
        attention_mask = attention_mask.narrow(-2, 0, q_len)
    return attention_mask


def _get_compressed_causal_mask(device: torch.device) -> torch.Tensor:
    """Return the mask required by the Ascend left-up causal sparse mode."""
    mask = _COMPRESSED_CAUSAL_MASKS.get(device)
    if mask is None:
        mask = torch.triu(
            torch.ones(
                (_COMPRESSED_CAUSAL_MASK_SIZE, _COMPRESSED_CAUSAL_MASK_SIZE),
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        )
        _COMPRESSED_CAUSAL_MASKS[device] = mask
    return mask


def _run_flash_attention(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Run Qwen3-VL-MoE text attention with the Ascend fused kernel."""
    import torch_npu  # pylint: disable=C0415

    shared_mask = attention_mask is not None and attention_mask.ndim == 2
    shared_window = attention_mask is None and (kwargs.get("sliding_window") is not None or "pre_tokens" in kwargs)
    if query.shape[-2] == key.shape[-2] and (shared_mask or shared_window):
        return npu_fusion_attention_forward(
            module, query, key, value, attention_mask,
            dropout=module.attention_dropout if module.training else 0.0,
            scaling=module.scaling,
            **kwargs,
        )
    if attention_mask is None:
        attention_mask = _get_compressed_causal_mask(query.device)
        sparse_mode = 2
    else:
        if attention_mask.ndim == 2:
            causal_mask = _cp_offset_causal_mask(query.shape[-2], key.shape[-2], 0, query.device)
            attention_mask = causal_mask[None, None] & attention_mask[:, None, None, :]
        if attention_mask.ndim == 4:
            attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        if attention_mask.dtype == torch.bool:
            attention_mask = torch.logical_not(attention_mask).to(query.device)
        else:
            attention_mask = attention_mask.bool().to(query.device)
        sparse_mode = 0

    sparse_kwargs = {}
    if sparse_mode == 0 and getattr(module, "is_causal", True) and query.shape[-2] == key.shape[-2]:
        sparse_kwargs["next_tockens"] = 0

    output = torch_npu.npu_fusion_attention(
        query,
        key,
        value,
        head_num=query.shape[1],
        input_layout="BNSD",
        atten_mask=attention_mask,
        keep_prob=1 - (0.0 if not module.training else module.attention_dropout),
        scale=module.scaling,
        sparse_mode=sparse_mode,
        **sparse_kwargs,
    )[0]
    return output.transpose(1, 2).contiguous(), None


def _async_ulysses_forward(
    module: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any | None = None,
    *,
    cp_mesh: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run Qwen3-VL-MoE text attention with asynchronous Ulysses exchange."""
    if past_key_values is not None:
        raise ValueError("Qwen3-VL-MoE async CP supports training without KV cache")

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    cos, sin = (term.unsqueeze(1) for term in position_embeddings)

    import torch_npu  # pylint: disable=C0415

    from hyper_parallel.components.functional.rms_norm import (  # pylint: disable=C0415
        rms_norm,
    )

    query = module.q_proj(hidden_states).view(hidden_shape)
    query = rms_norm(
        query, module.q_norm.weight, module.q_norm.variance_epsilon
    ).transpose(1, 2)
    query = torch_npu.npu_rotary_mul(query, cos, sin)
    query_pending = async_ulysses_seq_to_head_launch(
        query, _SEQUENCE_DIM, _HEAD_DIM, cp_mesh
    )

    key = module.k_proj(hidden_states).view(hidden_shape)
    key = rms_norm(
        key, module.k_norm.weight, module.k_norm.variance_epsilon
    ).transpose(1, 2)
    key = torch_npu.npu_rotary_mul(key, cos, sin)
    key_pending = async_ulysses_seq_to_head_launch(
        key, _SEQUENCE_DIM, _HEAD_DIM, cp_mesh
    )

    value = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_pending = async_ulysses_seq_to_head_launch(
        value, _SEQUENCE_DIM, _HEAD_DIM, cp_mesh
    )

    query = query_pending.wait()
    key = key_pending.wait()
    value = value_pending.wait()
    attention_mask = _prepare_attention_mask(attention_mask, query, key)
    attention_output, attention_weights = _run_flash_attention(
        module, query, key, value, attention_mask, **kwargs
    )
    attention_output = ulysses_head_to_seq(
        attention_output.transpose(1, 2).contiguous(),
        _SEQUENCE_DIM,
        _HEAD_DIM,
        cp_mesh,
    ).transpose(1, 2).contiguous()
    output = module.o_proj(attention_output.reshape(*input_shape, -1).contiguous())
    return output, attention_weights


@inner_wrapper
def qwen3_vl_moe_async_ulysses_cp_wrapper(
    target_module: Any,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
) -> _ForwardRewriteRequest:
    """Build the model-owned async Ulysses wrapper for text attention."""
    del mesh, tp_mesh, ep_mesh
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError("Qwen3-VL-MoE async Ulysses requires an active CP mesh")
    _require_text_attention(target_module)
    for name in ("num_attention_heads", "num_key_value_heads"):
        count = getattr(target_module.config, name)
        if count % cp_mesh.size():
            raise ValueError(
                f"Qwen3-VL-MoE async Ulysses requires {name} ({count}) to be "
                f"divisible by CP size ({cp_mesh.size()})"
            )

    original_forward = target_module.forward

    @functools.wraps(original_forward)
    def cp_forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run text attention using the bound context-parallel mesh."""
        return _async_ulysses_forward(
            target_module,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            cp_mesh=cp_mesh,
            **kwargs,
        )

    return _ForwardRewriteRequest(target_module, cp_forward)


@inner_wrapper
def qwen3_vl_moe_replicated_vision_cp_wrapper(
    target_module: Any,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
) -> _ForwardRewriteRequest:
    """Keep vision attention replicated while text tokens use CP."""
    del mesh, tp_mesh, ep_mesh
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError("Qwen3-VL-MoE replicated vision CP requires an active CP mesh")
    return _ForwardRewriteRequest(target_module, target_module.forward)


def _slice_dense_visual_features(
    features: torch.Tensor,
    global_mask: torch.Tensor,
    start: int,
    end: int,
) -> torch.Tensor:
    """Select mask-ordered visual features belonging to one local token shard."""
    global_mask = global_mask.to(torch.bool)
    local_mask = global_mask[:, start:end]
    if not local_mask.any():
        return features[:0]

    batch_size, seq_len = global_mask.shape
    local_positions = torch.arange(start, end, device=global_mask.device).unsqueeze(0).expand(batch_size, -1)
    flat_positions = (torch.arange(batch_size, device=global_mask.device).unsqueeze(1) * seq_len + local_positions)[
        local_mask
    ]
    feature_indices = global_mask.reshape(-1).to(torch.int64).cumsum(0)[flat_positions] - 1
    return features.index_select(0, feature_indices.to(features.device))


@inner_wrapper
def qwen3_vl_moe_text_input_cp_wrapper(
    target_module: Any,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
) -> _ForwardRewriteRequest:
    """Shard text inputs after Qwen3-VL has injected global visual features."""
    del mesh, tp_mesh, ep_mesh
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError("Qwen3-VL-MoE text input CP requires an active CP mesh")

    original_forward = target_module.forward
    target_module._hp_cp_shards_inputs = True  # pylint: disable=protected-access

    def cp_forward(
        input_ids: Any = None,
        attention_mask: Any = None,
        position_ids: Any = None,
        past_key_values: Any = None,
        inputs_embeds: Any = None,
        use_cache: Any = None,
        visual_pos_masks: Any = None,
        deepstack_visual_embeds: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Run the language model on the local CP token and visual-feature shard."""
        sequence = inputs_embeds if inputs_embeds is not None else input_ids
        if sequence is None:
            return original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                **kwargs,
            )

        seq_len = sequence.shape[1]
        cp_size = cp_mesh.size()
        if seq_len % cp_size:
            raise ValueError(f"Qwen3-VL-MoE sequence length ({seq_len}) must be divisible by CP size ({cp_size})")
        local_seq_len = seq_len // cp_size
        start = cp_mesh.get_local_rank() * local_seq_len
        end = start + local_seq_len

        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds[:, start:end, :].contiguous()
        if input_ids is not None:
            input_ids = input_ids[:, start:end].contiguous()
        if isinstance(position_ids, torch.Tensor):
            position_ids = position_ids[..., start:end].contiguous()
        if isinstance(visual_pos_masks, torch.Tensor):
            global_visual_pos_masks = visual_pos_masks
            visual_pos_masks = global_visual_pos_masks[:, start:end].contiguous()
            if deepstack_visual_embeds is not None:
                deepstack_visual_embeds = [
                    _slice_dense_visual_features(features, global_visual_pos_masks, start, end)
                    for features in deepstack_visual_embeds
                ]

        return original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

    return _ForwardRewriteRequest(target_module, cp_forward)


__all__ = [
    "qwen3_vl_moe_async_ulysses_cp_wrapper",
    "qwen3_vl_moe_replicated_vision_cp_wrapper",
    "qwen3_vl_moe_text_input_cp_wrapper",
]
