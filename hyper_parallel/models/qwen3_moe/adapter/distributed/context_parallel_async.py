# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""context_parallel_async: Qwen3-MoE asynchronous CP wrappers.

Moved from the generic ``distributed/context_parallel/wrappers.py`` in M5
(adjust doc §5.3, M5 step 4). Unlike the fused wrappers in
``context_parallel.py`` (which wrap the ``modules.GQAAttention`` built by
``replace_qwen3_moe_flash_attention`` and only swap its
``attention_interface``), the async wrappers target the ORIGINAL
HuggingFace-structure Qwen3-MoE attention module (``q_proj``/``k_proj``/
``v_proj``/``q_norm``/``k_norm``) and replace its whole forward with an
overlap-scheduled variant: projections and RoPE run locally, the K/V (or
Q/K/V) exchange is launched asynchronously and waited on right before the
fused attention kernel.

Forward-install discipline (05 §15.2.3): the wrappers never assign
``module.forward`` themselves; each validates its target and returns a
``_ForwardRewriteRequest`` that the generic
``distributed/_builder/forward_rewriter.py`` validates, commits and rolls
back atomically.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

import torch

from hyper_parallel.distributed._builder.forward_rewriter import (
    _ForwardRewriteRequest,
)
from hyper_parallel.distributed.context_parallel.attention import (
    _cp_offset_causal_mask,
)
from hyper_parallel.distributed.context_parallel.collectives import (
    _build_hybrid_cp_submeshes,
    async_cp_allgather_launch,
    async_ulysses_seq_to_head_launch,
    ulysses_head_to_seq,
)
from hyper_parallel.distributed.recipe_spec import inner_wrapper
from hyper_parallel.models.qwen3_moe.adapter.attention import (
    run_qwen3_moe_flash_attention,
)

_QWEN3_MOE_SEQ_DIM = 2
_QWEN3_MOE_HEAD_DIM = 1


def _fused_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Delegate to the merged generic implementation in ``functional.rms_norm``.

    Imported lazily because the generic functional module wraps the NPU-only
    kernel at top level; this module must stay CPU-importable.
    """
    from hyper_parallel.components.functional.rms_norm import (  # pylint: disable=C0415
        rms_norm,
    )

    return rms_norm(hidden_states, weight, epsilon)


def _require_qwen3_moe_attention(module: Any) -> None:
    """Validate the attributes required by the Qwen3-MoE attention wrapper."""
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
            "Qwen3-MoE async CP requires an attention module with attributes "
            f"{required}; missing {missing} on {type(module).__name__}"
        )


def _require_qwen3_moe_training_call(past_key_values: Any | None) -> None:
    if past_key_values is not None:
        raise ValueError(
            "Qwen3-MoE async CP currently supports training without KV cache; "
            "past_key_values must be None"
        )


def _qwen3_moe_position_terms(position_embeddings):
    cos, sin = position_embeddings
    return cos.unsqueeze(1), sin.unsqueeze(1)


def _qwen3_moe_project_query(module, hidden_states, hidden_shape, cos, sin):
    import torch_npu  # pylint: disable=C0415

    query = module.q_proj(hidden_states).view(hidden_shape)
    query = _fused_rms_norm(
        query,
        module.q_norm.weight,
        module.q_norm.variance_epsilon,
    ).transpose(1, 2)
    return torch_npu.npu_rotary_mul(query, cos, sin)


def _qwen3_moe_project_key(module, hidden_states, hidden_shape, cos, sin):
    import torch_npu  # pylint: disable=C0415

    key = module.k_proj(hidden_states).view(hidden_shape)
    key = _fused_rms_norm(
        key,
        module.k_norm.weight,
        module.k_norm.variance_epsilon,
    ).transpose(1, 2)
    return torch_npu.npu_rotary_mul(key, cos, sin)


def _qwen3_moe_project_value(module, hidden_states, hidden_shape):
    return module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)


def _prepare_qwen3_moe_attention_mask(
    attention_mask: torch.Tensor | None,
    query: torch.Tensor,
    key: torch.Tensor,
    query_offset: int,
) -> torch.Tensor | None:
    """Build the causal attention mask for a local Qwen3-MoE CP shard."""
    q_len = query.shape[_QWEN3_MOE_SEQ_DIM]
    kv_len = key.shape[_QWEN3_MOE_SEQ_DIM]
    if attention_mask is None:
        if q_len == kv_len and query_offset == 0:
            return None
        return _cp_offset_causal_mask(
            q_len,
            kv_len,
            query_offset,
            query.device,
        )
    if attention_mask.shape[-1] != kv_len:
        raise ValueError(
            "Qwen3-MoE CP attention_mask must cover the global KV sequence: "
            f"mask kv length={attention_mask.shape[-1]}, expected {kv_len}"
        )
    if attention_mask.ndim >= 2 and attention_mask.shape[-2] != q_len:
        if attention_mask.shape[-2] < query_offset + q_len:
            raise ValueError(
                "Qwen3-MoE CP attention_mask does not cover this rank's query "
                f"range [{query_offset}, {query_offset + q_len})"
            )
        attention_mask = attention_mask.narrow(-2, query_offset, q_len)
    return attention_mask


def _run_qwen3_moe_fused_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    kwargs,
):
    return run_qwen3_moe_flash_attention(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=0.0 if not module.training else module.attention_dropout,
        scaling=module.scaling,
        sliding_window=module.sliding_window,
        **kwargs,
    )


def _finish_qwen3_moe_attention(module, attention_output, input_shape):
    output = attention_output.reshape(*input_shape, -1).contiguous()
    return module.o_proj(output)


def _qwen3_moe_async_colossal_forward(
    module: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any | None = None,
    *,
    cp_mesh: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run Qwen3-MoE with async K/V AllGather and local Q."""
    _require_qwen3_moe_training_call(past_key_values)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    cos, sin = _qwen3_moe_position_terms(position_embeddings)

    query = _qwen3_moe_project_query(module, hidden_states, hidden_shape, cos, sin)
    key = _qwen3_moe_project_key(module, hidden_states, hidden_shape, cos, sin)
    key_pending = async_cp_allgather_launch(key, _QWEN3_MOE_SEQ_DIM, cp_mesh)
    value = _qwen3_moe_project_value(module, hidden_states, hidden_shape)
    value_pending = async_cp_allgather_launch(value, _QWEN3_MOE_SEQ_DIM, cp_mesh)

    key = key_pending.wait()
    value = value_pending.wait()
    query_offset = cp_mesh.get_local_rank() * query.shape[_QWEN3_MOE_SEQ_DIM]
    attention_mask = _prepare_qwen3_moe_attention_mask(
        attention_mask, query, key, query_offset
    )
    attention_output, attention_weights = _run_qwen3_moe_fused_attention(
        module,
        query,
        key,
        value,
        attention_mask,
        kwargs,
    )
    return _finish_qwen3_moe_attention(module, attention_output, input_shape), attention_weights


def _qwen3_moe_async_ulysses_forward(
    module: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any | None = None,
    *,
    cp_mesh: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run Qwen3-MoE with async Q/K/V sequence-to-head A2A."""
    _require_qwen3_moe_training_call(past_key_values)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    cos, sin = _qwen3_moe_position_terms(position_embeddings)

    query = _qwen3_moe_project_query(module, hidden_states, hidden_shape, cos, sin)
    query_pending = async_ulysses_seq_to_head_launch(
        query, _QWEN3_MOE_SEQ_DIM, _QWEN3_MOE_HEAD_DIM, cp_mesh
    )
    key = _qwen3_moe_project_key(module, hidden_states, hidden_shape, cos, sin)
    key_pending = async_ulysses_seq_to_head_launch(
        key, _QWEN3_MOE_SEQ_DIM, _QWEN3_MOE_HEAD_DIM, cp_mesh
    )
    value = _qwen3_moe_project_value(module, hidden_states, hidden_shape)
    value_pending = async_ulysses_seq_to_head_launch(
        value, _QWEN3_MOE_SEQ_DIM, _QWEN3_MOE_HEAD_DIM, cp_mesh
    )

    query = query_pending.wait()
    key = key_pending.wait()
    value = value_pending.wait()
    attention_mask = _prepare_qwen3_moe_attention_mask(
        attention_mask, query, key, query_offset=0
    )
    attention_output, attention_weights = _run_qwen3_moe_fused_attention(
        module,
        query,
        key,
        value,
        attention_mask,
        kwargs,
    )
    output_bnsd = attention_output.transpose(1, 2).contiguous()
    output_bnsd = ulysses_head_to_seq(
        output_bnsd, _QWEN3_MOE_SEQ_DIM, _QWEN3_MOE_HEAD_DIM, cp_mesh
    )
    attention_output = output_bnsd.transpose(1, 2).contiguous()
    return _finish_qwen3_moe_attention(module, attention_output, input_shape), attention_weights


def _qwen3_moe_async_hybrid_forward(
    module: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any | None = None,
    *,
    cp_mesh: Any,
    ulysses_degree: int,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run Qwen3-MoE with async Ulysses A2A and Colossal K/V gather."""
    _require_qwen3_moe_training_call(past_key_values)
    ulysses_mesh, colossal_mesh = _build_hybrid_cp_submeshes(
        cp_mesh, ulysses_degree
    )
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    cos, sin = _qwen3_moe_position_terms(position_embeddings)

    query = _qwen3_moe_project_query(module, hidden_states, hidden_shape, cos, sin)
    query_a2a = async_ulysses_seq_to_head_launch(
        query, _QWEN3_MOE_SEQ_DIM, _QWEN3_MOE_HEAD_DIM, ulysses_mesh
    )
    key = _qwen3_moe_project_key(module, hidden_states, hidden_shape, cos, sin)
    key_a2a = async_ulysses_seq_to_head_launch(
        key, _QWEN3_MOE_SEQ_DIM, _QWEN3_MOE_HEAD_DIM, ulysses_mesh
    )
    value = _qwen3_moe_project_value(module, hidden_states, hidden_shape)
    value_a2a = async_ulysses_seq_to_head_launch(
        value, _QWEN3_MOE_SEQ_DIM, _QWEN3_MOE_HEAD_DIM, ulysses_mesh
    )

    key = key_a2a.wait()
    key_gather = async_cp_allgather_launch(
        key, _QWEN3_MOE_SEQ_DIM, colossal_mesh
    )
    value = value_a2a.wait()
    value_gather = async_cp_allgather_launch(
        value, _QWEN3_MOE_SEQ_DIM, colossal_mesh
    )
    query = query_a2a.wait()
    key = key_gather.wait()
    value = value_gather.wait()

    query_offset = colossal_mesh.get_local_rank() * query.shape[_QWEN3_MOE_SEQ_DIM]
    attention_mask = _prepare_qwen3_moe_attention_mask(
        attention_mask, query, key, query_offset
    )
    attention_output, attention_weights = _run_qwen3_moe_fused_attention(
        module,
        query,
        key,
        value,
        attention_mask,
        kwargs,
    )
    output_bnsd = attention_output.transpose(1, 2).contiguous()
    output_bnsd = ulysses_head_to_seq(
        output_bnsd, _QWEN3_MOE_SEQ_DIM, _QWEN3_MOE_HEAD_DIM, ulysses_mesh
    )
    attention_output = output_bnsd.transpose(1, 2).contiguous()
    return _finish_qwen3_moe_attention(module, attention_output, input_shape), attention_weights


def _validate_qwen3_moe_cp_mesh(cp_mesh, wrapper_name):
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError(f"{wrapper_name} requires an active CP mesh")


def _validate_qwen3_moe_ulysses_heads(module, degree, wrapper_name):
    config = module.config
    for name in ("num_attention_heads", "num_key_value_heads"):
        count = getattr(config, name)
        if count % degree:
            raise ValueError(
                f"{wrapper_name} requires {name} ({count}) to be divisible "
                f"by Ulysses degree ({degree})"
            )


def _build_qwen3_moe_async_rewrite_request(
    target_module: Any,
    forward_fn: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    **forward_config: Any,
) -> _ForwardRewriteRequest:
    """Build the rewrite request for an asynchronous Qwen3-MoE CP forward."""
    _require_qwen3_moe_attention(target_module)
    original_forward = target_module.forward

    @functools.wraps(original_forward)
    def cp_forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Dispatch the original attention signature to the async CP forward."""
        return forward_fn(
            target_module,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            **forward_config,
            **kwargs,
        )

    return _ForwardRewriteRequest(target_module, cp_forward)


@inner_wrapper
def qwen3_moe_async_colossal_cp_wrapper(
    target_module: Any,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
) -> _ForwardRewriteRequest:
    """Return the rewrite request for the async Colossal Qwen3-MoE forward."""
    del mesh, tp_mesh, ep_mesh
    _validate_qwen3_moe_cp_mesh(cp_mesh, "qwen3_moe_async_colossal")
    return _build_qwen3_moe_async_rewrite_request(
        target_module,
        _qwen3_moe_async_colossal_forward,
        cp_mesh=cp_mesh,
    )


@inner_wrapper
def qwen3_moe_async_ulysses_cp_wrapper(
    target_module: Any,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
) -> _ForwardRewriteRequest:
    """Return the rewrite request for the async Pure Ulysses Qwen3-MoE forward."""
    del mesh, tp_mesh, ep_mesh
    _validate_qwen3_moe_cp_mesh(cp_mesh, "qwen3_moe_async_ulysses")
    _validate_qwen3_moe_ulysses_heads(
        target_module, cp_mesh.size(), "qwen3_moe_async_ulysses"
    )
    return _build_qwen3_moe_async_rewrite_request(
        target_module,
        _qwen3_moe_async_ulysses_forward,
        cp_mesh=cp_mesh,
    )


@inner_wrapper
def qwen3_moe_async_hybrid_cp_wrapper(
    target_module: Any,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
    ulysses_degree: int,
) -> _ForwardRewriteRequest:
    """Return the rewrite request for the async Hybrid Qwen3-MoE forward."""
    del mesh, tp_mesh, ep_mesh
    _validate_qwen3_moe_cp_mesh(cp_mesh, "qwen3_moe_async_hybrid")
    _build_hybrid_cp_submeshes(cp_mesh, ulysses_degree)
    _validate_qwen3_moe_ulysses_heads(
        target_module, ulysses_degree, "qwen3_moe_async_hybrid"
    )
    return _build_qwen3_moe_async_rewrite_request(
        target_module,
        _qwen3_moe_async_hybrid_forward,
        cp_mesh=cp_mesh,
        ulysses_degree=ulysses_degree,
    )


__all__ = [
    "qwen3_moe_async_colossal_cp_wrapper",
    "qwen3_moe_async_hybrid_cp_wrapper",
    "qwen3_moe_async_ulysses_cp_wrapper",
]
