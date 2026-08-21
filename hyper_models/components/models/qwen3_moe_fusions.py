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
"""Ascend fused forward replacements for Hugging Face Qwen3-MoE modules."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from functools import wraps
from types import MethodType
from typing import Any, Callable

import torch
from torch import nn

from hyper_models.components.distributed.cp_utils import (
    _cp_offset_causal_mask,
    flex_cp_allgather,
)
from hyper_models.components.distributed.injection import inner_wrapper
from hyper_models.components.model_transform import module_replacement

_COMPRESSED_CAUSAL_MASK_SIZE = 2048
_COMPRESSED_CAUSAL_MASKS: dict[torch.device, torch.Tensor] = {}


def _get_compressed_causal_mask(device: torch.device) -> torch.Tensor:
    """Return the cached mask required by NPU left-up causal sparse mode."""
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


class _GroupedMatmul(torch.autograd.Function):
    """Autograd wrapper adapting HF expert weights to grouped NPU matmul."""

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Apply grouped matmul and retain inputs for its explicit backward."""
        # torch_npu is optional outside Ascend environments.
        import torch_npu  # pylint: disable=C0415

        ctx.save_for_backward(inputs, weight)
        ctx.tokens_per_expert = tokens_per_expert
        grouped_weight = weight.transpose(1, 2).contiguous()
        return torch_npu.npu_grouped_matmul(
            [inputs],
            [grouped_weight],
            bias=None,
            group_list=tokens_per_expert,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Compute input and expert-weight gradients with grouped matmul."""
        # torch_npu is optional outside Ascend environments.
        import torch_npu  # pylint: disable=C0415

        inputs, weight = ctx.saved_tensors
        tokens_per_expert = ctx.tokens_per_expert
        grad_inputs = torch_npu.npu_grouped_matmul(
            [grad_output],
            [weight],
            bias=None,
            group_list=tokens_per_expert,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]
        grouped_grad_weight = torch_npu.npu_grouped_matmul(
            [inputs.transpose(0, 1)],
            [grad_output],
            bias=None,
            group_list=tokens_per_expert,
            split_item=3,
            group_type=2,
            group_list_type=1,
        )[0]
        grad_weight = grouped_grad_weight.transpose(1, 2).contiguous()
        return grad_inputs, grad_weight, None


def _fused_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Apply the Ascend fused RMSNorm kernel."""
    # torch_npu is optional outside Ascend environments.
    import torch_npu  # pylint: disable=C0415

    return torch_npu.npu_rms_norm(hidden_states, weight, epsilon=epsilon)[0]


def qwen3_moe_fused_rms_norm_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Replace ``Qwen3MoeRMSNorm.forward`` with fused NPU RMSNorm."""
    return _fused_rms_norm(hidden_states, module.weight, module.variance_epsilon)


def _run_qwen3_moe_flash_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Run Qwen3-MoE grouped-query attention with the Ascend Flash Attention kernel."""
    # torch_npu is optional outside Ascend environments.
    import torch_npu  # pylint: disable=C0415

    del kwargs

    if attention_mask is None:
        attention_mask = _get_compressed_causal_mask(query.device)
        sparse_mode = 2
    else:
        if attention_mask.ndim == 4:
            attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        if attention_mask.dtype == torch.bool:
            attention_mask = torch.logical_not(attention_mask).to(query.device)
        else:
            attention_mask = attention_mask.bool().to(query.device)
        sparse_mode = 0

    sparse_kwargs = {}
    if (
        sparse_mode == 0
        and getattr(module, "is_causal", True)
        and query.shape[-2] == key.shape[-2]
    ):
        sparse_kwargs["next_tockens"] = 0

    attention_output = torch_npu.npu_fusion_attention(
        query,
        key,
        value,
        head_num=query.shape[1],
        input_layout="BNSD",
        atten_mask=attention_mask,
        keep_prob=1 - dropout,
        scale=scaling,
        sparse_mode=sparse_mode,
        **sparse_kwargs,
    )[0]
    return attention_output.transpose(1, 2).contiguous(), None


def _prepare_qwen3_moe_attention_states(
    module: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    past_key_values: Any | None = None,
) -> tuple[
    tuple[int, ...],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Project and position Qwen3-MoE query, key, and value states."""
    # torch_npu is optional outside Ascend environments.
    import torch_npu  # pylint: disable=C0415

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)

    query_states = module.q_proj(hidden_states).view(hidden_shape)
    query_states = _fused_rms_norm(
        query_states,
        module.q_norm.weight,
        module.q_norm.variance_epsilon,
    ).transpose(1, 2)
    key_states = module.k_proj(hidden_states).view(hidden_shape)
    key_states = _fused_rms_norm(
        key_states,
        module.k_norm.weight,
        module.k_norm.variance_epsilon,
    ).transpose(1, 2)
    value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query_states = torch_npu.npu_rotary_mul(query_states, cos, sin)
    key_states = torch_npu.npu_rotary_mul(key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states,
            value_states,
            module.layer_idx,
        )

    return input_shape, query_states, key_states, value_states


def _project_qwen3_moe_attention_output(
    module: nn.Module,
    attention_output: torch.Tensor,
    input_shape: tuple[int, ...],
) -> torch.Tensor:
    """Restore the hidden shape and apply the output projection."""
    attention_output = attention_output.reshape(*input_shape, -1).contiguous()
    return module.o_proj(attention_output)


def qwen3_moe_flash_attention_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the non-CP Qwen3-MoE fused attention module implementation."""
    input_shape, query_states, key_states, value_states = (
        _prepare_qwen3_moe_attention_states(
            module,
            hidden_states,
            position_embeddings,
            past_key_values,
        )
    )

    attention_output, attention_weights = _run_qwen3_moe_flash_attention(
        module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not module.training else module.attention_dropout,
        scaling=module.scaling,
        sliding_window=module.sliding_window,
        **kwargs,
    )
    return (
        _project_qwen3_moe_attention_output(module, attention_output, input_shape),
        attention_weights,
    )


def qwen3_moe_grouped_moe_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Run Qwen3-MoE routing and experts with fused Ascend grouped kernels."""
    # torch_npu is optional outside Ascend environments.
    import torch_npu  # pylint: disable=C0415

    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    _, routing_weights, selected_experts = module.gate(hidden_states)
    permuted_states, row_ids_map = torch_npu.npu_moe_token_permute(
        hidden_states,
        selected_experts.to(torch.int32),
    )
    tokens_per_expert = torch.histc(
        selected_experts,
        bins=module.experts.num_experts,
        min=0,
        max=module.experts.num_experts,
    ).to(torch.int64)

    gate_up_output = _GroupedMatmul.apply(
        permuted_states,
        module.experts.gate_up_proj,
        tokens_per_expert,
    )
    activated_states = torch_npu.npu_swiglu(gate_up_output, dim=-1)
    expert_output = _GroupedMatmul.apply(
        activated_states,
        module.experts.down_proj,
        tokens_per_expert,
    )
    output = torch_npu.npu_moe_token_unpermute(
        expert_output,
        row_ids_map,
        probs=routing_weights,
    )
    return output.reshape(batch_size, sequence_length, hidden_dim)


def _replace_forward(module: nn.Module, forward: Callable) -> nn.Module:
    """Return a structure-preserving shallow module copy with a bound forward."""
    replacement = copy.copy(module)
    replacement.forward = MethodType(forward, replacement)
    return replacement


@inner_wrapper
def qwen3_moe_flash_attention_cp_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
) -> None:
    """Replace the exact Qwen3-MoE fused attention forward with its CP version."""
    del mesh, tp_mesh, ep_mesh
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError(
            "qwen3_moe_flash_attention_cp_wrapper requires an active CP mesh"
        )

    original_forward = target_module.forward
    installed_forward = getattr(original_forward, "__func__", original_forward)
    if installed_forward is not qwen3_moe_flash_attention_forward:
        raise ValueError(
            "qwen3_moe_flash_attention_cp_wrapper can only replace "
            "qwen3_moe_flash_attention_forward; apply "
            "replace_qwen3_moe_flash_attention first"
        )

    @wraps(original_forward)
    def cp_forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if attention_mask is not None:
            raise ValueError(
                "Qwen3-MoE fused CP attention currently requires an implicit "
                "causal mask; configure create_attention_mask_in_dataloader=false"
            )
        input_shape, query_states, key_states, value_states = (
            _prepare_qwen3_moe_attention_states(
                target_module,
                hidden_states,
                position_embeddings,
                past_key_values,
            )
        )
        query_length = query_states.shape[-2]
        query_offset = cp_mesh.get_local_rank() * query_length
        key_states, value_states = flex_cp_allgather(
            key_states.contiguous(),
            value_states.contiguous(),
            2,
            cp_mesh,
        )
        cp_causal_mask = _cp_offset_causal_mask(
            query_length,
            key_states.shape[-2],
            query_offset,
            query_states.device,
        )
        attention_output, attention_weights = _run_qwen3_moe_flash_attention(
            target_module,
            query_states,
            key_states,
            value_states,
            cp_causal_mask,
            dropout=(
                0.0
                if not target_module.training
                else target_module.attention_dropout
            ),
            scaling=target_module.scaling,
            sliding_window=target_module.sliding_window,
            **kwargs,
        )
        return (
            _project_qwen3_moe_attention_output(
                target_module,
                attention_output,
                input_shape,
            ),
            attention_weights,
        )

    target_module.forward = cp_forward


@module_replacement
def replace_qwen3_moe_rms_norm(
    *,
    module: nn.Module,
    module_fqn: str,
    context: Mapping[str, Any],
) -> nn.Module:
    """Build a structure-preserving Qwen3-MoE fused RMSNorm replacement."""
    del module_fqn, context
    return _replace_forward(module, qwen3_moe_fused_rms_norm_forward)


@module_replacement
def replace_qwen3_moe_flash_attention(
    *,
    module: nn.Module,
    module_fqn: str,
    context: Mapping[str, Any],
) -> nn.Module:
    """Build a structure-preserving Qwen3-MoE Flash Attention replacement."""
    del module_fqn, context
    return _replace_forward(module, qwen3_moe_flash_attention_forward)


@module_replacement
def replace_qwen3_moe_sparse_moe(
    *,
    module: nn.Module,
    module_fqn: str,
    context: Mapping[str, Any],
) -> nn.Module:
    """Build a structure-preserving Qwen3-MoE grouped-expert replacement."""
    del module_fqn, context
    return _replace_forward(module, qwen3_moe_grouped_moe_forward)
