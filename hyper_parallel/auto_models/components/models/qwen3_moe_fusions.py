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

from hyper_parallel.auto_models.components.distributed.cp_utils import (
    _cp_offset_causal_mask,
    flex_cp_allgather,
    ulysses_head_to_seq,
    ulysses_seq_to_head,
)
from hyper_parallel.auto_models.components.distributed.injection import inner_wrapper
from hyper_parallel.auto_models.components.model_transform import module_replacement
from hyper_parallel.auto_models.components.models.qwen3_moe_attention_common import (
    fused_rms_norm as _fused_rms_norm,
    run_qwen3_moe_flash_attention as _run_qwen3_moe_flash_attention,
)


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


def qwen3_moe_fused_rms_norm_forward(
    module: nn.Module,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Replace ``Qwen3MoeRMSNorm.forward`` with fused NPU RMSNorm."""
    return _fused_rms_norm(hidden_states, module.weight, module.variance_epsilon)


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


def _validate_qwen3_moe_flash_attention_cp_target(
    target_module: nn.Module,
    cp_mesh: Any,
    wrapper_name: str,
) -> Callable:
    """Validate a Qwen3-MoE fused CP wrapper target and return its forward."""
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError(f"{wrapper_name} requires an active CP mesh")

    original_forward = target_module.forward
    installed_forward = getattr(original_forward, "__func__", original_forward)
    if installed_forward is not qwen3_moe_flash_attention_forward:
        raise ValueError(
            f"{wrapper_name} can only replace "
            "qwen3_moe_flash_attention_forward; apply "
            "replace_qwen3_moe_flash_attention first"
        )
    return original_forward


def _validate_qwen3_moe_flash_attention_ulysses_heads(
    target_module: nn.Module,
    cp_size: int,
    wrapper_name: str,
) -> None:
    """Validate the Qwen3-MoE head counts required by Pure Ulysses."""
    config = getattr(target_module, "config", None)
    if config is None:
        raise ValueError(f"{wrapper_name} requires target_module.config")

    for name in ("num_attention_heads", "num_key_value_heads"):
        count = getattr(config, name, None)
        if count is None:
            count = getattr(target_module, name, None)
        if count is None:
            raise ValueError(
                f"{wrapper_name} requires {name} in target_module.config"
            )
        count = int(count)
        if count % cp_size:
            raise ValueError(
                f"{wrapper_name} requires {name} ({count}) to be divisible "
                f"by Ulysses degree ({cp_size})"
            )


def _prepare_qwen3_moe_flash_attention_ulysses_mask(
    attention_mask: torch.Tensor | None,
    query_length: int,
    key_length: int,
) -> torch.Tensor | None:
    """Validate an external mask after Ulysses has restored global sequence."""
    if attention_mask is None:
        return None
    if attention_mask.ndim < 2:
        raise ValueError(
            "Qwen3-MoE fused Ulysses attention_mask must include query and key "
            "dimensions"
        )
    if attention_mask.shape[-1] != key_length:
        raise ValueError(
            "Qwen3-MoE fused Ulysses attention_mask must cover the global key "
            f"sequence: mask key length={attention_mask.shape[-1]}, "
            f"expected {key_length}"
        )

    mask_query_length = attention_mask.shape[-2]
    if mask_query_length < query_length:
        raise ValueError(
            "Qwen3-MoE fused Ulysses attention_mask must cover the global query "
            f"sequence: mask query length={mask_query_length}, expected at least "
            f"{query_length}"
        )
    if mask_query_length != query_length:
        attention_mask = attention_mask.narrow(-2, 0, query_length)
    return attention_mask


def _prepare_qwen3_moe_flash_attention_cp_mask(
    attention_mask: torch.Tensor | None,
    query_length: int,
    key_length: int,
    query_offset: int,
    device: torch.device,
    allow_external_mask: bool,
) -> torch.Tensor:
    """Build an implicit CP mask or slice an external global allowed mask."""
    if attention_mask is None:
        causal_mask = _cp_offset_causal_mask(
            query_length,
            key_length,
            query_offset,
            device,
        )
        return causal_mask

    if not allow_external_mask:
        raise ValueError(
            "Qwen3-MoE fused CP attention requires an implicit causal mask; "
            "configure create_attention_mask_in_dataloader=false or use "
            "qwen3_moe_flash_attention_cp_mask_wrapper"
        )
    if attention_mask.ndim < 2:
        raise ValueError(
            "Qwen3-MoE fused CP attention_mask must include query and key dimensions"
        )
    if attention_mask.shape[-1] != key_length:
        raise ValueError(
            "Qwen3-MoE fused CP attention_mask must cover the global key sequence: "
            f"mask key length={attention_mask.shape[-1]}, expected {key_length}"
        )

    mask_query_length = attention_mask.shape[-2]
    if mask_query_length != query_length:
        query_end = query_offset + query_length
        if mask_query_length < query_end:
            raise ValueError(
                "Qwen3-MoE fused CP attention_mask does not cover this rank's query "
                f"range [{query_offset}, {query_end})"
            )
        attention_mask = attention_mask.narrow(
            -2,
            query_offset,
            query_length,
        )

    return attention_mask


def _run_qwen3_moe_flash_attention_cp(
    target_module: nn.Module,
    cp_mesh: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any | None,
    allow_external_mask: bool,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the common Qwen3-MoE fused CP attention implementation."""
    if attention_mask is not None and not allow_external_mask:
        raise ValueError(
            "Qwen3-MoE fused CP attention requires an implicit causal mask; "
            "configure create_attention_mask_in_dataloader=false or use "
            "qwen3_moe_flash_attention_cp_mask_wrapper"
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
    cp_attention_mask = _prepare_qwen3_moe_flash_attention_cp_mask(
        attention_mask,
        query_length,
        key_states.shape[-2],
        query_offset,
        query_states.device,
        allow_external_mask,
    )
    attention_output, attention_weights = _run_qwen3_moe_flash_attention(
        target_module,
        query_states,
        key_states,
        value_states,
        cp_attention_mask,
        dropout=(
            0.0
            if not target_module.training
            else target_module.attention_dropout
        ),
        scaling=target_module.scaling,
        sliding_window=target_module.sliding_window,
        **kwargs,
    )
    output = _project_qwen3_moe_attention_output(
        target_module,
        attention_output,
        input_shape,
    )
    return output, attention_weights


def _run_qwen3_moe_flash_attention_ulysses(
    target_module: nn.Module,
    cp_mesh: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any | None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run fused Qwen3-MoE attention with synchronous Pure Ulysses A2A."""
    input_shape, query_states, key_states, value_states = (
        _prepare_qwen3_moe_attention_states(
            target_module,
            hidden_states,
            position_embeddings,
            past_key_values,
        )
    )
    cp_size = cp_mesh.size()
    for name, states in (
        ("query", query_states),
        ("key", key_states),
        ("value", value_states),
    ):
        if states.shape[1] % cp_size:
            raise ValueError(
                f"{name} head count ({states.shape[1]}) must be divisible by "
                f"Ulysses degree ({cp_size})"
            )

    query_states, key_states, value_states = (
        ulysses_seq_to_head(states, 2, 1, cp_mesh)
        for states in (query_states, key_states, value_states)
    )
    global_query_length = query_states.shape[2]
    global_key_length = key_states.shape[2]
    ulysses_attention_mask = _prepare_qwen3_moe_flash_attention_ulysses_mask(
        attention_mask,
        global_query_length,
        global_key_length,
    )
    attention_output, attention_weights = _run_qwen3_moe_flash_attention(
        target_module,
        query_states,
        key_states,
        value_states,
        ulysses_attention_mask,
        dropout=(
            0.0
            if not target_module.training
            else target_module.attention_dropout
        ),
        scaling=target_module.scaling,
        sliding_window=target_module.sliding_window,
        **kwargs,
    )
    # The fused kernel returns BSHD, whereas the projections use BHSD.
    attention_output = ulysses_head_to_seq(
        attention_output,
        1,
        2,
        cp_mesh,
    )
    output = _project_qwen3_moe_attention_output(
        target_module,
        attention_output,
        input_shape,
    )
    return output, attention_weights


def _install_qwen3_moe_flash_attention_cp_forward(
    target_module: nn.Module,
    cp_mesh: Any,
    wrapper_name: str,
    allow_external_mask: bool,
) -> None:
    """Install the shared fused CP forward with the requested mask contract."""
    original_forward = _validate_qwen3_moe_flash_attention_cp_target(
        target_module,
        cp_mesh,
        wrapper_name,
    )

    @wraps(original_forward)
    def cp_forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Dispatch the model forward contract to the shared fused CP path."""
        return _run_qwen3_moe_flash_attention_cp(
            target_module,
            cp_mesh,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            allow_external_mask,
            **kwargs,
        )

    target_module.forward = cp_forward


@inner_wrapper
def qwen3_moe_flash_attention_cp_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
) -> None:
    """Install Qwen3-MoE fused CP attention with an implicit causal mask."""
    del mesh, tp_mesh, ep_mesh
    _install_qwen3_moe_flash_attention_cp_forward(
        target_module,
        cp_mesh,
        "qwen3_moe_flash_attention_cp_wrapper",
        allow_external_mask=False,
    )


@inner_wrapper
def qwen3_moe_flash_attention_cp_mask_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
) -> None:
    """Install Qwen3-MoE fused CP attention that accepts a global block mask."""
    del mesh, tp_mesh, ep_mesh
    _install_qwen3_moe_flash_attention_cp_forward(
        target_module,
        cp_mesh,
        "qwen3_moe_flash_attention_cp_mask_wrapper",
        allow_external_mask=True,
    )


@inner_wrapper
def qwen3_moe_flash_attention_ulysses_cp_wrapper(
    target_module: nn.Module,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
) -> None:
    """Install Qwen3-MoE fused attention with synchronous Pure Ulysses A2A.

    Args:
        target_module: Qwen3-MoE attention module whose forward is replaced.
        mesh: Active model mesh supplied by the injection framework.
        tp_mesh: Active tensor-parallel mesh, if configured.
        cp_mesh: Active context-parallel mesh used for Ulysses all-to-all.
        ep_mesh: Active expert-parallel mesh, if configured.
    """
    del mesh, tp_mesh, ep_mesh
    wrapper_name = "qwen3_moe_flash_attention_ulysses_cp_wrapper"
    if cp_mesh is None or cp_mesh.size() <= 1:
        raise ValueError(f"{wrapper_name} requires an active CP mesh")
    _validate_qwen3_moe_flash_attention_ulysses_heads(
        target_module,
        cp_mesh.size(),
        wrapper_name,
    )
    original_forward = _validate_qwen3_moe_flash_attention_cp_target(
        target_module,
        cp_mesh,
        wrapper_name,
    )

    @wraps(original_forward)
    def cp_forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the original attention signature through Pure Ulysses.

        Args:
            hidden_states: Local sequence shard hidden states.
            position_embeddings: Cosine and sine rotary embedding tensors.
            attention_mask: Optional mask covering the global sequence.
            past_key_values: Optional model cache passed through to projection.
            **kwargs: Additional fused attention arguments.
        """
        return _run_qwen3_moe_flash_attention_ulysses(
            target_module,
            cp_mesh,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            **kwargs,
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
