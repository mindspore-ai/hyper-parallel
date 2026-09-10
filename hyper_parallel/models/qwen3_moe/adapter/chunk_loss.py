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
"""Qwen3-MoE adapter for model-integrated Chunk Loss training."""

from __future__ import annotations

import functools
from typing import Any

# This family adapter targets the PyTorch-native Transformers Qwen3-MoE
# implementation and intentionally shares its tensor/module contracts.
# pylint: disable=forbidden-backend-import
import torch
from torch import nn

from hyper_parallel.components.losses.chunked_cross_entropy import (
    ChunkedCausalLMOutput,
    chunked_cross_entropy,
)
from hyper_parallel.distributed._builder.forward_rewriter import (  # pylint: disable=protected-access
    _ForwardRewriteRequest,
    _commit_forward_rewrite,
)


def _align_chunk_loss_inputs(
    hidden_states: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate pre-shifted targets and apply the optional loss mask."""
    if targets.dim() != 2 or hidden_states.shape[:2] != targets.shape:
        raise ValueError(
            "Qwen3-MoE Chunk Loss targets must match final hidden batch/sequence "
            f"dimensions, got hidden={tuple(hidden_states.shape)} and "
            f"targets={tuple(targets.shape)}"
        )
    if loss_mask is not None and loss_mask.shape != targets.shape:
        raise ValueError("Qwen3-MoE Chunk Loss mask must match targets")

    if loss_mask is not None:
        targets = targets.masked_fill(
            loss_mask.to(torch.bool).logical_not(),
            ignore_index,
        )
    return hidden_states, targets


def _qwen3_moe_chunk_loss_forward(
    model: nn.Module,
    *,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Any | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    output_router_logits: bool | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    chunk_loss_targets: torch.Tensor,
    chunk_loss_mask: torch.Tensor | None,
    chunk_loss_chunk_size: int,
    chunk_loss_ignore_index: int,
    **kwargs: Any,
) -> ChunkedCausalLMOutput:
    """Run Qwen3-MoE through final hidden states and skip full logits."""
    if past_key_values is not None or use_cache:
        raise ValueError("Qwen3-MoE Chunk Loss training does not support KV cache")
    if isinstance(logits_to_keep, torch.Tensor) or logits_to_keep != 0:
        raise ValueError("Qwen3-MoE Chunk Loss requires logits_to_keep=0")
    if kwargs.get("return_dict") is False:
        raise ValueError("Qwen3-MoE Chunk Loss requires return_dict=True")

    output_router_logits = (
        output_router_logits
        if output_router_logits is not None
        else bool(getattr(model.config, "output_router_logits", False))
    )
    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=inputs_embeds,
        use_cache=False,
        output_router_logits=output_router_logits,
        **kwargs,
    )
    hidden_states = outputs.last_hidden_state
    aligned_hidden, aligned_targets = _align_chunk_loss_inputs(
        hidden_states,
        chunk_loss_targets,
        chunk_loss_mask,
        chunk_loss_ignore_index,
    )
    loss_sum = chunked_cross_entropy(
        aligned_hidden,
        aligned_targets,
        model.lm_head.weight,
        chunk_size=chunk_loss_chunk_size,
        ignore_index=chunk_loss_ignore_index,
    )
    valid_token_count = aligned_targets.ne(chunk_loss_ignore_index).sum()

    aux_loss = None
    if output_router_logits:
        # Keep the concrete Transformers family import lazy so registry
        # discovery and CPU-only package import do not load accelerator hooks.
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (  # pylint: disable=import-outside-toplevel
            load_balancing_loss_func,
        )

        aux_loss = load_balancing_loss_func(
            outputs.router_logits,
            model.num_experts,
            model.num_experts_per_tok,
            attention_mask,
        )

    return ChunkedCausalLMOutput(
        loss_sum=loss_sum,
        valid_token_count=valid_token_count,
        aux_loss=aux_loss,
        aux_loss_coef=float(getattr(model, "router_aux_loss_coef", 0.0)),
        logits=None,
        past_key_values=getattr(outputs, "past_key_values", None),
        hidden_states=getattr(outputs, "hidden_states", None),
        attentions=getattr(outputs, "attentions", None),
        router_logits=getattr(outputs, "router_logits", None),
    )


def bind_chunk_loss(model: nn.Module) -> None:
    """Atomically install the Qwen3-MoE Chunk Loss training forward."""
    if getattr(model, "_hp_qwen3_moe_chunk_loss_bound", False):
        return
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type != "qwen3_moe":
        raise TypeError(
            "Qwen3-MoE Chunk Loss adapter requires config.model_type='qwen3_moe', "
            f"got {model_type!r}"
        )
    if not isinstance(getattr(model, "model", None), nn.Module):
        raise TypeError("Qwen3-MoE Chunk Loss adapter requires a decoder module at model.model")
    lm_head = getattr(model, "lm_head", None)
    if not isinstance(lm_head, nn.Linear) or lm_head.bias is not None:
        raise TypeError("Qwen3-MoE Chunk Loss adapter requires a bias-free nn.Linear lm_head")

    original_forward = model.forward

    @functools.wraps(original_forward)
    def _chunk_loss_aware_forward(*args: Any, **kwargs: Any) -> Any:
        if "chunk_loss_targets" not in kwargs:
            return original_forward(*args, **kwargs)
        if args:
            raise TypeError(
                "Qwen3-MoE Chunk Loss Trainer integration passes model inputs by keyword"
            )
        return _qwen3_moe_chunk_loss_forward(model, **kwargs)

    _commit_forward_rewrite(
        _ForwardRewriteRequest(
            model,
            _chunk_loss_aware_forward,
            companion_attrs={"_hp_qwen3_moe_chunk_loss_bound": True},
        )
    )


__all__ = ["bind_chunk_loss"]
