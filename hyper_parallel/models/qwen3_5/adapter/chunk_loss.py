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
"""Model-integrated chunked CE for Qwen3.5 dense and MoE text decoders."""

import functools
from typing import Any

# This adapter follows the native PyTorch Transformers model contract.
import torch
from torch import nn

from hyper_parallel.components.losses.chunked_cross_entropy import ChunkedCausalLMOutput, chunked_cross_entropy
from hyper_parallel.distributed._builder.forward_rewriter import (
    _ForwardRewriteRequest,
    _commit_forward_rewrite,
)


def _chunk_loss_forward(
    model: nn.Module,
    *,
    chunk_loss_targets: torch.Tensor,
    chunk_loss_mask: torch.Tensor | None,
    chunk_loss_chunk_size: int,
    chunk_loss_ignore_index: int,
    past_key_values: Any = None,
    use_cache: bool | None = None,
    output_router_logits: bool | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs: Any,
) -> ChunkedCausalLMOutput:
    """Keep CP-local hidden states and pre-shifted targets in the same order."""
    if past_key_values is not None or use_cache:
        raise ValueError("Qwen3.5 Chunk Loss training does not support KV cache")
    if isinstance(logits_to_keep, torch.Tensor) or logits_to_keep != 0:
        raise ValueError("Qwen3.5 Chunk Loss requires logits_to_keep=0")
    if kwargs.get("return_dict") is False:
        raise ValueError("Qwen3.5 Chunk Loss requires return_dict=True")
    is_moe = model.config.model_type == "qwen3_5_moe_text"
    if is_moe:
        output_router_logits = (
            output_router_logits if output_router_logits is not None else model.config.output_router_logits
        )
        kwargs["output_router_logits"] = output_router_logits
    outputs = model.model(past_key_values=None, use_cache=False, **kwargs)
    targets = chunk_loss_targets
    if chunk_loss_mask is not None:
        if chunk_loss_mask.shape != targets.shape:
            raise ValueError("Qwen3.5 Chunk Loss mask must match targets")
        targets = targets.masked_fill(~chunk_loss_mask.bool(), chunk_loss_ignore_index)
    # Targets have already been shifted globally before CP slicing. Never shift
    # again at either a CP boundary or a loss-chunk boundary.
    loss_sum = chunked_cross_entropy(
        outputs.last_hidden_state,
        targets,
        model.lm_head.weight,
        chunk_size=chunk_loss_chunk_size,
        ignore_index=chunk_loss_ignore_index,
    )
    aux_loss = None
    if is_moe and output_router_logits:
        # Keep optional MoE model imports out of dense-only registry discovery.
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # pylint: disable=C0415
            load_balancing_loss_func,
        )

        aux_loss = load_balancing_loss_func(
            outputs.router_logits, model.num_experts, model.num_experts_per_tok, kwargs.get("attention_mask"),
        )
    return ChunkedCausalLMOutput(
        loss_sum=loss_sum,
        valid_token_count=targets.ne(chunk_loss_ignore_index).sum(),
        aux_loss=aux_loss,
        aux_loss_coef=float(getattr(model, "router_aux_loss_coef", 0.0)),
        logits=None,
        hidden_states=getattr(outputs, "hidden_states", None),
        attentions=getattr(outputs, "attentions", None),
        router_logits=getattr(outputs, "router_logits", None),
    )


def bind_chunk_loss(model: nn.Module) -> None:
    """Bind the text-only training objective without changing inference forward."""
    if getattr(model, "_hp_qwen3_5_chunk_loss_bound", False):
        return
    if getattr(getattr(model, "config", None), "model_type", None) not in ("qwen3_5_text", "qwen3_5_moe_text"):
        raise TypeError("Qwen3.5 Chunk Loss requires a dense or MoE text model_type")
    if not isinstance(getattr(model, "model", None), nn.Module):
        raise TypeError("Qwen3.5 Chunk Loss requires a decoder at model.model")
    head = getattr(model, "lm_head", None)
    if not isinstance(head, nn.Linear) or head.bias is not None:
        raise TypeError("Qwen3.5 Chunk Loss requires a bias-free nn.Linear lm_head")
    original_forward = model.forward

    @functools.wraps(original_forward)
    def forward(*args: Any, **kwargs: Any) -> Any:
        if "chunk_loss_targets" not in kwargs:
            return original_forward(*args, **kwargs)
        if args:
            raise TypeError("Qwen3.5 Chunk Loss requires keyword model inputs")
        return _chunk_loss_forward(model, **kwargs)

    _commit_forward_rewrite(_ForwardRewriteRequest(
        model, forward, companion_attrs={"_hp_qwen3_5_chunk_loss_bound": True},
    ))
