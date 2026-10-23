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
"""Opt-in chunked CE for the Qwen image-text conditional-generation model."""

from __future__ import annotations

from functools import wraps
from typing import Any

# This adapter implements the Torch-only Transformers multimodal contract.
import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.losses.chunked_cross_entropy import ChunkedCausalLMOutput, chunked_cross_entropy
from hyper_parallel.distributed._builder.forward_rewriter import _ForwardRewriteRequest, _commit_forward_rewrite


def _chunk_loss_forward(
    model: nn.Module, *, chunk_loss_targets: torch.Tensor, chunk_loss_mask: torch.Tensor | None,
    chunk_loss_chunk_size: int, chunk_loss_ignore_index: int, past_key_values: Any = None,
    use_cache: bool | None = None, logits_to_keep: int | torch.Tensor = 0,
    output_router_logits: bool | None = None, **kwargs: Any,
) -> ChunkedCausalLMOutput:
    """Run the multimodal base model without materializing full vocabulary logits."""
    if past_key_values is not None or use_cache:
        raise ValueError("Qwen VLM Chunk Loss does not support KV cache")
    if isinstance(logits_to_keep, torch.Tensor) or logits_to_keep != 0:
        raise ValueError("Qwen VLM Chunk Loss requires logits_to_keep=0")
    if kwargs.get("return_dict") is False:
        raise ValueError("Qwen VLM Chunk Loss requires return_dict=True")
    router_enabled = (output_router_logits if output_router_logits is not None
                      else getattr(model.config.text_config, "output_router_logits", False))
    if router_enabled:
        raise NotImplementedError("Qwen VLM Chunk Loss does not yet aggregate router auxiliary loss")
    kwargs.pop("labels", None)
    outputs = model.model(past_key_values=None, use_cache=False, output_router_logits=False, **kwargs)
    targets = chunk_loss_targets
    if chunk_loss_mask is not None:
        if chunk_loss_mask.shape != targets.shape:
            raise ValueError("Qwen VLM Chunk Loss mask must match targets")
        targets = targets.masked_fill(~chunk_loss_mask.bool(), chunk_loss_ignore_index)
    # The batch adapter shifts globally before CP slicing; chunks never shift again.
    loss_sum = chunked_cross_entropy(
        outputs.last_hidden_state, targets, model.lm_head.weight,
        chunk_size=chunk_loss_chunk_size, ignore_index=chunk_loss_ignore_index,
    )
    return ChunkedCausalLMOutput(
        loss_sum=loss_sum, valid_token_count=targets.ne(chunk_loss_ignore_index).sum(), logits=None,
        hidden_states=getattr(outputs, "hidden_states", None), attentions=getattr(outputs, "attentions", None),
    )


def bind_chunk_loss(model: nn.Module) -> None:
    """Bind only the VLM training protocol, preserving ordinary inference forwards.

    Args:
        model: Qwen multimodal conditional-generation model to adapt.
    """
    if getattr(model, "_hp_qwen_vlm_chunk_loss_bound", False):
        return
    if getattr(getattr(model, "config", None), "model_type", None) != "qwen3_5_moe":
        raise TypeError("Qwen VLM Chunk Loss requires model_type='qwen3_5_moe'")
    if not isinstance(getattr(model, "model", None), nn.Module):
        raise TypeError("Qwen VLM Chunk Loss requires the multimodal base at model.model")
    head = getattr(model, "lm_head", None)
    if not isinstance(head, nn.Linear) or head.bias is not None:
        raise TypeError("Qwen VLM Chunk Loss requires a bias-free nn.Linear lm_head")
    original_forward = model.forward

    @wraps(original_forward)
    def forward(*args: Any, **kwargs: Any) -> Any:
        """Route training through chunked CE and preserve other forward calls."""
        if "chunk_loss_targets" not in kwargs or "vlm_cp_inputs" in kwargs:
            return original_forward(*args, **kwargs)
        if args:
            raise TypeError("Qwen VLM Chunk Loss requires keyword model inputs")
        return _chunk_loss_forward(model, **kwargs)

    _commit_forward_rewrite(_ForwardRewriteRequest(
        model, forward, companion_attrs={"_hp_qwen_vlm_chunk_loss_bound": True},
    ))
