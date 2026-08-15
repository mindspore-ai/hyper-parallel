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
"""Attention activation swap support for Hugging Face Qwen3-MoE models."""

import logging
from typing import Any, Optional

import torch
from torch import nn

from hyper_parallel.core.activation_checkpoint import CheckpointPolicy, SwapManager, swap_wrapper
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.platform import PlatformType

logger = logging.getLogger(__name__)
platform = get_platform()

_MIN_SWAP_TENSOR_BYTES = 1024 * 1024


def _qwen3_types() -> tuple[type[nn.Module], type[nn.Module], type[nn.Module]]:
    """Load Qwen3 classes only when the opt-in feature is requested."""
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (  # pylint: disable=C0415
            Qwen3MoeAttention,
            Qwen3MoeDecoderLayer,
            Qwen3MoeForCausalLM,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise ValueError(
            "activation_swap='attention' requires a Transformers build with Qwen3-MoE support"
        ) from exc
    return Qwen3MoeForCausalLM, Qwen3MoeDecoderLayer, Qwen3MoeAttention


def qwen3_attention_swap_policy(tensor: torch.Tensor) -> CheckpointPolicy:
    """Select large, independently-owned attention activations for swapping.

    Args:
        tensor: Tensor saved by autograd inside a Qwen3-MoE attention forward.

    Returns:
        ``MUST_SWAP`` for tensors that are safe and worthwhile to transfer;
        otherwise ``MUST_SAVE``.
    """
    if not tensor.requires_grad or tensor.dim() < 2:
        return CheckpointPolicy.MUST_SAVE
    storage_bytes = tensor.untyped_storage().size()
    tensor_bytes = tensor.numel() * tensor.element_size()
    if storage_bytes != tensor_bytes or tensor_bytes < _MIN_SWAP_TENSOR_BYTES:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.MUST_SWAP


def validate_attention_swap(
    activation_swap: str,
    *,
    activation_checkpoint: Optional[str] = None,
    enable_compile: bool = False,
    pp_size: int = 1,
) -> None:
    """Validate attention swap mode and incompatible model-build features.

    Args:
        activation_swap: Requested activation swap mode.
        activation_checkpoint: Activation recomputation mode.
        enable_compile: Whether graph compilation is enabled.
        pp_size: Configured pipeline-parallel world size.

    Raises:
        ValueError: If the mode is invalid or an unsupported combination is enabled.
    """
    if activation_swap not in ("none", "attention"):
        raise ValueError(
            "activation_swap must be one of ('none', 'attention'), "
            f"got {activation_swap!r}"
        )
    if activation_swap == "none":
        return
    if enable_compile:
        raise ValueError("activation_swap='attention' is incompatible with torch.compile")
    if activation_checkpoint not in (None, "off"):
        raise ValueError("activation_swap='attention' is incompatible with activation checkpointing")
    if pp_size != 1:
        raise ValueError("activation_swap='attention' does not support pipeline parallelism")


def _find_qwen3_moe_attentions(model: nn.Module) -> list[Any]:
    """Validate the supported HF model structure and return local attentions."""
    qwen3_model_type, qwen3_layer_type, qwen3_attention_type = _qwen3_types()
    if type(model) is not qwen3_model_type:
        raise ValueError(
            "activation_swap='attention' only supports Hugging Face Qwen3MoeForCausalLM for now, "
            f"got {type(model).__name__}"
        )
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not isinstance(layers, nn.ModuleList) or len(layers) != model.config.num_hidden_layers:
        raise ValueError(
            "activation_swap='attention' requires model.layers to contain exactly "
            f"{model.config.num_hidden_layers} Qwen3MoeDecoderLayer instances"
        )

    attentions = []
    for layer_index, layer in enumerate(layers):
        if type(layer) is not qwen3_layer_type:
            raise ValueError(
                "activation_swap='attention' requires every model.layers entry to be "
                f"Qwen3MoeDecoderLayer, got {type(layer).__name__} at index {layer_index}"
            )
        attention = getattr(layer, "self_attn", None)
        if type(attention) is not qwen3_attention_type:
            raise ValueError(
                "activation_swap='attention' requires every decoder layer self_attn to be "
                f"Qwen3MoeAttention, got {type(attention).__name__} at index {layer_index}"
            )
        attentions.append(attention)
    return attentions


def apply_qwen3_moe_attention_swap(model: nn.Module, activation_swap: str) -> nn.Module:
    """Wrap Qwen3-MoE attention modules and install layer-wise swap scheduling.

    Args:
        model: Model being prepared, before sharding and FSDP wrapping.
        activation_swap: Requested activation swap mode.

    Returns:
        The input model, patched in place when attention swapping is enabled.
    """
    if activation_swap != "attention":
        raise ValueError(
            "activation_swap must be one of ('none', 'attention'), "
            f"got {activation_swap!r}"
        )
    attentions = _find_qwen3_moe_attentions(model)
    wrapped_attentions = []
    for layer, attention in zip(model.model.layers, attentions):
        wrapped_attention = swap_wrapper(
            attention,
            policy_fn=qwen3_attention_swap_policy,
            group_swap=True,
        )
        layer.self_attn = wrapped_attention
        wrapped_attentions.append(wrapped_attention)

    swap_manager = SwapManager()
    for current_attention, next_attention in zip(wrapped_attentions, wrapped_attentions[1:]):
        swap_manager.set_forward_prefetch_layer(current_attention, next_attention)

    logger.info("Enabled attention activation swap for %d Qwen3-MoE layers", len(wrapped_attentions))
    return model


__all__ = [
    "apply_qwen3_moe_attention_swap",
    "qwen3_attention_swap_policy",
    "validate_attention_swap",
]
