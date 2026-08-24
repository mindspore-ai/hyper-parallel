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
"""Attention activation swap support for Hugging Face models."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

from hyper_parallel.core.activation_checkpoint import CheckpointPolicy, SwapManager, swap_wrapper

logger = logging.getLogger(__name__)

_MIN_SWAP_TENSOR_BYTES = 1024 * 1024


@dataclass
class _AttentionTarget:
    """One attention module and every parent attribute that references it."""

    module: nn.Module
    references: list[tuple[nn.Module, str]] = field(default_factory=list)


def attention_swap_policy(tensor: torch.Tensor) -> CheckpointPolicy:
    """Select large, independently-owned attention activations for swapping.

    Args:
        tensor: Tensor saved by autograd inside an attention forward pass.

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


def _record_attention_target(
    targets_by_id: dict[int, _AttentionTarget],
    parent: nn.Module,
    child_name: str,
    child: nn.Module,
) -> None:
    """Record a replaceable reference while deduplicating shared modules."""
    target = targets_by_id.setdefault(id(child), _AttentionTarget(child))
    target.references.append((parent, child_name))


def _find_attention_targets(model: nn.Module) -> list[_AttentionTarget]:
    """Find replaceable ``self_attn`` modules without assuming a layer path."""
    targets_by_id: dict[int, _AttentionTarget] = {}
    visited_modules: set[int] = set()

    def visit(parent: nn.Module) -> None:
        """Recursively record ``self_attn`` children, deduplicating shared modules.

        Args:
            parent: Module whose children are inspected.
        """
        parent_id = id(parent)
        if parent_id in visited_modules:
            return
        visited_modules.add(parent_id)

        for child_name, child in parent.named_children():
            if child_name == "self_attn":
                _record_attention_target(targets_by_id, parent, child_name, child)
                continue
            visit(child)

    visit(model)
    targets = list(targets_by_id.values())
    if not targets:
        raise ValueError(
            f"activation_swap='attention' found no self_attn modules in {type(model).__name__}; "
            "expected attention modules registered as self_attn"
        )
    return targets


def apply_attention_swap(model: nn.Module, activation_swap: str) -> nn.Module:
    """Wrap attention modules and install model-order swap scheduling.

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
    attention_targets = _find_attention_targets(model)
    wrapped_attentions = []
    for target in attention_targets:
        attention = target.module
        wrapped_attention = swap_wrapper(
            attention,
            policy_fn=attention_swap_policy,
            group_swap=True,
        )
        for parent, child_name in target.references:
            setattr(parent, child_name, wrapped_attention)
        wrapped_attentions.append(wrapped_attention)

    swap_manager = SwapManager()
    for current_attention, next_attention in zip(wrapped_attentions, wrapped_attentions[1:]):
        swap_manager.set_forward_prefetch_layer(current_attention, next_attention)

    logger.info(
        "Enabled attention activation swap for %d module(s) in %s",
        len(wrapped_attentions),
        type(model).__name__,
    )
    return model


__all__ = [
    "apply_attention_swap",
    "attention_swap_policy",
    "validate_attention_swap",
]
