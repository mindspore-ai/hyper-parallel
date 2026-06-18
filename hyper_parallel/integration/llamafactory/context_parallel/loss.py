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
"""Context-parallel loss helpers for HuggingFace causal LM models."""
from __future__ import annotations

from functools import wraps
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .inputs import get_cp_group, get_cp_rank


def _build_cp_shift_labels(
    labels: torch.Tensor,
    local_seq_len: int,
    cp_rank: int,
    ignore_index: int,
) -> torch.Tensor:
    """Build this CP rank's shifted labels from the full unsharded label sequence."""
    start = cp_rank * local_seq_len
    end = start + local_seq_len
    shift_labels = labels[..., start + 1 : min(end + 1, labels.size(-1))]
    if shift_labels.size(-1) < local_seq_len:
        pad_shape = (*shift_labels.shape[:-1], local_seq_len - shift_labels.size(-1))
        shift_labels = torch.cat((shift_labels, shift_labels.new_full(pad_shape, ignore_index)), dim=-1)
    return shift_labels.contiguous()


def _num_items_in_batch(
    local_shift_labels: torch.Tensor,
    ignore_index: int,
    group=None,
) -> torch.Tensor:
    """Count valid shifted labels across the requested process group."""
    local_tokens = local_shift_labels.ne(ignore_index).sum()
    if not dist.is_available() or not dist.is_initialized():
        return local_tokens
    tokens = local_tokens.clone()
    dist.all_reduce(tokens, group=group)
    return tokens


def _local_per_token_cross_entropy(
    logits: torch.Tensor,
    shift_labels: torch.Tensor,
    vocab_size: int,
    ignore_index: int,
) -> torch.Tensor:
    """Return unreduced per-token Causal LM CE for this CP rank's local logits."""
    local_logits = logits[..., :vocab_size].contiguous().float()
    local_labels = shift_labels.to(local_logits.device).contiguous()
    flat_loss = F.cross_entropy(
        local_logits.view(-1, vocab_size),
        local_labels.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    )
    return flat_loss.view_as(local_labels)


def _group_loss_for_logging(local_loss_sum: torch.Tensor, cp_num_items: torch.Tensor, cp_group=None) -> torch.Tensor:
    """Return the CP-group token mean to match per-DP-rank Trainer loss logging."""
    if not dist.is_available() or not dist.is_initialized():
        return local_loss_sum / torch.clamp(cp_num_items.to(local_loss_sum.device), min=1)

    group_loss_sum = local_loss_sum.detach().clone()
    dist.all_reduce(group_loss_sum, group=cp_group)
    return group_loss_sum / torch.clamp(cp_num_items.to(group_loss_sum.device), min=1)


def _wrap_loss_function(original_loss_function, cp_rank: int, cp_size: int, cp_group=None):
    """Patch only token alignment/normalization before delegating to the original loss."""

    @wraps(original_loss_function)
    def _hp_cp_loss_function(
        logits,
        labels,
        vocab_size: int,
        num_items_in_batch: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        shift_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if (
            shift_labels is not None
            or cp_size <= 1
            or not isinstance(logits, torch.Tensor)
            or not isinstance(labels, torch.Tensor)
            or logits.dim() < 3
            or labels.dim() < 2
        ):
            return original_loss_function(
                logits=logits,
                labels=labels,
                vocab_size=vocab_size,
                num_items_in_batch=num_items_in_batch,
                ignore_index=ignore_index,
                shift_labels=shift_labels,
                **kwargs,
            )

        local_seq_len = logits.size(-2)
        if local_seq_len * cp_size != labels.size(-1):
            return original_loss_function(
                logits=logits,
                labels=labels,
                vocab_size=vocab_size,
                num_items_in_batch=num_items_in_batch,
                ignore_index=ignore_index,
                shift_labels=shift_labels,
                **kwargs,
            )

        cp_shift_labels = _build_cp_shift_labels(labels, local_seq_len, cp_rank, ignore_index)
        # Fully_shard uses a flattened (DP * CP) mesh, so its gradient reduction
        # averages over CP peers as well. Scale each local token-loss shard by
        # cp_size to recover the same per-DP-sample gradient as full-sequence CE
        # without a post-backward CP gradient all-reduce.
        cp_num_items = _num_items_in_batch(cp_shift_labels, ignore_index, group=cp_group).to(logits.device)
        per_token_loss = _local_per_token_cross_entropy(logits, cp_shift_labels, vocab_size, ignore_index)
        local_loss_sum = per_token_loss.sum()
        backward_loss = local_loss_sum * cp_size / torch.clamp(cp_num_items, min=1)
        cp_group_token_mean = _group_loss_for_logging(local_loss_sum, cp_num_items, cp_group=cp_group)
        return backward_loss + (cp_group_token_mean - backward_loss).detach()

    return _hp_cp_loss_function


def _enable_context_parallel_loss_patch(model: nn.Module, hp_args) -> None:
    """Patch model-owned Causal LM loss functions for CP token alignment."""
    cp_size = getattr(hp_args, "cp_size", 1)
    if cp_size <= 1:
        return
    cp_rank = get_cp_rank(hp_args)
    cp_group = get_cp_group(hp_args)
    for module in model.modules():
        if getattr(module, "_hp_cp_loss_enabled", False) or not hasattr(module, "loss_function"):
            continue
        try:
            original_loss_function = module.loss_function
            module.loss_function = _wrap_loss_function(original_loss_function, cp_rank, cp_size, cp_group=cp_group)
        except (AttributeError, TypeError):
            continue
        module._hp_cp_loss_enabled = True  # pylint: disable=protected-access
