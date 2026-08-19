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
"""Transformers-compatible causal language-model loss for sharded logits."""

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.tensor_parallel.loss_parallel import (
    _get_loss_parallel_mesh,
    loss_parallel,
)
from hyper_models.components.loss.loss_parallel_ops import distributed_cross_entropy

_LOSS_PARALLEL_CHUNK_TOKENS = 128


def causal_lm_loss_parallel(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Compute causal LM loss without flattening a vocab-sharded DTensor.

    Transformers normally reshapes logits to ``[-1, global_vocab_size]``
    before cross entropy. A DTensor already stores only its local vocabulary
    shard, so that reshape would mix token and vocabulary axes. The distributed
    cross-entropy dispatcher accepts the original rank-3 tensor and performs
    the local flatten after reading its placements.

    Args:
        logits: Rank-3 logits whose final dimension may be vocabulary sharded.
        labels: Local labels aligned with the local sequence shard.
        vocab_size: Global vocabulary size retained for Transformers API parity.
        num_items_in_batch: Optional denominator for sum reduction.
        ignore_index: Label value excluded from the loss.
        shift_labels: Optional labels that have already been shifted.
        **kwargs: Additional Transformers loss arguments.

    Returns:
        The distributed causal language-model loss.
    """
    del vocab_size, kwargs
    tp_mesh = _get_loss_parallel_mesh()
    if tp_mesh is None:
        raise ValueError(
            "loss_parallel requires a TP mesh when computing causal LM loss"
        )
    local_logits = logits.to_local() if isinstance(logits, DTensor) else logits
    if shift_labels is None:
        labels = F.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()
    shift_labels = shift_labels.to(local_logits.device)

    local_vocab_size = local_logits.shape[-1]
    flat_logits = local_logits.reshape(-1, local_vocab_size)
    flat_labels = shift_labels.reshape(-1)
    loss = None
    for start in range(0, flat_labels.numel(), _LOSS_PARALLEL_CHUNK_TOKENS):
        end = min(start + _LOSS_PARALLEL_CHUNK_TOKENS, flat_labels.numel())

        def chunk_loss_fn(
            local_chunk: torch.Tensor,
            label_chunk: torch.Tensor,
        ) -> torch.Tensor:
            chunk_logits = DTensor.from_local(
                local_chunk.float(),
                tp_mesh,
                [Shard(-1)],
            )
            with loss_parallel(mesh=tp_mesh):
                return distributed_cross_entropy(
                    chunk_logits,
                    label_chunk,
                    ignore_index=ignore_index,
                    reduction="sum",
                )

        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            chunk_loss = checkpoint(
                chunk_loss_fn,
                flat_logits[start:end],
                flat_labels[start:end],
                use_reentrant=True,
            )
        loss = chunk_loss if loss is None else loss + chunk_loss

    if loss is None:
        raise ValueError("causal LM loss requires at least one target token")
    if num_items_in_batch is None:
        denominator = (flat_labels != ignore_index).sum().to(loss.dtype)
    else:
        denominator = num_items_in_batch
        if torch.is_tensor(denominator):
            denominator = denominator.to(loss.device)
    return loss / denominator


__all__ = ["causal_lm_loss_parallel"]
