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

# Transformers model adapters implement PyTorch-native model contracts.
import torch  # pylint: disable=forbidden-backend-import
import torch.nn.functional as F  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.tensor_parallel.loss_parallel import _get_loss_parallel_mesh
from hyper_parallel.components.losses._vocab_parallel_cross_entropy import (
    vocab_parallel_cross_entropy_local,
)


def causal_lm_loss_parallel(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Compute causal LM loss over a vocab-sharded logits shard, local-first.

    Transformers normally reshapes logits to ``[-1, global_vocab_size]``
    before cross entropy. With a vocab-sharded lm head only the local shard
    is materialized, so this adapter hands the local logits, the global
    vocabulary size, the loss-parallel mesh and its class-sharding axis
    straight to the AutoModels-private vocab-parallel cross entropy. No
    temporary ``DTensor`` is created.

    Args:
        logits: Rank-3 logits whose final dimension may be vocabulary sharded.
        labels: Local labels aligned with the local sequence shard.
        vocab_size: Global vocabulary size across all shard ranks.
        num_items_in_batch: Optional denominator for sum reduction.
        ignore_index: Label value excluded from the loss.
        shift_labels: Optional labels that have already been shifted.
        **kwargs: Additional Transformers loss arguments.

    Returns:
        The distributed causal language-model loss.
    """
    del kwargs
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
    if flat_labels.numel() == 0:
        raise ValueError("causal LM loss requires at least one target token")
    loss = vocab_parallel_cross_entropy_local(
        flat_logits.float(),
        flat_labels,
        vocab_size=vocab_size,
        mesh=tp_mesh,
        ignore_index=ignore_index,
        reduction="sum",
    )
    if num_items_in_batch is None:
        denominator = (flat_labels != ignore_index).sum().to(loss.dtype)
    else:
        denominator = num_items_in_batch
        if torch.is_tensor(denominator):
            denominator = denominator.to(loss.device)
    return loss / denominator


__all__ = ["causal_lm_loss_parallel"]
