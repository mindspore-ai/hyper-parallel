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
"""MoE utilities for distributed training."""

from typing import Any, Optional

import torch
import torch.distributed as dist
from torch import nn



def sync_and_update_expert_bias(
    moe: nn.Module,
    lr: float = 1e-3,
    tp_group: Optional[Any] = None,
    cp_group: Optional[Any] = None,
    dp_group: Optional[Any] = None,
    num_recomputations: int = 1,
) -> None:
    """Synchronize tokens_per_expert across distributed ranks, then update expert bias.

    This function handles distributed synchronization for expert bias updates in
    MoE (Mixture of Experts) models. It should be called after optimizer.step() in
    the training loop.

    The synchronization is needed because:
    - **DP**: Different ranks process different samples, need to aggregate global batch stats
    - **TP+SP**: Standard TP does not need sync (all ranks see same input).
      Only needed when TP is combined with sequence parallelism (TP+SP),
      where activations are sequence-sharded across TP ranks.
    - **CP**: Sequence dimension is sharded across ranks

    For pure EP scenario (no DP/TP/CP), call moe.update_expert_bias() directly.

    Args:
        moe: The MoE module whose expert bias should be updated.
        lr: Step size for the bias update. Defaults to ``1e-3``.
        tp_group: Tensor parallel process group (ProcessGroup or GroupInfo).
            Only needed when TP is combined with sequence parallelism (TP+SP),
            where activations are sequence-sharded across TP ranks.
            Standard TP (weight-only sharding) does not need this.
        cp_group: Context parallel process group (ProcessGroup or GroupInfo).
            Sync across sequence shards dimension.
        dp_group: Data parallel process group (ProcessGroup or GroupInfo).
            Sync across different samples to get global batch statistics.
        num_recomputations: Number of forward executions per optimizer step.
            Default ``1`` (normal training). Set to ``2`` when activation
            checkpoint is enabled.

    Example:
        >>> # DP scenario
        >>> sync_and_update_expert_bias(moe_layer, lr=1e-3, dp_group=dp_group_info)
        >>>
        >>> # TP×CP×DP scenario (reference: Megatron-LM)
        >>> sync_and_update_expert_bias(
        ...     moe_layer,
        ...     lr=1e-3,
        ...     tp_group=tp_group_info,
        ...     cp_group=cp_group_info,
        ...     dp_group=dp_group_info,
        ... )

    Note:
        Reference implementation: Megatron-LM megatron/core/transformer/moe/moe_utils.py
        uses TP×CP×DP group synchronization for global batch statistics.
        Model adapters may provide ``expert_bias_update_groups`` and
        ``expert_bias_update_rate`` to override generic groups and update speed.
    """
    groups = getattr(moe, "expert_bias_update_groups", None)
    if groups is None:
        groups = (tp_group, cp_group, dp_group)
    for group in groups:
        if group is not None:
            dist.all_reduce(moe.tokens_per_expert, group=getattr(group, "group", group))

    moe.update_expert_bias(
        lr=getattr(moe, "expert_bias_update_rate", lr),
        num_recomputations=num_recomputations,
    )
