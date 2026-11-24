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
"""Model-independent router load accounting, bias updates and statistics resources."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel.core.dtensor.dtensor import DTensor, SkipDTensorDispatch, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec


def local_router_metadata(
    tensor: torch.Tensor | None, states: torch.Tensor, tp_mesh: Any,
) -> torch.Tensor | None:
    """Align masks or logical sample IDs with the router's actual TP token slice.

    Args:
        tensor: Local metadata or metadata before the TP token slice.
        states: Router input ending in the hidden dimension.
        tp_mesh: Existing token-partition TP mesh, or None for replicated tokens.

    Returns:
        Metadata matching exactly the router input token dimensions.
    """
    if tensor is None or tensor.shape == states.shape[:-1]:
        return tensor
    length = states.shape[-2]
    expected = (*states.shape[:-2], length * tp_mesh.size()) if tp_mesh is not None else None
    if expected is None or tuple(tensor.shape) != expected:
        raise ValueError("router metadata must match local or TP-gathered token dimensions")
    return tensor.narrow(-1, tp_mesh.get_local_rank() * length, length)


def router_sharding(*, module: nn.Module, module_fqn: str, member: Any, context: Any) -> ModuleShardingSpec:
    """Declare complete replicated router state without another token boundary.

    Args:
        module: Actual router member.
        module_fqn: Diagnostic path supplied by the family builder.
        member: Resolved family member.
        context: Existing logical topology.
    """
    del module_fqn, member, context
    return ModuleShardingSpec(params={name: {} for name, _ in module.named_parameters()},
                              in_src={}, in_dst={}, is_boundary=False)


def validate_router_layout(*, module: nn.Module, sharding: ModuleShardingSpec, context: Any) -> None:
    """Reject parameter sharding or token redistribution inside a replicated router.

    Args:
        module: Router owning projection and optional correction state.
        sharding: Final family layout after user overrides.
        context: Existing logical topology.
    """
    del context
    if set(sharding.params) != {name for name, _ in module.named_parameters()}:
        raise ValueError("router requires complete parameter ownership")
    if any(value != Replicate() for axes in sharding.params.values() for value in axes.values()):
        raise ValueError("router parameters must be replicated across computation axes")
    if sharding.is_boundary or sharding.in_src or sharding.in_dst or sharding.out_src or sharding.out_dst:
        raise ValueError("router token redistribution belongs to the containing MoE")


class _AccumulateRouterLoad(torch.autograd.Function):
    """Count a training microbatch once, even when its forward is recomputed."""

    @staticmethod
    def forward(ctx: Any, weights: torch.Tensor, counts: torch.Tensor, accumulator: torch.Tensor) -> torch.Tensor:
        """Keep the forward unchanged; only a consumed backward records loads.

        Args:
            ctx: Autograd context holding the accumulator reference.
            weights: Routing-weight gradient carrier.
            counts: Immutable local microbatch counts.
        accumulator: Integer step-level expert counts.
        """
        ctx.save_for_backward(counts)
        ctx.accumulator = accumulator
        return weights

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Accumulate integer counts without retaining the model graph.

        Args:
            ctx: Context saved by forward.
            gradient: Unchanged gradient for the routing weights.
        """
        with SkipDTensorDispatch(), torch.no_grad():
            ctx.accumulator.add_(ctx.saved_tensors[0])
        return gradient, None, None



def accumulate_router_load(
        weights: torch.Tensor,
        selected_experts: torch.Tensor,
        token_mask: torch.Tensor | None,
        accumulator: torch.Tensor,
) -> torch.Tensor:
    """Attach expert load accumulation to the router's backward path.

    Args:
        weights: Routing weights used by expert combination.
        selected_experts: Actual selected indices, shaped [T, K].
        token_mask: Boolean token mask; true tokens contribute.
        accumulator: Integer [E] step counter.
    """
    selected_experts = selected_experts.to(torch.int64)
    num_tokens, top_k = selected_experts.shape
    valid = selected_experts.new_ones(num_tokens) if token_mask is None else token_mask.reshape(-1).long()
    counts = torch.zeros_like(accumulator).scatter_add(
        0, selected_experts.reshape(-1), valid.unsqueeze(-1).expand(-1, top_k).reshape(-1),
    )
    if torch.is_grad_enabled() and not weights.requires_grad:
        weights = weights.detach().requires_grad_()
    return _AccumulateRouterLoad.apply(weights, counts, accumulator)



@torch.no_grad()
def update_router_bias(bias: torch.Tensor, counts: torch.Tensor, update_rate: float) -> None:
    """Update expert correction bias, including a sharded parameter.

    Counts must already include all DP and token partitions. All-zero counts
    produce no update. Bias affects selection only.

    Args:
        bias: Expert correction parameter [E], possibly sharded.
        counts: Global int64 expert assignment counts [E].
        update_rate: Nonnegative step size for the sign update.
    """
    num_experts = counts.shape[-1]
    total = counts.sum(dim=-1, keepdim=True)
    # Compare in integer arithmetic so rounding cannot erase small load
    # differences before the sign update.
    direction = (total - counts * num_experts).sign()
    delta = (update_rate * direction.float()).to(device=bias.device, dtype=bias.dtype)
    if isinstance(bias, DTensor):
        delta = distribute_tensor(delta, bias.device_mesh, bias.placements)
    bias.add_(delta)


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
