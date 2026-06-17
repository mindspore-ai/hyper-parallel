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
"""Autograd-aware collective communication operations for SimpleFSDP.

These ``torch.autograd.Function`` wrappers make all-gather and reduce-scatter
visible to Dynamo's graph tracer while correctly propagating gradients in
backward. The forward performs all-gather (shard to full); the backward
automatically performs reduce-scatter (full gradient to gradient shard).

This is the small experimental bridge used before HyperParallel DTensor
``redistribute`` has first-class autograd semantics.
"""
import os

import torch
import torch.distributed as dist  # pylint: disable=C0415


def _use_functional_collectives() -> bool:
    """Return whether to prefer graph-friendly functional collectives."""
    flag = os.environ.get("HYPER_PARALLEL_SIMPLE_FSDP_FUNCTIONAL", "1")
    return flag.lower() not in ("0", "false", "off", "no")


def _functional_all_gather_tensor(input_tensor: torch.Tensor, group):
    """Run functional all-gather if the current PyTorch build exposes it."""
    if not _use_functional_collectives():
        return None
    try:
        # pylint: disable=C0415
        from torch.distributed import _functional_collectives as functional_collectives
    except (ImportError, AttributeError):
        return None
    all_gather_tensor = getattr(functional_collectives, "all_gather_tensor", None)
    if all_gather_tensor is None:
        return None
    try:
        return all_gather_tensor(input_tensor, 0, group)
    except TypeError:
        return all_gather_tensor(input_tensor, gather_dim=0, group=group)


def _functional_reduce_scatter_tensor(input_tensor: torch.Tensor, group):
    """Run functional reduce-scatter if the current PyTorch build exposes it."""
    if not _use_functional_collectives():
        return None
    try:
        # pylint: disable=C0415
        from torch.distributed import _functional_collectives as functional_collectives
    except (ImportError, AttributeError):
        return None
    reduce_scatter_tensor = getattr(functional_collectives, "reduce_scatter_tensor", None)
    if reduce_scatter_tensor is None:
        return None
    try:
        return reduce_scatter_tensor(input_tensor, "sum", 0, group)
    except TypeError:
        return reduce_scatter_tensor(input_tensor, reduceOp="sum", scatter_dim=0, group=group)


class AllGatherRSFunction(torch.autograd.Function):
    """Forward: all-gather. Backward: reduce-scatter with averaging.

    This implementation gathers along ``shard_dim`` by moving that dimension
    to the front, using ``all_gather_into_tensor``/``reduce_scatter_tensor``,
    then moving the dimension back.
    """

    @staticmethod
    def forward(ctx, shard, group, world_size: int, shard_dim: int):
        """All-gather a local shard into a full tensor."""
        ctx.group = group
        ctx.world_size = world_size
        ctx.shard_dim = shard_dim

        gather_input = shard.movedim(shard_dim, 0).contiguous()
        functional_gathered = _functional_all_gather_tensor(gather_input, group)
        if functional_gathered is not None:
            return functional_gathered.movedim(0, shard_dim).contiguous()

        full_shape = (gather_input.shape[0] * world_size,) + gather_input.shape[1:]
        gathered = torch.empty(full_shape, device=shard.device, dtype=shard.dtype)
        dist.all_gather_into_tensor(gathered, gather_input, group=group)
        return gathered.movedim(0, shard_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_full):
        """Reduce-scatter the full gradient back to the local shard shape."""
        world_size = ctx.world_size
        shard_dim = ctx.shard_dim
        scatter_input = grad_full.movedim(shard_dim, 0).contiguous()
        functional_grad_shard = _functional_reduce_scatter_tensor(scatter_input, ctx.group)
        if functional_grad_shard is not None:
            return (
                functional_grad_shard.div(world_size).movedim(0, shard_dim).contiguous(),
                None,
                None,
                None,
            )

        shard_size = scatter_input.shape[0] // world_size
        grad_shard = torch.empty(
            (shard_size,) + scatter_input.shape[1:],
            device=grad_full.device,
            dtype=grad_full.dtype,
        )
        dist.reduce_scatter_tensor(grad_shard, scatter_input, group=ctx.group)
        return (grad_shard / world_size).movedim(0, shard_dim).contiguous(), None, None, None


def all_gather_rs(shard: torch.Tensor, group, world_size: int, shard_dim: int) -> torch.Tensor:
    """All-gather a shard, with automatic reduce-scatter in backward.

    Args:
        shard: Local parameter shard.
        group: Torch process group.
        world_size: Number of ranks in ``group``.
        shard_dim: Tensor dimension that was sharded.

    Returns:
        Full replicated parameter tensor.
    """
    return AllGatherRSFunction.apply(shard, group, world_size, shard_dim)
