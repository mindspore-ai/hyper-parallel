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
"""PyTorch helpers used by context-parallel implementations."""

# Context parallel is Torch-only after removal of the backend abstraction.
# pylint: disable=forbidden-backend-import,missing-public-type-hints

from __future__ import annotations

from typing import Sequence

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_func
from torch import Tensor
from torch.nn import Module

Function = torch.autograd.Function


def _normalize_dim(dim: int, ndim: int) -> int:
    """Normalize a possibly negative dimension index."""
    return dim + ndim if dim < 0 else dim


def _move_dim_to_front(tensor: Tensor, dim: int) -> Tensor:
    """Move ``dim`` to the front while keeping the other dimensions ordered."""
    dim = _normalize_dim(dim, tensor.dim())
    if dim == 0:
        return tensor.contiguous()
    perm = [dim] + [index for index in range(tensor.dim()) if index != dim]
    return tensor.permute(perm).contiguous()


def _move_dim_from_front(tensor: Tensor, dim: int) -> Tensor:
    """Move the leading dimension back to ``dim``."""
    dim = _normalize_dim(dim, tensor.dim())
    if dim == 0:
        return tensor.contiguous()
    perm = [dim] + [index for index in range(tensor.dim()) if index != dim]
    inverse = [0] * len(perm)
    for index, value in enumerate(perm):
        inverse[value] = index
    return tensor.permute(inverse).contiguous()


def _a2a_reconstruct(out_perm: Tensor, concat_dim: int) -> Tensor:
    """Reconstruct an all-to-all output from its leading-rank layout."""
    chunk_in_perm = concat_dim + 1
    recon_perm = list(range(1, chunk_in_perm)) + [0] + list(range(chunk_in_perm, out_perm.dim()))
    reconstructed = out_perm.permute(recon_perm).contiguous()
    shape = list(reconstructed.shape)
    merged = shape[concat_dim] * shape[concat_dim + 1]
    return reconstructed.reshape(shape[:concat_dim] + [merged] + shape[concat_dim + 2:])


class _AsyncA2AWait(torch.autograd.Function):
    """Wait for a pre-launched all-to-all and preserve its backward overlap."""

    @staticmethod
    def forward(ctx, tensor, work, out_perm, group, world_size, concat_dim, split_dim, handle_box):
        """Wait and reconstruct the forward all-to-all output."""
        ctx.group = group
        ctx.world_size = world_size
        ctx.concat_dim = concat_dim
        ctx.split_dim = split_dim
        ctx.handle_box = handle_box
        ctx.input_shape = tensor.shape
        work.wait()
        return _a2a_reconstruct(out_perm, concat_dim)

    @staticmethod
    def backward(ctx, grad_output):
        """Launch the reverse all-to-all when overlap was requested."""
        if ctx.handle_box is not None:
            grad_output = grad_output.contiguous()
            shape = list(grad_output.shape)
            seq_dim = ctx.concat_dim
            full_size = shape[seq_dim]
            ndim = len(shape) + 1
            grad_perm = grad_output.reshape(
                shape[:seq_dim]
                + [ctx.world_size, full_size // ctx.world_size]
                + shape[seq_dim + 1:]
            ).permute(
                [seq_dim] + list(range(seq_dim)) + list(range(seq_dim + 1, ndim))
            ).contiguous()
            out_perm = torch.empty_like(grad_perm)
            work = dist.all_to_all_single(out_perm, grad_perm, group=ctx.group, async_op=True)
            ctx.handle_box.append((work, out_perm))
        return grad_output.new_zeros(ctx.input_shape), None, None, None, None, None, None, None


class _AsyncAllGatherWait(torch.autograd.Function):
    """Wait for a pre-launched all-gather and provide reduce-scatter backward."""

    @staticmethod
    def forward(ctx, tensor, work, out_perm, group, world_size, gather_dim, handle_box):
        """Wait and reconstruct the gathered tensor."""
        ctx.group = group
        ctx.world_size = world_size
        ctx.gather_dim = gather_dim
        ctx.handle_box = handle_box
        ctx.input_shape = tensor.shape
        work.wait()
        return _move_dim_from_front(out_perm, gather_dim)

    @staticmethod
    def backward(ctx, grad_output):
        """Reduce-scatter the gathered gradient."""
        grad_perm = _move_dim_to_front(grad_output.contiguous(), ctx.gather_dim)
        output_shape = list(grad_perm.shape)
        if output_shape[0] % ctx.world_size != 0:
            raise ValueError(
                "all_gather backward expected gathered dimension to be divisible by world_size, "
                f"got {output_shape[0]} and {ctx.world_size}."
            )
        output_shape[0] //= ctx.world_size
        output = torch.empty(output_shape, dtype=grad_perm.dtype, device=grad_perm.device)
        work = dist.reduce_scatter_tensor(output, grad_perm, group=ctx.group, async_op=True)
        if ctx.handle_box is not None:
            ctx.handle_box.append((work, output, ctx.gather_dim))
            return grad_output.new_zeros(ctx.input_shape), None, None, None, None, None, None
        work.wait()
        return _move_dim_from_front(output, ctx.gather_dim), None, None, None, None, None, None


class _P2PExchange(torch.autograd.Function):
    """Symmetrically exchange a tensor with one peer in forward and backward."""

    @staticmethod
    def forward(ctx, tensor: Tensor, peer_rank: int, group):
        """Exchange the forward tensor."""
        ctx.peer_rank = peer_rank
        ctx.group = group
        send_buffer = tensor.contiguous()
        receive_buffer = torch.empty_like(send_buffer)
        requests = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, send_buffer, peer_rank, group),
                dist.P2POp(dist.irecv, receive_buffer, peer_rank, group),
            ]
        )
        for request in requests:
            request.wait()
        return receive_buffer

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """Exchange the backward gradient."""
        send_buffer = grad_output.contiguous()
        receive_buffer = torch.empty_like(send_buffer)
        requests = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, send_buffer, ctx.peer_rank, ctx.group),
                dist.P2POp(dist.irecv, receive_buffer, ctx.peer_rank, ctx.group),
            ]
        )
        for request in requests:
            request.wait()
        return receive_buffer, None, None


def is_tensor(value) -> bool:
    """Return whether ``value`` is a PyTorch tensor."""
    return isinstance(value, Tensor)


def register_forward_pre_hook(module: Module, hook, *, with_kwargs: bool = False):
    """Register a module forward pre-hook."""
    return module.register_forward_pre_hook(hook, with_kwargs=with_kwargs)


def register_full_backward_pre_hook(module: Module, hook):
    """Register a module full backward pre-hook."""
    return module.register_full_backward_pre_hook(hook)


def get_rank() -> int:
    """Return the current distributed rank."""
    return dist.get_rank()


def all_to_all_single(input_tensor: Tensor, output_shape: Sequence[int], group, async_op: bool = False):
    """Run a fixed-shape all-to-all collective."""
    output = torch.empty(output_shape, device=input_tensor.device, dtype=input_tensor.dtype)
    work = dist.all_to_all_single(output, input_tensor, group=group, async_op=async_op)
    return output, work


def all_gather_single(input_tensor: Tensor, output_shape: Sequence[int], group, async_op: bool = False):
    """Run an all-gather-into-tensor collective."""
    output = torch.empty(output_shape, device=input_tensor.device, dtype=input_tensor.dtype)
    work = dist.all_gather_into_tensor(output, input_tensor, group=group, async_op=async_op)
    return output, work


def reduce_scatter_single(input_tensor: Tensor, output_shape: Sequence[int], group, async_op: bool = False):
    """Run a reduce-scatter-into-tensor collective."""
    output = torch.empty(output_shape, device=input_tensor.device, dtype=input_tensor.dtype)
    work = dist.reduce_scatter_tensor(output, input_tensor, group=group, async_op=async_op)
    return output, work


def differentiable_async_a2a_wait(
    tensor: Tensor, work, out_perm: Tensor, group, world_size: int, concat_dim: int, split_dim: int, handle_box=None
) -> Tensor:
    """Wait for an asynchronous all-to-all while preserving autograd."""
    return _AsyncA2AWait.apply(tensor, work, out_perm, group, world_size, concat_dim, split_dim, handle_box)


def differentiable_async_allgather_wait(
    tensor: Tensor, work, out_perm: Tensor, group, world_size: int, gather_dim: int, handle_box=None
) -> Tensor:
    """Wait for an asynchronous all-gather while preserving autograd."""
    return _AsyncAllGatherWait.apply(tensor, work, out_perm, group, world_size, gather_dim, handle_box)


def differentiable_all_to_all_single(
    input_tensor: Tensor, input_splits: Sequence[int], output_splits: Sequence[int], group
) -> Tensor:
    """Run a differentiable variable-split all-to-all collective."""
    output = torch.empty(
        sum(output_splits), *input_tensor.shape[1:], dtype=input_tensor.dtype, device=input_tensor.device
    )
    return dist_func.all_to_all_single(
        output,
        input_tensor,
        output_split_sizes=list(output_splits),
        input_split_sizes=list(input_splits),
        group=group,
    )


def differentiable_all_gather_concat(
    tensor: Tensor, group, concat_size: int, concat_dim: int, rank_list=None
) -> Tensor:
    """Differentiably gather and concatenate tensors in mesh rank order."""
    del concat_size
    tensor = tensor.contiguous()
    output = list(dist_func.all_gather(tensor, group=group))
    if rank_list is not None:
        group_ranks = dist.get_process_group_ranks(group)
        if tuple(rank_list) != tuple(group_ranks):
            rank_to_index = {int(rank): index for index, rank in enumerate(group_ranks)}
            output = [output[rank_to_index[int(rank)]] for rank in rank_list]
    return torch.cat(output, dim=concat_dim)


def p2p_exchange(tensor: Tensor, peer_rank: int, group=None) -> Tensor:
    """Symmetrically exchange a tensor with ``peer_rank``."""
    if peer_rank == dist.get_rank(group):
        return tensor
    return _P2PExchange.apply(tensor, peer_rank, group)


def cat(tensors, dim: int = 0) -> Tensor:
    """Concatenate tensors along ``dim``."""
    return torch.cat(tensors, dim=dim)
