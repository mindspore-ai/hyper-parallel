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
"""Torch-native differentiable collectives shared by the distributed module.

These helpers wrap ``torch.distributed`` (and its autograd-aware
``torch.distributed.nn.functional`` counterparts) with the small amount of
glue the distributed builders and context-parallel collectives need:
contiguity normalization, string reduce-op resolution, and concatenation.
"""

from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.nn import functional as dist_func


class _TorchContiguousGrad(torch.autograd.Function):  # pylint: disable=abstract-method
    """Autograd identity that materializes gradients before upstream collectives."""

    @staticmethod
    def forward(ctx: Any, tensor: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        """Return the input unchanged in the forward pass.

        Args:
            ctx: Autograd context required by ``torch.autograd.Function``.
            tensor: Tensor produced by the differentiable collective.

        Returns:
            The input tensor without a forward copy.
        """
        del ctx
        return tensor

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        """Return a contiguous gradient to the preceding autograd node.

        Args:
            ctx: Autograd context required by ``torch.autograd.Function``.
            grad_output: Gradient from the collective output consumer.

        Returns:
            A contiguous gradient tensor.
        """
        del ctx
        return grad_output.contiguous()


class _TorchP2PExchangeFunction(torch.autograd.Function):
    """Symmetric bidirectional P2P: send local tensor to peer, receive peer's tensor."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, peer_rank: int, group) -> torch.Tensor:  # pylint: disable=arguments-differ
        """Perform symmetric bidirectional P2P exchange with peer_rank."""
        ctx.peer_rank = peer_rank
        ctx.group = group
        send_buf = tensor.contiguous()
        recv_buf = torch.empty_like(send_buf)
        reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, send_buf, peer_rank, group),
            dist.P2POp(dist.irecv, recv_buf, peer_rank, group),
        ])
        for req in reqs:
            req.wait()
        return recv_buf

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Perform symmetric P2P exchange for the backward gradient pass."""
        send_buf = grad_output.contiguous()
        recv_buf = torch.empty_like(send_buf)
        reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, send_buf, ctx.peer_rank, ctx.group),
            dist.P2POp(dist.irecv, recv_buf, ctx.peer_rank, ctx.group),
        ])
        for req in reqs:
            req.wait()
        return recv_buf, None, None


# Mapping from string op names to torch.distributed.ReduceOp
_OP_MAP = {
    'sum': dist.ReduceOp.SUM,
    'prod': dist.ReduceOp.PRODUCT,
    'max': dist.ReduceOp.MAX,
    'min': dist.ReduceOp.MIN,
    # convert tensor elements to int32 and use MIN
    'all': dist.ReduceOp.MIN,
    # 'avg' is typically handled by SUM followed by division in current implementation logic
    'avg': dist.ReduceOp.SUM,
}

# Try to add AVG for 'mean' if supported by current torch version
if hasattr(dist.ReduceOp, "AVG"):
    _OP_MAP['mean'] = dist.ReduceOp.AVG
else:
    # Fallback for older torch versions if necessary, though this might require manual division upstream
    # Assuming standard behavior where 'mean' implies native AVG support or upstream handling
    _OP_MAP['mean'] = dist.ReduceOp.SUM


def _ensure_contiguous(x):
    """Return a contiguous copy of *x* if not already contiguous."""
    if torch.compiler.is_compiling():
        return x.contiguous()
    if not x.is_contiguous() or x.storage_offset() != 0:
        return x.contiguous()
    return x


def differentiable_all_gather_concat(data, group, concat_size, concat_dim, rank_list=None):  # pylint: disable=unused-argument
    """Differentiable all-gather followed by concatenation along ``concat_dim``."""
    data = _ensure_contiguous(data)
    output = [
        _TorchContiguousGrad.apply(tensor)
        for tensor in dist_func.all_gather(data, group=group)
    ]
    if rank_list is not None:
        group_ranks = dist.get_process_group_ranks(group)
        if tuple(rank_list) != tuple(group_ranks):
            rank_to_idx = {int(rank): idx for idx, rank in enumerate(group_ranks)}
            output = [output[rank_to_idx[int(rank)]] for rank in rank_list]
    return torch.cat(output, dim=concat_dim)


def differentiable_all_to_all(input_data, output_shape, group):
    """Differentiable all-to-all with equal splits, returning a new output tensor."""
    input_data = _ensure_contiguous(input_data)
    output_tensor = torch.empty(output_shape, device=input_data.device, dtype=input_data.dtype)
    output_tensor = dist_func.all_to_all_single(
        output_tensor,
        input_data,
        group=group
    )
    return output_tensor


def differentiable_all_reduce(data, op, group):
    """Differentiable all-reduce; ``op`` accepts a string name or ``dist.ReduceOp``."""
    data = _ensure_contiguous(data)
    # Resolve the op from string to ReduceOp enum if necessary
    reduce_op = _OP_MAP.get(op, dist.ReduceOp.SUM) if isinstance(op, str) else op
    return dist_func.all_reduce(data, op=reduce_op, group=group)


def differentiable_reduce_scatter(data, dev_num, axis, op, group):
    """Differentiable reduce-scatter: chunk ``data`` along ``axis`` then reduce-scatter."""
    data = _ensure_contiguous(data)
    input_tuple = torch.chunk(data, dev_num, dim=axis)
    output_tensor = torch.empty(input_tuple[0].shape, device=data.device, dtype=data.dtype)

    # Resolve the op from string to ReduceOp enum
    reduce_op = _OP_MAP.get(op, dist.ReduceOp.SUM) if isinstance(op, str) else op

    output_tensor = dist_func.reduce_scatter(output_tensor, input_tuple, op=reduce_op, group=group)

    # Keep manual handling for 'avg' string as it maps to SUM in _OP_MAP
    if op == 'avg':
        output_tensor = output_tensor / dev_num
    return output_tensor


def differentiable_all_to_all_single(input_tensor, input_splits, output_splits, group):
    """Variable-split all-to-all with autograd support for EP token dispatch/combine."""
    out_total = sum(output_splits)
    output = torch.empty(
        out_total, *input_tensor.shape[1:],
        dtype=input_tensor.dtype, device=input_tensor.device,
    )
    output = dist_func.all_to_all_single(
        output, input_tensor,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )
    return output


def p2p_exchange(tensor, peer_rank: int, group=None):
    """Exchange a tensor with ``peer_rank`` (identity when the peer is self)."""
    if peer_rank == dist.get_rank(group):
        return tensor
    return _TorchP2PExchangeFunction.apply(tensor, peer_rank, group)
