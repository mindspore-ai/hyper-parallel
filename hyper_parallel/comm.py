# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""HyperParallel communication gateway (torch-first).

This module is the single place where ``core/`` resolves the small set of
communication helpers that historically lived behind ``platform.*``.  The
default implementations are plain torch; the MindSpore runtime monkeypatches
the module-level names (see ``hyper_parallel.platform.mindspore.patch``),
so callers must access them as ``comm.<fn>`` (module-qualified) rather than
``from hyper_parallel.comm import <fn>`` — the latter would bind the torch
implementation at import time and defeat the patch.
"""

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_func

__all__ = [
    "differentiable_all_gather_concat",
    "p2p_exchange",
    "construct_strided_slice",
]


def _ensure_contiguous(x):
    """Return a contiguous copy of *x* if not already contiguous."""
    if torch.compiler.is_compiling():
        return x.contiguous()
    if not x.is_contiguous() or x.storage_offset() != 0:
        return x.contiguous()
    return x


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


def differentiable_all_gather_concat(data, group, concat_size, concat_dim, rank_list=None):
    """Differentiable all-gather along ``concat_dim``, reordered by ``rank_list``."""
    data = _ensure_contiguous(data)
    output = list(dist_func.all_gather(data, group=group))
    if rank_list is not None:
        group_ranks = dist.get_process_group_ranks(group)
        if tuple(rank_list) != tuple(group_ranks):
            rank_to_idx = {int(rank): idx for idx, rank in enumerate(group_ranks)}
            output = [output[rank_to_idx[int(rank)]] for rank in rank_list]
    return torch.cat(output, dim=concat_dim)


def p2p_exchange(tensor, peer_rank: int, group=None):
    """Differentiable symmetric P2P exchange (send local tensor, receive peer's tensor)."""
    if peer_rank == dist.get_rank(group):
        return tensor
    return _TorchP2PExchangeFunction.apply(tensor, peer_rank, group)


def construct_strided_slice(x, begin, end, stride):
    """Construct a strided slice operation on a tensor.

    MindSpore-only capability; the torch default path does not implement it.
    """
    raise NotImplementedError("Unsupported construct_strided_slice for torch platform")
