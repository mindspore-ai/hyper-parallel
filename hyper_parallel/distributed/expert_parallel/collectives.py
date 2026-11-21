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

"""expert_parallel.collectives: backend-dispatched EP all_to_all.

NCCL/HCCL use the ragged a2a (``_EPAllToAllUneven``). Gloo and
other backends that do not support ragged a2a use pad-to-max +
``all_to_all_single`` (``_EPAllToAllPadded``). Both paths are numerically
equivalent (padding only adds filler rows that do not participate in
computation).

Split out of components/distributed/ep_utils.py in stage 4e.
"""

from typing import Any, Optional
import torch
import torch.distributed as dist
from torch import _check, _check_is_size
from torch.distributed import _functional_collectives

from hyper_parallel.core.dtensor._utils import differentiable_all_to_all_single_async
from hyper_parallel.core.utils.communication import _is_tracing_or_compiling

_UNEVEN_A2A_BACKENDS = ("nccl", "hccl")



def ep_exchange_counts(counts: torch.Tensor, group: Any) -> torch.Tensor:
    """Exchange integer count rows per peer, waiting before reading the result.

    Args:
        counts: Non-differentiable send counts in process-group rank order.
            Shape [EP] exchanges one count per peer; [EP, fields] may include
            per-peer protocol metadata alongside each count.
        group: Expert-parallel process group.

    Returns:
        Counts received from each peer in the same rank order.
    """
    if _is_tracing_or_compiling():
        return _functional_collectives.wait_tensor(
            _functional_collectives.all_to_all_single(counts, None, None, group)
        )
    received = torch.empty_like(counts)
    dist.all_to_all_single(received, counts, group=group)
    return received


def ep_split_sizes(counts: torch.Tensor, total: Optional[int] = None) -> list[int]:
    """Read per-call segment sizes, retaining data dependencies during capture.

    Args:
        counts: One-dimensional integer tensor of segment lengths.
        total: Expected number of rows, when known from a payload's shape.

    Returns:
        Concrete eager sizes or symbolic sizes evaluated by the captured graph.
    """
    if not _is_tracing_or_compiling():
        return counts.tolist()
    sizes = counts.cpu().tolist()
    for size in sizes:
        _check_is_size(size)
    if total is not None:
        _check(sum(sizes) == total)
    return sizes


def _backend_supports_uneven_a2a(group) -> bool:
    return dist.get_backend(group) in _UNEVEN_A2A_BACKENDS


class _EPAllToAllUneven(torch.autograd.Function):  # pylint: disable=abstract-method
    """Ragged all_to_all (NCCL/HCCL production path): split by send/recv counts.

    forward:  split(x, send_counts) -> dist.all_to_all(out_list, in_list) -> cat
    backward: swap send/recv counts and run the ragged all_to_all again
              (a2a is self-inverse).
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        send_counts: list[int],
        recv_counts: list[int],
        group: Any,
    ) -> torch.Tensor:  # pylint: disable=arguments-differ
        """Run the ragged all_to_all and retain the counts for backward.

        Args:
            ctx: Autograd context retaining the exchange plan.
            x: Peer-major payload, split along dimension zero.
            send_counts: Rows sent to each peer.
            recv_counts: Rows received from each peer.
            group: Existing expert-parallel process group.
        """
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.group = group
        out = x.new_empty((sum(recv_counts),) + tuple(x.shape[1:]))
        dist.all_to_all(list(out.split(recv_counts)), list(x.split(send_counts)), group=group)
        return out

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None]:  # pylint: disable=arguments-differ
        """Swap send/recv counts and re-run the self-inverse ragged all_to_all.

        Args:
            ctx: Saved forward exchange plan.
            grad_output: Gradients in received peer-major order.
        """
        grad = _EPAllToAllUneven.apply(
            grad_output.contiguous(), ctx.recv_counts, ctx.send_counts, ctx.group)
        return grad, None, None, None


class _EPAllToAllPadded(torch.autograd.Function):  # pylint: disable=abstract-method
    """pad-to-max + all_to_all_single (gloo test path).

    forward:  pad each dest chunk to the global max(counts) (a2a_single
              requires equal-length chunks per rank -> pad_to must be
              globally consistent, obtained via all_reduce MAX)
              -> a2a_single -> unpad by recv_counts;
    backward: pad by recv_counts -> a2a_single (equal-length self-inverse)
              -> unpad by send_counts.
    """

    @staticmethod
    def _pad_and_exchange(x, counts, pad_to, group):
        """Pad each chunk to pad_to, run equal-length a2a_single, return [ep*pad_to, ...]."""
        chunks = []
        for chunk, n in zip(x.split(counts), counts):
            if n < pad_to:
                pad = x.new_zeros((pad_to - n,) + tuple(x.shape[1:]))
                chunk = torch.cat([chunk, pad])
            chunks.append(chunk)
        send = torch.cat(chunks).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=group)
        return recv

    @staticmethod
    def _unpad(recv, counts, pad_to):
        """Take the valid rows from the equal-length buffer per counts and cat."""
        pieces = []
        for i, n in enumerate(counts):
            if n > 0:
                pieces.append(recv[i * pad_to: i * pad_to + n])
        if not pieces:
            return recv.new_zeros((0,) + tuple(recv.shape[1:]))
        return torch.cat(pieces)

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        send_counts: list[int],
        recv_counts: list[int],
        group: Any,
    ) -> torch.Tensor:  # pylint: disable=arguments-differ
        """Exchange padded expert-token chunks and retain counts for backward.

        Args:
            ctx: Autograd context retaining counts and padding size.
            x: Peer-major payload, split along dimension zero.
            send_counts: Valid rows sent to each peer.
            recv_counts: Valid rows received from each peer.
            group: Existing expert-parallel process group.
        """
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.group = group
        local_max = max([*send_counts, *recv_counts, 1])
        pad_to = x.new_tensor([local_max], dtype=torch.int64)
        dist.all_reduce(pad_to, op=dist.ReduceOp.MAX, group=group)
        ctx.pad_to = pad_to = int(pad_to.item())
        recv = _EPAllToAllPadded._pad_and_exchange(x, send_counts, pad_to, group)
        return _EPAllToAllPadded._unpad(recv, recv_counts, pad_to)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None]:  # pylint: disable=arguments-differ
        """Reverse the exchange: pad by recv_counts, a2a_single, unpad by send_counts.

        Args:
            ctx: Saved forward counts and padding size.
            grad_output: Gradients in received peer-major order.
        """
        # backward = reversed a2a: pad by recv_counts -> a2a_single -> unpad by send_counts
        recv = _EPAllToAllPadded._pad_and_exchange(
            grad_output.contiguous(), ctx.recv_counts, ctx.pad_to, ctx.group)
        grad = _EPAllToAllPadded._unpad(recv, ctx.send_counts, ctx.pad_to)
        return grad, None, None, None


def ep_all_to_all(
    x: torch.Tensor,
    send_counts: list[int],
    recv_counts: list[int],
    group: Any,
) -> torch.Tensor:
    """Unified entry for EP token exchange (autograd-differentiable).

    NCCL/HCCL use variable splits. Other backends pad each segment for the exchange.
    The eager entry returns a materialized tensor. Graph capture records the
    functional exchange and its consumer wait.

    Args:
        x: Payload split along dimension zero.
        send_counts: Rows sent to each peer, in process-group rank order.
        recv_counts: Rows received from each peer, in process-group rank order.
        group: Existing expert-parallel process group.

    Returns:
        Received payload with sum(recv_counts) rows.
    """
    if _is_tracing_or_compiling():
        # Keep the autograd wrapper; its first consumer inserts the stream wait.
        return _functional_collectives.all_to_all_single_autograd(x, recv_counts, send_counts, group)
    if not _backend_supports_uneven_a2a(group):
        return _EPAllToAllPadded.apply(x, send_counts, recv_counts, group)
    return _EPAllToAllUneven.apply(x, send_counts, recv_counts, group)


def ep_all_to_all_async(
    x: torch.Tensor,
    send_counts: list[int],
    recv_counts: list[int],
    group: Any,
) -> torch.Tensor:
    """Non-blocking variant of :func:`ep_all_to_all` (lazy wait).

    On backends whose ``all_to_all_single`` accepts unequal splits the exchange
    is issued through ``differentiable_all_to_all_single_async``, which returns
    an ``AsyncCollectiveTensor``: the ``wait_tensor`` op is only enqueued when a
    non-view op first consumes the result.  Issuing independent work between the
    exchange and that first read therefore overlaps with the in-flight transfer
    (forward and backward alike).

    Backends without that support use the padded blocking exchange.

    Args:
        x: Input tensor, split along dim 0 by ``send_counts``.
        send_counts: Rows sent to each EP rank.
        recv_counts: Rows received from each EP rank.
        group: EP process group.
    Returns:
        The exchanged rows, materialized lazily on the async path.
    """
    if not _backend_supports_uneven_a2a(group):
        return _EPAllToAllPadded.apply(x, send_counts, recv_counts, group)
    return differentiable_all_to_all_single_async(x, send_counts, recv_counts, group)
