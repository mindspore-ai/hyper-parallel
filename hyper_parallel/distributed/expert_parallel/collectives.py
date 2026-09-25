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

NCCL/HCCL use the ragged a2a (``_EPAllToAllUneven``, zero-padding); gloo and
other backends that do not support ragged a2a use pad-to-max +
``all_to_all_single`` (``_EPAllToAllPadded``). Both paths are numerically
equivalent (padding only adds filler rows that do not participate in
computation).

When every per-peer count is equal -- which is exactly what a balanced routing
plan produces -- the split sizes carry no information, so ``HP_EP_EQUAL_A2A``
swaps the ragged exchange for the plain equal-length ``all_to_all_single``,
whose backend kernel is cheaper than alltoallv.  The rows, their order and the
values are identical on both paths.  On top of the kernel, the exchange has to
stay *lazy*: the champion issues its token exchange through the async entry and
hides it behind the shared-expert MLP, and a plain c10d collective orders the
stream it is called on at issue time, which closes that window.  ``=1``
therefore means split-free *and* lazy (``_EPAllToAllEqualLazy``: the exchange is
issued on a stream of its own and the caller waits on an event), while ``=eager``
forces the split-free call back onto the caller's stream for an A/B of the two.

Split out of components/distributed/ep_utils.py in stage 4e.
"""

import os
import threading
from typing import Any, Callable, Optional
import torch
import torch.distributed as dist

from hyper_parallel.core.dtensor._utils import (
    differentiable_all_to_all_single_async,
    get_device_handle,
)

_UNEVEN_A2A_BACKENDS = ("nccl", "hccl")

# A ragged exchange reaches NCCL/HCCL as alltoallv, which takes a per-peer count
# vector and runs a count-driven kernel.  Under a balanced routing plan every
# per-peer count is the same, so the counts add nothing and the plain
# equal-length all_to_all_single (alltoall) can be issued instead.  Opt-in while
# it is being measured (HP_EP_EQUAL_A2A).
_EQUAL_A2A_OFF = "off"
_EQUAL_A2A_LAZY = "lazy"
_EQUAL_A2A_EAGER = "eager"
_EQUAL_A2A_MODES = {
    "0": _EQUAL_A2A_OFF,
    "1": _EQUAL_A2A_LAZY,
    "eager": _EQUAL_A2A_EAGER,
}
_EQUAL_A2A_RAW = os.environ.get("HP_EP_EQUAL_A2A", "0")

# Ragged (uneven) exchange kernel: the list form issues one
# ``dist.all_to_all`` with per-peer tensors, which HCCL serves through
# alltoallv; ``HP_EP_A2A_SINGLE=1`` issues the identical exchange as one
# ``all_to_all_single`` with split sizes, the form the micro benchmark
# measures at ~2x the bandwidth (106.4 vs 52.9 GB/s). Default off.  Scope: this
# only affects THIS synchronous entry; the training MoE path calls
# ``ep_all_to_all_async``, whose ``_AsyncA2ALazyBwd`` already issues the split-sized
# ``all_to_all_single``, so the knob is a no-op there (measured: neutral).
_A2A_SINGLE = os.environ.get("HP_EP_A2A_SINGLE", "0") == "1"

# Lazily created, then reused for every lazy exchange (see
# :func:`_lazy_a2a_resources`): the stream the split-free collective is issued
# on, the event that hands it the payload, and the event that hands the result
# back to the consumer.
_LAZY_A2A_STREAM = None
_LAZY_A2A_READY_EVENT = None
_LAZY_A2A_DONE_EVENT = None
_LAZY_A2A_LOCK = threading.Lock()


def _backend_supports_uneven_a2a(group) -> bool:
    return dist.get_backend(group) in _UNEVEN_A2A_BACKENDS


def _equal_a2a_mode() -> str:
    """Resolve ``HP_EP_EQUAL_A2A`` to the split-free exchange's mode.

    ``"0"`` (the default) keeps the ragged exchange, ``"1"`` takes the
    split-free exchange and defers its wait, and ``"eager"`` takes the
    split-free exchange while ordering the caller's stream on it at issue time
    -- the behaviour ``"1"`` used to have, kept so the kernel's gain and the
    lost overlap can be measured apart.

    Returns:
        One of ``"off"``, ``"lazy"``, ``"eager"``.

    Raises:
        ValueError: If the knob is none of the accepted values; a typo must fail
            loudly instead of silently leaving the fast path off.
    """
    raw = _EQUAL_A2A_RAW.strip()
    try:
        return _EQUAL_A2A_MODES[raw]
    except KeyError:
        raise ValueError(
            f"HP_EP_EQUAL_A2A must be one of "
            f"{sorted(_EQUAL_A2A_MODES)}, but got {raw!r}"
        ) from None


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
        """Run the ragged all_to_all and retain the counts for backward."""
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.group = group
        out = x.new_empty((sum(recv_counts),) + tuple(x.shape[1:]))
        if _A2A_SINGLE:
            dist.all_to_all_single(
                out,
                x.contiguous(),
                output_split_sizes=[int(count) for count in recv_counts],
                input_split_sizes=[int(count) for count in send_counts],
                group=group,
            )
        else:
            dist.all_to_all(list(out.split(recv_counts)),
                            list(x.split(send_counts)), group=group)
        return out

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None]:  # pylint: disable=arguments-differ
        """Swap send/recv counts and re-run the self-inverse ragged all_to_all."""
        grad = _EPAllToAllUneven.apply(
            grad_output.contiguous(), ctx.recv_counts, ctx.send_counts, ctx.group)
        return grad, None, None, None


def _equal_a2a_exchange(
    x: torch.Tensor,
    rows_per_peer: int,
    ep_size: int,
    group: Any,
) -> torch.Tensor:
    """Move ``rows_per_peer`` rows to each peer and return the peer-major result.

    The split-free call: the backend derives the chunking from the tensor shape
    alone, and the received chunks are concatenated in peer order.  Both the
    eager and the lazy equal-length exchange issue this, so a change here cannot
    make the two disagree about the layout.
    """
    out = x.new_empty((rows_per_peer * ep_size,) + tuple(x.shape[1:]))
    dist.all_to_all_single(out, x, group=group)
    return out


class _EPAllToAllEqual(torch.autograd.Function):  # pylint: disable=abstract-method
    """Equal-length ``all_to_all_single`` (NCCL/HCCL path when counts are uniform).

    Without split sizes every peer gets ``rows_per_peer`` rows -- the backend
    derives the chunking from the tensor shape alone -- and the received chunks
    are concatenated in peer order. That is exactly the layout
    :class:`_EPAllToAllUneven` produces when all counts are equal, so this path
    is interchangeable with it (see :func:`_equal_a2a_rows` for when it applies).

    forward:  all_to_all_single(no splits) -> [ep_size * rows_per_peer, ...]
    backward: the same exchange; an equal-length a2a is its own inverse, and
              with a uniform plan the reverse chunks are the same size.

    ``forward`` issues the exchange on the caller's stream, which orders that
    stream on the transfer before it returns; :class:`_EPAllToAllEqualLazy` is
    the same exchange with the ordering left to the caller.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        rows_per_peer: int,
        ep_size: int,
        group: Any,
    ) -> torch.Tensor:  # pylint: disable=arguments-differ
        """Run the equal-length all_to_all and retain its geometry for backward.

        Args:
            ctx: Autograd context of this exchange.
            x: Payload to exchange; it holds ``ep_size * rows_per_peer`` rows.
            rows_per_peer: Rows handed to each peer.
            ep_size: Number of ranks in ``group``.
            group: Process group to exchange over.

        Returns:
            The received rows, concatenated in peer order.
        """
        ctx.rows_per_peer = rows_per_peer
        ctx.ep_size = ep_size
        ctx.group = group
        return _equal_a2a_exchange(x, rows_per_peer, ep_size, group)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None]:  # pylint: disable=arguments-differ
        """Re-run the self-inverse equal-length exchange on the output gradient.

        Args:
            ctx: Autograd context of the forward exchange.
            grad_output: Output gradient, one row per exchanged row.

        Returns:
            The input gradient, followed by ``None`` for the non-tensor arguments.
        """
        grad = _equal_a2a_exchange(
            grad_output.contiguous(), ctx.rows_per_peer, ctx.ep_size, ctx.group)
        return grad, None, None, None


def _lazy_a2a_resources() -> tuple[Any, Any, Any]:
    """Return the ``(stream, ready_event, done_event)`` the lazy exchange reuses.

    One stream and two events for the whole process, which is deliberate: a
    fresh event per exchange drains the runtime's event pool after a few hundred
    exchanges (the run then dies in ``AclQueryEventRecordedStatus``), and one
    stream is enough because every record and wait below is enqueued in host
    order -- a wait therefore never observes a *stale* record, at worst a later
    one, which only makes it wait longer than it strictly had to.

    Returns:
        The shared comm stream, the event that hands the payload over to it, and
        the event that hands the result back.
    """
    global _LAZY_A2A_STREAM, _LAZY_A2A_READY_EVENT, _LAZY_A2A_DONE_EVENT  # pylint: disable=global-statement
    with _LAZY_A2A_LOCK:
        if _LAZY_A2A_DONE_EVENT is None:
            _LAZY_A2A_STREAM = get_device_handle().Stream()
            _LAZY_A2A_READY_EVENT = get_device_handle().Event()
            _LAZY_A2A_DONE_EVENT = get_device_handle().Event()
        return _LAZY_A2A_STREAM, _LAZY_A2A_READY_EVENT, _LAZY_A2A_DONE_EVENT


def _record_stream(tensor: torch.Tensor, stream: Any) -> None:
    """Tell the caching allocator that ``stream`` touches ``tensor``'s storage.

    Wrapped rather than called inline so the exchange can be driven with a fake
    stream in unit tests: ``Tensor.record_stream`` rejects anything that is not a
    real device stream.
    """
    tensor.record_stream(stream)


def _issue_split_free_a2a(out: torch.Tensor, x: torch.Tensor, group: Any) -> Any:
    """Issue the split-free exchange on the shared comm stream.

    ``dist.all_to_all_single`` without split sizes is the one call shape whose
    backend kernel is the plain alltoall, and it is a *blocking* collective:
    c10d orders the stream it is called on after the transfer before returning,
    which is what would close the caller's overlap window.  Issuing it inside a
    stream context moves that ordering onto the comm stream.  c10d only orders
    the collective against the stream it is called on, so the payload's
    producers are re-connected to that stream explicitly through the ready
    event.

    Args:
        out: Receive buffer, ``[ep_size * rows_per_peer, ...]``; written here.
        x: Contiguous payload, ``[ep_size * rows_per_peer, ...]``; read here.
        group: Process group to exchange over.

    Returns:
        The event recorded when the exchange has completed; the consumer's
        stream waits on it (see :class:`_PendingEqualA2A`).
    """
    compute_stream = get_device_handle().current_stream()
    stream, ready_event, done_event = _lazy_a2a_resources()
    stream_context = get_device_handle().stream
    ready_event.record(compute_stream)
    with stream_context(stream):
        ready_event.wait(stream)
        dist.all_to_all_single(out, x, group=group)
        # Both buffers outlive the call by different means: the receive buffer
        # was allocated on the compute stream, and the payload is a temporary of
        # the caller's dispatch, so its storage may be freed and reused while
        # this stream is still reading it.
        _record_stream(out, stream)
        _record_stream(x, stream)
        done_event.record(stream)
    return done_event


class _PendingEqualA2A:
    """Handle to a split-free exchange that is still in flight.

    ``wait()`` is the only way to read the result, and *where* it is called is
    the caller's decision: the exchange is issued while the dispatch runs, but
    the consumer's stream is only ordered on it at ``wait()``, so independent
    work issued in between (the shared-expert MLP an ``overlap_fn`` runs) works
    against the transfer instead of after it.  Reading the exchange without
    waiting is not an option -- the handle is not a tensor.

    Attributes:
        completed: Whether a stream has already been ordered on the exchange.
    """

    __slots__ = ("completed", "_event", "_tensor")

    def __init__(self, tensor: torch.Tensor, event: Any, completed: bool = False) -> None:
        """Wrap an issued-but-unwaited exchange result."""
        self._tensor = tensor
        self._event = event
        self.completed = completed

    def wait(self) -> torch.Tensor:
        """Order the current stream after the exchange and return its result."""
        if not self.completed:
            self._event.wait(get_device_handle().current_stream())
            self.completed = True
        return self._tensor

    def squeeze(self, dim: int) -> "_PendingEqualA2A":
        """Return a handle to the squeezed result (a view, so the wait is pending).

        The dispatched expert indices arrive as ``[rows, 1]`` and their consumers
        want ``[rows]``.  Squeezing is metadata-only, so it can be taken here
        without materializing: a read here would enqueue the wait before the
        caller reaches its independent work.

        Args:
            dim: Dimension to squeeze, as ``Tensor.squeeze`` takes it.

        Returns:
            A handle to the squeezed result; it still holds the exchange's result
            as pending as this one does.
        """
        return _PendingEqualA2A(self._tensor.squeeze(dim), self._event, self.completed)

    def __repr__(self) -> str:
        """Name the pending exchange and what is known about it without reading it."""
        return (f"PendingEqualA2A(shape={tuple(self._tensor.shape)}, "
                f"completed={self.completed})")


def wait_ep_all_to_all(value: Any) -> Any:
    """Materialize a pending EP exchange.

    Every consumer of an :func:`ep_all_to_all_async` result must pass it through
    here before reading it: with ``HP_EP_EQUAL_A2A=1`` and a uniform plan that
    entry returns a :class:`_PendingEqualA2A` instead of a tensor.

    Args:
        value: The pending handle, an already-materialized tensor from any other
            path of :func:`ep_all_to_all_async`, or ``None``.

    Returns:
        The exchange result as a regular tensor; ``value`` unchanged when it is
        not a pending handle.
    """
    if isinstance(value, _PendingEqualA2A):
        return value.wait()
    return value


class _EPAllToAllEqualLazy(torch.autograd.Function):  # pylint: disable=abstract-method
    """Split-free equal-length exchange issued on the comm stream, waited lazily.

    Same exchange as :class:`_EPAllToAllEqual` and just as differentiable, but
    the forward leaves the caller's stream alone: it issues on the shared comm
    stream and returns the buffer together with the event that says when the
    buffer is final, so the caller can synchronize when it needs the values
    rather than when it issues the exchange.  Both travel out of ``forward`` --
    the buffer as the autograd output, the event as a second, non-tensor output.

    The backward is the plain split-free exchange on the caller's stream: the
    engine consumes its result immediately (it accumulates it onto the input's
    ``grad``), so there is nothing to defer.
    """

    @staticmethod
    def forward(  # pylint: disable=arguments-differ
        ctx: Any,
        x: torch.Tensor,
        rows_per_peer: int,
        ep_size: int,
        group: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Issue the split-free exchange and return ``(buffer, done event)``.

        Args:
            ctx: Autograd context of this exchange.
            x: Contiguous payload, ``[ep_size * rows_per_peer, ...]``.
            rows_per_peer: Rows handed to each peer.
            ep_size: Number of ranks in ``group``.
            group: Process group to exchange over.

        Returns:
            The receive buffer the exchange writes -- the output the autograd
            graph is built on -- and the event that says when it is final.
        """
        ctx.rows_per_peer = rows_per_peer
        ctx.ep_size = ep_size
        ctx.group = group
        out = x.new_empty((rows_per_peer * ep_size,) + tuple(x.shape[1:]))
        return out, _issue_split_free_a2a(out, x, group)

    @staticmethod
    def backward(  # pylint: disable=arguments-differ
        ctx: Any,
        grad_output: torch.Tensor,
        _event_grad: Any,
    ) -> tuple[torch.Tensor, None, None, None]:
        """Re-run the self-inverse exchange, on the caller's stream this time.

        Args:
            ctx: Autograd context of the forward exchange.
            grad_output: Output gradient, one row per exchanged row.
            _event_grad: Gradient slot of the forward's event output; unused.

        Returns:
            The input gradient, followed by ``None`` for the non-tensor arguments.
        """
        grad = _equal_a2a_exchange(
            grad_output.contiguous(), ctx.rows_per_peer, ctx.ep_size, ctx.group)
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
        """Exchange padded expert-token chunks and retain counts for backward."""
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
        """Reverse the exchange: pad by recv_counts, a2a_single, unpad by send_counts."""
        # backward = reversed a2a: pad by recv_counts -> a2a_single -> unpad by send_counts
        recv = _EPAllToAllPadded._pad_and_exchange(
            grad_output.contiguous(), ctx.recv_counts, ctx.pad_to, ctx.group)
        grad = _EPAllToAllPadded._unpad(recv, ctx.send_counts, ctx.pad_to)
        return grad, None, None, None


def _equal_a2a_rows(
    x: torch.Tensor,
    send_counts: list[int],
    recv_counts: list[int],
    group: Any,
) -> Optional[int]:
    """Rows per peer when the exchange may take the equal-length path, else None.

    The equal-length path is only equivalent to the ragged one when the plan is
    uniform *and* the payload really holds one such chunk per peer: the split
    sizes then carry no information, and the plain all-to-all -- which derives
    its chunking from the tensor shape alone -- moves exactly the rows the
    ragged call would.  Anything else (unequal counts, a payload whose row count
    disagrees with the counts, a non-contiguous payload, the knob off, a backend
    without a ragged all-to-all) yields ``None`` so the caller keeps the path it
    runs today.

    Args:
        x: Payload to exchange, split along dim 0.
        send_counts: Rows sent to each EP rank.
        recv_counts: Rows received from each EP rank.
        group: EP process group.

    Returns:
        The uniform per-peer row count, or ``None`` when the fast path does not
        apply.
    """
    if _equal_a2a_mode() == _EQUAL_A2A_OFF or not _backend_supports_uneven_a2a(group):
        return None
    ep_size = len(send_counts)
    if ep_size == 0 or len(recv_counts) != ep_size or not x.is_contiguous():
        return None
    rows_per_peer = send_counts[0]
    # The split-free call takes the chunk size from the tensor shape and needs
    # the receive buffer to be the input's size, so a uniform plan is only
    # equivalent when both count lists hold that same count.
    uniform = (all(count == rows_per_peer for count in send_counts)
               and all(count == rows_per_peer for count in recv_counts))
    if not uniform or x.shape[0] != rows_per_peer * ep_size:
        return None
    return rows_per_peer


def ep_all_to_all(
    x: torch.Tensor,
    send_counts: list[int],
    recv_counts: list[int],
    group: Any,
) -> torch.Tensor:
    """Unified entry for EP token exchange (autograd-differentiable).

    send_counts/recv_counts: list[int], length ep_size, row counts per dest/src rank.
    NCCL/HCCL -> ragged a2a (zero-padding); other backends (gloo test path) -> pad-to-max.
    With ``HP_EP_EQUAL_A2A`` on, a uniform plan is exchanged with the equal-length
    a2a instead (same rows, same order, cheaper kernel).  This entry always
    returns a materialized tensor; :func:`ep_all_to_all_async` is the lazy one.
    """
    if not _backend_supports_uneven_a2a(group):
        return _EPAllToAllPadded.apply(x, send_counts, recv_counts, group)
    rows_per_peer = _equal_a2a_rows(x, send_counts, recv_counts, group)
    if rows_per_peer is not None:
        return _EPAllToAllEqual.apply(x, rows_per_peer, len(send_counts), group)
    return _EPAllToAllUneven.apply(x, send_counts, recv_counts, group)


def ep_all_to_all_async(
    x: torch.Tensor,
    send_counts: list[int],
    recv_counts: list[int],
    group: Any,
    *,
    allow_pending: bool = True,
) -> Any:
    """Non-blocking variant of :func:`ep_all_to_all` (lazy wait).

    On backends whose ``all_to_all_single`` accepts unequal splits the exchange
    is issued through ``differentiable_all_to_all_single_async``, which returns
    an ``AsyncCollectiveTensor``: the ``wait_tensor`` op is only enqueued when a
    non-view op first consumes the result.  Issuing independent work between the
    exchange and that first read therefore overlaps with the in-flight transfer
    (forward and backward alike).

    Backends without that support fall back to the blocking path, so the result
    always carries the same values — only the schedule differs.

    With ``HP_EP_EQUAL_A2A=1`` and a uniform plan the exchange is the split-free
    equal-length one instead -- the cheaper backend kernel -- and it stays lazy:
    :class:`_EPAllToAllEqualLazy` issues it on a stream of its own, so the two
    properties are no longer a trade.  The result is then a
    :class:`_PendingEqualA2A` *handle* rather than a tensor, and every consumer
    must read it through :func:`wait_ep_all_to_all`; that call is where the
    consumer's stream is ordered on the transfer, which is what the caller wants
    to place after its overlapped work.  With ``=eager`` the same split-free
    exchange is issued on the caller's stream and a plain tensor comes back --
    the behaviour ``=1`` used to have, kept for measuring the kernel's gain and
    the lost overlap apart.

    Args:
        x: Input tensor, split along dim 0 by ``send_counts``.
        send_counts: Rows sent to each EP rank.
        recv_counts: Rows received from each EP rank.
        group: EP process group.
        allow_pending: Whether a uniform plan may return a pending handle.  A
            caller whose next step is a view chain over the exchanged buffer
            (the fused states+indices dispatch) passes ``False`` and gets the
            eager split-free tensor instead.

    Returns:
        The exchanged rows, materialized lazily on the async path.
    """
    if not _backend_supports_uneven_a2a(group):
        return _EPAllToAllPadded.apply(x, send_counts, recv_counts, group)
    rows_per_peer = _equal_a2a_rows(x, send_counts, recv_counts, group)
    if rows_per_peer is not None:
        ep_size = len(send_counts)
        if _equal_a2a_mode() == _EQUAL_A2A_LAZY and allow_pending:
            tensor, event = _EPAllToAllEqualLazy.apply(x, rows_per_peer, ep_size, group)
            return _PendingEqualA2A(tensor, event)
        return _EPAllToAllEqual.apply(x, rows_per_peer, ep_size, group)
    return differentiable_all_to_all_single_async(
        x, send_counts, recv_counts, group,
    )
