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
# The distributed doubles below mirror torch's own API names (``ReduceOp``, ``Stream``,
# ``Event``, ...) on purpose: the code under test calls them verbatim.
# pylint: disable=invalid-name
"""Unit tests for the chunked routed exchange (``HP_EP_DISPATCH_CHUNKS``).

The routed all-to-all is the one collective of the MoE step with no independent
work to hide behind (the expert GEMM hard-depends on the arrived tokens), so
``HP_EP_DISPATCH_CHUNKS > 1`` splits the expert-major stream into contiguous
chunks and software-pipelines them.  What these tests pin:

* the per-chunk count/slice algebra is self-consistent *across* ranks (the
  contract the ragged all-to-all needs), reconstructs the original counts, and
  needs no extra collective to learn a chunk's traffic;
* ``1`` (and unset) executes the unchanged schedule bit-for-bit, while ``2/3/4``
  produce the same routed output (grouped GEMM over smaller groups is not
  bit-identical, so the comparison is a tight tolerance);
* the schedule really is pipelined: chunk ``c + 1``'s dispatch is issued before
  chunk ``c``'s experts run, and chunk ``c``'s combine before chunk ``c + 1``'s;
* the lazy wait survives (nothing materializes an exchange before its consumer);
* an invalid knob fails loudly.

No process group is created: every exchange is driven by a barrier world inside
this process, in which each rank runs in its own thread.  The world implements
the ``all_to_all`` contract (including the zero-padding gloo fallback), so a
chunk mis-derivation shows up as a hard shape assertion rather than as a wrong
number.
"""
import os
import threading
import unittest
from unittest import mock

import torch
from torch import nn

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from hyper_parallel.distributed.expert_parallel import collectives as ep_collectives  # noqa: E402
from hyper_parallel.distributed.expert_parallel import experts as ep_experts  # noqa: E402
from hyper_parallel.distributed.expert_parallel.experts import (  # noqa: E402
    _chunk_rank_group,
    _chunk_rank_range,
    _ep_dispatch_chunks,
    _resolve_dispatch_chunks,
    bind_local_expert_forward,
    ep_routed_dispatch,
    ep_routed_experts_and_combine,
    ep_routed_forward,
)

_EP_SIZE = 4
_LOCAL_EXPERTS = 2
_HIDDEN = 4
_INTERMEDIATE = 3
_TOKEN_COUNT = 4
_TOP_K = 2
_BATCH, _SEQ = 2, 2
_BARRIER_TIMEOUT = 20.0

# Unequal routing plan: destination 2 is fed only by rank 1 and destination 1 is
# fed only by rank 0, so a chunk plan built from "an equal slice of the stream"
# rather than from the exchanged counts cannot pass.
_SLOTS = {
    0: [0, 1, 0, 3, 1, 0, 2, 3],
    1: [4, 5, 6, 4, 5, 6, 7, 4],
    2: [6, 7, 7, 6, 6, 7, 7, 6],
    3: [1, 0, 0, 1, 0, 1, 1, 0],
}

_REFERENCE_MODE = "reference"
_FORWARD_MODE = "forward"
_SPLIT_MODE = "split"
_ISSUE_MODE = "issue"


class _VirtualEpGroup:
    """Size+rank EP group double: carries the rank the calling thread runs as."""

    def __init__(self, rank, size):
        self.rank = rank
        self._size = size

    def size(self):
        """Return the EP group size."""
        return self._size


class _VirtualEpWorld:
    """One-process EP group double for the ragged token exchange.

    Every rank runs in its own thread and every collective is a barrier: a rank
    posts the chunks it splits out of its send buffer, waits for its peers, and
    fills its receive buffer from their chunks -- the ``dist.all_to_all``
    contract :class:`_EPAllToAllUneven` implements.  The per-source shape check
    is what makes a chunk plan self-consistent *across* ranks: a rank that
    derives a chunk boundary its peers do not derive fails right here.

    ``wrap_async`` hands back a real ``AsyncCollectiveTensor``, which pins the
    lazy-wait behaviour (an exchange is issued, not consumed) without a process
    group.
    """

    def __init__(self, ep_size, wrap_async=False):
        self.ep_size = ep_size
        self.wrap_async = wrap_async
        self.barrier = threading.Barrier(ep_size, timeout=_BARRIER_TIMEOUT)
        self._lock = threading.Lock()
        self._calls = {}
        self._index = {}
        self.counts_calls = []
        self.trace = []
        self.pending_flags = []
        self.handles = {rank: [] for rank in range(ep_size)}
        self.snapshots = []
        self.rows = {rank: [0, 0] for rank in range(ep_size)}

    def _record(self, rank, event):
        """Append one schedule event (rank order is preserved per rank)."""
        with self._lock:
            self.trace.append((rank, event))
            if event == "gemm":
                self.snapshots.append(
                    (rank, [handle.completed for handle in self.handles[rank]]))

    def record_pending(self, allowed: bool) -> None:
        """Record an exchange that explicitly asked for (or refused) a lazy handle."""
        with self._lock:
            self.pending_flags.append(allowed)

    def events(self, rank):
        """Return one rank's schedule events, in order."""
        with self._lock:
            return [event for event_rank, event in self.trace if event_rank == rank]

    def gemm_snapshots(self, rank):
        """Return the handle states recorded when this rank's expert calls started."""
        with self._lock:
            return [flags for event_rank, flags in self.snapshots if event_rank == rank]

    def _next_call(self, rank):
        """Reserve this rank's next exchange slot (ranks call them in lockstep)."""
        with self._lock:
            index = self._index.get(rank, 0)
            self._index[rank] = index + 1
            return index

    def _post(self, rank, index, pieces, recv_counts):
        """Publish this rank's chunks for ``index`` and wait for every peer."""
        with self._lock:
            self._calls[(rank, index)] = pieces
        self.barrier.wait()

    def counts_exchange(self, rank, output_tensor, input_tensor):
        """Stand-in for the counts ``all_to_all_single`` of _prepare_ep_dispatch."""
        with self._lock:
            self.counts_calls.append(rank)
        index = self._next_call(rank)
        self._post(rank, index, list(input_tensor.split(1)), [1] * self.ep_size)
        output_tensor.copy_(torch.cat(
            [self._calls[(src, index)][rank] for src in range(self.ep_size)]))

    def token_exchange(self, rank, tensor, send_counts, recv_counts, kind):
        """Stand-in for ``ep_all_to_all`` / ``ep_all_to_all_async``."""
        index = self._next_call(rank)
        self._post(rank, index, list(tensor.split(send_counts)), recv_counts)
        buffer = tensor.new_empty((sum(recv_counts),) + tuple(tensor.shape[1:]))
        offset = 0
        with self._lock:
            self.rows[rank][0] += tensor.shape[0]
            self.rows[rank][1] += buffer.shape[0]
        for src in range(self.ep_size):
            piece = self._calls[(src, index)][rank]
            assert piece.shape[0] == recv_counts[src], (
                f"rank {rank} exchange {index}: chunk from rank {src} has "
                f"{piece.shape[0]} rows, this rank's chunk plan expects {recv_counts[src]}")
            buffer[offset:offset + recv_counts[src]] = piece
            offset += recv_counts[src]
        self._record(rank, kind)
        if self.wrap_async:
            from torch.distributed._functional_collectives import (  # pylint: disable=C0415
                AsyncCollectiveTensor,
            )
            handle = AsyncCollectiveTensor(buffer)
            with self._lock:
                self.handles[rank].append(handle)
            return handle
        return buffer


class _VirtualEpDist:
    """Stand-in for the ``dist`` module ``experts`` resolves rank/counts through."""

    def __init__(self, world):
        self._world = world

    def get_rank(self, group=None):
        """Return the rank of the double group passed in."""
        return group.rank

    def all_to_all_single(self, output, input_tensor, group=None):
        """Route the per-rank dispatch counts through the world."""
        self._world.counts_exchange(group.rank, output, input_tensor)


class _PaddedEpWorld:
    """One-process gloo double: equal-length collectives + pad-to-max a2a.

    gloo has no ragged all-to-all, so :func:`ep_all_to_all` falls back to
    ``pad-to-max + all_to_all_single``.  That fallback is what the CPU tests
    would run on, so this world implements the two collectives it needs
    (``all_reduce`` for the padded length, ``all_to_all_single`` with equal
    splits) and nothing else.
    """

    def __init__(self, ep_size):
        self.ep_size = ep_size
        self.barrier = threading.Barrier(ep_size, timeout=_BARRIER_TIMEOUT)
        self._lock = threading.Lock()
        self._calls = {}
        self._index = {}
        self._maxima = {}

    def _next_call(self, rank):
        """Reserve this rank's next collective slot."""
        with self._lock:
            index = self._index.get(rank, 0)
            self._index[rank] = index + 1
            return index

    def all_reduce(self, tensor, op=None, group=None):
        """MAX reduction of a single-element tensor (the padded length)."""
        del op
        index = self._next_call(group.rank)
        with self._lock:
            self._maxima[(group.rank, index)] = int(tensor.item())
        self.barrier.wait()
        tensor.fill_(max(self._maxima[(rank, index)] for rank in range(self.ep_size)))

    def all_to_all_single(self, output, input_tensor, input_splits=None,
                          output_splits=None, group=None):
        """Equal-split all_to_all_single (one row block per rank)."""
        del input_splits, output_splits
        rank = group.rank
        index = self._next_call(rank)
        pieces = list(input_tensor.split(input_tensor.shape[0] // self.ep_size))
        with self._lock:
            self._calls[(rank, index)] = pieces
        self.barrier.wait()
        output.copy_(torch.cat(
            [self._calls[(src, index)][rank] for src in range(self.ep_size)]))

    def get_backend(self, group=None):
        """Report a backend without a ragged all-to-all."""
        del group
        return "gloo"


class _PaddedEpDist:
    """Stand-in for the ``dist`` module ``collectives`` resolves through."""

    ReduceOp = torch.distributed.ReduceOp

    def __init__(self, world):
        self._world = world

    def get_backend(self, group=None):
        """Forward the backend query."""
        return self._world.get_backend(group)

    def all_reduce(self, tensor, op=None, group=None):
        """Forward the MAX reduction."""
        return self._world.all_reduce(tensor, op, group)

    def all_to_all_single(self, output, input_tensor, input_splits=None,
                          output_splits=None, group=None):
        """Forward the equal-split exchange."""
        return self._world.all_to_all_single(
            output, input_tensor, input_splits, output_splits, group)


def _build_moe(rank, world):
    """Build one rank's MoE double with the real bound SwiGLU expert entry.

    Every rank holds its own expert weights (as a real EP rank does) and every
    expert call is traced, so the schedule the pipeline claims to emit is
    observable.
    """
    module = nn.Module()
    experts = nn.Module()
    experts.num_experts = _LOCAL_EXPERTS * _EP_SIZE
    experts.local_expert_count = _LOCAL_EXPERTS
    generator = torch.Generator().manual_seed(100 + rank)
    experts.w1 = nn.Parameter(torch.randn(_LOCAL_EXPERTS, _INTERMEDIATE, _HIDDEN,
                                          generator=generator) * 0.2)
    experts.w3 = nn.Parameter(torch.randn(_LOCAL_EXPERTS, _INTERMEDIATE, _HIDDEN,
                                          generator=generator) * 0.2)
    experts.w2 = nn.Parameter(torch.randn(_LOCAL_EXPERTS, _HIDDEN, _INTERMEDIATE,
                                          generator=generator) * 0.2)
    experts.act_fn = torch.nn.functional.silu
    module.experts = experts
    bind_local_expert_forward(module, _EP_SIZE)
    bound_forward = module.experts.forward

    def traced_forward(*args, **kwargs):
        """Record the experts call that the pipeline schedules."""
        world._record(rank, "gemm")  # pylint: disable=protected-access
        return bound_forward(*args, **kwargs)

    module.experts.forward = traced_forward
    return module


def _scenario(seed=7):
    """Fresh per-rank inputs: hidden states, routing slots and top-k weights."""
    hidden = {
        rank: torch.randn(_BATCH, _SEQ, _HIDDEN, generator=torch.Generator().manual_seed(seed + rank))
        for rank in range(_EP_SIZE)
    }
    topk = {
        rank: torch.tensor(_SLOTS[rank], dtype=torch.int64).view(_TOKEN_COUNT, _TOP_K)
        for rank in range(_EP_SIZE)
    }
    weights = {
        rank: torch.rand(_TOKEN_COUNT, _TOP_K, generator=torch.Generator().manual_seed(41 + rank))
        for rank in range(_EP_SIZE)
    }
    return hidden, topk, weights


def _run_ranks(mode, chunk_count, *, fused=False, wrap_async=False, seed=7,
               trace_handles=False, differentiable=False, backward=False):
    """Drive every virtual rank once and return its outputs and schedule.

    Args:
        mode: ``"reference"`` (the unchunked composition, called directly),
            ``"forward"`` (:func:`ep_routed_forward`), ``"split"``
            (:func:`ep_routed_dispatch` + :func:`ep_routed_experts_and_combine`)
            or ``"issue"`` (dispatch only, no experts).
        chunk_count: Value the ``HP_EP_DISPATCH_CHUNKS`` knob takes.
        fused: Value of the fused states+indices dispatch switch.
        wrap_async: Hand every exchange back as an unfinished async handle.
        seed: Input seed.
        trace_handles: Also return the per-rank pending split-path states.
        differentiable: Run the exchange as a real autograd Function (its
            backward is the reverse exchange).  Without it the world's plain
            tensor ops entangle the ranks' graphs, which only matters when the
            test backpropagates.
        backward: Mark the inputs as leaves, backpropagate each rank's sum, and
            return the hidden-state gradients.

    Returns:
        ``(outputs, states, world, grads)``: the per-rank routed output, the
        per-rank :class:`EPRoutedState` when requested, the world double, and
        the per-rank input gradients when requested.
    """
    hidden, topk, weights = _scenario(seed)
    for rank in range(_EP_SIZE):
        hidden[rank].requires_grad_(backward)
    world = _VirtualEpWorld(_EP_SIZE, wrap_async=wrap_async)
    outputs, states, grads, failures = {}, {}, {}, []

    def run_rank(rank):
        """One virtual rank: build the module and run the requested entry."""
        try:
            module = _build_moe(rank, world)
            ep_group = _VirtualEpGroup(rank, _EP_SIZE)
            def router(module_, hidden_states):
                """Return this rank's pre-computed plan (the trace rebuilds it)."""
                del module_, hidden_states
                return topk[rank], weights[rank]
            if mode == _REFERENCE_MODE:
                outputs[rank] = _unchunked_reference(module, hidden[rank], router, ep_group, rank)
            elif mode == _FORWARD_MODE:
                outputs[rank] = ep_routed_forward(
                    module, hidden[rank], router_fn=router, ep_group=ep_group)
            elif mode == _SPLIT_MODE:
                state = ep_routed_dispatch(
                    module, hidden[rank], router_fn=router, ep_group=ep_group)
                states[rank] = state
                outputs[rank] = ep_routed_experts_and_combine(module, state, ep_group)
            elif mode == _ISSUE_MODE:
                states[rank] = ep_routed_dispatch(
                    module, hidden[rank], router_fn=router, ep_group=ep_group)
            else:
                raise AssertionError(f"unknown mode {mode!r}")
            if backward:
                outputs[rank].sum().backward()
                grads[rank] = hidden[rank].grad.clone()
        except BaseException as exc:  # pylint: disable=broad-except
            failures.append((rank, exc))

    def exchange(kind):
        """Build the stand-in :func:`ep_all_to_all` / ``..._async`` for this run."""
        def call(x, send_counts, recv_counts, group, **kwargs):
            """Hand one exchange to the world (as a Function when differentiable)."""
            if "allow_pending" in kwargs:
                world.record_pending(kwargs["allow_pending"])
            if differentiable:
                return _VirtualA2A.apply(group.rank, x, send_counts, recv_counts, world, kind)
            return world.token_exchange(group.rank, x, send_counts, recv_counts, kind)
        return call

    with mock.patch.multiple(
            ep_experts,
            ep_all_to_all_async=exchange("async"),
            ep_all_to_all=exchange("sync"),
            dist=_VirtualEpDist(world),
            _DISPATCH_CHUNKS_RAW=str(chunk_count),
            _FUSED_DISPATCH_ENABLED=fused):
        threads = [threading.Thread(target=run_rank, args=(rank,)) for rank in range(_EP_SIZE)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    if failures:
        raise failures[0][1]
    return outputs, states if trace_handles else None, world, grads if backward else None


class _VirtualA2A(torch.autograd.Function):
    """Differentiable stand-in for the EP all-to-all.

    The world's plain tensor ops connect every receiver's graph to its peers'
    inputs, which is not the production contract: the real exchange is an
    autograd Function whose backward is the reverse exchange, so each rank
    backpropagates through its own graph only.  This double has exactly that
    contract.
    """

    @staticmethod
    def forward(ctx, rank, tensor, send_counts, recv_counts, world, kind):  # pylint: disable=arguments-differ
        """Run the world's ragged exchange and retain the counts for backward."""
        ctx.rank = rank
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.world = world
        ctx.kind = kind
        return world.token_exchange(rank, tensor, send_counts, recv_counts, kind)

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        """Reverse the exchange, exactly as the real a2a does."""
        grad = ctx.world.token_exchange(
            ctx.rank, grad_output.contiguous(), ctx.recv_counts, ctx.send_counts, ctx.kind)
        return None, grad, None, None, None, None


def _unchunked_reference(module, hidden_states, router_fn, ep_group, rank):
    """The pre-change routed branch, called straight through its primitives.

    This is the composition :func:`ep_routed_forward` ran before the chunked
    path existed; comparing against it is what makes "``HP_EP_DISPATCH_CHUNKS``
    unset is bit-identical" a checked claim rather than a code-reading exercise.
    """
    topk_indices, topk_weights = router_fn(module, hidden_states)
    dispatch = ep_experts._prepare_ep_dispatch(  # pylint: disable=protected-access
        hidden_states,
        topk_indices,
        topk_weights,
        local_expert_count=_LOCAL_EXPERTS,
        global_expert_count=_LOCAL_EXPERTS * _EP_SIZE,
        ep_size=_EP_SIZE,
        ep_group=ep_group,
    )
    source_indices = dispatch.source_indices
    flattened_weights = dispatch.expert_weights
    dispatch_order = dispatch.dispatch_order
    dispatched_states = dispatch.states
    dispatched_indices = dispatch.expert_indices
    send_counts = dispatch.send_counts
    receive_counts = dispatch.receive_counts
    combined = ep_experts._run_ep_local_experts(  # pylint: disable=protected-access
        module,
        dispatched_states,
        dispatched_indices,
        send_counts,
        receive_counts,
        ep_group,
        rank * _LOCAL_EXPERTS,
    )
    return ep_experts._aggregate_ep_outputs(  # pylint: disable=protected-access
        combined,
        flattened_weights,
        source_indices,
        dispatch_order,
        (_BATCH, _SEQ, _HIDDEN),
    )


def _transpose(plan, ep_size):
    """Row ``i`` of the transposed plan: what rank ``i`` receives from each peer."""
    return [[plan[src][rank] for src in range(ep_size)] for rank in range(ep_size)]


class TestDispatchChunkPlan(unittest.TestCase):
    """The per-chunk count/slice algebra, checked without any exchange."""

    def setUp(self):
        """Build the unequal send plan the cases derive their chunks from."""
        self.ep_size = _EP_SIZE
        self.send_plan = {
            rank: [sum(1 for slot in _SLOTS[rank] if slot // _LOCAL_EXPERTS == dest)
                   for dest in range(self.ep_size)]
            for rank in range(self.ep_size)
        }
        self.rows = {rank: len(_SLOTS[rank]) for rank in range(self.ep_size)}

    def _plan(self, rank, chunk_count):
        """Return rank ``rank``'s chunk plan for the uneven scenario."""
        send_counts = self.send_plan[rank]
        receive_counts = _transpose(
            [self.send_plan[src] for src in range(self.ep_size)], self.ep_size)[rank]
        return _ep_dispatch_chunks(
            send_counts, receive_counts, self.ep_size, chunk_count, rank)

    def test_scenario_is_uneven(self):
        """The scenario must stay uneven, or the chunk cases prove nothing."""
        self.assertEqual(self.send_plan, {
            0: [5, 3, 0, 0], 1: [0, 0, 5, 3], 2: [0, 0, 0, 8], 3: [8, 0, 0, 0],
        })

    def test_chunk_counts_reconstruct_the_originals(self):
        """Every chunk's counts add back up to the full send/receive counts."""
        for chunk_count in (2, 3, 4):
            for rank in range(self.ep_size):
                plan = self._plan(rank, chunk_count)
                with self.subTest(chunk_count=chunk_count, rank=rank):
                    self.assertEqual(len(plan.send_counts), chunk_count)
                    self.assertEqual(len(plan.row_ranges), chunk_count)
                    send_totals = [
                        sum(chunk[dest] for chunk in plan.send_counts)
                        for dest in range(self.ep_size)]
                    self.assertEqual(send_totals, self.send_plan[rank])
                    receive_totals = [
                        sum(chunk[src] for chunk in plan.recv_counts)
                        for src in range(self.ep_size)]
                    self.assertEqual(
                        receive_totals,
                        _transpose([self.send_plan[src] for src in range(self.ep_size)],
                                   self.ep_size)[rank])

    def test_chunk_counts_are_consistent_across_ranks(self):
        """Sender chunk counts equal the receiver chunk counts for the same chunk.

        This is the contract the ragged all-to-all needs: chunk ``c`` of rank
        ``i`` and chunk ``c`` of rank ``j`` must agree on how many rows travel
        from ``i`` to ``j``.
        """
        for chunk_count in (2, 3, 4):
            plans = [self._plan(rank, chunk_count) for rank in range(self.ep_size)]
            for src in range(self.ep_size):
                for dest in range(self.ep_size):
                    for chunk in range(chunk_count):
                        with self.subTest(chunk_count=chunk_count, src=src, dest=dest,
                                          chunk=chunk):
                            self.assertEqual(
                                plans[src].send_counts[chunk][dest],
                                plans[dest].recv_counts[chunk][src])

    def test_row_ranges_partition_the_stream(self):
        """The chunk slices are contiguous, in order, and cover every row once."""
        for chunk_count in (2, 3, 4):
            for rank in range(self.ep_size):
                plan = self._plan(rank, chunk_count)
                with self.subTest(chunk_count=chunk_count, rank=rank):
                    ranges = sorted(plan.row_ranges)
                    self.assertEqual(ranges[0][0], 0)
                    self.assertEqual(ranges[-1][1], self.rows[rank])
                    for (_, end), (next_start, _) in zip(ranges, ranges[1:]):
                        self.assertEqual(end, next_start)
                    for (start, end), chunk_send in zip(plan.row_ranges, plan.send_counts):
                        self.assertEqual(end - start, sum(chunk_send))

    def test_rows_in_a_chunk_all_go_to_that_chunk_rank_group(self):
        """A chunk's slice holds exactly the rows of its destination rank group."""
        for chunk_count in (2, 3, 4):
            for rank in range(self.ep_size):
                plan = self._plan(rank, chunk_count)
                own_group = _chunk_rank_group(rank, self.ep_size, chunk_count)
                destinations = [
                    sum(self.send_plan[rank][:dest]) for dest in range(self.ep_size + 1)]
                for chunk, (start, end) in enumerate(plan.row_ranges):
                    expected_group = (chunk + own_group) % chunk_count
                    first, last = _chunk_rank_range(expected_group, self.ep_size, chunk_count)
                    with self.subTest(chunk_count=chunk_count, rank=rank, chunk=chunk):
                        self.assertEqual((start, end), (destinations[first], destinations[last]))
                        for dest in range(self.ep_size):
                            expected = self.send_plan[rank][dest] if first <= dest < last else 0
                            self.assertEqual(plan.send_counts[chunk][dest], expected)

    def test_one_chunk_moves_the_whole_stream(self):
        """``chunk_count=1`` degenerates to the unchunked plan."""
        for rank in range(self.ep_size):
            plan = self._plan(rank, 1)
            with self.subTest(rank=rank):
                self.assertEqual(plan.send_counts[0], self.send_plan[rank])
                self.assertEqual(plan.row_ranges, [(0, self.rows[rank])])

    def test_rank_that_routes_nothing_gets_empty_chunks(self):
        """A rank with no rows at all still gets a full, consistent plan."""
        plan = _ep_dispatch_chunks([0, 0], [0, 0], 2, 2, 0)
        self.assertEqual(plan.row_ranges, [(0, 0), (0, 0)])
        self.assertEqual(plan.send_counts, [[0, 0], [0, 0]])
        self.assertEqual(plan.recv_counts, [[0, 0], [0, 0]])

    def test_rank_groups_partition_every_rank(self):
        """The groups are contiguous, non-empty, and own every rank exactly once."""
        for ep_size, chunk_count in ((4, 2), (4, 3), (4, 4), (8, 3), (8, 5), (2, 2), (1, 1)):
            groups = [_chunk_rank_group(rank, ep_size, chunk_count) for rank in range(ep_size)]
            covered = []
            for group in range(chunk_count):
                first, last = _chunk_rank_range(group, ep_size, chunk_count)
                covered.extend(range(first, last))
                with self.subTest(ep_size=ep_size, chunk_count=chunk_count, group=group):
                    self.assertLess(first, last)
                    for rank in range(first, last):
                        self.assertEqual(groups[rank], group)
            with self.subTest(ep_size=ep_size, chunk_count=chunk_count):
                self.assertEqual(covered, list(range(ep_size)))


class TestDispatchChunkKnob(unittest.TestCase):
    """``HP_EP_DISPATCH_CHUNKS`` parsing: default, clamping and rejection."""

    def _resolve(self, raw, ep_size):
        """Resolve the knob with ``raw`` as its value."""
        with mock.patch.object(ep_experts, "_DISPATCH_CHUNKS_RAW", raw):
            return _resolve_dispatch_chunks(ep_size)

    def test_default_and_one_keep_the_unchunked_schedule(self):
        """Unset (1) resolves to a single chunk for every group size."""
        for raw in ("1", " 1 "):
            for ep_size in (1, 2, 8):
                with self.subTest(raw=raw, ep_size=ep_size):
                    self.assertEqual(self._resolve(raw, ep_size), 1)

    def test_value_is_used_up_to_the_ep_size(self):
        """A valid chunk count is used as given while it fits the group."""
        self.assertEqual(self._resolve("2", 4), 2)
        self.assertEqual(self._resolve("3", 4), 3)
        self.assertEqual(self._resolve("4", 4), 4)

    def test_value_above_the_ep_size_is_clamped(self):
        """More chunks than ranks would only add empty slices, so clamp."""
        self.assertEqual(self._resolve("8", 4), 4)
        self.assertEqual(self._resolve("2", 1), 1)

    def test_invalid_values_are_rejected(self):
        """Zero, negative and non-integer values fail with the knob named."""
        for raw in ("0", "-1", "", "abc", "2.5", "1e3", "0x2"):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, "HP_EP_DISPATCH_CHUNKS"):
                    self._resolve(raw, 4)

    def test_unset_environment_imports_as_one(self):
        """The module-level default is the unchunked schedule."""
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HP_EP_DISPATCH_CHUNKS", None)
            self.assertEqual(os.environ.get("HP_EP_DISPATCH_CHUNKS", "1"), "1")
        self.assertEqual(ep_experts._DISPATCH_CHUNKS_RAW,  # pylint: disable=protected-access
                         os.environ.get("HP_EP_DISPATCH_CHUNKS", "1"))


class TestChunkedRoutedForward(unittest.TestCase):
    """End-to-end equivalence of the chunked schedule on a virtual EP group."""

    def _assert_close(self, actual, expected, label):
        """Assert two routed outputs agree within float32 regrouping noise."""
        difference = (actual - expected).abs().max().item()
        self.assertTrue(
            torch.allclose(actual, expected, rtol=1e-5, atol=1e-6),
            f"{label}: max abs difference {difference:.3e}")

    def test_unchunked_path_is_bit_identical_to_the_previous_composition(self):
        """``HP_EP_DISPATCH_CHUNKS=1`` reproduces the pre-change routed branch."""
        reference, _, _, _ = _run_ranks(_REFERENCE_MODE, 1)
        unchunked, _, world, _ = _run_ranks(_FORWARD_MODE, 1)
        for rank in range(_EP_SIZE):
            with self.subTest(rank=rank):
                self.assertTrue(
                    torch.equal(unchunked[rank], reference[rank]),
                    f"rank {rank}: unchunked routed output differs from the "
                    f"unchunked composition by "
                    f"{(unchunked[rank] - reference[rank]).abs().max().item():.3e}")
        for rank in range(_EP_SIZE):
            with self.subTest(rank=rank, schedule="unchunked"):
                self.assertEqual(world.events(rank), ["sync", "sync", "gemm", "sync"])

    def test_chunked_forward_matches_the_unchunked_output(self):
        """2/3/4 chunks route the same tokens to the same places."""
        reference, _, _, _ = _run_ranks(_REFERENCE_MODE, 1)
        for chunk_count in (2, 3, 4):
            outputs, _, _, _ = _run_ranks(_FORWARD_MODE, chunk_count)
            for rank in range(_EP_SIZE):
                with self.subTest(chunk_count=chunk_count, rank=rank):
                    self._assert_close(outputs[rank], reference[rank],
                                       f"chunks={chunk_count} rank={rank}")

    def test_chunked_split_path_matches_the_unchunked_output(self):
        """The overlap_shared_expert split point keeps the same routed output."""
        reference, _, _, _ = _run_ranks(_REFERENCE_MODE, 1)
        for chunk_count in (2, 3, 4):
            outputs, states, _, _ = _run_ranks(_SPLIT_MODE, chunk_count, trace_handles=True)
            for rank in range(_EP_SIZE):
                with self.subTest(chunk_count=chunk_count, rank=rank):
                    self._assert_close(outputs[rank], reference[rank],
                                       f"split chunks={chunk_count} rank={rank}")
            # ``trace_handles=True`` makes the helper return the per-rank states; the checker
            # still reads its ``states if trace_handles else None`` as Optional.
            # pylint: disable=unsubscriptable-object
            self.assertIsNotNone(states[0].chunks)
            self.assertIsNone(states[0].received_states)
            self.assertEqual(len(states[0].chunks.plan.row_ranges), chunk_count)

    def test_chunked_fused_dispatch_matches_the_unchunked_output(self):
        """The fused states+indices exchange works per chunk as well."""
        reference, _, _, _ = _run_ranks(_REFERENCE_MODE, 1)
        outputs, _, world, _ = _run_ranks(_SPLIT_MODE, 2, fused=True)
        for rank in range(_EP_SIZE):
            with self.subTest(rank=rank):
                self._assert_close(outputs[rank], reference[rank], f"fused rank={rank}")
        for rank in range(_EP_SIZE):
            with self.subTest(rank=rank, schedule="fused"):
                self.assertEqual(world.events(rank),
                                 ["async", "async", "gemm", "async", "gemm", "async"])
        # The fused unpack is a view chain over the exchanged buffer, which only
        # a materializing exchange result can carry: the fused path must keep
        # asking for one (``allow_pending=False``) instead of the lazy handle
        # the split-free exchange returns for a uniform plan.
        self.assertEqual(world.pending_flags, [False] * (2 * _EP_SIZE),
                         f"fused exchanges asked for pending handles: {world.pending_flags}")

    def test_cohesive_path_ignores_the_fused_switch_for_its_chunks(self):
        """``ep_routed_forward`` never fused its exchange; its chunks do not either."""
        _, _, world, _ = _run_ranks(_FORWARD_MODE, 2, fused=True)
        for rank in range(_EP_SIZE):
            with self.subTest(rank=rank):
                self.assertEqual(world.events(rank),
                                 ["async", "async", "async", "async",
                                  "gemm", "async", "gemm", "async"])

    def test_chunked_backward_matches_the_unchunked_gradient(self):
        """Autograd stays correct: the reverse exchange runs once per chunk."""
        gradients = {}
        for chunk_count in (1, 2):
            _, _, _, grads = _run_ranks(_FORWARD_MODE, chunk_count,
                                        differentiable=True, backward=True)
            gradients[chunk_count] = grads

        for rank in range(_EP_SIZE):
            with self.subTest(rank=rank):
                self.assertTrue(bool(gradients[1][rank].abs().sum() > 0),
                                f"rank {rank}: the gradient is all zero, so this case proves nothing")
                self.assertTrue(
                    torch.allclose(gradients[2][rank], gradients[1][rank],
                                   rtol=1e-5, atol=1e-6),
                    f"rank {rank}: chunked gradient differs by "
                    f"{(gradients[2][rank] - gradients[1][rank]).abs().max().item():.3e}")

    def test_pipeline_issues_the_next_dispatch_before_the_experts(self):
        """The schedule is the pipeline: dispatch(c+1) before GEMM(c) before combine(c)."""
        for chunk_count in (2, 3, 4):
            for mode in (_FORWARD_MODE, _SPLIT_MODE):
                _, _, world, _ = _run_ranks(mode, chunk_count)
                with self.subTest(chunk_count=chunk_count, mode=mode):
                    events = world.events(0)
                    expected = []
                    if mode == _FORWARD_MODE:
                        expected.extend(["async", "async"])
                        for chunk in range(chunk_count):
                            if chunk + 1 < chunk_count:
                                expected.extend(["async", "async"])
                            expected.extend(["gemm", "async"])
                    else:
                        expected.extend(["async", "async"] * chunk_count)
                        for _ in range(chunk_count):
                            expected.extend(["gemm", "async"])
                    self.assertEqual(events, expected)

    def test_no_chunk_exchange_carries_more_rows_than_the_unchunked_one(self):
        """Chunking splits the same traffic; it never adds rows or counts calls."""
        _, _, unchunked, _ = _run_ranks(_FORWARD_MODE, 1)
        for chunk_count in (2, 3, 4):
            _, _, world, _ = _run_ranks(_FORWARD_MODE, chunk_count)
            for rank in range(_EP_SIZE):
                with self.subTest(chunk_count=chunk_count, rank=rank):
                    self.assertEqual(world.rows[rank], unchunked.rows[rank])
                    self.assertEqual(sorted(world.counts_calls), sorted(unchunked.counts_calls))

    def test_chunk_exchanges_are_not_materialized_before_their_consumer(self):
        """The chunked dispatch is issued, not consumed: its handles stay pending."""
        _, states, world, _ = _run_ranks(_ISSUE_MODE, 2, wrap_async=True,
                                         trace_handles=True)
        # ``trace_handles=True`` makes the helper return the per-rank states; the checker
        # still reads its ``states if trace_handles else None`` as Optional.
        # pylint: disable=unsubscriptable-object
        for rank in range(_EP_SIZE):
            handles = (states[rank].chunks.received_states
                       + states[rank].chunks.received_indices)
            self.assertEqual(len(handles), 2 * 2)
            for handle in handles:
                with self.subTest(rank=rank):
                    self.assertIs(handle.completed, False)
        self.assertFalse(any(handle.completed
                             for handles in world.handles.values() for handle in handles))

    def test_chunk_combines_are_not_materialized_before_the_aggregation(self):
        """A combine issued after chunk ``c`` stays pending while chunk ``c + 1`` runs."""
        outputs, _, world, _ = _run_ranks(_SPLIT_MODE, 2, wrap_async=True)
        reference, _, _, _ = _run_ranks(_REFERENCE_MODE, 1)
        snapshots = world.gemm_snapshots(0)
        self.assertEqual(len(snapshots), 2)
        # Chunk 0's experts consumed chunk 0's dispatch handles, and chunk 0's
        # combine -- issued after them -- is still pending when chunk 1's experts
        # start.  Had anything read it in between, the combine would serialize
        # against chunk 1's GEMM instead of overlapping it.
        #
        # The expert indices travel through a squeeze view, so the handle that
        # records the wait for them is that view, not the wrapper in the list;
        # the cases below therefore assert on the handles the code consumes
        # directly (the states, and each chunk's combine output).
        self.assertEqual(snapshots[0], [False, False, False, False])
        self.assertIs(snapshots[1][0], True)
        self.assertIs(snapshots[1][-1], False)
        for rank in range(_EP_SIZE):
            with self.subTest(rank=rank):
                self._assert_close(outputs[rank], reference[rank], f"async chunks rank={rank}")
        self.assertTrue(all(handle.completed for handle in world.handles[0][-2:]))
        self.assertTrue(all(handle.completed
                            for handle in (world.handles[0][0], world.handles[0][2])))


class TestChunkedPaddedFallback(unittest.TestCase):
    """The gloo pad-to-max fallback consumes the chunked counts unchanged."""

    def _run(self, chunk_count):
        """Return the padded exchange's output for every rank and chunk."""
        send_plan = [
            [3, 1, 0, 4],
            [0, 2, 5, 0],
            [1, 0, 0, 2],
            [4, 4, 1, 1],
        ]
        plans = [
            _ep_dispatch_chunks(send_plan[rank],
                                _transpose(send_plan, _EP_SIZE)[rank],
                                _EP_SIZE, chunk_count, rank)
            for rank in range(_EP_SIZE)
        ]
        tensors = [
            {chunk: torch.randn(sum(plans[rank].send_counts[chunk]), 3,
                                generator=torch.Generator().manual_seed(3 + rank))
             for chunk in range(chunk_count)}
            for rank in range(_EP_SIZE)
        ]
        padded = _PaddedEpWorld(_EP_SIZE)
        ragged = _VirtualEpWorld(_EP_SIZE)
        results = {"padded": {}, "ragged": {}}
        failures = []

        def run_rank(rank):
            """Exchange every chunk of this rank through both worlds."""
            try:
                group = _VirtualEpGroup(rank, _EP_SIZE)
                ragged_group = _VirtualEpGroup(rank, _EP_SIZE)
                for chunk in range(chunk_count):
                    results["padded"][(rank, chunk)] = ep_collectives.ep_all_to_all(
                        tensors[rank][chunk],
                        plans[rank].send_counts[chunk],
                        plans[rank].recv_counts[chunk],
                        group,
                    )
                    results["ragged"][(rank, chunk)] = ragged.token_exchange(
                        rank, tensors[rank][chunk],
                        plans[rank].send_counts[chunk],
                        plans[rank].recv_counts[chunk], "async")
            except BaseException as exc:  # pylint: disable=broad-except
                failures.append((rank, exc))

        with mock.patch.object(ep_collectives, "dist", _PaddedEpDist(padded)):
            threads = [threading.Thread(target=run_rank, args=(rank,))
                       for rank in range(_EP_SIZE)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        if failures:
            raise failures[0][1]
        return send_plan, plans, results

    def test_padded_fallback_matches_the_ragged_exchange(self):
        """pad-to-max + a2a_single moves exactly the rows the ragged a2a moves."""
        for chunk_count in (2, 4):
            send_plan, plans, results = self._run(chunk_count)
            with self.subTest(chunk_count=chunk_count):
                self.assertEqual(
                    [sum(row) for row in send_plan],
                    [sum(plans[rank].send_counts[chunk][dest]
                         for chunk in range(chunk_count)
                         for dest in range(_EP_SIZE))
                     for rank in range(_EP_SIZE)])
            for rank in range(_EP_SIZE):
                for chunk in range(chunk_count):
                    with self.subTest(chunk_count=chunk_count, rank=rank, chunk=chunk):
                        self.assertTrue(torch.equal(
                            results["padded"][(rank, chunk)],
                            results["ragged"][(rank, chunk)]))

    def test_zero_count_chunks_survive_the_fallback(self):
        """A chunk whose counts are all zero on this rank stays a zero-row result."""
        _, plans, results = self._run(4)
        empty = [(rank, chunk)
                 for rank in range(_EP_SIZE)
                 for chunk in range(4)
                 if not any(plans[rank].send_counts[chunk])
                 and not any(plans[rank].recv_counts[chunk])]
        self.assertTrue(empty, "the scenario must contain a locally empty chunk")
        for rank, chunk in empty:
            with self.subTest(rank=rank, chunk=chunk):
                self.assertEqual(results["padded"][(rank, chunk)].shape[0], 0)


if __name__ == "__main__":
    unittest.main()
