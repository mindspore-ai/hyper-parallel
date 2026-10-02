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
"""Unit tests for the split-free exchange's laziness across the dispatch split.

``HP_EP_EQUAL_A2A=1`` buys the cheaper alltoall kernel by issuing the exchange
without split sizes, and it must not pay for it with the shared-expert overlap:
the exchange is issued on a communication stream of its own, so the caller's
stream is only ordered on it where the caller reads the result.  These tests pin
that schedule end to end, on the split the overlap uses:

* the knob off leaves the schedule and the collectives exactly as they are today
  (the two ragged dispatch exchanges and the ragged combine, through the async
  entry);
* the knob on routes all three exchanges of the split through the split-free
  collective, and the dispatch exchanges are issued before -- and waited after --
  the work the caller runs between dispatch and experts+combine;
* the routed output and its gradient are identical either way.

The knob is default-off, so the arm the champion runs today is the first bullet;
the eager mode and the caller that refuses a pending handle (the fused
states+indices dispatch, whose unpack is a view chain) are pinned in
``test_ep_equal_a2a.py`` at the exchange level.

No process group is created: every exchange is driven by a barrier world inside
this process, in which each rank runs in its own thread, and the platform's
stream/event API is faked by a double that traces what was enqueued where.
"""

import collections
import os
import threading
import unittest
from typing import Any, List, NamedTuple, Optional
from unittest import mock

import torch
from torch import nn

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from hyper_parallel.distributed.expert_parallel import collectives as ep_collectives  # noqa: E402
from hyper_parallel.distributed.expert_parallel import experts as ep_experts  # noqa: E402
from hyper_parallel.distributed.expert_parallel.experts import (  # noqa: E402
    bind_local_expert_forward,
    ep_routed_dispatch,
    ep_routed_experts_and_combine,
)
from tests.common.mark_utils import arg_mark  # noqa: E402

_EP_SIZE = 4
_LOCAL_EXPERTS = 2
_HIDDEN = 4
_INTERMEDIATE = 3
_TOKEN_COUNT = 4
_TOP_K = 2
_BATCH, _SEQ = 2, 2
_ROWS_PER_PEER = 2
_BARRIER_TIMEOUT = 20.0
_CPU_MARKS = {"plat_marks": ["cpu_linux", "cpu_macos"], "level_mark": "level0",
              "card_mark": "allcards", "essential_mark": "essential"}

# A balanced plan: every rank dispatches two rows to each of the four ranks (two
# local experts per rank, so destination rank = expert // 2), rotated so the rows
# are distinguishable per rank.  The equal path is only applicable to this shape,
# which is exactly what a balanced router produces.
_SLOTS = {
    rank: [(expert + rank) % (_LOCAL_EXPERTS * _EP_SIZE)
           for expert in range(_TOKEN_COUNT * _TOP_K)]
    for rank in range(_EP_SIZE)
}

_RAGGED = "ragged"
_EQUAL = "equal"
_COUNTS = "counts"


class _VirtualEpGroup:
    """Size+rank EP group double: carries the rank the calling thread runs as."""

    def __init__(self, rank: int, size: int) -> None:
        """Bind the double to one rank of a group of ``size`` ranks."""
        self.rank = rank
        self._size = size

    def size(self) -> int:
        """Return the EP group size."""
        return self._size


class _VirtualStream:
    """Stand-in for a device stream: names itself in the trace it writes to."""

    def __init__(self, name: str, world: "_VirtualEpWorld") -> None:
        """Bind the stream to its name and the world that owns the trace."""
        self.name = name
        self._world = world


class _VirtualEvent:
    """Stand-in for a device event: traces every record and wait it receives."""

    def __init__(self, name: str, world: "_VirtualEpWorld") -> None:
        """Bind the event to its name and the world that owns the trace."""
        self.name = name
        self._world = world

    def record(self, stream: _VirtualStream) -> None:
        """Trace one ``record`` on ``stream``."""
        self._world.mark(f"record:{self.name}<-{stream.name}")

    def wait(self, stream: _VirtualStream) -> None:
        """Trace one ``wait`` on ``stream``."""
        self._world.mark(f"wait:{self.name}->{stream.name}")


class _VirtualStreamContext:
    """The context manager ``platform.get_stream_context()`` hands out."""

    def __init__(self, stream: _VirtualStream, world: "_VirtualEpWorld") -> None:
        """Hold the stream the caller wants to run on."""
        self._stream = stream
        self._world = world

    def __enter__(self) -> _VirtualStream:
        """Trace entering the stream."""
        self._world.mark(f"enter@{self._stream.name}")
        return self._stream

    def __exit__(self, *exc: Any) -> bool:
        """Trace leaving the stream."""
        self._world.mark(f"exit@{self._stream.name}")
        return False


class _DeviceHandleDouble:
    """Expose the fake stream/event factories under the torch device-module names.

    ``collectives`` now reaches the accelerator through
    ``get_device_handle()`` (``torch.npu`` / ``torch.cuda``) rather than through
    the retired platform object, so the tests bind their traced fakes to that
    handle instead of patching a module-level ``platform``.
    """

    def __init__(self, streams: Any) -> None:
        """Wrap the fake that owns the stream, event and trace bookkeeping."""
        self._streams = streams

    def Stream(self) -> Any:  # noqa: N802 - mirrors torch.npu/torch.cuda
        """Return the shared comm stream."""
        return self._streams.new_stream()

    def Event(self) -> Any:  # noqa: N802 - mirrors torch.npu/torch.cuda
        """Return a fresh event."""
        return self._streams.new_event()

    def current_stream(self) -> Any:
        """Return the stream the calling rank computes on."""
        return self._streams.get_current_stream()

    @property
    def stream(self) -> Any:
        """Return the factory used as a stream context manager."""
        return self._streams.get_stream_context()


class _VirtualEpWorld:
    """One-process EP group double: the collectives the split issues, plus a trace.

    Every rank runs in its own thread and every collective is a barrier: a rank
    publishes the chunks it splits out of its send buffer, waits for its peers,
    and fills its receive buffer from theirs.  Three contracts are implemented --
    the counts exchange (a 1-D split-free ``all_to_all_single``), the ragged
    ``all_to_all`` the knob-off path issues, and the split-free payload exchange
    the knob-on path issues -- and each is recorded, so a case can tell which
    path ran *and* in which order relative to the caller's overlapped work.
    """

    def __init__(self, ep_size: int) -> None:
        """Build a world of ``ep_size`` ranks."""
        self.ep_size = ep_size
        self.barrier = threading.Barrier(ep_size, timeout=_BARRIER_TIMEOUT)
        self._lock = threading.Lock()
        self._calls = {}
        self._index = {}
        self._kinds = {}
        self.new_stream_calls = 0
        self.new_event_calls = 0
        self._stream = _VirtualStream("comm", self)
        self._compute = _VirtualStream("compute", self)
        self._ranks = {}

    def bind(self, rank: int) -> None:
        """Attribute this thread's trace entries to ``rank``."""
        with self._lock:
            self._ranks[threading.get_ident()] = rank

    def mark(self, entry: str) -> None:
        """Append one trace entry under the calling thread's rank."""
        with self._lock:
            self._kinds.setdefault("trace", {}).setdefault(
                self._ranks.get(threading.get_ident(), 0), []).append(entry)

    def entries(self, rank: int) -> List[str]:
        """Return rank ``rank``'s stream/event trace entries, in order."""
        with self._lock:
            return list(self._kinds.get("trace", {}).get(rank, []))

    def calls(self, rank: int) -> List[str]:
        """Return the collectives rank ``rank`` issued, in order."""
        with self._lock:
            return list(self._kinds.get(rank, []))

    def new_stream(self) -> _VirtualStream:
        """Hand out the comm stream, counting the request."""
        with self._lock:
            self.new_stream_calls += 1
        return self._stream

    def new_event(self) -> _VirtualEvent:
        """Hand out a fresh event, counting the request."""
        with self._lock:
            self.new_event_calls += 1
            name = f"event{self.new_event_calls}"
        return _VirtualEvent(name, self)

    def get_current_stream(self) -> _VirtualStream:
        """Report the stream the calling rank computes on."""
        return self._compute

    def get_stream_context(self) -> Any:
        """Return the factory that enters a given stream."""
        return lambda stream: _VirtualStreamContext(stream, self)

    def record_stream(self, tensor: torch.Tensor, stream: _VirtualStream) -> None:
        """Stand in for ``_record_stream`` (a real one rejects a fake stream)."""
        del tensor
        self.mark(f"record_stream@{stream.name}")

    def get_rank(self, group: Any = None) -> int:
        """Return the rank of the double group passed in."""
        return group.rank

    def _record(self, rank: int, kind: str) -> None:
        """Append one issued collective to this rank's trace."""
        with self._lock:
            self._kinds.setdefault(rank, []).append(kind)

    def _next_call(self, rank: int) -> int:
        """Reserve this rank's next collective slot (ranks call them in lockstep)."""
        with self._lock:
            index = self._index.get(rank, 0)
            self._index[rank] = index + 1
            return index

    def _post(self, rank: int, index: int, pieces: List[torch.Tensor]) -> None:
        """Publish this rank's chunks for ``index`` and wait for every peer."""
        with self._lock:
            self._calls[(rank, index)] = pieces
        self.barrier.wait()

    def _gather(self, rank: int, index: int) -> List[torch.Tensor]:
        """Return, in source-rank order, the chunks this rank received."""
        return [self._calls[(src, index)][rank] for src in range(self.ep_size)]

    def all_to_all_single(self, output: torch.Tensor, input_tensor: torch.Tensor,
                          input_splits: Optional[List[int]] = None,
                          output_splits: Optional[List[int]] = None,
                          group: Any = None) -> None:
        """Split-free contract: the counts exchange, or the equal payload exchange.

        The dispatch counts travel as a 1-D tensor of one row per rank; the token
        payload as a 2-D tensor, and its chunking comes from the shape alone --
        which is the whole point of the fast path, so passing split sizes here is
        a hard failure rather than a different layout.
        """
        rank = group.rank
        index = self._next_call(rank)
        if input_tensor.dim() == 1:
            self._record(rank, _COUNTS)
            self._post(rank, index, list(input_tensor.split(1)))
            output.copy_(torch.cat(self._gather(rank, index)))
            return
        assert input_splits is None and output_splits is None, (
            f"the equal path must not pass split sizes, but got "
            f"input_splits={input_splits}, output_splits={output_splits}")
        self._record(rank, _EQUAL)
        rows = input_tensor.shape[0] // self.ep_size
        self._post(rank, index, list(input_tensor.split(rows)))
        output.copy_(torch.cat(self._gather(rank, index)))

    def all_to_all(self, output_list: List[torch.Tensor], input_list: List[torch.Tensor],
                   group: Any = None) -> None:
        """Ragged contract (the knob-off path): chunk ``i`` goes to rank ``i``."""
        rank = group.rank
        index = self._next_call(rank)
        self._record(rank, _RAGGED)
        self._post(rank, index, list(input_list))
        for src, slot in enumerate(output_list):
            piece = self._calls[(src, index)][rank]
            assert piece.shape[0] == slot.shape[0], (
                f"rank {rank} exchange {index}: rank {src} sent {piece.shape[0]} rows "
                f"where this rank has room for {slot.shape[0]}")
            slot.copy_(piece)

    def ragged_exchange(self, tensor: torch.Tensor, send_counts: List[int],
                        recv_counts: List[int], group: Any) -> torch.Tensor:
        """Stand-in for the platform's lazy exchange: the ragged contract itself."""
        buffer = tensor.new_empty((sum(recv_counts),) + tuple(tensor.shape[1:]))
        self.all_to_all(list(buffer.split(recv_counts)), list(tensor.split(send_counts)), group)
        return buffer


class _VirtualRaggedA2A(torch.autograd.Function):
    """Differentiable stand-in for the platform's lazy ragged exchange.

    The world's plain tensor ops connect every receiver's graph to its peers'
    inputs, which is not the production contract, and they would build views and
    modify them in place outside an autograd Function -- so this double has the
    real exchange's contract instead: forward is the ragged exchange, backward is
    the same exchange with the counts swapped.
    """

    @staticmethod
    def forward(ctx: Any, rank: int, tensor: torch.Tensor, send_counts: List[int],  # pylint: disable=arguments-differ
                recv_counts: List[int], world: _VirtualEpWorld, group: Any) -> torch.Tensor:
        """Run the world's ragged exchange and retain the counts for backward."""
        ctx.rank = rank
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.world = world
        ctx.group = group
        return world.ragged_exchange(tensor, send_counts, recv_counts, group)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:  # pylint: disable=arguments-differ
        """Reverse the exchange, exactly as the ragged all-to-all does."""
        grad = ctx.world.ragged_exchange(
            grad_output.contiguous(), ctx.recv_counts, ctx.send_counts, ctx.group)
        return None, grad, None, None, None, None


class _VirtualEpDist:
    """Stand-in for the ``dist`` module ``collectives``/``experts`` resolve through."""

    ReduceOp = torch.distributed.ReduceOp

    def __init__(self, world: _VirtualEpWorld) -> None:
        """Forward every collective to ``world``."""
        self._world = world

    def get_backend(self, group: Any = None) -> str:
        """Report a backend whose ``all_to_all`` accepts unequal splits."""
        del group
        return "hccl"

    def get_rank(self, group: Any = None) -> int:
        """Forward the rank query."""
        return self._world.get_rank(group)

    def all_to_all(self, output_list: List[torch.Tensor], input_list: List[torch.Tensor],
                   group: Any = None) -> None:
        """Forward the ragged exchange to the world."""
        return self._world.all_to_all(output_list, input_list, group)

    def all_to_all_single(self, output: torch.Tensor, input_tensor: torch.Tensor,
                          input_splits: Optional[List[int]] = None,
                          output_splits: Optional[List[int]] = None,
                          group: Any = None) -> None:
        """Forward the split-free exchanges (counts and payload) to the world."""
        return self._world.all_to_all_single(
            output, input_tensor, input_splits, output_splits, group)


class _Run(NamedTuple):
    """One driven split path: the routed outputs plus what each rank issued."""

    outputs: dict
    states: dict
    world: _VirtualEpWorld
    lazy_calls: list
    grads: dict


def _build_moe(rank: int) -> nn.Module:
    """Build one rank's MoE double with the real bound SwiGLU expert entry."""
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
    return module


def _scenario(seed: int = 11) -> tuple:
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


def _run_split(knob: str, *, backward: bool = False, issue_only: bool = False) -> _Run:
    """Drive every virtual rank through dispatch -> overlapped work -> experts+combine.

    Args:
        knob: Value of ``HP_EP_EQUAL_A2A`` for the run.
        backward: Backpropagate each rank's sum and collect the input gradients.
        issue_only: Stop after the dispatch, so the pending state it returned can
            be inspected before its consumers wait for it.

    Returns:
        The :class:`_Run` holding every rank's routed output, its
        :class:`EPRoutedState`, the world double and the calls the stubbed async
        entry received (empty unless the knob was off).
    """
    hidden, topk, weights = _scenario()
    for rank in range(_EP_SIZE):
        hidden[rank].requires_grad_(backward)
    world = _VirtualEpWorld(_EP_SIZE)
    dist_double = _VirtualEpDist(world)
    outputs, states, lazy_calls, grads, failures = {}, {}, [], {}, []

    def lazy_exchange(x: torch.Tensor, send_counts: List[int], recv_counts: List[int],
                      group: Any) -> torch.Tensor:
        """Stand-in for the platform's lazy exchange, driving the ragged contract."""
        lazy_calls.append((group.rank, tuple(send_counts), tuple(recv_counts)))
        return _VirtualRaggedA2A.apply(
            group.rank, x, send_counts, recv_counts, world, group)

    def run_rank(rank: int) -> None:
        """One virtual rank: dispatch, run the overlapped work, then finish."""
        def router(_module: Any, _hidden: torch.Tensor) -> tuple:
            """Hand the case's fixed routing plan to the dispatch."""
            return topk[rank], weights[rank]

        try:
            world.bind(rank)
            module = _build_moe(rank)
            ep_group = _VirtualEpGroup(rank, _EP_SIZE)
            state = ep_routed_dispatch(
                module, hidden[rank], router_fn=router, ep_group=ep_group)
            states[rank] = state
            world.mark("overlap")  # the shared-expert MLP the split exists for
            if issue_only:
                return
            outputs[rank] = ep_routed_experts_and_combine(module, state, ep_group)
            if backward:
                outputs[rank].sum().backward()
                grads[rank] = hidden[rank].grad.clone()
        except BaseException as exc:  # pylint: disable=broad-except
            failures.append((rank, exc))

    with mock.patch.object(ep_experts, "dist", dist_double), \
            mock.patch.object(ep_collectives, "dist", dist_double), \
            mock.patch.object(ep_collectives, "_EQUAL_A2A_RAW", knob), \
            mock.patch.multiple(ep_collectives,
                                _record_stream=world.record_stream,
                                _LAZY_A2A_STREAM=None,
                                _LAZY_A2A_READY_EVENT=None,
                                _LAZY_A2A_DONE_EVENT=None), \
            mock.patch.object(ep_collectives, "get_device_handle", lambda: _DeviceHandleDouble(world)), \
            mock.patch.object(ep_collectives,
                              "differentiable_all_to_all_single_async",
                              side_effect=lazy_exchange):
        threads = [threading.Thread(target=run_rank, args=(rank,)) for rank in range(_EP_SIZE)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    if failures:
        raise failures[0][1]
    return _Run(outputs, states, world, lazy_calls, grads)


def _positions(entries: List[str], entry: str) -> List[int]:
    """Return the positions of ``entry`` in ``entries``."""
    return [position for position, value in enumerate(entries) if value == entry]


class TestSplitPathWithTheEqualKnob(unittest.TestCase):
    """Which collectives the dispatch split (``overlap_shared_expert``) issues."""

    @arg_mark(**_CPU_MARKS)
    def test_scenario_is_balanced(self):
        """The plan the cases run on must be uniform, or the equal path cannot apply.

        Feature: test scenario precondition.
        Description: Count the rows every rank dispatches to every peer.
        Expectation: Each rank sends exactly ``_ROWS_PER_PEER`` rows to each peer.
        """
        for rank, slots in _SLOTS.items():
            counts = [sum(1 for slot in slots if slot // _LOCAL_EXPERTS == dest)
                      for dest in range(_EP_SIZE)]
            with self.subTest(rank=rank):
                self.assertEqual(counts, [_ROWS_PER_PEER] * _EP_SIZE,
                                 f"rank {rank} routes {counts}, not a balanced plan")

    @arg_mark(**_CPU_MARKS)
    def test_knob_off_keeps_todays_ragged_split(self):
        """Unchanged behaviour: the split exchanges its tokens ragged, as today.

        Feature: ``HP_EP_EQUAL_A2A`` default on the dispatch split.
        Description: Run dispatch -> overlap -> experts+combine with the knob off.
        Expectation: The counts exchange and the three ragged exchanges of the split
            run through the async entry, and the split-free payload exchange is never
            used.
        """
        run = _run_split("0")
        issued = {rank: run.world.calls(rank) for rank in range(_EP_SIZE)}
        self.assertEqual(issued, {rank: [_COUNTS, _RAGGED, _RAGGED, _RAGGED]
                                  for rank in range(_EP_SIZE)},
                         f"the knob-off split must stay ragged, but the ranks issued {issued}")
        uniform = (_ROWS_PER_PEER,) * _EP_SIZE
        self.assertEqual(
            collections.Counter(run.lazy_calls),
            collections.Counter({(rank, uniform, uniform): 3 for rank in range(_EP_SIZE)}),
            f"every exchange must go through the lazy entry, got {run.lazy_calls}")

    @arg_mark(**_CPU_MARKS)
    def test_knob_on_sends_the_whole_split_over_the_split_free_exchange(self):
        """The knob swaps all three exchanges of the split, not just one.

        Feature: ``HP_EP_EQUAL_A2A=1`` on the dispatch split.
        Description: Run the same split with the knob on.
        Expectation: The counts exchange is followed by three split-free payload
            exchanges and no ragged one.
        """
        run = _run_split("1")
        issued = {rank: run.world.calls(rank) for rank in range(_EP_SIZE)}
        self.assertEqual(issued, {rank: [_COUNTS, _EQUAL, _EQUAL, _EQUAL]
                                  for rank in range(_EP_SIZE)},
                         f"the knob-on split must be split-free, but the ranks issued {issued}")
        self.assertEqual(run.lazy_calls, [],
                         f"the lazy ragged entry must not be used, got {run.lazy_calls}")

    @arg_mark(**_CPU_MARKS)
    def test_knob_on_waits_only_after_the_overlapped_work(self):
        """The wait is placed by the caller, after the work the split overlaps.

        Feature: lazy split-free exchange, schedule.
        Description: Read the traced stream/event order of one rank's split.
        Expectation: Both dispatch exchanges are issued before the overlap marker and
            waited after it, and the combine is issued only afterwards -- so nothing
            orders the compute stream on the transfer before the overlap closed.
        """
        run = _run_split("1")
        for rank in range(_EP_SIZE):
            with self.subTest(rank=rank):
                entries = run.world.entries(rank)
                marker = entries.index("overlap")
                issued_at = _positions(entries, "record:event2<-comm")
                waited_at = _positions(entries, "wait:event2->compute")
                self.assertEqual(len(issued_at), 3,
                                 f"rank {rank} issued {len(issued_at)} exchanges, "
                                 f"expected the two dispatch exchanges and the combine: "
                                 f"{entries}")
                self.assertEqual(len(waited_at), 3,
                                 f"rank {rank} waited {len(waited_at)} times, trace={entries}")
                self.assertLess(issued_at[1], marker,
                                f"both dispatch exchanges must be issued before the "
                                f"overlap: {entries}")
                self.assertLess(marker, waited_at[0],
                                f"the first wait must land after the overlap: {entries}")
                self.assertLess(waited_at[1], issued_at[2],
                                f"the dispatch must be waited before the combine is "
                                f"issued: {entries}")
                self.assertLess(issued_at[2], waited_at[2],
                                f"the combine must be waited after it is issued: {entries}")

    @arg_mark(**_CPU_MARKS)
    def test_pending_dispatch_state_is_a_handle_that_the_consumer_waits(self):
        """The dispatched values reach the consumer as a pending handle.

        Feature: lazy split-free exchange, dispatch state.
        Description: Stop after the dispatch and inspect the state it returned.
        Expectation: Both exchanged tensors are pending handles, and waiting them
            yields the tensors their consumers read.
        """
        run = _run_split("1", issue_only=True)
        state = run.states[0]
        self.assertIsInstance(state.received_states, ep_collectives._PendingEqualA2A,  # pylint: disable=W0212
                              f"the dispatched states must be pending, "
                              f"got {type(state.received_states)}")
        self.assertIsInstance(state.received_indices, ep_collectives._PendingEqualA2A,  # pylint: disable=W0212
                              f"the dispatched indices must be pending, "
                              f"got {type(state.received_indices)}")
        self.assertFalse(state.received_states.completed,
                         "the dispatch must not have waited for the exchange itself")
        with mock.patch.object(
                ep_collectives, "get_device_handle",
                lambda: mock.MagicMock(current_stream=run.world.get_current_stream)):
            states = ep_collectives.wait_ep_all_to_all(state.received_states)
            indices = ep_collectives.wait_ep_all_to_all(state.received_indices)
        self.assertTrue(state.received_states.completed,
                        "waiting the handle must mark it completed")
        rows = _ROWS_PER_PEER * _EP_SIZE
        self.assertEqual(tuple(states.shape), (rows, _HIDDEN),
                         f"the exchanged states are {tuple(states.shape)}, "
                         f"expected [{rows}, {_HIDDEN}]")
        self.assertEqual(tuple(indices.shape), (rows,),
                         f"the exchanged indices are {tuple(indices.shape)}, "
                         f"expected [{rows}]")

    @arg_mark(**_CPU_MARKS)
    def test_both_paths_agree_on_the_routed_output_and_its_gradient(self):
        """Knob on and knob off differ in the schedule only, not in the numbers.

        Feature: split-free equivalence through the whole split.
        Description: Run both arms of the knob, backpropagated, on the same inputs.
        Expectation: Every rank's routed output and hidden-state gradient are identical.
        """
        ragged = _run_split("0", backward=True)
        split_free = _run_split("1", backward=True)
        for rank in range(_EP_SIZE):
            with self.subTest(rank=rank, direction="forward"):
                self.assertTrue(
                    torch.equal(ragged.outputs[rank], split_free.outputs[rank]),
                    f"rank {rank}: ragged={ragged.outputs[rank].flatten().tolist()}, "
                    f"split-free={split_free.outputs[rank].flatten().tolist()}")
            with self.subTest(rank=rank, direction="backward"):
                self.assertTrue(
                    torch.equal(ragged.grads[rank], split_free.grads[rank]),
                    f"rank {rank}: ragged grad={ragged.grads[rank].flatten().tolist()}, "
                    f"split-free grad={split_free.grads[rank].flatten().tolist()}")


if __name__ == "__main__":
    unittest.main()
