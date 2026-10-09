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
"""Unit tests for the equal-length EP all-to-all fast path (``HP_EP_EQUAL_A2A``).

A balanced routing plan hands every peer the same number of rows, so the ragged
split sizes of the token exchange carry no information and the plain
equal-length ``all_to_all_single`` can be issued instead (a cheaper backend
kernel than alltoallv).  The swap is only safe if it moves exactly the rows the
ragged path moves, in exactly the same order, so that is what these tests pin:

* the knob off keeps the ragged path bit-for-bit (forward and backward);
* the knob on with a uniform plan takes the equal path, and its output *and* its
  backward are bit-identical (``torch.equal``) to the ragged path's;
* on the async entry the same exchange is *lazy*: it is issued on a
  communication stream of its own and the caller's stream is ordered on it by
  ``wait_ep_all_to_all``, where the caller decides -- the split-free kernel must
  not cost the shared-expert overlap;
* a non-uniform plan falls back to the ragged path instead of silently
  re-chunking the payload;
* a plan whose counts are uniform but do not describe the payload falls back as
  well, so no wrong layout can be produced;
* gloo (the padded fallback) is untouched, and an invalid knob fails loudly.

No process group is created: every exchange is driven by a barrier world inside
this process, in which each rank runs in its own thread.  The world implements
both collective contracts -- the ragged ``all_to_all`` and the split-free
``all_to_all_single`` -- and checks the rows it hands over against the buffer
sizes the receiver derived, so a wrong layout is a hard failure rather than a
wrong number.  The platform's stream/event API is faked by a double that traces
what was recorded and waited where, so the laziness is checked as a schedule
rather than inferred from the code.
"""
import collections
import contextlib
import os
import threading
import unittest
from typing import Any, Callable, List, NamedTuple, Optional
from unittest import mock
import torch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

from hyper_parallel.distributed.expert_parallel import collectives as ep_collectives  # noqa: E402
from tests.common.mark_utils import arg_mark  # noqa: E402

_EP_SIZE = 4
_HIDDEN = 3
_UNIFORM_ROWS = 3
_BARRIER_TIMEOUT = 20.0
_SYNC = "sync"
_ASYNC = "async"
_HCCL = "hccl"
_GLOO = "gloo"
_CPU_MARKS = {"plat_marks": ["cpu_linux", "cpu_macos"], "level_mark": "level0",
              "card_mark": "allcards", "essential_mark": "essential"}


class _VirtualEpGroup:
    """Size+rank EP group double: carries the rank the calling thread runs as."""

    def __init__(self, rank: int, size: int) -> None:
        """Bind the double to one rank of a group of ``size`` ranks."""
        self.rank = rank
        self._size = size

    def size(self) -> int:
        """Return the EP group size."""
        return self._size


class _VirtualEpWorld:
    """One-process EP group double implementing both all-to-all contracts.

    Every rank runs in its own thread and every collective is a barrier: a rank
    publishes the chunks it splits out of its send buffer, waits for its peers,
    and copies their chunks into its own buffer, recording which collective it
    issued.  The ragged contract (``all_to_all``) checks each received chunk
    against the size of the buffer slot it is destined for, and the equal
    contract (``all_to_all_single``) refuses split sizes outright -- that is the
    whole point of the fast path -- and checks the received row count against
    the buffer the caller allocated.
    """

    def __init__(self, ep_size: int, kind: str = "equal") -> None:
        """Build a world of ``ep_size`` ranks labelling its exchanges ``kind``."""
        self.ep_size = ep_size
        self.kind = kind
        self.barrier = threading.Barrier(ep_size, timeout=_BARRIER_TIMEOUT)
        self._lock = threading.Lock()
        self._calls = {}
        self._index = {}
        self._kinds = {}

    def _next_call(self, rank: int) -> int:
        """Reserve this rank's next exchange slot (ranks call them in lockstep)."""
        with self._lock:
            index = self._index.get(rank, 0)
            self._index[rank] = index + 1
            return index

    def _record(self, rank: int, kind: str) -> None:
        """Append one issued collective to this rank's trace."""
        with self._lock:
            self._kinds.setdefault(rank, []).append(kind)

    def calls(self, rank: int) -> List[str]:
        """Return the collectives rank ``rank`` issued, in order."""
        with self._lock:
            return list(self._kinds.get(rank, []))

    def all_to_all(self, output_list: List[torch.Tensor], input_list: List[torch.Tensor],
                   group: Any = None) -> None:
        """Ragged contract: chunk ``i`` of the input goes to rank ``i``."""
        rank = group.rank
        index = self._next_call(rank)
        self._record(rank, "ragged")
        with self._lock:
            self._calls[(rank, index)] = list(input_list)
        self.barrier.wait()
        for src, slot in enumerate(output_list):
            piece = self._calls[(src, index)][rank]
            assert piece.shape[0] == slot.shape[0], (
                f"rank {rank} exchange {index}: rank {src} sent {piece.shape[0]} rows "
                f"where this rank has room for {slot.shape[0]}")
            slot.copy_(piece)

    def all_to_all_single(self, output: torch.Tensor, input_tensor: torch.Tensor,
                          input_splits: Optional[List[int]] = None,
                          output_splits: Optional[List[int]] = None,
                          group: Any = None) -> None:
        """Equal contract: no split sizes, ``shape[0] // ep_size`` rows per peer."""
        assert input_splits is None and output_splits is None, (
            f"the equal path must not pass split sizes, but got "
            f"input_splits={input_splits}, output_splits={output_splits}")
        rank = group.rank
        index = self._next_call(rank)
        self._record(rank, self.kind)
        rows = input_tensor.shape[0] // self.ep_size
        pieces = list(input_tensor.split(rows))
        with self._lock:
            self._calls[(rank, index)] = pieces
        self.barrier.wait()
        received = torch.cat([self._calls[(src, index)][rank] for src in range(self.ep_size)])
        assert received.shape[0] == output.shape[0], (
            f"rank {rank} exchange {index}: received {received.shape[0]} rows into a "
            f"{output.shape[0]}-row buffer")
        output.copy_(received)


class _VirtualGlooWorld(_VirtualEpWorld):
    """One-process gloo double: the two collectives the padded fallback needs.

    The padded path pads every chunk to the world-wide maximum and therefore
    also issues its exchange without split sizes, so it reuses the equal-length
    exchange above; the extra collective is the MAX ``all_reduce`` that learns
    that common length.  Its exchange is labelled ``padded``, which is how a
    case tells the fallback apart from the fast path.
    """

    def __init__(self, ep_size: int) -> None:
        """Build a world of ``ep_size`` ranks whose exchange is labelled padded."""
        super().__init__(ep_size, kind="padded")
        self._maxima = {}

    def all_reduce(self, tensor: torch.Tensor, group: Any = None) -> None:
        """MAX-reduce the padded length across the world."""
        rank = group.rank
        index = self._next_call(rank)
        with self._lock:
            self._maxima[(rank, index)] = int(tensor.item())
        self.barrier.wait()
        tensor.fill_(max(self._maxima[(rank, index)] for rank in range(self.ep_size)))


class _VirtualEpDist:
    """Stand-in for the ``dist`` module ``collectives`` resolves through."""

    ReduceOp = torch.distributed.ReduceOp

    def __init__(self, world: _VirtualEpWorld) -> None:
        """Forward every collective to ``world``."""
        self._world = world

    def get_backend(self, group: Any = None) -> str:
        """Report a backend whose all_to_all accepts unequal splits."""
        del group
        return _HCCL

    def all_to_all(self, output_list: List[torch.Tensor], input_list: List[torch.Tensor],
                   group: Any = None) -> None:
        """Forward the ragged exchange to the world."""
        return self._world.all_to_all(output_list, input_list, group)

    def all_to_all_single(self, output: torch.Tensor, input_tensor: torch.Tensor,
                          input_splits: Optional[List[int]] = None,
                          output_splits: Optional[List[int]] = None,
                          group: Any = None) -> None:
        """Forward the equal-length exchange to the world."""
        return self._world.all_to_all_single(
            output, input_tensor, input_splits, output_splits, group)


class _VirtualGlooDist:
    """Stand-in for a backend without a ragged all-to-all (the gloo test path)."""

    ReduceOp = torch.distributed.ReduceOp

    def __init__(self, world: _VirtualGlooWorld) -> None:
        """Forward every collective to ``world``."""
        self._world = world

    def get_backend(self, group: Any = None) -> str:
        """Report a backend that has no ragged all-to-all."""
        del group
        return _GLOO

    def all_reduce(self, tensor: torch.Tensor, op: Any = None, group: Any = None) -> None:
        """Forward the MAX reduction of the padded length."""
        del op
        return self._world.all_reduce(tensor, group)

    def all_to_all_single(self, output: torch.Tensor, input_tensor: torch.Tensor,
                          input_splits: Optional[List[int]] = None,
                          output_splits: Optional[List[int]] = None,
                          group: Any = None) -> None:
        """Forward the equal-length exchange the padded path issues."""
        return self._world.all_to_all_single(
            output, input_tensor, input_splits, output_splits, group)


class _BackendOnlyDist:
    """Stand-in for ``dist`` when only the backend query matters."""

    def __init__(self, backend: str) -> None:
        """Report ``backend`` for every query."""
        self._backend = backend

    def get_backend(self, group: Any = None) -> str:
        """Report the configured backend name."""
        del group
        return self._backend


class _FakeStream:
    """Stand-in for a device stream: names itself in the trace it writes to."""

    def __init__(self, name: str, streams: "_FakeStreams") -> None:
        """Bind the stream to its name and the trace owner."""
        self.name = name
        self._streams = streams

    def record_stream(self, tensor: torch.Tensor) -> None:
        """Stand in for ``Tensor.record_stream`` on this stream."""
        del tensor
        self._streams.mark(f"record_stream@{self.name}")


class _FakeEvent:
    """Stand-in for a device event: records every record and wait it receives."""

    def __init__(self, name: str, streams: "_FakeStreams") -> None:
        """Bind the event to its name and the trace owner."""
        self.name = name
        self._streams = streams

    def record(self, stream: _FakeStream) -> None:
        """Trace one ``record`` on ``stream``."""
        self._streams.mark(f"record:{self.name}<-{stream.name}")

    def wait(self, stream: _FakeStream) -> None:
        """Trace one ``wait`` on ``stream``."""
        self._streams.mark(f"wait:{self.name}->{stream.name}")


class _FakeStreamContext:
    """The context manager ``platform.get_stream_context()`` hands out."""

    def __init__(self, stream: _FakeStream, streams: "_FakeStreams") -> None:
        """Hold the stream the caller wants to run on."""
        self._stream = stream
        self._streams = streams

    def __enter__(self) -> _FakeStream:
        """Trace entering the stream."""
        self._streams.mark(f"enter@{self._stream.name}")
        return self._stream

    def __exit__(self, *exc: Any) -> bool:
        """Trace leaving the stream."""
        self._streams.mark(f"exit@{self._stream.name}")
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


class _FakeStreams:
    """Fake of the platform's stream/event API with a trace and call counters.

    The lazy exchange must reuse one stream and two events for the whole
    process -- a fresh event per exchange drains the runtime's event pool -- so
    the counters are the assertion: an implementation that allocates per
    exchange shows up as more than one ``new_stream`` call.  Every record, wait
    and stream entry is traced with the rank that caused it, which is how the
    tests pin the schedule (the exchange must be issued and waited on where the
    caller wants to wait, not where the collective is issued).
    """

    def __init__(self) -> None:
        """Start with no trace, no callers bound and both factories uncalled."""
        self._comm = _FakeStream("comm", self)
        self._compute = _FakeStream("compute", self)
        self._lock = threading.Lock()
        self._ranks = {}
        self._trace = []
        self.new_stream_calls = 0
        self.new_event_calls = 0

    def bind(self, rank: int) -> None:
        """Attribute this thread's trace entries to ``rank``."""
        with self._lock:
            self._ranks[threading.get_ident()] = rank

    def mark(self, entry: str) -> None:
        """Append one trace entry under the calling thread's rank."""
        with self._lock:
            self._trace.append((self._ranks.get(threading.get_ident(), 0), entry))

    def entries(self, rank: int) -> List[str]:
        """Return rank ``rank``'s trace entries, in order."""
        with self._lock:
            return [entry for entry_rank, entry in self._trace if entry_rank == rank]

    def new_stream(self) -> _FakeStream:
        """Hand out the comm stream, counting the request."""
        with self._lock:
            self.new_stream_calls += 1
        return self._comm

    def new_event(self) -> _FakeEvent:
        """Hand out a fresh event, counting the request."""
        with self._lock:
            self.new_event_calls += 1
            name = f"event{self.new_event_calls}"
        return _FakeEvent(name, self)

    def get_current_stream(self) -> _FakeStream:
        """Report the stream the calling rank computes on."""
        return self._compute

    def get_stream_context(self) -> Callable:
        """Return the factory that enters a given stream."""
        return lambda stream: _FakeStreamContext(stream, self)

    def record_stream(self, tensor: torch.Tensor, stream: _FakeStream) -> None:
        """Stand in for ``_record_stream`` (a real one rejects a fake stream)."""
        stream.record_stream(tensor)


class _Run(NamedTuple):
    """One driven exchange: per-rank results plus what each rank issued."""

    outputs: dict
    grads: dict
    world: object
    lazy_calls: list
    pending: dict


def _transpose(plan: List[List[int]]) -> List[List[int]]:
    """Row ``i`` of the transposed plan: what rank ``i`` receives from each peer."""
    return [[plan[src][rank] for src in range(len(plan))] for rank in range(len(plan))]


def _inputs(send_plan: List[List[int]], backward: bool) -> dict:
    """Per-rank payloads whose row values identify which rows arrived where."""
    inputs = {}
    for rank, counts in enumerate(send_plan):
        rows = sum(counts)
        row_ids = torch.arange(rows, dtype=torch.float32) + rank * 1000
        inputs[rank] = row_ids.unsqueeze(1).repeat(1, _HIDDEN).requires_grad_(backward)
    return inputs


def _weights(recv_plan: List[List[int]]) -> dict:
    """Per-rank gradient weights: distinct values, so a wrong row order shows up."""
    return {
        rank: torch.arange(sum(counts), dtype=torch.float32) + rank * 100
        for rank, counts in enumerate(recv_plan)
    }


def _expected_exchange(send_plan: List[List[int]], recv_plan: List[List[int]],
                       inputs: dict) -> dict:
    """Peer-major reference: what each rank receives, from which rows, in order.

    Rank ``dest``'s result is the concatenation, in source-rank order, of the
    block each source split out of its input for ``dest`` -- the contract both
    exchange paths implement.
    """
    expected = {}
    for dest, dest_recv in enumerate(recv_plan):
        pieces = []
        for src, counts in enumerate(send_plan):
            start = sum(counts[:dest])
            pieces.append(inputs[src][start:start + dest_recv[src]])
        expected[dest] = torch.cat(pieces)
    return expected


def _expected_grad(send_plan: List[List[int]], recv_plan: List[List[int]], weights: dict) -> dict:
    """Reference gradient: the same exchange run backwards over the weights."""
    expected = {}
    for src, counts in enumerate(send_plan):
        pieces = []
        for dest, dest_recv in enumerate(recv_plan):
            start = sum(dest_recv[:src])
            rows = weights[dest][start:start + counts[dest]]
            pieces.append(rows.unsqueeze(1).repeat(1, _HIDDEN))
        expected[src] = torch.cat(pieces)
    return expected


@contextlib.contextmanager
def _equal_a2a_scope(dist_double: Any, knob: str, streams: _FakeStreams):
    """Install everything the exchange resolves its world and its streams through.

    The lazy exchange reads the platform's stream/event API and the ``dist``
    module at call time, so both are patched for the duration of a case: the
    ``dist`` double routes the collectives into the virtual world, and the stream
    double traces what was enqueued where.  The three resource caches are reset
    so each case starts from "nothing created yet".

    Args:
        dist_double: Stand-in for the ``dist`` module ``collectives`` resolves.
        knob: Value of ``HP_EP_EQUAL_A2A`` for the run.
        streams: Stream/event double to install.
    """
    with mock.patch.object(ep_collectives, "dist", dist_double), \
            mock.patch.object(ep_collectives, "_EQUAL_A2A_RAW", knob), \
            mock.patch.multiple(ep_collectives,
                                _record_stream=streams.record_stream,
                                _LAZY_A2A_STREAM=None,
                                _LAZY_A2A_READY_EVENT=None,
                                _LAZY_A2A_DONE_EVENT=None), \
            mock.patch.object(ep_collectives, "get_device_handle", lambda: _DeviceHandleDouble(streams)):
        yield


def _drive(send_plan: List[List[int]], knob: str, *, entry: str = _SYNC, backward: bool = False,
           backend: str = _HCCL, world: Optional[_VirtualEpWorld] = None,
           streams: Optional[_FakeStreams] = None) -> _Run:
    """Run every virtual rank's exchange once, all ranks in lockstep.

    Args:
        send_plan: ``send_plan[rank][dest]`` rows rank ``rank`` dispatches to ``dest``.
        knob: Value of ``HP_EP_EQUAL_A2A`` for the run.
        entry: ``"sync"`` for :func:`ep_all_to_all`, ``"async"`` for
            :func:`ep_all_to_all_async`.
        backward: Backpropagate a weighted sum and collect the input gradients.
        backend: Backend the ``dist`` double reports.
        world: World double to drive; a fresh one by default.
        streams: Stream/event double to install; a fresh one by default.

    Returns:
        The :class:`_Run` holding each rank's output, its input gradient when
        requested, the world double, the calls the stubbed lazy exchange
        received (empty unless the async entry fell back to it), and the raw
        exchange results -- a pending handle where the lazy equal path was
        taken -- before the wait.
    """
    ep_size = len(send_plan)
    recv_plan = _transpose(send_plan)
    inputs = _inputs(send_plan, backward)
    weights = _weights(recv_plan)
    world = _VirtualEpWorld(ep_size) if world is None else world
    streams = _FakeStreams() if streams is None else streams
    dist_double = _VirtualGlooDist(world) if backend == _GLOO else _VirtualEpDist(world)
    exchange = ep_collectives.ep_all_to_all if entry == _SYNC else ep_collectives.ep_all_to_all_async
    outputs, grads, lazy_calls, pending, failures = {}, {}, [], {}, []

    def lazy_exchange(x: torch.Tensor, send_counts: List[int], recv_counts: List[int],
                      group: Any) -> torch.Tensor:
        """Stand-in for the platform's lazy exchange: record it, return the buffer."""
        lazy_calls.append((group.rank, tuple(send_counts), tuple(recv_counts)))
        return x.new_empty((sum(recv_counts),) + tuple(x.shape[1:]))

    def run_rank(rank: int) -> None:
        """One virtual rank: issue the exchange and, when asked, backpropagate."""
        try:
            streams.bind(rank)
            issued = exchange(
                inputs[rank], send_plan[rank], recv_plan[rank], _VirtualEpGroup(rank, ep_size))
            pending[rank] = issued
            received = ep_collectives.wait_ep_all_to_all(issued)
            outputs[rank] = received
            if backward:
                (received * weights[rank].unsqueeze(1)).sum().backward()
                grads[rank] = inputs[rank].grad.clone()
        except BaseException as exc:  # pylint: disable=broad-except
            failures.append((rank, exc))

    with _equal_a2a_scope(dist_double, knob, streams), \
            mock.patch.object(ep_collectives,
                              "differentiable_all_to_all_single_async",
                              side_effect=lazy_exchange):
        threads = [threading.Thread(target=run_rank, args=(rank,)) for rank in range(ep_size)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    if failures:
        raise failures[0][1]
    return _Run(outputs, grads, world, lazy_calls, pending)


class TestEqualA2ADispatch(unittest.TestCase):
    """Which collective each entry issues on a uniform / uneven routing plan."""

    @staticmethod
    def _uniform_plan() -> List[List[int]]:
        """The balanced-router shape: every peer gets the same row count."""
        return [[_UNIFORM_ROWS] * _EP_SIZE for _ in range(_EP_SIZE)]

    def _assert_matches_reference(self, outputs: dict, send_plan: List[List[int]],
                                  label: str) -> None:
        """Assert every rank received the rows the plan says it should."""
        expected = _expected_exchange(send_plan, _transpose(send_plan),
                                      _inputs(send_plan, backward=False))
        for rank in range(len(send_plan)):
            with self.subTest(label=label, rank=rank):
                self.assertTrue(
                    torch.equal(outputs[rank], expected[rank]),
                    f"{label}: rank {rank} received {outputs[rank].flatten().tolist()}, "
                    f"the plan says {expected[rank].flatten().tolist()}")

    def _assert_grad_matches_reference(self, grads: dict, send_plan: List[List[int]],
                                       label: str) -> None:
        """Assert every rank's gradient came back in its own input row order."""
        recv_plan = _transpose(send_plan)
        expected = _expected_grad(send_plan, recv_plan, _weights(recv_plan))
        for rank in range(len(send_plan)):
            with self.subTest(label=label, rank=rank):
                self.assertTrue(
                    torch.equal(grads[rank], expected[rank]),
                    f"{label}: rank {rank} gradient is {grads[rank].flatten().tolist()}, "
                    f"the plan says {expected[rank].flatten().tolist()}")

    @arg_mark(**_CPU_MARKS)
    def test_knob_off_keeps_the_ragged_path(self):
        """Unset (or 0) issues exactly the ragged exchange it issues today.

        Feature: ``HP_EP_EQUAL_A2A`` default.
        Description: Run the sync entry on a uniform plan with the knob off.
        Expectation: Every rank issues the ragged collective, forward and backward alike.
        """
        send_plan = self._uniform_plan()
        run = _drive(send_plan, "0", backward=True)
        for rank in range(_EP_SIZE):
            with self.subTest(rank=rank):
                self.assertEqual(run.world.calls(rank), ["ragged", "ragged"],
                                 f"rank {rank} issued {run.world.calls(rank)}")
        self._assert_matches_reference(run.outputs, send_plan, "knob off")
        self._assert_grad_matches_reference(run.grads, send_plan, "knob off")

    @arg_mark(**_CPU_MARKS)
    def test_knob_on_with_uniform_counts_takes_the_equal_path(self):
        """A uniform plan issues the split-free collective, forward and backward.

        Feature: ``HP_EP_EQUAL_A2A`` on a balanced plan.
        Description: Run the sync entry on a uniform plan with the knob on.
        Expectation: Every rank issues the split-free collective, with the planned rows.
        """
        send_plan = self._uniform_plan()
        run = _drive(send_plan, "1", backward=True)
        for rank in range(_EP_SIZE):
            with self.subTest(rank=rank):
                self.assertEqual(run.world.calls(rank), ["equal", "equal"],
                                 f"rank {rank} issued {run.world.calls(rank)}")
        self._assert_matches_reference(run.outputs, send_plan, "knob on")
        self._assert_grad_matches_reference(run.grads, send_plan, "knob on")

    @arg_mark(**_CPU_MARKS)
    def test_equal_path_is_bit_identical_to_the_ragged_path(self):
        """Both paths move the same rows: same output, same backward, no tolerance.

        Feature: equal-length exchange equivalence.
        Description: Compare the two arms of the knob on the same uniform plan.
        Expectation: Output and input gradient are identical element-wise.
        """
        send_plan = self._uniform_plan()
        ragged = _drive(send_plan, "0", backward=True)
        equal = _drive(send_plan, "1", backward=True)
        for rank in range(_EP_SIZE):
            with self.subTest(rank=rank, direction="forward"):
                self.assertTrue(
                    torch.equal(ragged.outputs[rank], equal.outputs[rank]),
                    f"rank {rank}: ragged={ragged.outputs[rank].flatten().tolist()}, "
                    f"equal={equal.outputs[rank].flatten().tolist()}")
            with self.subTest(rank=rank, direction="backward"):
                self.assertTrue(
                    torch.equal(ragged.grads[rank], equal.grads[rank]),
                    f"rank {rank}: ragged grad={ragged.grads[rank].flatten().tolist()}, "
                    f"equal grad={equal.grads[rank].flatten().tolist()}")

    @arg_mark(**_CPU_MARKS)
    def test_knob_on_with_uneven_counts_keeps_the_ragged_path(self):
        """An uneven plan must not be re-chunked into equal pieces.

        Feature: ``HP_EP_EQUAL_A2A`` on an uneven plan.
        Description: Run the sync entry on a plan whose per-peer counts all differ,
            while the totals still balance (8 rows sent and received by each rank).
        Expectation: Every rank issues the ragged collective and moves the planned rows.
        """
        send_plan = [[4, 3, 1, 0], [0, 4, 3, 1], [1, 0, 4, 3], [3, 1, 0, 4]]
        run = _drive(send_plan, "1", backward=True)
        for rank in range(_EP_SIZE):
            with self.subTest(rank=rank):
                self.assertEqual(run.world.calls(rank), ["ragged", "ragged"],
                                 f"rank {rank} issued {run.world.calls(rank)}")
        self._assert_matches_reference(run.outputs, send_plan, "uneven plan")
        self._assert_grad_matches_reference(run.grads, send_plan, "uneven plan")

    @arg_mark(**_CPU_MARKS)
    def test_uniform_counts_that_do_not_describe_the_payload_fall_back(self):
        """Uniform counts for the wrong payload size stay on the ragged path.

        Feature: equal-path payload guard.
        Description: Ask for 3 rows per peer while the payload holds 5 rows.  The
            split-free call takes its chunk size from the tensor shape, so it would
            exchange rows the counts never asked for.
        Expectation: The guard rejects the plan and the ragged path reports the mismatch.
        """
        payload = torch.zeros(5, _HIDDEN)
        world = _VirtualEpWorld(1)
        with mock.patch.object(ep_collectives, "dist", _VirtualEpDist(world)), \
                mock.patch.object(ep_collectives, "_EQUAL_A2A_RAW", "1"):
            self.assertIsNone(
                ep_collectives._equal_a2a_rows(
                    payload, [3], [3], _VirtualEpGroup(0, 1)),
                "a payload that does not hold the counted rows must not take the equal path")
            with self.assertRaisesRegex(RuntimeError, "split_sizes"):
                ep_collectives.ep_all_to_all(payload, [3], [3], _VirtualEpGroup(0, 1))
        self.assertEqual(world.calls(0), [],
                         f"nothing may be exchanged here, but rank 0 issued {world.calls(0)}")

    @arg_mark(**_CPU_MARKS)
    def test_gloo_backend_ignores_the_knob(self):
        """The padded fallback is not touched: even on a uniform plan it pads.

        Feature: padded fallback isolation.
        Description: Run the sync entry on a gloo backend with the knob on.
        Expectation: The padded exchange runs and moves the planned rows.
        """
        send_plan = self._uniform_plan()
        world = _VirtualGlooWorld(_EP_SIZE)
        run = _drive(send_plan, "1", backend=_GLOO, world=world)
        for rank in range(_EP_SIZE):
            with self.subTest(rank=rank):
                self.assertEqual(run.world.calls(rank), ["padded"],
                                 f"rank {rank} issued {run.world.calls(rank)}")
        self._assert_matches_reference(run.outputs, send_plan, "gloo fallback")

    @arg_mark(**_CPU_MARKS)
    def test_async_entry_takes_the_lazy_equal_path_on_a_uniform_plan(self):
        """The async entry issues the split-free exchange and keeps it pending.

        Feature: ``HP_EP_EQUAL_A2A=1`` on the async entry.
        Description: Run ``ep_all_to_all_async`` on a uniform plan with the knob on.
        Expectation: The lazy ragged exchange is not used, every rank issues the
            split-free collective, and what comes back is a pending handle whose wait
            yields the planned rows.
        """
        send_plan = self._uniform_plan()
        streams = _FakeStreams()
        run = _drive(send_plan, "1", entry=_ASYNC, streams=streams)
        for rank in range(_EP_SIZE):
            with self.subTest(rank=rank):
                self.assertEqual(run.world.calls(rank), ["equal"],
                                 f"rank {rank} issued {run.world.calls(rank)}")
                self.assertIsInstance(
                    run.pending[rank], ep_collectives._PendingEqualA2A,  # pylint: disable=W0212
                    f"rank {rank} must get a pending handle, got {type(run.pending[rank])}")
        self.assertEqual(run.lazy_calls, [],
                         f"the lazy exchange was still used: {run.lazy_calls}")
        self._assert_matches_reference(run.outputs, send_plan, "async equal")

    @arg_mark(**_CPU_MARKS)
    def test_async_entry_returns_a_tensor_in_eager_mode(self):
        """``HP_EP_EQUAL_A2A=eager`` keeps the split-free kernel, not the laziness.

        Feature: ``HP_EP_EQUAL_A2A`` eager mode on the async entry.
        Description: Run ``ep_all_to_all_async`` on a uniform plan with the knob at
            ``eager`` -- the behaviour ``1`` used to have.
        Expectation: The split-free collective runs and a materialized tensor comes
            back, so this mode is the eager arm of the kernel-vs-latency A/B.
        """
        send_plan = self._uniform_plan()
        run = _drive(send_plan, "eager", entry=_ASYNC)
        for rank in range(_EP_SIZE):
            with self.subTest(rank=rank):
                self.assertEqual(run.world.calls(rank), ["equal"],
                                 f"rank {rank} issued {run.world.calls(rank)}")
                self.assertIsInstance(
                    run.pending[rank], torch.Tensor,
                    f"rank {rank} must get a tensor, got {type(run.pending[rank])}")
        self._assert_matches_reference(run.outputs, send_plan, "async equal eager")

    @arg_mark(**_CPU_MARKS)
    def test_a_caller_that_cannot_defer_gets_a_materialized_exchange(self):
        """``allow_pending=False`` refuses the handle and keeps the eager kernel.

        Feature: pending-handle opt-out (the fused states+indices dispatch).
        Description: Ask the async entry for a uniform plan with the handle refused.
        Expectation: The split-free collective runs, its result is a tensor, and the
            caller's stream was ordered on it before the call returned.
        """
        send_plan = self._uniform_plan()
        world = _VirtualEpWorld(_EP_SIZE)
        streams = _FakeStreams()
        results, failures = {}, []

        def issue(rank: int) -> None:
            """One rank: issue through the real entry with the handle refused."""
            try:
                streams.bind(rank)
                results[rank] = ep_collectives.ep_all_to_all_async(
                    _inputs(send_plan, backward=False)[rank],
                    send_plan[rank], _transpose(send_plan)[rank],
                    _VirtualEpGroup(rank, _EP_SIZE),
                    allow_pending=False,
                )
            except BaseException as exc:  # pylint: disable=broad-except
                failures.append((rank, exc))

        with mock.patch.object(ep_collectives, "dist", _VirtualEpDist(world)), \
                mock.patch.object(ep_collectives, "_EQUAL_A2A_RAW", "1"), \
                mock.patch.object(ep_collectives, "get_device_handle", lambda: _DeviceHandleDouble(streams)):
            threads = [threading.Thread(target=issue, args=(rank,)) for rank in range(_EP_SIZE)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        if failures:
            raise failures[0][1]
        for rank in range(_EP_SIZE):
            with self.subTest(rank=rank):
                self.assertIsInstance(
                    results[rank], torch.Tensor,
                    f"rank {rank} must get a materialized tensor, got {type(results[rank])}")
        self.assertEqual(streams.new_stream_calls, 0,
                         f"an eager exchange must not touch the comm stream, "
                         f"but it was created {streams.new_stream_calls} times")

    @arg_mark(**_CPU_MARKS)
    def test_async_entry_keeps_the_lazy_exchange_when_the_knob_is_off(self):
        """Unchanged behaviour: the knob off leaves the lazy path in place.

        Feature: ``HP_EP_EQUAL_A2A`` default on the async entry.
        Description: Run ``ep_all_to_all_async`` on a uniform plan with the knob off.
        Expectation: Every rank hands its counts to the lazy exchange and issues no other
            collective.
        """
        send_plan = self._uniform_plan()
        run = _drive(send_plan, "0", entry=_ASYNC)
        issued = {rank: run.world.calls(rank) for rank in range(_EP_SIZE)}
        self.assertEqual(issued, {rank: [] for rank in range(_EP_SIZE)},
                         f"no bare collective may be issued, but the ranks issued {issued}")
        # ``lazy_calls`` is appended from each rank's own thread, so compare as a
        # multiset -- the ranks are what matter, not the order they finished in.
        self.assertEqual(
            collections.Counter(run.lazy_calls),
            collections.Counter(
                (rank, (_UNIFORM_ROWS,) * _EP_SIZE, (_UNIFORM_ROWS,) * _EP_SIZE)
                for rank in range(_EP_SIZE)),
            f"the lazy exchange must take every rank's counts, got {run.lazy_calls}")

    @arg_mark(**_CPU_MARKS)
    def test_async_entry_keeps_the_lazy_exchange_on_an_uneven_plan(self):
        """An uneven plan keeps the lazy exchange, with its counts passed through.

        Feature: ``HP_EP_EQUAL_A2A`` on an uneven plan through the async entry.
        Description: Rank 0 sends two equal chunks but receives a 2/4 split, so the
            guard -- which needs both lists uniform -- must decline.
        Expectation: The lazy exchange is used with the uneven counts and no bare
            collective is issued.
        """
        send_plan = [[2, 2], [4, 0]]
        run = _drive(send_plan, "1", entry=_ASYNC)
        self.assertEqual(
            collections.Counter(run.lazy_calls),
            collections.Counter([(0, (2, 2), (2, 4)), (1, (4, 0), (2, 0))]),
            f"the lazy exchange must take the uneven counts, got {run.lazy_calls}")
        issued = {rank: run.world.calls(rank) for rank in range(len(send_plan))}
        self.assertEqual(issued, {rank: [] for rank in range(len(send_plan))},
                         f"the equal path must not be taken, but the ranks issued {issued}")


class TestLazyEqualA2A(unittest.TestCase):
    """The lazy split-free exchange: where it is issued, where it is waited, and
    the resources it must not allocate per exchange."""

    _ROWS = 3

    def _payload(self, *, width: int = _HIDDEN) -> torch.Tensor:
        """Payload whose row values identify each row's position."""
        rows = torch.arange(self._ROWS, dtype=torch.float32) + 1000
        return rows.unsqueeze(1).repeat(1, width)

    @staticmethod
    def _issue(payload: torch.Tensor, *, allow_pending: bool = True) -> Any:
        """Issue one single-rank exchange through the real async entry.

        The ``dist`` double of a single-rank world makes the exchange an
        identity, so what the case checks is the schedule and the handle, not
        the routing (that is covered by the multi-rank cases).
        """
        return ep_collectives.ep_all_to_all_async(
            payload, [TestLazyEqualA2A._ROWS], [TestLazyEqualA2A._ROWS],  # pylint: disable=W0212
            _VirtualEpGroup(0, 1), allow_pending=allow_pending)

    @arg_mark(**_CPU_MARKS)
    def test_issue_puts_the_exchange_on_the_comm_stream(self):
        """The split-free call runs on the comm stream, not on the caller's.

        Feature: lazy equal exchange, issue side.
        Description: Issue one exchange and read the recorded stream/event trace.
        Expectation: The payload is handed to the comm stream through the ready
            event, the collective runs inside the comm stream context, and the
            compute stream is not ordered on the transfer at issue time.
        """
        world, streams = _VirtualEpWorld(1), _FakeStreams()
        with _equal_a2a_scope(_VirtualEpDist(world), "1", streams):
            issued = self._issue(self._payload())
            trace = streams.entries(0)
        self.assertIsInstance(issued, ep_collectives._PendingEqualA2A,  # pylint: disable=W0212
                              f"the lazy equal path must return a handle, got {type(issued)}")
        self.assertFalse(issued.completed, "a just-issued exchange must still be pending")
        self.assertEqual(world.calls(0), ["equal"],
                         f"rank 0 must issue the split-free collective, got {world.calls(0)}")
        self.assertEqual(
            trace,
            ["record:event1<-compute",   # the payload is handed to the comm stream
             "enter@comm",
             "wait:event1->comm",
             "record_stream@comm",       # the receive buffer survives the stream switch
             "record_stream@comm",       # ... and so does the payload's storage
             "record:event2<-comm",      # the exchange is done
             "exit@comm"],
            f"unexpected issue trace: {trace}")

    @arg_mark(**_CPU_MARKS)
    def test_wait_is_the_callers_decision(self):
        """Independent work fits between the issue and the wait.

        Feature: lazy equal exchange, wait side.
        Description: Issue the exchange, mark the caller's independent work, then
            wait and read the result.
        Expectation: The completion is recorded before the marker and the consumer
            stream is ordered on it only after the marker; the rows are the payload's.
        """
        world, streams = _VirtualEpWorld(1), _FakeStreams()
        payload = self._payload()
        with _equal_a2a_scope(_VirtualEpDist(world), "1", streams):
            issued = self._issue(payload)
            streams.mark("overlap")  # where the shared-expert MLP would run
            received = ep_collectives.wait_ep_all_to_all(issued)
            trace = streams.entries(0)
        marker = trace.index("overlap")
        self.assertLess(trace.index("record:event2<-comm"), marker,
                        f"the exchange must be complete before the overlap: {trace}")
        self.assertLess(marker, trace.index("wait:event2->compute"),
                        f"the wait must land after the overlap, not at issue: {trace}")
        self.assertTrue(issued.completed, "waiting must mark the handle completed")
        self.assertTrue(torch.equal(received, payload),
                        f"received={received.flatten().tolist()}, "
                        f"payload={payload.flatten().tolist()}")

    @arg_mark(**_CPU_MARKS)
    def test_repeated_waits_order_the_consumer_stream_once(self):
        """Waiting twice is the same as waiting once (idempotent handle).

        Feature: lazy equal exchange, wait idempotency.
        Description: Wait the same handle twice.
        Expectation: The consumer stream is ordered on the exchange exactly once.
        """
        world, streams = _VirtualEpWorld(1), _FakeStreams()
        with _equal_a2a_scope(_VirtualEpDist(world), "1", streams):
            issued = self._issue(self._payload())
            first = ep_collectives.wait_ep_all_to_all(issued)
            second = ep_collectives.wait_ep_all_to_all(issued)
            trace = streams.entries(0)
        self.assertIs(first, second, "both waits must return the same buffer")
        self.assertEqual(trace.count("wait:event2->compute"), 1,
                         f"the consumer stream was ordered {trace.count('wait:event2->compute')} "
                         f"times, trace={trace}")

    @arg_mark(**_CPU_MARKS)
    def test_squeeze_keeps_the_exchange_pending(self):
        """The expert-index squeeze is a view and must not force the wait.

        Feature: lazy equal exchange, view-only squeeze.
        Description: Squeeze the trailing dim of a ``[rows, 1]`` payload, then wait.
        Expectation: No consumer wait is traced before the explicit one, and the
            squeezed rows are the payload's rows.
        """
        world, streams = _VirtualEpWorld(1), _FakeStreams()
        payload = self._payload(width=1)
        with _equal_a2a_scope(_VirtualEpDist(world), "1", streams):
            issued = self._issue(payload).squeeze(-1)
            squeezed_trace = streams.entries(0)
            received = ep_collectives.wait_ep_all_to_all(issued)
        self.assertIsInstance(issued, ep_collectives._PendingEqualA2A,  # pylint: disable=W0212
                              f"squeeze must stay pending, got {type(issued)}")
        self.assertNotIn("wait:event2->compute", squeezed_trace,
                         f"squeeze must not materialize the exchange: {squeezed_trace}")
        self.assertEqual(tuple(received.shape), (self._ROWS,),
                         f"the squeezed result must be [{self._ROWS}], "
                         f"got {tuple(received.shape)}")
        self.assertTrue(torch.equal(received, payload.squeeze(-1)),
                        f"received={received.tolist()}, payload={payload.squeeze(-1).tolist()}")

    @arg_mark(**_CPU_MARKS)
    def test_the_comm_stream_and_events_are_created_once(self):
        """One stream and two events serve every exchange of the process.

        Feature: lazy equal exchange resource reuse.
        Description: Run four virtual ranks in lockstep, each issuing one exchange.
        Expectation: The platform's factories were asked for one stream and two
            events in total, so no exchange allocates its own.
        """
        send_plan = [[_UNIFORM_ROWS] * _EP_SIZE for _ in range(_EP_SIZE)]
        streams = _FakeStreams()
        run = _drive(send_plan, "1", entry=_ASYNC, streams=streams)
        self.assertEqual(streams.new_stream_calls, 1,
                         f"the comm stream must be created once, "
                         f"but it was requested {streams.new_stream_calls} times")
        self.assertEqual(streams.new_event_calls, 2,
                         f"the two events must be created once each, "
                         f"but they were requested {streams.new_event_calls} times")
        self.assertEqual({rank: run.world.calls(rank) for rank in range(_EP_SIZE)},
                         {rank: ["equal"] for rank in range(_EP_SIZE)},
                         "every rank must have issued exactly one exchange")

    @arg_mark(**_CPU_MARKS)
    def test_lazy_output_and_gradient_are_bit_identical_to_the_ragged_path(self):
        """The lazy exchange moves the rows the ragged one moves, forward and back.

        Feature: lazy split-free equivalence.
        Description: Compare the eager ragged entry with the lazy equal entry on the
            same uniform plan, both backpropagated.
        Expectation: Output and input gradient are identical element-wise.
        """
        send_plan = [[_UNIFORM_ROWS] * _EP_SIZE for _ in range(_EP_SIZE)]
        ragged = _drive(send_plan, "0", backward=True)
        lazy = _drive(send_plan, "1", entry=_ASYNC, backward=True)
        for rank in range(_EP_SIZE):
            with self.subTest(rank=rank, direction="forward"):
                self.assertTrue(
                    torch.equal(ragged.outputs[rank], lazy.outputs[rank]),
                    f"rank {rank}: ragged={ragged.outputs[rank].flatten().tolist()}, "
                    f"lazy={lazy.outputs[rank].flatten().tolist()}")
            with self.subTest(rank=rank, direction="backward"):
                self.assertTrue(
                    torch.equal(ragged.grads[rank], lazy.grads[rank]),
                    f"rank {rank}: ragged grad={ragged.grads[rank].flatten().tolist()}, "
                    f"lazy grad={lazy.grads[rank].flatten().tolist()}")


class TestEqualA2AGuard(unittest.TestCase):
    """When the equal path applies: uniformity, payload and backend checks."""

    def _rows(self, send_counts: List[int], recv_counts: List[int], row_count: int, *,
              knob: str = "1", backend: str = _HCCL, contiguous: bool = True) -> Optional[int]:
        """Resolve the guard for one payload and plan."""
        if contiguous:
            payload = torch.zeros(row_count, _HIDDEN)
        else:
            payload = torch.zeros(row_count, _HIDDEN * 2)[:, :_HIDDEN]
        group = _VirtualEpGroup(0, len(send_counts))
        with mock.patch.object(ep_collectives, "dist", _BackendOnlyDist(backend)), \
                mock.patch.object(ep_collectives, "_EQUAL_A2A_RAW", knob):
            return ep_collectives._equal_a2a_rows(payload, send_counts, recv_counts, group)

    @arg_mark(**_CPU_MARKS)
    def test_uniform_plan_matching_the_payload_is_accepted(self):
        """The balanced-router shape is the case the knob exists for.

        Feature: equal-path guard acceptance.
        Description: Resolve the guard for uniform plans of several payload sizes.
        Expectation: Each is accepted with the payload's per-peer row count.
        """
        for row_count in (0, 4, 12, 64):
            with self.subTest(row_count=row_count):
                self.assertEqual(
                    self._rows([row_count // 4] * 4, [row_count // 4] * 4, row_count),
                    row_count // 4, f"a uniform {row_count}-row payload must be accepted")

    @arg_mark(**_CPU_MARKS)
    def test_knob_off_is_never_accepted(self):
        """With the knob off the guard must not select the fast path at all.

        Feature: ``HP_EP_EQUAL_A2A`` gating.
        Description: Resolve the guard for a uniform plan with the knob off.
        Expectation: The fast path is declined, so the ragged path stays in place.
        """
        self.assertIsNone(self._rows([3, 3], [3, 3], 6, knob="0"),
                          "HP_EP_EQUAL_A2A=0 must keep the ragged path")

    @arg_mark(**_CPU_MARKS)
    def test_uneven_plan_is_rejected(self):
        """Any per-peer count that differs keeps the ragged path.

        Feature: equal-path guard uniformity.
        Description: Resolve the guard for plans with one differing send or receive count.
        Expectation: Both are declined, so the payload is never re-chunked.
        """
        self.assertIsNone(self._rows([3, 4], [4, 3], 7),
                          "an uneven send count must keep the ragged path")
        self.assertIsNone(self._rows([3, 3], [3, 4], 6),
                          "an uneven receive count must keep the ragged path")

    @arg_mark(**_CPU_MARKS)
    def test_unequal_send_and_receive_counts_are_rejected(self):
        """The split-free call needs the receive buffer to be the input's size.

        Feature: equal-path guard send/receive symmetry.
        Description: Resolve the guard for uniform counts that differ between the two
            directions (3 sent, 2 received per peer).
        Expectation: The plan is declined rather than sized wrongly.
        """
        self.assertIsNone(self._rows([3, 3], [2, 2], 6),
                          "uniform but different send/receive counts must keep the ragged path")

    @arg_mark(**_CPU_MARKS)
    def test_payload_disagreeing_with_the_counts_is_rejected(self):
        """Uniform counts are not enough: the payload must hold them.

        Feature: equal-path guard payload check.
        Description: Resolve the guard for uniform plans whose counts do not describe
            the payload's row count.
        Expectation: Both mismatching payloads are declined.
        """
        self.assertIsNone(self._rows([3, 3], [3, 3], 5),
                          "a 5-row payload cannot hold 3 rows per peer")
        self.assertIsNone(self._rows([3, 3], [3, 3], 12),
                          "a 12-row payload does not match 3 rows per peer")

    @arg_mark(**_CPU_MARKS)
    def test_mismatched_count_lengths_are_rejected(self):
        """The counts must describe the group, not a shorter or longer one.

        Feature: equal-path guard plan length.
        Description: Resolve the guard for count lists whose length is not the group size.
        Expectation: Both malformed plans are declined.
        """
        self.assertIsNone(self._rows([3, 3], [3], 6),
                          "a receive count list of the wrong length must be rejected")
        self.assertIsNone(self._rows([], [], 0),
                          "an empty plan must be rejected")

    @arg_mark(**_CPU_MARKS)
    def test_gloo_backend_is_rejected(self):
        """A backend without a ragged all-to-all keeps its padded path.

        Feature: equal-path guard backend check.
        Description: Resolve the guard for a uniform plan on a gloo backend.
        Expectation: The plan is declined so the padded fallback is used unchanged.
        """
        self.assertIsNone(self._rows([3, 3], [3, 3], 6, backend=_GLOO),
                          "gloo must keep the padded path")

    @arg_mark(**_CPU_MARKS)
    def test_non_contiguous_payload_is_rejected(self):
        """The split-free call needs a contiguous payload; anything else keeps today's path.

        Feature: equal-path guard contiguity check.
        Description: Resolve the guard for a uniform plan over a tensor view.
        Expectation: The plan is declined instead of silently copying the payload.
        """
        self.assertIsNone(self._rows([3, 3], [3, 3], 6, contiguous=False),
                          "a non-contiguous payload must keep the ragged path")


class TestEqualA2AKnob(unittest.TestCase):
    """``HP_EP_EQUAL_A2A`` parsing: default off, whitespace tolerated, typos fatal."""

    def _mode(self, raw: str) -> str:
        """Resolve the knob with ``raw`` as its value."""
        with mock.patch.object(ep_collectives, "_EQUAL_A2A_RAW", raw):
            return ep_collectives._equal_a2a_mode()

    @arg_mark(**_CPU_MARKS)
    def test_default_is_off(self):
        """The module-level default keeps the fast path disabled.

        Feature: ``HP_EP_EQUAL_A2A`` default value.
        Description: Read the knob from an environment that does not set it.
        Expectation: The module default and the documented default are both "0", and
            they resolve to the off mode.
        """
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HP_EP_EQUAL_A2A", None)
            self.assertEqual(os.environ.get("HP_EP_EQUAL_A2A", "0"), "0",
                             "the documented default must be '0'")
        self.assertEqual(ep_collectives._EQUAL_A2A_RAW,  # pylint: disable=protected-access
                         os.environ.get("HP_EP_EQUAL_A2A", "0"),
                         "the module must read the knob defaulting to '0'")
        self.assertEqual(self._mode(ep_collectives._EQUAL_A2A_RAW),  # pylint: disable=W0212
                         ep_collectives._EQUAL_A2A_OFF,  # pylint: disable=W0212
                         "an unset knob must resolve to the off mode")

    @arg_mark(**_CPU_MARKS)
    def test_values_resolve_to_their_modes(self):
        """``0`` / ``1`` / ``eager`` (with surrounding whitespace) are the valid values.

        Feature: ``HP_EP_EQUAL_A2A`` parsing.
        Description: Resolve the knob for its documented values and padded forms.
        Expectation: "0" turns the fast path off, "1" defers its wait and "eager"
            takes the same exchange without the deferral.
        """
        for raw, expected in (("0", "off"), ("1", "lazy"), (" 1 ", "lazy"),
                              ("0 ", "off"), ("eager", "eager"), (" eager ", "eager")):
            with self.subTest(raw=raw):
                self.assertEqual(self._mode(raw), expected,
                                 f"HP_EP_EQUAL_A2A={raw!r} must resolve to {expected}")

    @arg_mark(**_CPU_MARKS)
    def test_invalid_values_are_rejected(self):
        """A typo fails loudly instead of silently leaving the fast path off.

        Feature: ``HP_EP_EQUAL_A2A`` validation.
        Description: Resolve the knob for values that are none of the accepted ones.
        Expectation: Every one of them raises a ValueError naming the knob.
        """
        for raw in ("", "2", "true", "yes", "-1", "on", "lazyish", "Eager"):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, "HP_EP_EQUAL_A2A"):
                    self._mode(raw)


if __name__ == "__main__":
    unittest.main()
