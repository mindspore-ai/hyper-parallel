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
"""Two-thread comm/compute overlap orchestrator.

This module provides :class:`CommComputeOverlap`, a helper that wraps
MoE-style dispatch / combine phases with four synchronization hooks
(``A``, ``B``, ``C``, ``D``) and drives a forward + backward pass on two
threads with deterministic comm-first dispatch ordering via
:class:`HookCoordinator`.

The mechanism is independent of any specific pipeline schedule.  It is
typically driven by the ``OVERLAP_B_F`` callback registered on a
schedule (e.g. ``ScheduleInterleaved1F1B(overlap_b_f=True)``), but the
same orchestrator could be reused by other concurrent-dispatch overlap
scenarios (TP+CP, FSDP prefetch, etc.) without modification.

Every rendezvous is a strict COMM + COMPUTE pair — including layer
boundaries — so the NCCL kernel is always enqueued before the paired
compute kernel::

    [A] ─► dispatch ─► [B] ─► module ─► [C] ─► combine ─► [D] ─► (Attention) ─► [A_next]

At layer boundaries the D / A hooks coordinate combine (COMM) with the
other thread's Attention (COMPUTE), preserving overlap across layers.

Typical integration::

    overlap = CommComputeOverlap()

    # Wrap the expert-parallel dispatch / combine callables:
    wrapped_dispatch = overlap.wrap_dispatch(original_dispatch)
    wrapped_combine  = overlap.wrap_combine(original_combine)

    # At schedule time, run forward and backward in parallel:
    overlap.run(
        fwd_fn=lambda: fwd_stage.forward_one_chunk(mb, *args),
        bwd_fn=lambda: bwd_stage.backward_one_chunk(mb, loss=loss),
    )
"""
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from hyper_parallel.platform import get_platform
from hyper_parallel.core.pipeline_parallel.hook_coordinator import HookCoordinator

platform = get_platform()

_WORKER_STOP = object()


@dataclass
class _BwdTask:
    """One backward task submitted to the persistent overlap worker."""

    bwd_fn: Callable[[], None]
    done: threading.Event
    exc_box: list[Exception]


class CommComputeOverlap:
    """Orchestrator for two-thread comm/compute overlap.

    Manages a :class:`HookCoordinator` and provides helpers to insert the
    four synchronization hooks (``A``, ``B``, ``C``, ``D``) around MoE
    dispatch / combine phases and to run forward + backward concurrently
    with deterministic comm-first kernel launch ordering.  The backward side
    uses one persistent daemon worker thread per orchestrator, avoiding a
    Python thread create/join cycle for every overlap window.

    Example:
        >>> overlap = CommComputeOverlap()
        >>> wrapped_dispatch = overlap.wrap_dispatch(ep_dispatch_fn)
        >>> wrapped_combine  = overlap.wrap_combine(ep_combine_fn, is_last_layer=is_last)
        >>> overlap.run(fwd_fn, bwd_fn)  # doctest: +SKIP
    """

    def __init__(self) -> None:
        self._coordinator = HookCoordinator()
        self._worker_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_lock = threading.Lock()
        self._run_lock = threading.Lock()

    @property
    def coordinator(self) -> HookCoordinator:
        """The underlying :class:`HookCoordinator` instance."""
        return self._coordinator

    # ------------------------------------------------------------------
    # Wrapping helpers
    # ------------------------------------------------------------------

    def wrap_dispatch(self, dispatch_fn: Callable) -> Callable:
        """Return a wrapped version of ``dispatch_fn`` bracketed by hooks A/B.

        The returned callable inserts synchronization hooks on the **first
        positional tensor argument** before and after the call::

            A ─► dispatch_fn ─► B

        Args:
            dispatch_fn: The original dispatch callable.

        Returns:
            A new callable with the same signature.
        """
        coordinator = self._coordinator

        def _wrapped(*args, **kwargs):
            first, rest = args[0], args[1:]
            first = platform.differentiable_sync_hook(first, "A", coordinator)
            result = dispatch_fn(first, *rest, **kwargs)
            if isinstance(result, tuple):
                hooked = platform.differentiable_sync_hook(result[0], "B", coordinator)
                return (hooked,) + result[1:]
            return platform.differentiable_sync_hook(result, "B", coordinator)

        return _wrapped

    def wrap_combine(self, combine_fn: Callable, is_last_layer: bool = False) -> Callable:
        """Return a wrapped version of ``combine_fn`` bracketed by hooks C/D.

        The returned callable inserts synchronization hooks on the **first
        positional tensor argument** before and after the call::

            C ─► combine_fn ─► D

        Args:
            combine_fn:    The original combine callable.
            is_last_layer: If ``True``, the closing D hook is tagged
                ``"D_LAST"`` so the rendezvous is skipped both in
                forward (no Attention follows the last layer) and in
                backward (this is the first BWD hook to fire and
                combine.bwd has already dispatched freely).  Tagging
                this hook statically replaces the old runtime cycle
                counter and BWD-D-skip mechanisms.

        Returns:
            A new callable with the same signature.
        """
        coordinator = self._coordinator
        d_hook = "D_LAST" if is_last_layer else "D"

        def _wrapped(*args, **kwargs):
            first, rest = args[0], args[1:]
            first = platform.differentiable_sync_hook(first, "C", coordinator)
            result = combine_fn(first, *rest, **kwargs)
            if isinstance(result, tuple):
                hooked = platform.differentiable_sync_hook(result[0], d_hook, coordinator)
                return (hooked,) + result[1:]
            return platform.differentiable_sync_hook(result, d_hook, coordinator)

        return _wrapped

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _ensure_worker(self) -> None:
        """Start the persistent backward worker if it is not already alive."""
        with self._worker_lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="hp-overlap-bwd-worker",
                daemon=True,
            )
            self._worker_thread.start()

    def _submit_bwd_task(self, task: _BwdTask) -> None:
        """Submit one backward task to the persistent worker."""
        self._ensure_worker()
        self._worker_queue.put(task)

    def _worker_loop(self) -> None:
        """Run submitted backward tasks serially on the persistent worker."""
        while True:
            task = self._worker_queue.get()
            try:
                if task is _WORKER_STOP:
                    return
                self._run_bwd_task(task)
            finally:
                self._worker_queue.task_done()

    def _run_bwd_task(self, task: _BwdTask) -> None:
        """Execute one backward task and preserve the existing cleanup contract."""
        coordinator = self._coordinator
        try:
            try:
                task.bwd_fn()
            except Exception as exc:  # pylint: disable=W0718
                task.exc_box.append(exc)
                # BWD died — disable the coordinator so any FWD rendezvous
                # waiting on a barrier/event unblocks immediately.  Without
                # this the FWD thread hangs forever at the very first hook
                # it reaches and the outer ``finally`` never runs.
                coordinator.disable()
            finally:
                # Graceful one-party-left: this BWD chunk is done.  If the
                # paired FWD chunk has MORE hooks (e.g. more layers) it would
                # otherwise block forever on the 2-party barrier waiting for
                # a partner that has exited.  ``depart`` aborts the barrier
                # and flags the coordinator so FWD's remaining hooks run
                # solo.  Required for correctness, not just on error.
                coordinator.depart()
        finally:
            task.done.set()

    def close(self) -> None:
        """Stop the persistent backward worker after its current task finishes."""
        with self._worker_lock:
            thread = self._worker_thread
            if thread is None:
                return
            if not thread.is_alive():
                self._worker_thread = None
                return
            self._worker_queue.put(_WORKER_STOP)
        thread.join()
        with self._worker_lock:
            if self._worker_thread is thread:
                self._worker_thread = None

    def run(
        self,
        fwd_fn: Callable[[], None],
        bwd_fn: Callable[[], None],
    ) -> None:
        """Run ``fwd_fn`` and ``bwd_fn`` with comm/compute overlap.

        Enables the coordinator, submits the backward pass to the persistent
        daemon worker, and waits for both sides to complete.  Layer-boundary
        handling is encoded statically by the ``is_last_layer`` flag passed
        to :meth:`wrap_combine` at wrap time, so no per-call layer count is
        needed here.

        Args:
            fwd_fn: Callable that executes the forward pass.
            bwd_fn: Callable that executes the backward pass.  If it needs
                    a specific device stream, wrap that logic inside
                    ``bwd_fn``.

        Raises:
            RuntimeError: If the backward thread raises an exception, it
                is re-raised on the main thread after the worker task finishes.
        """
        with self._run_lock:
            self._coordinator.enable()
            done = threading.Event()
            exc_box: list[Exception] = []
            try:
                self._submit_bwd_task(_BwdTask(bwd_fn, done, exc_box))
            except Exception:
                self._coordinator.disable()
                raise

            fwd_exc: list[Exception] = []
            try:
                fwd_fn()
            except Exception as exc:  # pylint: disable=W0718
                fwd_exc.append(exc)
                # Symmetric: if FWD dies, unblock BWD so it can exit.
                self._coordinator.disable()
            finally:
                # Graceful one-party-left, mirroring the worker task: if the
                # paired BWD chunk has MORE hooks, ``depart`` lets it drain its
                # remaining rendezvous solo instead of hanging on the barrier.
                # Must precede waiting on the task so a still-running BWD is
                # released.
                self._coordinator.depart()
                done.wait()
                # Full reset after both sides are done (idempotent with any
                # earlier disable on the FWD error path).
                self._coordinator.disable()

            if exc_box:
                raise RuntimeError(
                    "Exception in backward thread during dual-pipe overlap"
                ) from exc_box[0]
            if fwd_exc:
                raise fwd_exc[0]
