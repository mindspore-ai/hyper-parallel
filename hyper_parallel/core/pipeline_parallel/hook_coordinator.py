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
"""Event-based rendezvous coordinator for dual-pipe comm/compute overlap.

In a dual-pipe schedule two Python threads drive a pair of forward/backward
passes concurrently so that a communication op on one side can overlap with
a compute op on the other.

This module provides :class:`HookCoordinator`, which guarantees that the
``COMM`` side of each rendezvous dispatches its op *before* the paired
``COMPUTE`` side.  The coordinator combines:

    1. ``threading.Barrier(2)`` — both threads must arrive at the rendezvous
       point before either proceeds.
    2. **Per-rendezvous** ``threading.Event`` — a fresh event is created for
       each rendezvous by the ``COMM`` side, eliminating stale-state races
       between consecutive rendezvous points.

The 3-hook design (``CA``, ``B``, ``C``) merges the old ``D`` + ``A`` layer
boundary hooks into a single ``CA`` hook, removing the ``NONE`` role and
allowing combine + Attention to run without barrier interruption.
"""
import threading
from enum import Enum
from typing import Optional


class HookRole(Enum):
    """Role of the caller at a rendezvous point.

    Each rendezvous is a meeting between two threads.  Their roles decide the
    ordering of op dispatch after the rendezvous returns.
    """

    COMM = "comm"
    """Caller will dispatch a communication op after the rendezvous returns.
    The coordinator lets the ``COMM`` side proceed first so the NCCL kernel
    is enqueued before any paired compute kernel."""

    COMPUTE = "compute"
    """Caller will dispatch a compute op after the rendezvous returns.  The
    call blocks until the paired ``COMM`` side has signalled dispatch
    completion via :meth:`HookCoordinator.notify_dispatched`."""


class HookCoordinator:
    """Per-rendezvous Event coordinator for dual-pipe comm/compute overlap.

    Two threads call :meth:`rendezvous` at synchronization points, declaring
    their :class:`HookRole` for the op they are about to dispatch.  The
    coordinator guarantees:

        1. Both threads are at the rendezvous point before either proceeds
           (via :class:`threading.Barrier`).
        2. If the pair is ``COMM`` + ``COMPUTE``, the ``COMM`` side dispatches
           its op before the ``COMPUTE`` side (via a fresh
           :class:`threading.Event` per rendezvous).

    Each ``COMM`` rendezvous creates a **fresh Event**, stored both in the
    shared ``_comm_dispatched`` slot (read by the ``COMPUTE`` side after the
    barrier) and in a per-thread ``_my_event`` (used by the subsequent
    :meth:`notify_dispatched` call at the next hook).  This eliminates the
    stale-state race that a single long-lived Event would suffer when
    ``notify`` at hook *k+1* races with ``clear`` at hook *k+1* from the
    other thread.

    Example:
        >>> coordinator = HookCoordinator()
        >>> coordinator.enable()
        >>>
        >>> # Forward thread, at a hook point whose next op is comm:
        >>> coordinator.rendezvous(HookRole.COMM)
        >>> dispatch_comm_op()                             # doctest: +SKIP
        >>> coordinator.notify_dispatched(HookRole.COMM)
        >>>
        >>> # Backward thread, at a hook point whose next op is compute:
        >>> coordinator.rendezvous(HookRole.COMPUTE)       # blocks
        >>> dispatch_compute_op()                          # doctest: +SKIP
        >>>
        >>> coordinator.disable()
    """

    def __init__(self) -> None:
        self._barrier: Optional[threading.Barrier] = None
        self._comm_dispatched = threading.Event()
        # Per-thread storage: after a COMM rendezvous the calling thread
        # saves the freshly created Event here so that the *next*
        # notify_dispatched(COMM) call (at the following hook) can set it.
        # Threads that have not yet done a COMM rendezvous have no ``.evt``
        # attribute, so their notify is naturally a no-op.
        self._my_event: threading.local = threading.local()
        self._state_lock = threading.Lock()
        self._enabled = False

    def enable(self) -> None:
        """Enable coordination for a new dual-pipe session.

        Layer-boundary handling (skipping the rendezvous at the
        last-layer ``D`` hook in both forward and backward) is encoded
        statically by tagging that hook ``"D_LAST"`` at wrap time, so
        no per-session layer count is needed here.
        """
        with self._state_lock:
            self._barrier = threading.Barrier(2)
            self._comm_dispatched = threading.Event()
            self._my_event = threading.local()
            self._enabled = True

    def disable(self) -> None:
        """Disable coordination and unblock any waiting threads.

        Safe to call multiple times and from either thread.
        """
        with self._state_lock:
            self._enabled = False
            barrier = self._barrier
        if barrier is not None:
            barrier.abort()
        # Release any compute-side waiter so it does not hang on shutdown.
        self._comm_dispatched.set()

    def is_enabled(self) -> bool:
        """Return whether coordination is currently active."""
        return self._enabled

    def rendezvous(self, role: HookRole) -> None:
        """Synchronize with the paired thread and enforce comm-first ordering.

        - ``COMM``: creates a **fresh** :class:`threading.Event`, stores it
          in the shared slot and in per-thread storage, then waits on the
          barrier and returns.  The caller must dispatch its comm op and
          then call :meth:`notify_dispatched`.
        - ``COMPUTE``: waits on the barrier, then blocks on the shared
          event until the paired ``COMM`` side has notified.

        Args:
            role: The role for the op to be dispatched after this call.
        """
        if not isinstance(role, HookRole):
            raise TypeError(
                f"Argument 'role' must be HookRole, but got type {type(role)}."
            )
        if not self._enabled:
            return

        # COMM side: create a per-rendezvous event *before* the barrier.
        # After the barrier both threads reference the same fresh event.
        if role is HookRole.COMM:
            evt = threading.Event()
            self._comm_dispatched = evt
            self._my_event.evt = evt

        try:
            self._barrier.wait()
        except threading.BrokenBarrierError:
            return

        # COMPUTE side: wait for the paired COMM to notify.
        if role is HookRole.COMPUTE:
            self._comm_dispatched.wait()

    def notify_dispatched(self, role: HookRole) -> None:
        """Signal that the caller has finished dispatching its comm op.

        Sets the per-rendezvous event that was saved during the most recent
        ``COMM`` rendezvous on this thread, then clears the saved reference
        so that subsequent calls are no-ops until the next ``COMM``
        rendezvous.

        If this thread has never done a ``COMM`` rendezvous (e.g. the
        backward thread's first hook, whose preceding op was a free-running
        ``combine.bwd`` not initiated by any rendezvous), the call is
        silently ignored — no stale event can be polluted.

        Args:
            role: The role of the caller that just finished dispatching.
        """
        if not isinstance(role, HookRole):
            raise TypeError(
                f"Argument 'role' must be HookRole, but got type {type(role)}."
            )
        if not self._enabled:
            return
        if role is HookRole.COMM:
            evt = getattr(self._my_event, "evt", None)
            if evt is not None:
                evt.set()
                self._my_event.evt = None
