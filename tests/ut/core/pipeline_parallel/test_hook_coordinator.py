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
"""Unit tests for HookCoordinator rendezvous, depart, and disabled bypass."""
import threading
import unittest

from hyper_parallel.core.pipeline_parallel.hook_coordinator import (
    HookCoordinator,
    HookRole,
)


class TestHookCoordinatorRecomputeBypass(unittest.TestCase):
    """Verify ``rendezvous`` short-circuits when the coordinator is disabled."""

    @staticmethod
    def _call_with_timeout(target, timeout=10.0):
        """Run ``target`` in a daemon thread; return whether it completed."""
        done = []

        def _wrapper():
            target()
            done.append(True)

        t = threading.Thread(target=_wrapper, daemon=True)
        t.start()
        t.join(timeout=timeout)
        return bool(done)

    def test_rendezvous_no_op_when_disabled(self):
        """Disabled coordinator: ``rendezvous`` returns immediately."""
        coord = HookCoordinator()
        # not enabled yet — short-circuit on ``not self._enabled``.
        completed = self._call_with_timeout(
            lambda: coord.rendezvous(HookRole.COMM),
        )
        assert completed, \
            ("rendezvous on a disabled coordinator should return "
             "immediately, but it blocked past the timeout")


class TestHookCoordinatorDepart(unittest.TestCase):
    """Verify ``depart`` lets a partner with unequal hook counts drain solo.

    The dual-pipe barrier is a hard 2-party rendezvous: the forward and
    backward threads must fire the *same* number of hooks or one blocks
    forever.  When the paired chunks have different layer counts the counts
    differ by ``4 * |layers_fwd - layers_bwd|``.  ``depart`` converts that
    deadlock into a graceful solo drain of the longer side.
    """

    @staticmethod
    def _call_with_timeout(target, timeout=10.0):
        """Run ``target`` in a daemon thread; return whether it completed."""
        done = []

        def _wrapper():
            target()
            done.append(True)

        t = threading.Thread(target=_wrapper, daemon=True)
        t.start()
        t.join(timeout=timeout)
        return bool(done)

    @staticmethod
    def _drive_two_threads(coord, spec_a, spec_b, timeout=30.0):
        """Run two threads firing ``(count, role)`` specs; ``depart`` when done.

        A ``COMM`` thread does ``rendezvous(COMM); notify_dispatched(COMM)``
        per step (mirroring a hook that dispatches a collective); a
        ``COMPUTE`` thread does ``rendezvous(COMPUTE)``.  Each thread calls
        :meth:`HookCoordinator.depart` in a ``finally`` so the partner with
        more steps drains its excess hooks solo.  Returns the set of thread
        names (``{"a", "b"}``) that completed within ``timeout`` — a missing
        name means that thread deadlocked.
        """
        done = set()
        done_lock = threading.Lock()

        def _make(name, count, role):
            def _run():
                try:
                    for _ in range(count):
                        coord.rendezvous(role)
                        if role is HookRole.COMM:
                            coord.notify_dispatched(role)
                finally:
                    coord.depart()
                with done_lock:
                    done.add(name)

            return _run

        thread_a = threading.Thread(target=_make("a", *spec_a), daemon=True)
        thread_b = threading.Thread(target=_make("b", *spec_b), daemon=True)
        coord.enable()
        try:
            thread_a.start()
            thread_b.start()
            thread_a.join(timeout=timeout)
            thread_b.join(timeout=timeout)
        finally:
            coord.disable()
        return done

    def test_depart_makes_lone_rendezvous_return(self):
        """After ``depart`` a lone ``rendezvous`` returns immediately (solo)."""
        coord = HookCoordinator()
        coord.enable()
        coord.depart()
        try:
            completed = self._call_with_timeout(
                lambda: coord.rendezvous(HookRole.COMM),
            )
        finally:
            coord.disable()
        assert completed, \
            ("rendezvous after depart() should return immediately so the "
             "longer chunk runs solo, but it blocked past the timeout")

    def test_depart_releases_parked_compute_waiter(self):
        """``depart`` unblocks a ``COMPUTE`` caller already parked on the event.

        A ``COMPUTE`` rendezvous that has passed the barrier waits on
        ``_comm_dispatched`` for the partner's notify.  If the partner
        departs instead of notifying, ``depart``'s ``_comm_dispatched.set()``
        must release the waiter.
        """
        coord = HookCoordinator()
        coord.enable()
        # Pre-park: a COMM rendezvous installs a fresh unset event into the
        # shared slot, then a lone COMPUTE waiter blocks on it.
        coord._my_event.evt = None  # pylint: disable=W0212
        parked_evt = threading.Event()
        coord._comm_dispatched = parked_evt  # pylint: disable=W0212

        def _wait_on_event():
            parked_evt.wait()

        waiter = threading.Thread(target=_wait_on_event, daemon=True)
        waiter.start()
        coord.depart()
        waiter.join(timeout=10.0)
        try:
            assert not waiter.is_alive(), \
                ("depart() must set _comm_dispatched to release a COMPUTE "
                 "waiter parked past the barrier, but the waiter is still alive")
        finally:
            coord.disable()

    def test_enable_resets_departed(self):
        """A new ``enable`` session clears the departed flag."""
        coord = HookCoordinator()
        coord.enable()
        coord.depart()
        assert coord._departed is True, \
            ("depart() should set _departed=True, "  # pylint: disable=W0212
             f"got {coord._departed}")  # pylint: disable=W0212
        coord.enable()
        try:
            assert coord._departed is False, \
                ("enable() must reset _departed for the new session, "  # pylint: disable=W0212
                 f"got {coord._departed}")  # pylint: disable=W0212
        finally:
            coord.disable()

    def test_comm_longer_no_deadlock(self):
        """FWD-style (COMM) thread longer than BWD-style (COMPUTE): no hang.

        Mirrors ``layers(fwd) > layers(bwd)`` — the original hard-deadlock
        direction, where the backward thread returns normally and the
        forward thread used to block forever on the barrier.
        """
        coord = HookCoordinator()
        done = self._drive_two_threads(
            coord, (9, HookRole.COMM), (5, HookRole.COMPUTE),
        )
        assert done == {"a", "b"}, \
            (f"unequal counts (COMM=9 > COMPUTE=5) must both drain via "
             f"depart(), but only {done} completed within the timeout")

    def test_compute_longer_no_deadlock(self):
        """BWD-style (COMPUTE) thread longer than FWD-style (COMM): no hang.

        Mirrors ``layers(bwd) > layers(fwd)`` — the forward thread departs
        first and the backward thread drains its excess hooks solo.
        """
        coord = HookCoordinator()
        done = self._drive_two_threads(
            coord, (5, HookRole.COMM), (9, HookRole.COMPUTE),
        )
        assert done == {"a", "b"}, \
            (f"unequal counts (COMPUTE=9 > COMM=5) must both drain via "
             f"depart(), but only {done} completed within the timeout")

    def test_balanced_counts_still_complete(self):
        """No regression: equal hook counts still pair and complete."""
        coord = HookCoordinator()
        done = self._drive_two_threads(
            coord, (7, HookRole.COMM), (7, HookRole.COMPUTE),
        )
        assert done == {"a", "b"}, \
            (f"balanced counts (7 == 7) must both complete via the normal "
             f"barrier pairing, but only {done} completed within the timeout")


if __name__ == "__main__":
    unittest.main()
