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
"""Unit tests for HookCoordinator recompute bypass + context_fn helper."""
import threading
import unittest

from hyper_parallel.core.pipeline_parallel.hook_coordinator import (
    HookCoordinator,
    HookRole,
)


class TestHookCoordinatorRecomputeFlag(unittest.TestCase):
    """Verify the per-thread ``set_recomputing`` / ``is_recomputing`` API."""

    def test_default_is_false(self):
        """A fresh coordinator reports not-recomputing on any thread."""
        coord = HookCoordinator()
        assert coord.is_recomputing() is False, \
            (f"freshly constructed coordinator should report "
             f"is_recomputing=False, got {coord.is_recomputing()}")

    def test_set_true_then_false_round_trip(self):
        """``set_recomputing(True)`` followed by ``False`` returns to default."""
        coord = HookCoordinator()
        coord.set_recomputing(True)
        assert coord.is_recomputing() is True, \
            (f"after set_recomputing(True), is_recomputing should be True, "
             f"got {coord.is_recomputing()}")
        coord.set_recomputing(False)
        assert coord.is_recomputing() is False, \
            (f"after set_recomputing(False), is_recomputing should be False, "
             f"got {coord.is_recomputing()}")

    def test_per_thread_isolation(self):
        """Flag on thread A does not leak to thread B (``threading.local``)."""
        coord = HookCoordinator()
        coord.set_recomputing(True)
        # is_recomputing on this thread is True.
        assert coord.is_recomputing() is True

        other_thread_state = []

        def _probe_from_other_thread():
            other_thread_state.append(coord.is_recomputing())

        t = threading.Thread(target=_probe_from_other_thread)
        t.start()
        t.join(timeout=10.0)
        assert other_thread_state == [False], \
            (f"set_recomputing on main thread should not leak to a "
             f"second thread; other thread saw is_recomputing="
             f"{other_thread_state}, expected [False]")


class TestHookCoordinatorRecomputeBypass(unittest.TestCase):
    """Verify ``rendezvous`` / ``notify_dispatched`` short-circuit under recompute."""

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

    def test_rendezvous_bypass_when_recomputing(self):
        """``set_recomputing(True)`` makes lone ``rendezvous`` return immediately.

        Without the bypass, ``rendezvous(COMM)`` would create a fresh
        Event and call ``self._barrier.wait()`` on a ``Barrier(2)`` with
        no second participant — blocking forever.  With the bypass the
        call must early-return on ``is_recomputing()``.

        ``set_recomputing`` is per-thread (``threading.local``), so the
        flag must be set **on the same thread** that calls
        ``rendezvous`` — mirrors the real usage where the BWD thread's
        recompute re-run brackets its own hooks.
        """
        coord = HookCoordinator()
        coord.enable()

        def _bracketed_rendezvous():
            coord.set_recomputing(True)
            try:
                coord.rendezvous(HookRole.COMM)
            finally:
                coord.set_recomputing(False)

        try:
            completed = self._call_with_timeout(_bracketed_rendezvous)
        finally:
            coord.disable()
        assert completed, \
            ("rendezvous(COMM) on a recomputing thread should bypass "
             "the Barrier(2) and return immediately, but it blocked "
             "past the timeout — recompute bypass is broken")

    def test_notify_dispatched_bypass_when_recomputing(self):
        """``notify_dispatched`` is a quick no-op when recomputing.

        ``notify_dispatched`` does not normally block, but the bypass
        must keep it from poking the per-thread event slot during the
        re-run (which would race with the original forward's saved
        event reference).
        """
        coord = HookCoordinator()
        coord.enable()
        coord.set_recomputing(True)
        try:
            # No exception and no per-thread event mutation — observable
            # by checking that ``_my_event.evt`` remains unset.
            coord.notify_dispatched(HookRole.COMM)
        finally:
            coord.disable()
        evt_attr = getattr(coord._my_event, "evt", None)  # pylint: disable=W0212
        assert evt_attr is None, \
            (f"notify_dispatched during recompute must not touch the "
             f"per-thread event slot, but _my_event.evt={evt_attr}")


class TestRecomputeContextFn(unittest.TestCase):
    """Verify the ``recompute_context_fn`` helper for ``ms.recompute``."""

    def test_returns_two_context_managers(self):
        """``ms.recompute`` unpacks the result as ``(forward_ctx, recompute_ctx)``."""
        coord = HookCoordinator()
        result = coord.recompute_context_fn()
        assert isinstance(result, tuple) and len(result) == 2, \
            (f"recompute_context_fn must return a 2-tuple "
             f"(forward_ctx, recompute_ctx); got type={type(result)}, "
             f"len={len(result) if isinstance(result, tuple) else 'N/A'}")
        fwd_ctx, rec_ctx = result
        for name, ctx in (("forward_ctx", fwd_ctx), ("recompute_ctx", rec_ctx)):
            assert hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__"), \
                (f"{name} must be a context manager; got type={type(ctx)} "
                 f"(missing __enter__/__exit__)")

    def test_forward_ctx_is_a_noop(self):
        """``forward_ctx`` must NOT mutate the recomputing flag.

        The original forward pass needs its sync hooks to fire
        normally, which means ``is_recomputing()`` must stay ``False``
        inside ``with forward_ctx:``.
        """
        coord = HookCoordinator()
        fwd_ctx, _ = coord.recompute_context_fn()
        assert coord.is_recomputing() is False
        with fwd_ctx:
            assert coord.is_recomputing() is False, \
                (f"forward_ctx must NOT set is_recomputing — original "
                 f"forward needs hooks to fire; got is_recomputing="
                 f"{coord.is_recomputing()} inside the CM")
        assert coord.is_recomputing() is False

    def test_recompute_ctx_brackets_flag(self):
        """``recompute_ctx`` sets the flag on enter and clears on exit."""
        coord = HookCoordinator()
        _, rec_ctx = coord.recompute_context_fn()
        assert coord.is_recomputing() is False
        with rec_ctx:
            assert coord.is_recomputing() is True, \
                (f"recompute_ctx must set is_recomputing=True on enter; "
                 f"got {coord.is_recomputing()}")
        assert coord.is_recomputing() is False, \
            (f"recompute_ctx must clear is_recomputing on exit; "
             f"got {coord.is_recomputing()}")

    def test_recompute_ctx_clears_flag_on_exception(self):
        """``set_recomputing(False)`` must fire even if the re-run raises.

        ``ms.recompute``'s ``with recompute_ctx:`` wraps the re-run.
        If the re-run raises, the CM's ``__exit__`` still runs — and
        our implementation uses ``try/finally`` so the flag clears.
        """
        coord = HookCoordinator()
        _, rec_ctx = coord.recompute_context_fn()
        with self.assertRaises(RuntimeError):
            with rec_ctx:
                assert coord.is_recomputing() is True
                raise RuntimeError("simulated recompute failure")
        assert coord.is_recomputing() is False, \
            (f"recompute_ctx must clear is_recomputing even when the "
             f"wrapped block raises; got {coord.is_recomputing()}")

    def test_each_call_returns_fresh_context_managers(self):
        """``ms.recompute`` calls ``context_fn()`` once per invocation.

        Reusing a single CM instance across calls would break
        ``contextlib.contextmanager``'s single-use contract.  Each
        call must yield brand-new CM instances.
        """
        coord = HookCoordinator()
        a_fwd, a_rec = coord.recompute_context_fn()
        b_fwd, b_rec = coord.recompute_context_fn()
        assert a_fwd is not b_fwd, \
            (f"forward_ctx must be a fresh instance each call; "
             f"got the same object: id(a)={id(a_fwd)}, id(b)={id(b_fwd)}")
        assert a_rec is not b_rec, \
            (f"recompute_ctx must be a fresh instance each call; "
             f"got the same object: id(a)={id(a_rec)}, id(b)={id(b_rec)}")


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


class TestHookCoordinatorGateOpener(unittest.TestCase):
    """Verify the one-shot gate opener that fixes the mixed-recompute deadlock.

    Under per-layer recompute the FWD thread is parked on the recompute gate.
    If the gate only opened at the first ``recompute_ctx`` exit, a mixed chunk
    whose LAST layer is NOT recomputed would deadlock: that layer's backward
    fires a rendezvous before any re-run, while the FWD thread it needs as a
    barrier partner is still gated, and the re-run that would open the gate
    never runs.  Opening the gate on the BWD thread's first NON-suppressed
    rendezvous releases the FWD thread exactly when it is needed.
    """

    def test_first_rendezvous_opens_parked_gate(self):
        """A FWD thread parked on the gate is released by BWD's first rendezvous."""
        coord = HookCoordinator()
        coord.enable()
        gate = threading.Event()
        coord.set_gate_opener(gate.set)

        fwd_released = []

        def _parked_fwd():
            gate.wait()
            fwd_released.append(True)

        fwd_thread = threading.Thread(target=_parked_fwd, daemon=True)
        fwd_thread.start()
        # BWD's first rendezvous fires the opener (before it blocks on the
        # lone barrier), opening the gate and releasing the FWD thread.
        bwd_thread = threading.Thread(
            target=lambda: coord.rendezvous(HookRole.COMPUTE), daemon=True,
        )
        bwd_thread.start()
        fwd_thread.join(timeout=0.5)
        coord.disable()  # release the lone BWD rendezvous parked on the barrier
        assert fwd_released == [True], \
            ("BWD's first rendezvous must open the FWD gate (mixed-recompute "
             f"deadlock fix), but the FWD thread stayed parked: {fwd_released}")

    def test_opener_suppressed_during_recompute(self):
        """A suppressed (recompute re-run) rendezvous must NOT open the gate.

        The opener must fire on the first *real* rendezvous — i.e. after any
        initial re-run — otherwise the gate would open mid re-run and lose the
        serialization that protects the re-run from the FWD forward.
        """
        coord = HookCoordinator()
        coord.enable()
        fired = []
        coord.set_gate_opener(lambda: fired.append(True))

        def _suppressed_rendezvous():
            coord.set_recomputing(True)
            try:
                coord.rendezvous(HookRole.COMM)  # early-returns, opener skipped
            finally:
                coord.set_recomputing(False)

        thread = threading.Thread(target=_suppressed_rendezvous, daemon=True)
        thread.start()
        thread.join(timeout=0.5)
        still_armed = coord._gate_opener  # pylint: disable=W0212
        coord.disable()
        assert not fired, \
            (f"a recompute-suppressed rendezvous must not open the gate, "
             f"but the opener fired: {fired}")
        assert still_armed is not None, \
            ("the opener must stay armed after a suppressed rendezvous (it "
             "fires on the first REAL rendezvous), but it was cleared")

    def test_opener_is_one_shot_and_enable_resets(self):
        """The opener fires once, and a new ``enable`` session re-arms to None."""
        coord = HookCoordinator()
        coord.enable()
        coord.set_gate_opener(lambda: None)
        thread = threading.Thread(
            target=lambda: coord.rendezvous(HookRole.COMM), daemon=True,
        )
        thread.start()
        thread.join(timeout=0.3)
        cleared_after_fire = coord._gate_opener is None  # pylint: disable=W0212
        coord.disable()
        assert cleared_after_fire, \
            "the gate opener must be one-shot (cleared after it fires)"
        # A fresh session must start with no opener armed.
        coord.set_gate_opener(lambda: None)
        coord.enable()
        reset_by_enable = coord._gate_opener is None  # pylint: disable=W0212
        coord.disable()
        assert reset_by_enable, \
            "enable() must reset the gate opener for the new session"


if __name__ == "__main__":
    unittest.main()
