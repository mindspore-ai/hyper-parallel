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
        t.join(timeout=1.0)
        assert other_thread_state == [False], \
            (f"set_recomputing on main thread should not leak to a "
             f"second thread; other thread saw is_recomputing="
             f"{other_thread_state}, expected [False]")


class TestHookCoordinatorRecomputeBypass(unittest.TestCase):
    """Verify ``rendezvous`` / ``notify_dispatched`` short-circuit under recompute."""

    @staticmethod
    def _call_with_timeout(target, timeout=0.5):
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


if __name__ == "__main__":
    unittest.main()
