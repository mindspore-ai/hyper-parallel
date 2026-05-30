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
import threading
from contextlib import contextmanager, nullcontext
from typing import Callable, Optional, Tuple

from hyper_parallel.platform import get_platform
from hyper_parallel.core.activation_checkpoint.activation_checkpoint import checkpoint
from hyper_parallel.core.pipeline_parallel.hook_coordinator import HookCoordinator

platform = get_platform()


class CommComputeOverlap:
    """Orchestrator for two-thread comm/compute overlap.

    Manages a :class:`HookCoordinator` and provides helpers to insert the
    four synchronization hooks (``A``, ``B``, ``C``, ``D``) around MoE
    dispatch / combine phases and to run forward + backward concurrently
    with deterministic comm-first kernel launch ordering.

    Example:
        >>> overlap = CommComputeOverlap()
        >>> wrapped_dispatch = overlap.wrap_dispatch(ep_dispatch_fn)
        >>> wrapped_combine  = overlap.wrap_combine(ep_combine_fn, is_last_layer=is_last)
        >>> overlap.run(fwd_fn, bwd_fn)  # doctest: +SKIP
    """

    def __init__(self) -> None:
        self._coordinator = HookCoordinator()
        # Flipped on the first call to make_recompute_context_fn /
        # wrap_checkpoint.  Tells run() to gate the FWD thread on the BWD
        # thread's recompute completion, so the re-run executes serially
        # (no FWD-thread random / a2a / instance-state interleaving).
        self._has_recompute: bool = False
        # Fresh per-run() Event when _has_recompute is True; ``set()`` by
        # the first recompute_ctx exit per run, which opens the FWD gate.
        self._fwd_gate: Optional[threading.Event] = None

    @property
    def coordinator(self) -> HookCoordinator:
        """The underlying :class:`HookCoordinator` instance."""
        return self._coordinator

    # ------------------------------------------------------------------
    # Recompute integration
    # ------------------------------------------------------------------

    def make_recompute_context_fn(self) -> Callable[[], Tuple[object, object]]:
        """Build a ``context_fn`` for the activation-checkpoint recompute path.

        Returns a factory matching the
        ``ms.recompute(use_reentrant=False, context_fn=...)`` /
        ``torch.utils.checkpoint(use_reentrant=False, context_fn=...)``
        contract.  The recompute side of the returned pair brackets the
        backward-time forward re-run with two effects:

        1. ``HookCoordinator.set_recomputing(True/False)`` — the dual-pipe
           A/B/C/D/CHUNK_* sync hooks become no-ops on the re-run, so the
           barrier participant count stays balanced.  Without this the
           re-run double-fires every rendezvous and deadlocks the two-thread
           schedule.
        2. ``CommComputeOverlap._fwd_gate.set()`` on scope exit — a SAFETY NET
           for the FWD-thread gate installed by :meth:`run`.  The gate's
           primary trigger is the coordinator's one-shot opener, which fires
           on the BWD thread's first NON-suppressed rendezvous (the re-run's
           hooks are suppressed by effect 1, so the first real rendezvous is
           the grad phase that follows the *first* re-run); see
           :meth:`run` and :meth:`HookCoordinator.set_gate_opener`.  Until the
           gate opens the FWD thread is parked on ``gate.wait()``, so the
           initial re-run owns the RNG, the EP a2a group, and the
           ``ExpertParallel`` instance state exclusively.

        Marks the owning overlap instance as ``_has_recompute = True`` so
        :meth:`run` installs the gate around ``fwd_fn``.

        Note:
            Multiple checkpointed segments per ``run()`` (per-layer / mixed
            fine-grained checkpoint, not just one chunk-level wrap) are
            supported.  The gate opens at the first BWD grad-phase rendezvous,
            so segments ``2..N`` re-run while the FWD thread is parked on the
            2-party barrier (it cannot advance past a hook without its BWD
            partner, which is busy re-running), serializing each re-run record
            against the FWD forward record.  This is validated by the
            ``test_pp_overlap_moe_recompute_per_layer`` /
            ``test_pp_overlap_moe_recompute_mixed`` system tests.

        Returns:
            A no-arg callable returning ``(forward_ctx, recompute_ctx)`` on
            every invocation, suitable as the ``context_fn`` argument to
            :func:`hyper_parallel.core.activation_checkpoint.checkpoint`.
        """
        self._has_recompute = True
        coord = self._coordinator
        overlap = self

        def factory() -> Tuple[object, object]:

            @contextmanager
            def _recompute_scope():
                coord.set_recomputing(True)
                try:
                    yield
                finally:
                    coord.set_recomputing(False)
                    gate = overlap._fwd_gate    # pylint: disable=W0212
                    if gate is not None:
                        gate.set()

            return nullcontext(), _recompute_scope()

        return factory

    def wrap_checkpoint(self, fn: Callable, **ckpt_kwargs) -> Callable:
        """Return ``fn`` wrapped in :func:`checkpoint` with the overlap-aware ``context_fn``.

        Convenience for the common case of running a module / callable under
        activation checkpoint while a dual-thread overlap schedule is also
        active.  The returned callable forwards positional and keyword
        arguments to ``fn`` and routes the recompute through
        :meth:`make_recompute_context_fn`.

        Args:
            fn: The callable (typically an ``nn.Cell``/``nn.Module`` forward,
                or a chunk function) whose execution should be checkpointed.
            **ckpt_kwargs: Extra keyword arguments forwarded once to
                :func:`hyper_parallel.core.activation_checkpoint.checkpoint`
                — e.g. ``policy_fn`` for SAC, ``swap_inputs=True``.  Runtime
                arguments to ``fn`` are passed via the returned callable's
                own ``*args, **kwargs``.

        Returns:
            A callable with the same signature as ``fn`` that executes under
            activation checkpoint with overlap-aware recompute bracketing.
        """
        ctx_fn = self.make_recompute_context_fn()

        def _wrapped(*args, **kwargs):
            return checkpoint(fn, *args, context_fn=ctx_fn, **ckpt_kwargs, **kwargs)

        return _wrapped

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

    def run(
        self,
        fwd_fn: Callable[[], None],
        bwd_fn: Callable[[], None],
    ) -> None:
        """Run ``fwd_fn`` and ``bwd_fn`` with comm/compute overlap.

        Enables the coordinator, spawns the backward pass on a daemon
        thread, and waits for both to complete.  Layer-boundary handling
        is encoded statically by the ``is_last_layer`` flag passed to
        :meth:`wrap_combine` at wrap time, so no per-call layer count is
        needed here.

        When recompute integration is active (:meth:`make_recompute_context_fn`
        or :meth:`wrap_checkpoint` has been called on this instance), the
        FWD thread is gated on the BWD thread's first ``recompute_ctx``
        exit so the BWD re-run runs serially, then ``fwd_fn`` and the
        BWD grad phase overlap.  Pipeline cycle time is unchanged in the
        common chunk-level checkpoint case (see
        :meth:`make_recompute_context_fn` for the analysis and the
        single-segment limitation).

        Args:
            fwd_fn: Callable that executes the forward pass.
            bwd_fn: Callable that executes the backward pass.  If it needs
                    a specific device stream, wrap that logic inside
                    ``bwd_fn``.

        Raises:
            RuntimeError: If the backward thread raises an exception, it
                is re-raised on the main thread after ``join``.
        """
        self._coordinator.enable()

        # Reset the per-run FWD gate.  When _has_recompute is True the
        # gate opens on the BWD thread's first recompute_ctx exit (or as
        # a safety net at the end of bwd_fn).  Otherwise the gate stays
        # None and fwd_fn starts immediately.
        if self._has_recompute:
            self._fwd_gate = threading.Event()
            # Open the gate at the BWD thread's FIRST grad-phase rendezvous,
            # not at the first recompute_ctx exit.  For mixed per-layer
            # recompute — where a non-recomputed layer's backward fires
            # rendezvous BEFORE the first re-run — the recompute_ctx-exit
            # trigger opens the gate too late: the FWD thread stays parked
            # while the BWD thread blocks on the barrier waiting for it, and
            # the re-run that would open the gate never runs (circular
            # deadlock).  The first-rendezvous trigger releases FWD exactly
            # when BWD needs it as a barrier partner, while still holding FWD
            # through any initial re-run (whose suppressed hooks are not a
            # real rendezvous).  See HookCoordinator.set_gate_opener.
            self._coordinator.set_gate_opener(self._fwd_gate.set)
            original_fwd_fn = fwd_fn
            gate_for_fwd = self._fwd_gate

            def _gated_fwd_fn():
                gate_for_fwd.wait()
                original_fwd_fn()

            fwd_fn = _gated_fwd_fn
        else:
            self._fwd_gate = None

        exc_box: list = []
        coordinator = self._coordinator
        gate = self._fwd_gate

        def _bwd_target():
            try:
                bwd_fn()
            except Exception as exc:  # pylint: disable=W0718
                exc_box.append(exc)
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
                # solo.  Required for correctness, not just on error: BWD's
                # normal return previously left FWD hanging whenever
                # ``layers(fwd) > layers(bwd)``.
                coordinator.depart()
                # Safety net: release the FWD gate even if no recompute
                # fired (e.g. this bwd_fn did not traverse a checkpointed
                # segment, or it died before reaching one).  Idempotent
                # with the recompute_ctx-driven set().
                if gate is not None:
                    gate.set()

        thread = threading.Thread(target=_bwd_target, daemon=True)
        thread.start()

        fwd_exc: list = []
        try:
            fwd_fn()
        except Exception as exc:  # pylint: disable=W0718
            fwd_exc.append(exc)
            # Symmetric: if FWD dies, unblock BWD so it can exit.
            coordinator.disable()
        finally:
            # Graceful one-party-left, mirroring ``_bwd_target``: if the
            # paired BWD chunk has MORE hooks, ``depart`` lets it drain its
            # remaining rendezvous solo instead of hanging on the barrier.
            # Must precede ``join`` so a still-running BWD is released.
            coordinator.depart()
            thread.join()
            # Full reset after both threads are done (idempotent with any
            # earlier disable on the FWD error path).
            coordinator.disable()

        if exc_box:
            raise RuntimeError(
                "Exception in backward thread during dual-pipe overlap"
            ) from exc_box[0]
        if fwd_exc:
            raise fwd_exc[0]
