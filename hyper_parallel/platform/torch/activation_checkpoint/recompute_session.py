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
"""Recompute session mechanism for PyTorch activation checkpointing.

This module implements a session-based recomputation layer on top of PyTorch's
``saved_tensors_hooks`` mechanism.  It enables dx/dw split backward passes
(and pipeline-parallel micro-batch schedules) to share a single recompute run,
with explicit retain / clear lifecycle management.

The design mirrors MindSpore PR mindspore/mindspore#92629 but is built using
PyTorch's native ``torch.autograd.graph.saved_tensors_hooks``.
"""
from __future__ import annotations

import contextlib
import uuid
import weakref
from collections import defaultdict
from contextvars import ContextVar
from typing import Any, Callable, List, Optional, Tuple
from weakref import ReferenceType, WeakSet

import torch


# ---------------------------------------------------------------------------
# Exception for early-stop recomputation
# ---------------------------------------------------------------------------

class _StopRecomputationError(Exception):
    """Raised to abort recomputation early once all needed tensors are ready.

    This is a control-flow mechanism, not an error condition.
    """


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

class _RecomputeSession:
    """Lightweight data carrier identifying a recompute session.

    Args:
        session_id: Stable key shared by the producing re-run and the
            consuming backward.
        retain_on_unpack: When ``True``, unpack returns recomputed tensors
            without popping them so that a later backward can consume them.
    """

    __slots__ = ("session_id", "retain_on_unpack")

    def __init__(self, session_id: str, retain_on_unpack: bool = False) -> None:
        self.session_id = session_id
        self.retain_on_unpack = retain_on_unpack


class _Handle:
    """Opaque pointer linking a ``_Holder`` to its recomputed tensor."""

    __slots__ = ()


class _Holder:
    """Per-saved-tensor container carried through PyTorch's pack/unpack hooks.

    One ``_Holder`` is created per ``pack_hook`` call during the forward pass.
    On recomputation, ``_Handle`` objects are stored in ``handles`` keyed by
    session_id, and the actual tensor is stored in
    ``_CheckpointFrame.recomputed[key]``.
    """

    __slots__ = ("handles", "__weakref__")

    def __init__(self) -> None:
        self.handles: dict[str, Optional[_Handle]] = {}


class _CheckpointFrame:
    """Internal ledger for a single checkpointed region.

    Similar to PyTorch's ``_CheckpointFrame`` but keyed by *session_id*
    instead of autograd graph-task id.  This allows recomputation data to be
    shared across multiple backward passes within the same session.

    Args:
        recompute_fn: Callable that re-runs the forward function.
        early_stop: If ``True``, raise ``_StopRecomputationError`` once all
            tensors needed by the current session have been recomputed.
    """

    def __init__(
        self,
        recompute_fn: Callable[..., None],
        early_stop: bool = True,
    ) -> None:
        self.recompute_fn: Callable[..., None] = recompute_fn
        self.saved_args: List[Any] = []
        self.saved_kwargs: dict[str, Any] = {}

        # Weak refs to _Holder objects created during forward.
        self.weak_holders: List[ReferenceType] = []

        # session_id -> { _Handle -> recomputed tensor }
        self.recomputed: dict[str, dict[_Handle, torch.Tensor]] = defaultdict(dict)

        # session_id -> whether recomputation has already run
        self.is_recomputed: dict[str, bool] = defaultdict(bool)

        # session_id -> number of tensors saved during recomputation so far
        self.recomp_counter: dict[str, int] = defaultdict(int)

        self.early_stop: bool = early_stop
        self.forward_completed: bool = False
        self.ignore_saved_mismatch: bool = False

    # -- Input management ----------------------------------------------------

    def save_inputs(self, *args: Any, **kwargs: Any) -> None:
        """Detach and store positional and keyword arguments for recomputation."""
        self.saved_args = [_detach_tensor(a) for a in args]
        self.saved_kwargs = {
            k: _detach_tensor(v) for k, v in kwargs.items()
        }

    def get_inputs(self) -> Tuple[tuple, dict]:
        """Return the ``(args, kwargs)`` tuple saved by :meth:`save_inputs`."""
        return tuple(self.saved_args), self.saved_kwargs


# ---------------------------------------------------------------------------
# Per-checkpoint-block handle (returned to pipeline code)
# ---------------------------------------------------------------------------

class _RecomputeHandle:
    """Per-checkpoint-block handle that can be eagerly fired by pipeline code.

    Args:
        frame: The ``_CheckpointFrame`` this handle is associated with.
        recompute_fn: The forward re-run callable (captured separately so
            that the handle can be used even after the frame's own
            ``recompute_fn`` reference is cleared).
        early_stop: Whether to raise ``_StopRecomputationError`` once all
            tensors needed by the session are ready.
    """

    def __init__(
        self,
        frame: _CheckpointFrame,
        recompute_fn: Callable[..., None],
        early_stop: bool = True,
    ) -> None:
        self._frame = frame
        self._recompute_fn = recompute_fn
        self._early_stop = early_stop

    def recompute(self, session_id: str) -> None:
        """Run recomputation for *session_id* if not already done.

        Args:
            session_id: The session key under which to cache the recomputed
                activations.
        """
        frame = self._frame
        if frame.is_recomputed[session_id]:
            return

        args, kwargs = frame.get_inputs()

        try:
            with _RecomputationSessionHook(
                weakref.ref(frame), session_id
            ), torch.autograd.enable_grad():
                self._recompute_fn(*args, **kwargs)
        except _StopRecomputationError:
            pass

        frame.is_recomputed[session_id] = True

    def clear_session(self, session_id: str) -> None:
        """Release all cached recomputation data for *session_id*.

        Args:
            session_id: The session key whose cached data should be cleared.
        """
        frame = self._frame
        frame.recomputed.pop(session_id, None)
        frame.is_recomputed.pop(session_id, None)
        frame.recomp_counter.pop(session_id, None)
        # Null out handles in all live holders for this session.
        for weak_holder in frame.weak_holders:
            holder = weak_holder()
            if holder is not None and session_id in holder.handles:
                holder.handles[session_id] = None


# ---------------------------------------------------------------------------
# Context variables
# ---------------------------------------------------------------------------

_recompute_session: ContextVar[Optional[_RecomputeSession]] = ContextVar(
    "_recompute_session", default=None
)

# session_id -> WeakSet of _RecomputeHandle objects registered for that session
_recompute_session_handles: defaultdict[str, WeakSet] = defaultdict(WeakSet)

# When set, new _RecomputeHandle objects are appended to this list.
_recompute_handle_collector: ContextVar[Optional[list]] = ContextVar(
    "_recompute_handle_collector", default=None
)


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _recompute_session_ctx(
    session_id: Optional[str] = None,
    retain_on_unpack: bool = False,
):
    """Activate a recompute session for the scope of this context manager.

    Inside the scope, ``_recompute_session`` is set so that
    ``_CheckpointSessionHook.unpack_hook`` uses *session_id* as the key
    for looking up (and optionally retaining) recomputed activations.

    Args:
        session_id: Stable session key.  If ``None``, a UUID is generated.
        retain_on_unpack: When ``True``, unpack does not pop the recomputed
            tensor from the cache, allowing a subsequent backward to reuse it.

    Yields:
        The ``session_id`` string that is active for the scope.
    """
    if session_id is None:
        session_id = str(uuid.uuid4())
    session = _RecomputeSession(session_id=session_id, retain_on_unpack=retain_on_unpack)
    token = _recompute_session.set(session)
    try:
        yield session_id
    finally:
        _recompute_session.reset(token)


@contextlib.contextmanager
def _recompute_handle_collector_ctx():
    """Context manager that collects ``_RecomputeHandle`` objects created inside.

    Yields:
        A list that will be populated with one ``_RecomputeHandle`` per
        checkpointed block executed during the forward pass within this scope.
    """
    collector: list = []
    token = _recompute_handle_collector.set(collector)
    try:
        yield collector
    finally:
        _recompute_handle_collector.reset(token)


def _clear_recompute_session(session_id: str) -> None:
    """Release retained recompute data for *session_id*.

    Pops all handles from the session registry and calls
    ``clear_session(session_id)`` on each, freeing cached tensors.

    Args:
        session_id: The session key whose cached recompute data is cleared.
    """
    handles = _recompute_session_handles.pop(session_id, None)
    if handles is not None:
        for handle in list(handles):
            handle.clear_session(session_id)


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

class _RecomputationSessionHook(torch.autograd.graph.saved_tensors_hooks):
    """Mounted during recomputation: stores tensors in the frame's cache.

    During recomputation the pack hook detaches the tensor, creates a
    ``_Handle``, stores the tensor in ``frame.recomputed[key]``, and
    associates the handle with the corresponding holder.  The unpack hook
    is a pass-through.

    Args:
        target_frame_ref: Weak reference to the ``_CheckpointFrame``.
        session_id: The session key under which to cache recomputed tensors.
    """

    def __init__(
        self,
        target_frame_ref: ReferenceType,
        session_id: str,
    ) -> None:
        def pack_hook(x: torch.Tensor) -> torch.Tensor:
            frame = target_frame_ref()
            if frame is None:
                raise RuntimeError(
                    "CheckpointFrame has been garbage collected during recomputation."
                )

            frame.recomp_counter[session_id] += 1
            recomp_idx = frame.recomp_counter[session_id] - 1

            # If recomputation produces more tensors than the original forward
            # saved, either silently ignore or error.
            if recomp_idx >= len(frame.weak_holders):
                if not frame.early_stop and not frame.forward_completed:
                    # Allow the extra tensor through without caching.
                    frame.ignore_saved_mismatch = True
                    return x
                raise RuntimeError(
                    "Recompute session: more tensors were saved during "
                    "recomputation than during the original forward pass."
                )

            holder = frame.weak_holders[recomp_idx]()

            # The holder may have been cleared already (e.g. backward within
            # forward).  In that case we don't need to save.
            if holder is not None:
                handle = _Handle()
                holder.handles[session_id] = handle
                frame.recomputed[session_id][handle] = x.detach()

            # Early stop: once we've recomputed as many tensors as were saved
            # during forward, abort the rest of the recomputation.
            if frame.early_stop and frame.recomp_counter[session_id] == len(
                frame.weak_holders
            ):
                raise _StopRecomputationError

            return x

        def unpack_hook(x: torch.Tensor) -> torch.Tensor:
            return x

        super().__init__(pack_hook, unpack_hook)


class _CheckpointSessionHook(torch.autograd.graph.saved_tensors_hooks):
    """Mounted during the forward pass: creates placeholders and triggers
    recomputation on unpack.

    Pack creates a ``_Holder`` placeholder.  Unpack reads the active
    session from the ``_recompute_session`` ContextVar, triggers
    recomputation if needed, retrieves the tensor, and either retains or
    discards it based on ``retain_on_unpack``.

    Args:
        frame: The ``_CheckpointFrame`` for this checkpointed region.
    """

    def __init__(self, frame: _CheckpointFrame) -> None:
        def pack_hook(x: torch.Tensor) -> _Holder:
            holder = _Holder()
            frame.weak_holders.append(weakref.ref(holder))
            return holder

        def unpack_hook(holder: _Holder) -> torch.Tensor:
            session = _recompute_session.get()
            if session is None:
                raise RuntimeError(
                    "checkpoint_with_session: unpack triggered outside a "
                    "recompute session context.  Wrap backward in "
                    "_recompute_session_ctx()."
                )
            key = session.session_id

            # Trigger recomputation if not done yet.
            if not frame.is_recomputed[key]:
                args, kwargs = frame.get_inputs()
                try:
                    with _RecomputationSessionHook(
                        weakref.ref(frame), key
                    ), torch.autograd.enable_grad():
                        frame.recompute_fn(*args, **kwargs)
                except _StopRecomputationError:
                    pass
                frame.is_recomputed[key] = True

            if key not in holder.handles:
                raise RuntimeError(
                    f"checkpoint_with_session: session '{key}' has no handle "
                    "for this holder.  The recomputation may have saved a "
                    "different number of tensors than the original forward."
                )

            handle = holder.handles[key]
            if handle is None:
                raise RuntimeError(
                    "checkpoint_with_session: unpack triggered for a tensor "
                    "that has already been unpacked once in this session.  "
                    "If you need to access the tensor multiple times, use "
                    "retain_on_unpack=True."
                )

            if handle not in frame.recomputed[key]:
                raise RuntimeError(
                    "checkpoint_with_session: handle not found in recomputed "
                    f"cache for session '{key}'."
                )

            ret = frame.recomputed[key][handle]

            if session.retain_on_unpack:
                # Keep the tensor in the cache so a later backward can reuse it.
                pass
            else:
                # Consume: pop from cache and null the handle.
                frame.recomputed[key].pop(handle, None)
                holder.handles[key] = None

            return ret

        super().__init__(pack_hook, unpack_hook)


# ---------------------------------------------------------------------------
# Entry function
# ---------------------------------------------------------------------------

def _noop_context_fn() -> Tuple[contextlib.nullcontext, contextlib.nullcontext]:
    """Return a pair of null contexts (forward, recompute)."""
    return contextlib.nullcontext(), contextlib.nullcontext()


def checkpoint_with_session(
    function: Callable[..., Any],
    *args: Any,
    context_fn: Optional[Callable[[], Tuple[Any, Any]]] = None,
    use_reentrant: bool = False,
    **kwargs: Any,
) -> Any:
    """Run *function* inside a checkpointed region that supports recompute sessions.

    When a recompute session is active (i.e. ``_recompute_session`` is set),
    this function creates a ``_CheckpointFrame``, saves the inputs, mounts a
    ``_CheckpointSessionHook``, and runs the forward.  The resulting
    ``_RecomputeHandle`` is registered in the session handle registry and
    optionally collected by ``_recompute_handle_collector_ctx``.

    When no session is active, falls back to
    ``torch.utils.checkpoint.checkpoint(use_reentrant=False)``.

    Args:
        function: The function to checkpoint.
        *args: Positional arguments forwarded to *function*.
        context_fn: Optional callable returning a pair of context managers
            (forward_ctx, recompute_ctx).  Defaults to ``_noop_context_fn``.
        use_reentrant: Must be ``False``.  ``True`` raises ``ValueError``
            because session-based checkpointing requires the non-reentrant
            path.
        **kwargs: Keyword arguments forwarded to *function*.

    Returns:
        The output of *function(\*args, \*\*kwargs)*.

    Raises:
        ValueError: If ``use_reentrant=True``.
    """
    if use_reentrant:
        raise ValueError(
            "checkpoint_with_session does not support use_reentrant=True.  "
            "Session-based checkpointing requires the non-reentrant path."
        )

    session = _recompute_session.get()

    # If no session is active, fall back to vanilla torch checkpoint.
    if session is None:
        from torch.utils.checkpoint import checkpoint as _torch_checkpoint  # pylint: disable=C0415
        return _torch_checkpoint(
            function, *args, use_reentrant=False, **kwargs
        )

    # -- Session is active: use our custom machinery. ------------------------
    if context_fn is None:
        context_fn = _noop_context_fn

    forward_context, recompute_context = context_fn()

    def recompute_fn(*recompute_args: Any, **recompute_kwargs: Any) -> None:
        with recompute_context:
            function(*recompute_args, **recompute_kwargs)

    frame = _CheckpointFrame(recompute_fn, early_stop=True)
    frame.save_inputs(*args, **kwargs)

    handle = _RecomputeHandle(frame, recompute_fn, early_stop=True)

    # Register handle in the session-wide registry.
    _recompute_session_handles[session.session_id].add(handle)

    # Also register in the collector if one is active.
    collector = _recompute_handle_collector.get()
    if collector is not None:
        collector.append(handle)

    with _CheckpointSessionHook(frame), forward_context:
        ret = function(*args, **kwargs)

    frame.forward_completed = True
    return ret


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detach_tensor(value: Any) -> Any:
    """Detach a tensor (preserving ``requires_grad``); return non-tensors as-is."""
    if isinstance(value, torch.Tensor):
        detached = value.detach()
        detached.requires_grad = value.requires_grad
        return detached
    return value
