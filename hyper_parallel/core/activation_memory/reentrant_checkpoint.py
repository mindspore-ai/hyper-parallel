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
"""Eager reentrant checkpoint with explicit scheduled replay and local SAVE graphs."""
import contextlib
import contextvars
import threading
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Sequence, Union

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from .checkpoint import _ExecutionState, _noop_context_fn, _RECOMPUTE_COLLECTOR, _RECOMPUTE_SESSION
from .checkpoint_exclude import _ExcludeCache
from .reentrant_exclude import current_frame, execution_frame


_SCOPE = contextvars.ContextVar("hyper_reentrant_checkpoint_scope", default=None)
_SESSIONS = {}
_SESSIONS_LOCK = threading.RLock()


def _read_context(context_ref: Callable[[], Any]) -> Any:
    """Dereference the context without retaining its autograd graph."""
    return context_ref()


@dataclass(frozen=True)
class _InputSlot:
    """Index of a Tensor in the positional Function inputs."""

    index: int


class ReentrantCheckpointHandle:
    """One checkpoint invocation owned by an explicit stage-forward scope.

    In a split stage, DX targets activations and must run before DW. DW targets
    each handle's ``backward_trigger`` as well as the stage parameters. The
    trigger makes closure-only parameter regions reachable without a synthetic
    parameter edge that would invoke a leaf hook with ``None``. During DW,
    captured parameters accumulate naturally in the retained replay graph;
    explicit parameter inputs return their gradients through their original
    outer edges. Distinct contributions to a shared parameter may each invoke
    its hook, as they do in ordinary reentrant backward.
    """

    def __init__(self, function: Callable, template: list, spec: Any, params: Any,
                 preserve_rng_state: bool, context_fn: Callable) -> None:
        """Capture invocation inputs and state without retaining transient activations."""
        self.function = function
        self.template = template
        self.spec = spec
        self.params = params
        self.split = params is not None
        self.preserve_rng_state = preserve_rng_state
        self.context_fn = context_fn
        self.exclude_cache = _ExcludeCache()
        self.entries = []
        self.replaying = False
        self.phase = "ordinary"
        self.replay_outputs = None
        self.replay_inputs = ()
        self.state = None
        self.context_ref: Optional[Callable[[], Any]] = None
        self.input_count = 0
        self.param_input_mask = ()
        self.backward_trigger = None
        self.released = False
        self.retain_graph = False
        self.active_session = None
        self.output_spec = None
        self.output_template = ()
        self.excluded_output_mask = {}
        self.recording_exclude = False

    def _call(self, tensors):
        leaves = [tensors[item.index] if isinstance(item, _InputSlot) else item for item in self.template]
        args, kwargs = tree_unflatten(leaves, self.spec)
        with execution_frame(self):
            return self.function(*args, **kwargs)

    def _gradients(self, values: tuple, grads: tuple) -> tuple:
        """Compute replay-input gradients without returning captured parameter edges."""
        if self.split and self.phase == "dx":
            targets = [item for item, is_param in zip(self.replay_inputs, self.param_input_mask)
                       if item.requires_grad and not is_param]
            if not targets:
                return (None,) * self.input_count
            calculated = iter(torch.autograd.grad(values, targets, grads, retain_graph=True, allow_unused=True))
            return tuple(next(calculated) if item.requires_grad and not is_param else None
                         for item, is_param in zip(self.replay_inputs, self.param_input_mask))

        for item in self.replay_inputs:
            if item.requires_grad:
                item.grad = None
        torch.autograd.backward(values, grads, retain_graph=self.retain_graph)
        return tuple(item.grad if item.requires_grad else None for item in self.replay_inputs)

    def recompute(self) -> None:
        """Replay this frame once; retain its graph until DW or explicit cleanup."""
        if self.released:
            raise RuntimeError("Cannot replay a released reentrant checkpoint")
        if self.replay_outputs is not None:
            return
        context_ref = self.context_ref
        if context_ref is None:
            raise RuntimeError("Reentrant checkpoint output graph was released before replay")
        ctx = _read_context(context_ref)
        if ctx is None:
            raise RuntimeError("Reentrant checkpoint output graph was released before replay")
        inputs = ctx.saved_tensors
        self.replay_inputs = tuple(
            tensor.detach().requires_grad_(tensor.requires_grad) if tensor.is_floating_point() or tensor.is_complex()
            else tensor.detach() for tensor in inputs
        )
        self.replaying = True
        try:
            def replay(*unused: Any) -> None:
                """Store the differentiable replay output for the owning frame."""
                del unused
                self.replay_outputs = self._call(self.replay_inputs)

            with torch.enable_grad():
                self.state.recompute_fn(replay, {})
        finally:
            self.replaying = False

    def set_phase(self, phase: str) -> None:
        """Select the phase used by this frame's engine callbacks (dx or dw)."""
        if not self.split or phase not in {"dx", "dw"}:
            raise ValueError("Split reentrant handles require phase='dx' or phase='dw'")
        if self.released:
            raise RuntimeError("Cannot set phase on a released reentrant checkpoint")
        self.phase = phase

    def release(self) -> None:
        """Release all retained replay and excluded local graphs."""
        self.exclude_cache.clear()
        for entry in self.entries:
            entry.tape.release()
            entry.input_bindings.clear()
            entry.output = None
        self.entries.clear()
        self.excluded_output_mask.clear()
        self.replay_outputs = None
        self.replay_inputs = ()
        self.released = True

    def bind_session(self, session_id: Any) -> None:
        """Register retained state for cleanup through the unified session API."""
        activation = _RECOMPUTE_SESSION.get()
        if activation is not None and activation.session_id != session_id:
            raise ValueError("Active session does not match the reentrant handle session")
        with _SESSIONS_LOCK:
            if self.active_session is not None and self.active_session.session_id != session_id:
                raise RuntimeError("Concurrent reentrant recompute sessions on one frame are unsupported")
            _SESSIONS.setdefault(session_id, weakref.WeakSet()).add(self)
            if activation is not None:
                self.active_session = activation
                self.retain_graph = activation.retain_on_unpack


def clear_session(session_id: Any) -> None:
    """Release any reentrant handles registered with a recompute session."""
    with _SESSIONS_LOCK:
        handles = _SESSIONS.pop(session_id, ())
    for handle in handles:
        handle.release()


def activate_session(activation: Any) -> None:
    """Bind a session's retention policy before autograd enters worker threads."""
    with _SESSIONS_LOCK:
        for handle in _SESSIONS.get(activation.session_id, ()):
            if handle.active_session is not None and handle.active_session is not activation:
                raise RuntimeError("Concurrent reentrant recompute sessions on one frame are unsupported")
            handle.active_session = activation
            handle.retain_graph = activation.retain_on_unpack


def deactivate_session(activation: Any) -> None:
    """Unbind the policy without discarding registered frames or retained graph."""
    with _SESSIONS_LOCK:
        for handle in _SESSIONS.get(activation.session_id, ()):
            if handle.active_session is activation:
                handle.active_session = None
                handle.retain_graph = False


class _Checkpoint(torch.autograd.Function):
    """Expose input edges and a split-backward trigger to the outer engine."""

    @staticmethod
    def forward(ctx: Any, frame: ReentrantCheckpointHandle, dummy: torch.Tensor,
                *tensors: torch.Tensor) -> tuple:
        """Record one no-grad invocation and attach its replay frame to outputs."""
        del dummy
        ctx.frame = frame
        ctx.save_for_backward(*tensors[:frame.input_count])
        frame.context_ref = weakref.ref(ctx)
        frame.state = _ExecutionState.capture(frame.context_fn, frame.preserve_rng_state, tensors[:frame.input_count])
        with torch.no_grad(), frame.state.forward_context:
            output = frame._call(tensors[:frame.input_count])  # pylint: disable=protected-access
        frame.state.check_device_state_preserved(frame.preserve_rng_state)
        leaves, frame.output_spec = tree_flatten(output)
        frame.output_template = tuple(
            _InputSlot(index) if isinstance(value, torch.Tensor) else value
            for index, value in enumerate(leaves)
        )
        outputs = tuple(value.detach() for value in leaves if isinstance(value, torch.Tensor))
        if not outputs:
            raise ValueError("Reentrant checkpoint requires at least one Tensor output")
        originals = (value for value in leaves if isinstance(value, torch.Tensor))
        def differentiable(original: torch.Tensor) -> bool:
            """Match only the same live detached object, never a reused Python id."""
            matched = frame.excluded_output_mask.get(id(original))
            if matched is not None and matched[0]() is original:
                return matched[1]
            return original.is_floating_point() or original.is_complex()

        ctx.mark_non_differentiable(*(
            detached for detached, original in zip(outputs, originals) if not differentiable(original)
        ))
        frame.excluded_output_mask.clear()
        return outputs

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple:
        """Return target-limited gradients to the outer engine."""
        try:
            return _Checkpoint._run_backward(ctx, *grad_outputs)
        except BaseException:
            if not ctx.frame.retain_graph:
                ctx.frame.release()
            raise

    @staticmethod
    def _run_backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple:
        """Perform the replay and route gradients according to the explicit phase."""
        frame = ctx.frame
        if frame.split and frame.phase == "ordinary":
            raise RuntimeError("Set dx or dw phase on reentrant handles before split backward")
        frame.recompute()
        outputs = tuple(value for value in tree_flatten(frame.replay_outputs)[0] if isinstance(value, torch.Tensor))
        selected = [
            (value, grad) for value, grad in zip(outputs, grad_outputs)
            if value.requires_grad and grad is not None
        ]
        input_grads = (frame._gradients(*zip(*selected))  # pylint: disable=protected-access
                       if selected else (None,) * frame.input_count)
        if (not frame.split and not frame.retain_graph) or (frame.split and frame.phase == "dw"):
            frame.release()
        return (None, None, *input_grads, None)


@contextlib.contextmanager
def reentrant_checkpoint_scope(
    parameters: Union[Sequence[torch.Tensor], Callable[[], Sequence[torch.Tensor]]],
) -> Iterator[list]:
    """Collect split checkpoint frames during stage forward using explicit parameters.

    Args:
        parameters: Stable iterable or zero-argument provider of stage parameter Tensors.
            A provider is only evaluated if a reentrant checkpoint runs in the scope.

    Yields:
        List of invocation handles. Set each handle's phase before DX and DW;
        include every ``backward_trigger`` in DW's autograd ``inputs`` tuple.
        Release handles on cancellation; DW releases them after consumption.
    """
    if _SCOPE.get() is not None:
        raise ValueError("Nested reentrant checkpoint scopes are not supported")
    scope = [parameters, [], None]
    token = _SCOPE.set(scope)
    try:
        yield scope[1]
    finally:
        _SCOPE.reset(token)


def _checkpoint_inputs(args: tuple, kwargs: dict) -> tuple:
    """Index Tensor leaves once while preserving the original pytree structure."""
    leaves, spec = tree_flatten((args, kwargs))
    inputs = []
    template = []
    for item in leaves:
        if isinstance(item, torch.Tensor):
            template.append(_InputSlot(len(inputs)))
            inputs.append(item)
        else:
            template.append(item)
    return tuple(inputs), template, spec


def _scope_parameters(scope: Optional[list]) -> Optional[tuple]:
    """Resolve and deduplicate scoped parameters only on the first checkpoint."""
    if scope is None:
        return None
    if scope[2] is None:
        supplied = tuple(scope[0]() if callable(scope[0]) else scope[0])
        if any(not isinstance(param, torch.Tensor) for param in supplied):
            raise ValueError("Scope parameters must be Tensors")
        scope[2] = tuple(dict((id(param), param) for param in supplied if param.requires_grad).values())
    return scope[2]


def _collect_frame(frame: ReentrantCheckpointHandle, scope: Optional[list]) -> Optional[list]:
    """Register a frame for the current scope or unified collector."""
    if scope is not None:
        scope[1].append(frame)
        return scope[1]
    collector = _RECOMPUTE_COLLECTOR.get()
    if collector is not None:
        collector.append(frame)
    return collector


def checkpoint(function: Callable, *args: Any, preserve_rng_state: bool = True,
               context_fn: Callable = _noop_context_fn, **kwargs: Any) -> Any:
    """Run an eager reentrant checkpoint with localized excluded-region graphs.

    Ordinary backward accumulates captured parameter gradients naturally.
    Unscoped ``autograd.grad`` does not filter these nested accumulations by
    its target list; use ``backward`` for ordinary execution and an explicit
    scope for staged DX/DW. Nested reentrant checkpoints are unsupported.
    """
    if current_frame() is not None:
        raise ValueError("Nested reentrant checkpoints are not supported")
    if not torch.is_grad_enabled():
        return function(*args, **kwargs)
    inputs, template, spec = _checkpoint_inputs(args, kwargs)
    scope = _SCOPE.get()
    params = _scope_parameters(scope)
    frame = ReentrantCheckpointHandle(function, template, spec, params, preserve_rng_state, context_fn)
    frame.input_count = len(inputs)
    frame.param_input_mask = tuple(id(item) in {id(param) for param in params or ()} for item in inputs)
    frame.backward_trigger = torch.empty(0, requires_grad=True) if frame.split else None
    collector = _collect_frame(frame, scope)
    try:
        result = _Checkpoint.apply(frame, torch.empty(0, requires_grad=True), *inputs, frame.backward_trigger)
    except BaseException:
        frame.release()
        if collector is not None:
            collector.remove(frame)
        raise
    tensors = (result,) if isinstance(result, torch.Tensor) else result
    iterator = iter(tensors)
    return tree_unflatten(
        [next(iterator) if isinstance(item, _InputSlot) else item for item in frame.output_template],
        frame.output_spec,
    )
