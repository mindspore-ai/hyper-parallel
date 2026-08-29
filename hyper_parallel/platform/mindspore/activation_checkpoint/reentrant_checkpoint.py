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
"""Experimental reentrant activation checkpoint with excluded regions.

This module is intentionally independent from the public activation checkpoint
entry points.  It provides a hook-free prototype in which the outer checkpoint
uses a custom backward and every excluded region retains a small, ordinary
PyNative autograd graph.  During checkpoint replay, a custom autograd bridge
returns the cached excluded output and routes its backward through the retained
local graph.

The implementation currently supports PyNative execution with Tensor leaves in
nested input and output containers. Selective checkpoint policies and activation
swapping are intentionally not supported yet.
"""
import contextlib
import contextvars
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import threading
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple
import weakref

from mindspore import Tensor
from mindspore._c_expression import run_backward
from mindspore._c_expression.amp import get_curr_amp_strategy
from mindspore.common._grad_function import _Function
from mindspore.common.api import _no_grad, _pynative_executor
from mindspore.common.generator import get_rng_state, set_rng_state
from mindspore.train.amp import AmpDecorator

from hyper_parallel.core.activation_checkpoint.recompute_state import create_recompute_contexts
from hyper_parallel.platform.mindspore.activation_checkpoint.activation_swap import ActivationWrapper


_DEFAULT_REPLAY_KEY = object()


@lru_cache(maxsize=1)
def _get_tensor_type() -> type:
    """Return the MindSpore Tensor type."""
    return Tensor


def _is_tensor(value: Any) -> bool:
    """Return whether *value* is a MindSpore Tensor."""
    return isinstance(value, _get_tensor_type())


def _tensor_leaves(value: Any) -> List[Any]:
    """Collect Tensor leaves from supported Python containers."""
    if _is_tensor(value):
        return [value]
    leaves = []
    if isinstance(value, (tuple, list)):
        for item in value:
            leaves.extend(_tensor_leaves(item))
    elif isinstance(value, dict):
        for item in value.values():
            leaves.extend(_tensor_leaves(item))
    return leaves


def _map_tensors(function: Callable[[Any], Any], value: Any) -> Any:
    """Apply *function* to Tensor leaves while preserving container types."""
    if _is_tensor(value):
        return function(value)
    if isinstance(value, list):
        return [_map_tensors(function, item) for item in value]
    if isinstance(value, tuple):
        items = [_map_tensors(function, item) for item in value]
        if hasattr(value, "_fields"):
            return type(value)(*items)
        return tuple(items)
    if isinstance(value, dict):
        return type(value)((key, _map_tensors(function, item)) for key, item in value.items())
    return value


def _detach_tree(value: Any) -> Any:
    """Detach every Tensor leaf in *value*."""
    return _map_tensors(lambda tensor: tensor.detach(), value)


def _validate_output(output: Any) -> None:
    """Validate the output structure supported by the custom bridge."""
    if _tensor_leaves(output):
        return
    raise ValueError(
        "reentrant checkpoint requires at least one Tensor output leaf"
    )


def _normalize_sensitivities(output: Any, sensitivity: Any) -> List[Any]:
    """Flatten sensitivities in the same order as output Tensor leaves."""
    output_tensors = _tensor_leaves(output)
    sensitivity_tensors = _tensor_leaves(sensitivity)
    if len(output_tensors) != len(sensitivity_tensors):
        raise RuntimeError(
            "The number of reentrant checkpoint output sensitivities does not match its Tensor outputs: "
            f"expected {len(output_tensors)}, but got {len(sensitivity_tensors)}"
        )
    return sensitivity_tensors


def _mark_requires_grad(tensors: Sequence[Any]) -> None:
    """Mark local VJP roots as requiring gradients."""
    for tensor in tensors:
        requires_grad = getattr(tensor, "requires_grad_", None)
        if callable(requires_grad):
            requires_grad()
        else:
            tensor._requires_grad = True  # pylint: disable=protected-access


class _VjpTape:
    """Retain one ordinary PyNative graph and expose a repeatable VJP."""

    def __init__(self, output: Any, inputs: Sequence[Any], parameters: Sequence[Any]) -> None:
        """Initialize a tape from a recorded output and its gradient targets."""
        _validate_output(output)
        self.output = output
        self.inputs = tuple(inputs)
        self.parameters = tuple(parameters)
        self._released = False

    @classmethod
    def record(
        cls,
        function: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        parameters: Sequence[Any],
    ) -> "_VjpTape":
        """Execute *function* with PyNative grad recording enabled.

        Args:
            function: Callable to record.
            args: Positional arguments passed to the callable.
            kwargs: Keyword arguments passed to the callable.
            parameters: Parameters included as gradient targets.
        """
        input_tensors = _tensor_leaves((args, kwargs))
        _mark_requires_grad(input_tensors)
        previous_enable_grad = _pynative_executor.enable_grad()
        previous_grad_flag = _pynative_executor.grad_flag()
        _pynative_executor.set_enable_grad(True)
        _pynative_executor.set_grad_flag(True)
        try:
            output = function(*args, **kwargs)
        finally:
            _pynative_executor.set_grad_flag(previous_grad_flag)
            _pynative_executor.set_enable_grad(previous_enable_grad)
        return cls(output, input_tensors, parameters)

    def vjp(
        self,
        sensitivity: Any,
        keep_graph: bool,
        accumulate_parameters: bool = False,
    ) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        """Calculate gradients for recorded inputs and parameters.

        Args:
            sensitivity: Output gradients supplied by the caller.
            keep_graph: Whether to retain the local graph for another traversal.
            accumulate_parameters: Accumulate parameter gradients through nested hooks.
        """
        if self._released:
            raise RuntimeError("The retained reentrant checkpoint graph has already been released")
        output_tensors = tuple(_tensor_leaves(self.output))
        sensitivity_tensors = tuple(_normalize_sensitivities(self.output, sensitivity))
        targets = self.inputs + self.parameters
        if accumulate_parameters:
            gradients = run_backward(
                output_tensors,
                sensitivity_tensors,
                keep_graph,
                False,
                (),
                allow_unreachable=True,
                accumulate_grad=True,
            )
            if gradients is None:
                gradients = tuple(getattr(tensor, "grad", None) for tensor in self.inputs)
                for tensor in self.inputs:
                    tensor.grad = None
            if not keep_graph:
                self.release()
            return tuple(gradients), (None,) * len(self.parameters)
        gradients = run_backward(
            output_tensors,
            sensitivity_tensors,
            keep_graph,
            False,
            targets,
            allow_unreachable=True,
            accumulate_grad=False,
        )
        input_count = len(self.inputs)
        input_grads = tuple(gradients[:input_count])
        parameter_grads = tuple(gradients[input_count:])
        if not keep_graph:
            self.release()
        return input_grads, parameter_grads

    def release(self) -> None:
        """Release all Python references keeping the PyNative graph alive."""
        self.output = None
        self.inputs = ()
        self.parameters = ()
        self._released = True


@dataclass(frozen=True)
class _SessionState:
    """Describe the active recompute session on the current thread."""

    session_id: Any
    retain_graph: bool


_CURRENT_HANDLE_COLLECTOR: contextvars.ContextVar[Optional[List[Any]]] = contextvars.ContextVar(
    "hyper_parallel_reentrant_handle_collector",
    default=None,
)
_CURRENT_SESSION: contextvars.ContextVar[Optional[_SessionState]] = contextvars.ContextVar(
    "hyper_parallel_reentrant_session",
    default=None,
)
_IN_REENTRANT_TOP_LEVEL_BACKWARD: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "hyper_parallel_in_reentrant_top_level_backward",
    default=False,
)

_SESSION_LOCK = threading.RLock()
_SESSION_CALLS: Dict[Any, weakref.WeakSet] = defaultdict(weakref.WeakSet)


@contextlib.contextmanager
def reentrant_recompute_handle_collector_ctx() -> Iterator[List[Any]]:
    """Collect reentrant checkpoint handles created by forward execution."""
    handles = []
    token = _CURRENT_HANDLE_COLLECTOR.set(handles)
    try:
        yield handles
    finally:
        _CURRENT_HANDLE_COLLECTOR.reset(token)


@contextlib.contextmanager
def reentrant_recompute_session_ctx(
    session_id: Any,
    retain_on_unpack: bool = False,
) -> Iterator[None]:
    """Bind replay selection and graph retention to the current thread.

    Args:
        session_id: Stable identifier shared by pre-fired replay and backward.
        retain_on_unpack: Retain both outer replay and excluded local graphs for
            a later consumer such as the weight-gradient phase.
    """
    state = _SessionState(session_id=session_id, retain_graph=retain_on_unpack)
    token = _CURRENT_SESSION.set(state)
    try:
        yield
    finally:
        _CURRENT_SESSION.reset(token)


def clear_reentrant_recompute_session(session_id: Any) -> None:
    """Release all checkpoint graphs retained by *session_id*.

    Args:
        session_id: Identifier of the replay session to clear.
    """
    with _SESSION_LOCK:
        calls = list(_SESSION_CALLS.pop(session_id, ()))
    for call in calls:
        call.clear_recompute_session(session_id)


def is_in_reentrant_top_level_backward() -> bool:
    """Return whether execution is inside a top-level reentrant backward."""
    return _IN_REENTRANT_TOP_LEVEL_BACKWARD.get()


@contextlib.contextmanager
def reentrant_recompute_backward_compat_ctx() -> Iterator[None]:
    """Preserve shared-parameter accumulation order for a top-level backward."""
    token = _IN_REENTRANT_TOP_LEVEL_BACKWARD.set(True)
    try:
        yield
    finally:
        _IN_REENTRANT_TOP_LEVEL_BACKWARD.reset(token)


class _CheckpointInvocation:
    """Own excluded-region graphs belonging to one checkpoint call."""

    def __init__(self) -> None:
        """Initialize an empty ordered call registry."""
        self.exclude_entries: Dict[int, List[_ExcludeEntry]] = defaultdict(list)

    def add_entry(self, wrapper_id: int, entry: "_ExcludeEntry") -> None:
        """Append one original-forward excluded call.

        Args:
            wrapper_id: Identity of the exclusion wrapper.
            entry: Retained graph for one invocation.
        """
        self.exclude_entries[wrapper_id].append(entry)

    def get_entry(self, wrapper_id: int, index: int) -> "_ExcludeEntry":
        """Return the excluded call matching replay order.

        Args:
            wrapper_id: Identity of the exclusion wrapper.
            index: Invocation index within the wrapper.
        """
        entries = self.exclude_entries.get(wrapper_id)
        if entries is None or index >= len(entries):
            raise RuntimeError("Checkpoint replay executed an unmatched checkpoint exclusion wrapper call")
        return entries[index]

    def clear(self) -> None:
        """Release all retained excluded-region graphs."""
        for entries in self.exclude_entries.values():
            for entry in entries:
                entry.release()
        self.exclude_entries.clear()


class _ExecutionState:
    """Expose one checkpoint invocation and phase to nested wrappers."""

    def __init__(self, invocation: _CheckpointInvocation, recomputing: bool) -> None:
        """Initialize forward or replay state."""
        self.invocation = invocation
        self.recomputing = recomputing
        self._positions: Dict[int, int] = defaultdict(int)

    def next_entry(self, wrapper_id: int) -> "_ExcludeEntry":
        """Consume the next original call entry for *wrapper_id*.

        Args:
            wrapper_id: Identity of the exclusion wrapper.
        """
        index = self._positions[wrapper_id]
        entry = self.invocation.get_entry(wrapper_id, index)
        self._positions[wrapper_id] += 1
        return entry


_CURRENT_EXECUTION_STATE: contextvars.ContextVar[Optional[_ExecutionState]] = contextvars.ContextVar(
    "hyper_parallel_reentrant_execution_state",
    default=None,
)


@contextlib.contextmanager
def _execution_state_ctx(state: _ExecutionState) -> Iterator[None]:
    """Install a checkpoint execution phase in the current dynamic scope."""
    token = _CURRENT_EXECUTION_STATE.set(state)
    try:
        yield
    finally:
        _CURRENT_EXECUTION_STATE.reset(token)


@contextlib.contextmanager
def _stack_contexts(contexts: Sequence[Any]) -> Iterator[None]:
    """Enter a sequence of contexts in order and exit them in reverse."""
    with contextlib.ExitStack() as stack:
        for context in contexts:
            stack.enter_context(context)
        yield


def _create_phase_contexts(
    context_fn: Optional[Callable[[], Tuple[Any, Any]]],
) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
    """Create framework and optional user contexts for both phases."""
    pairs = [create_recompute_contexts()]
    if context_fn is not None:
        user_pair = context_fn()
        if not isinstance(user_pair, tuple) or len(user_pair) != 2:
            raise ValueError("context_fn must return a (forward_context, recompute_context) tuple")
        pairs.append(user_pair)
    return tuple(pair[0] for pair in pairs), tuple(pair[1] for pair in pairs)


class _ExcludeEntry:
    """Retain one excluded forward output and its local VJP graph."""

    def __init__(
        self,
        output: Any,
        tape: _VjpTape,
        parameter_count: int,
        rng_state_after: Optional[Any],
    ) -> None:
        """Initialize one excluded call entry."""
        self.output = output
        self.tape = tape
        self.input_count = len(tape.inputs)
        self.parameter_count = parameter_count
        self.rng_state_after = rng_state_after

    def vjp(self, sensitivity: Any, keep_graph: bool) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        """Run the retained excluded-region backward graph.

        Args:
            sensitivity: Output gradients supplied by the bridge.
            keep_graph: Whether to retain the graph for another traversal.
        """
        return self.tape.vjp(sensitivity, keep_graph)

    def restore_rng_state(self) -> None:
        """Advance replay RNG to the state after the skipped region."""
        if self.rng_state_after is None:
            return
        set_rng_state(self.rng_state_after)

    def release(self) -> None:
        """Release output and local VJP graph references."""
        self.output = None
        self.tape.release()


@lru_cache(maxsize=1)
def _get_exclude_replay_function() -> type:
    """Create the private MindSpore custom autograd bridge lazily."""
    class _ExcludeReplayFunction(_Function):
        """Return a cached output and connect it to replay inputs in backward."""

        @staticmethod
        def forward(ctx: Any, entry: _ExcludeEntry, input_count: int, *tensors: Any) -> Any:
            """Return the cached output without executing the excluded module.

            Args:
                ctx: Autograd context for the bridge call.
                entry: Retained excluded-region graph.
                input_count: Number of activation inputs.
                tensors: Activation and parameter tensors attached to the bridge.
            """
            del tensors
            ctx.entry = entry
            ctx.input_count = input_count
            return _detach_tree(entry.output)

        @staticmethod
        def backward(ctx: Any, *grad_outputs: Any) -> Tuple[Any, ...]:
            """Route the bridge VJP through the original excluded graph.

            Args:
                ctx: Autograd context created in forward.
                grad_outputs: Gradients of the cached outputs.
            """
            session = _CURRENT_SESSION.get()
            keep_graph = session is not None and session.retain_graph
            sensitivity = grad_outputs[0] if len(grad_outputs) == 1 else grad_outputs
            input_grads, parameter_grads = ctx.entry.vjp(sensitivity, keep_graph)
            if len(input_grads) != ctx.input_count:
                raise RuntimeError(
                    "Checkpoint exclusion input count changed between original forward and replay"
                )
            return (None, None, *input_grads, *parameter_grads)

    return _ExcludeReplayFunction


def _apply_exclude_replay(
    entry: _ExcludeEntry,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    parameters: Sequence[Any],
) -> Any:
    """Apply an autograd bridge from current replay tensors to cached output."""
    entry.restore_rng_state()
    input_tensors = tuple(_tensor_leaves((args, kwargs)))
    if len(input_tensors) != entry.input_count:
        raise RuntimeError(
            "Checkpoint exclusion Tensor input count changed between original forward and replay: "
            f"expected {entry.input_count}, but got {len(input_tensors)}"
        )
    if len(parameters) != entry.parameter_count:
        raise RuntimeError(
            "Checkpoint exclusion parameter count changed between original forward and replay: "
            f"expected {entry.parameter_count}, but got {len(parameters)}"
        )
    differentiable_tensors = input_tensors + tuple(parameters)
    if not differentiable_tensors:
        return _detach_tree(entry.output)
    replay_function = _get_exclude_replay_function()
    return replay_function.apply(entry, len(input_tensors), *differentiable_tensors)


class ReentrantCheckpointExcludeWrapper(ActivationWrapper):
    """Execute a region once and reuse its output during reentrant replay."""

    def __init__(self, module: Callable[..., Any], save_rng_state: bool = True) -> None:
        """Initialize a reentrant checkpoint exclusion wrapper."""
        if not callable(module):
            raise ValueError("module must be a MindSpore Cell or callable")
        super().__init__(module, track_overlaps=False)
        self.save_rng_state = save_rng_state

    def construct(self, *args: Any, **kwargs: Any) -> Any:
        """Record a local VJP in forward and insert a bridge during replay."""
        state = _CURRENT_EXECUTION_STATE.get()
        if state is None:
            return self._wrapped_module(*args, **kwargs)  # pylint: disable=not-callable
        parameters = tuple(self._wrapped_module.trainable_params())
        if state.recomputing:
            entry = state.next_entry(id(self))
            return _apply_exclude_replay(entry, args, kwargs, parameters)

        detached_args = _detach_tree(args)
        detached_kwargs = _detach_tree(kwargs)
        tape = _VjpTape.record(self._wrapped_module, detached_args, detached_kwargs, parameters)
        rng_state_after = None
        if self.save_rng_state:
            rng_state_after = get_rng_state()
        entry = _ExcludeEntry(tape.output, tape, len(parameters), rng_state_after)
        state.invocation.add_entry(id(self), entry)
        return tape.output


class _ReentrantCheckpointCall:
    """Own the state and retained graphs for one checkpoint invocation."""

    def __init__(
        self,
        module: Callable[..., Any],
        context_fn: Optional[Callable[[], Tuple[Any, Any]]],
        save_rng_state: bool,
    ) -> None:
        """Initialize one checkpoint invocation."""
        self._wrapped_module = module
        self.context_fn = context_fn
        self.save_rng_state = save_rng_state
        self._invocation = _CheckpointInvocation()
        self._original_args: Optional[Tuple[Any, ...]] = None
        self._original_kwargs: Optional[Dict[str, Any]] = None
        self._rng_state = None
        self._amp_strategy = None
        self._forward_contexts: Tuple[Any, ...] = ()
        self._recompute_contexts: Tuple[Any, ...] = ()
        self._replay_tapes: Dict[Any, _VjpTape] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _validate_inputs(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        """Validate that the checkpoint invocation has a Tensor input leaf."""
        if not _tensor_leaves((args, kwargs)):
            raise ValueError("reentrant checkpoint requires at least one Tensor input")

    def run_forward(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Execute the original checkpoint forward without recording its graph.

        Args:
            args: Positional checkpoint arguments.
            kwargs: Keyword checkpoint arguments.
        """
        self._validate_inputs(args, kwargs)
        self._original_args = args
        self._original_kwargs = dict(kwargs)
        self._forward_contexts, self._recompute_contexts = _create_phase_contexts(self.context_fn)
        if self.save_rng_state:
            self._rng_state = get_rng_state()
        self._amp_strategy = get_curr_amp_strategy()
        execution_state = _ExecutionState(self._invocation, recomputing=False)
        with _stack_contexts(self._forward_contexts), _execution_state_ctx(execution_state), _no_grad():
            output = self._wrapped_module(*args, **kwargs)
        _validate_output(output)
        collector = _CURRENT_HANDLE_COLLECTOR.get()
        if collector is not None:
            collector.append(self)
        return output

    @contextlib.contextmanager
    def _replay_environment(self) -> Iterator[None]:
        """Restore RNG, AMP, user contexts, and recompute state for replay."""
        previous_rng_state = get_rng_state() if self.save_rng_state else None
        if self.save_rng_state:
            set_rng_state(self._rng_state)
        _pynative_executor.set_is_run_recompute(True)
        execution_state = _ExecutionState(self._invocation, recomputing=True)
        try:
            with _stack_contexts(self._recompute_contexts), _execution_state_ctx(execution_state):
                if self._amp_strategy is None:
                    yield
                else:
                    with AmpDecorator(
                        self._amp_strategy.get_amp_level(),
                        self._amp_strategy.get_amp_dtype(),
                        self._amp_strategy.get_white_list(),
                        self._amp_strategy.get_black_list(),
                    ):
                        yield
        finally:
            _pynative_executor.set_is_run_recompute(False)
            if previous_rng_state is not None:
                set_rng_state(previous_rng_state)

    def _record_replay(self) -> _VjpTape:
        """Replay the checkpoint function once and retain its PyNative graph."""
        if self._original_args is None or self._original_kwargs is None:
            raise RuntimeError("Cannot replay a reentrant checkpoint before its original forward")
        replay_args = _detach_tree(self._original_args)
        replay_kwargs = _detach_tree(self._original_kwargs)
        with self._replay_environment():
            replay_parameters = tuple(self._wrapped_module.trainable_params())
            return _VjpTape.record(
                self._wrapped_module,
                replay_args,
                replay_kwargs,
                replay_parameters,
            )

    def recompute(self, session_id: Any) -> None:
        """Pre-fire replay and retain its graph under *session_id*.

        Args:
            session_id: Identifier used by the later backward traversal.
        """
        with self._lock:
            if session_id not in self._replay_tapes:
                self._replay_tapes[session_id] = self._record_replay()
        with _SESSION_LOCK:
            _SESSION_CALLS[session_id].add(self)

    def _get_replay_tape(self, replay_key: Any) -> _VjpTape:
        """Get or lazily record the replay tape for one backward session."""
        with self._lock:
            tape = self._replay_tapes.get(replay_key)
            if tape is None:
                tape = self._record_replay()
                self._replay_tapes[replay_key] = tape
        if replay_key is not _DEFAULT_REPLAY_KEY:
            with _SESSION_LOCK:
                _SESSION_CALLS[replay_key].add(self)
        return tape

    def backward(self, sensitivity: Any) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        """Replay if needed and calculate checkpoint gradients.

        Args:
            sensitivity: Output gradients supplied to the custom backward.
        """
        session = _CURRENT_SESSION.get()
        replay_key = session.session_id if session is not None else _DEFAULT_REPLAY_KEY
        retain_graph = session is not None and session.retain_graph
        tape = self._get_replay_tape(replay_key)
        input_grads, parameter_grads = tape.vjp(
            sensitivity,
            keep_graph=retain_graph,
            accumulate_parameters=True,
        )
        if not retain_graph:
            self._finish_replay(replay_key)
        return input_grads, parameter_grads

    def _finish_replay(self, replay_key: Any) -> None:
        """Drop replay and excluded graphs after their terminal consumer."""
        with self._lock:
            self._replay_tapes.pop(replay_key, None)
        self._invocation.clear()
        self._original_args = None
        self._original_kwargs = None

    def clear_recompute_session(self, session_id: Any) -> None:
        """Release replay state retained by one schedule session.

        Args:
            session_id: Identifier of the replay session to clear.
        """
        with self._lock:
            tape = self._replay_tapes.pop(session_id, None)
        if tape is not None:
            tape.release()
        self._invocation.clear()
        self._original_args = None
        self._original_kwargs = None


@lru_cache(maxsize=1)
def _get_reentrant_checkpoint_function() -> type:
    """Create the outer custom autograd checkpoint node lazily."""
    class _ReentrantCheckpointFunction(_Function):
        """Run forward without a graph and replay it from custom backward."""

        @staticmethod
        def forward(
            ctx: Any,
            call: _ReentrantCheckpointCall,
            input_count: int,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
            *tensors: Any,
        ) -> Any:
            """Execute and retain one original checkpoint invocation.

            Args:
                ctx: Autograd context for this checkpoint call.
                call: Per-invocation checkpoint state.
                input_count: Number of activation inputs.
                args: Original positional arguments.
                kwargs: Original keyword arguments.
                tensors: Activation tensors attached to the node.
            """
            del tensors
            ctx.call = call
            ctx.input_count = input_count
            return call.run_forward(args, kwargs)

        @staticmethod
        def backward(ctx: Any, *grad_outputs: Any) -> Tuple[Any, ...]:
            """Replay the checkpoint and return its gradients.

            Args:
                ctx: Autograd context created in forward.
                grad_outputs: Gradients of checkpoint outputs.
            """
            sensitivity = grad_outputs[0] if len(grad_outputs) == 1 else grad_outputs
            input_grads, _ = ctx.call.backward(sensitivity)
            if len(input_grads) != ctx.input_count:
                raise RuntimeError(
                    "Checkpoint Tensor input count changed between original forward and replay"
                )
            return (None, None, None, None, *input_grads)

    return _ReentrantCheckpointFunction


class ReentrantCheckpointWrapper(ActivationWrapper):
    """Wrap a MindSpore Cell or method with hook-free reentrant checkpointing."""

    def __init__(
        self,
        module: Callable[..., Any],
        context_fn: Optional[Callable[[], Tuple[Any, Any]]] = None,
        save_rng_state: bool = True,
    ) -> None:
        """Initialize a reentrant checkpoint wrapper."""
        if not callable(module):
            raise ValueError("module must be a MindSpore Cell or callable")
        super().__init__(module)
        self.context_fn = context_fn
        self.save_rng_state = save_rng_state

    def construct(self, *args: Any, **kwargs: Any) -> Any:
        """Create and execute one per-invocation reentrant checkpoint call."""
        if not _pynative_executor.enable_grad():
            return self._wrapped_module(*args, **kwargs)  # pylint: disable=not-callable
        call = _ReentrantCheckpointCall(
            self._wrapped_module,
            context_fn=self.context_fn,
            save_rng_state=self.save_rng_state,
        )
        input_tensors = tuple(_tensor_leaves((args, kwargs)))
        checkpoint_function = _get_reentrant_checkpoint_function()
        return checkpoint_function.apply(
            call,
            len(input_tensors),
            args,
            kwargs,
            *input_tensors,
        )


def reentrant_checkpoint_wrapper(
    module: Callable[..., Any],
    context_fn: Optional[Callable[[], Tuple[Any, Any]]] = None,
    save_rng_state: bool = True,
) -> ReentrantCheckpointWrapper:
    """Wrap *module* with the experimental hook-free reentrant checkpoint.

    Args:
        module: Cell or callable to checkpoint.
        context_fn: Optional factory returning forward and replay contexts.
        save_rng_state: Preserve default generators across replay.
    """
    return ReentrantCheckpointWrapper(
        module,
        context_fn=context_fn,
        save_rng_state=save_rng_state,
    )


def reentrant_checkpoint_exclude_wrapper(
    module: Callable[..., Any],
    save_rng_state: bool = True,
) -> ReentrantCheckpointExcludeWrapper:
    """Wrap a callable so its region is not executed during reentrant replay.

    Args:
        module: Cell or callable to exclude.
        save_rng_state: Advance default generators as in the original forward.
    """
    return ReentrantCheckpointExcludeWrapper(module, save_rng_state=save_rng_state)


__all__ = [
    "ReentrantCheckpointExcludeWrapper",
    "ReentrantCheckpointWrapper",
    "clear_reentrant_recompute_session",
    "reentrant_recompute_backward_compat_ctx",
    "reentrant_checkpoint_exclude_wrapper",
    "reentrant_checkpoint_wrapper",
    "reentrant_recompute_handle_collector_ctx",
    "reentrant_recompute_session_ctx",
    "is_in_reentrant_top_level_backward",
]
