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

import contextlib
import contextvars
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import importlib
import threading
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple
import weakref

from mindspore import Parameter, Tensor, nn
from mindspore._c_expression import run_backward
from mindspore._c_expression.amp import get_curr_amp_strategy
from mindspore.common._grad_function import _Function
from mindspore.common.api import _no_grad, _pynative_executor
from mindspore.common.generator import get_rng_state, set_rng_state
from mindspore.train.amp import AmpDecorator

from hyper_parallel.core.activation_checkpoint.recompute_state import create_recompute_contexts
from hyper_parallel.platform.mindspore.activation_checkpoint.activation_swap import ActivationWrapper
from hyper_parallel.platform.mindspore.reentrant_mode import (
    ReentrantBackwardMode,
    get_reentrant_backward_mode,
    should_expose_reentrant_parameters,
)


_DEFAULT_REPLAY_KEY = object()
_checkpoint_exclude = importlib.import_module(
    "hyper_parallel.platform.mindspore.activation_checkpoint.checkpoint_exclude_wrapper"
)


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


def _collect_parameters(value: Any, parameters: List[Any], seen: set) -> None:
    """Collect parameters reachable from Cell arguments without duplicates."""
    if isinstance(value, Parameter):
        candidates = (value,)
    elif isinstance(value, nn.Cell):
        candidates = value.trainable_params()
    elif isinstance(value, (tuple, list)):
        for item in value:
            _collect_parameters(item, parameters, seen)
        return
    elif isinstance(value, dict):
        for item in value.values():
            _collect_parameters(item, parameters, seen)
        return
    else:
        return

    for parameter in candidates:
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        parameters.append(parameter)


def _collect_callable_parameters(
    module: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Tuple[Any, ...]:
    """Collect module-owned and externally supplied Cell parameters."""
    parameters = []
    seen = set()
    for parameter in module.trainable_params():
        parameter_id = id(parameter)
        if parameter_id not in seen:
            seen.add(parameter_id)
            parameters.append(parameter)
    _collect_parameters((args, kwargs), parameters, seen)
    return tuple(parameters)


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


@dataclass(frozen=True)
class _DetachedInputs:
    """Store detached arguments and tensor leaves collected in one traversal."""

    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    tensors: Tuple[Any, ...]
    tensor_inputs: Tuple[Tuple[Tuple[Tuple[str, Any], ...], Any], ...]


def _detach_inputs(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> _DetachedInputs:
    """Detach inputs, mark tensor roots, and collect paths in one traversal."""
    tensors = []
    tensor_inputs = []

    def detach(value: Any, path: Tuple[Tuple[str, Any], ...]) -> Any:
        if isinstance(value, Tensor):
            detached = value.detach()
            requires_grad = getattr(detached, "requires_grad_", None)
            if callable(requires_grad):
                requires_grad()
            else:
                detached._requires_grad = True  # pylint: disable=protected-access
            tensors.append(detached)
            tensor_inputs.append((path, detached))
            return detached
        if isinstance(value, list):
            return [detach(item, path + (("index", index),)) for index, item in enumerate(value)]
        if isinstance(value, tuple):
            items = [
                detach(item, path + (("index", index),))
                for index, item in enumerate(value)
            ]
            if hasattr(value, "_fields"):
                return type(value)(*items)
            return tuple(items)
        if isinstance(value, dict):
            return type(value)(
                (key, detach(item, path + (("key", key),)))
                for key, item in value.items()
            )
        return value

    detached_args = tuple(
        detach(value, (("arg", index),))
        for index, value in enumerate(args)
    )
    detached_kwargs = {
        key: detach(value, (("kwarg", key),))
        for key, value in kwargs.items()
    }
    return _DetachedInputs(
        args=detached_args,
        kwargs=detached_kwargs,
        tensors=tuple(tensors),
        tensor_inputs=tuple(tensor_inputs),
    )


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
    """Mark local backward roots as requiring gradients."""
    for tensor in tensors:
        requires_grad = getattr(tensor, "requires_grad_", None)
        if callable(requires_grad):
            requires_grad()
        else:
            tensor._requires_grad = True  # pylint: disable=protected-access


class _ReplayGraph:
    """Retain one ordinary PyNative graph and expose its backward."""

    def __init__(self, output: Any, inputs: Sequence[Any], parameters: Sequence[Any]) -> None:
        """Initialize a tape from a recorded output and its gradient targets."""
        _validate_output(output)
        self.output = output
        self.inputs = tuple(inputs)
        self.input_count = len(self.inputs)
        self.parameters = tuple(parameters)
        self._released = False

    @classmethod
    def record(
        cls,
        function: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        parameters: Sequence[Any],
        input_tensors: Optional[Sequence[Any]] = None,
    ) -> "_ReplayGraph":
        """Execute *function* with PyNative grad recording enabled.

        Args:
            function: Callable to record.
            args: Positional arguments passed to the callable.
            kwargs: Keyword arguments passed to the callable.
            parameters: Parameters included as gradient targets.
            input_tensors: Pre-collected differentiable input leaves.
        """
        if input_tensors is None:
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

    def backward(
        self,
        sensitivity: Any,
        keep_graph: bool,
    ) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        """Return gradients selected by the active dx/dw replay mode.

        Args:
            sensitivity: Output gradients supplied by the caller.
            keep_graph: Whether to retain the local graph for another traversal.
        """
        if self._released:
            raise RuntimeError("The retained reentrant checkpoint graph has already been released")
        output_tensors = tuple(_tensor_leaves(self.output))
        sensitivity_tensors = tuple(_normalize_sensitivities(self.output, sensitivity))
        mode = get_reentrant_backward_mode()
        if mode is ReentrantBackwardMode.DX:
            targets = self.inputs
            parameter_grads = (None,) * len(self.parameters)
        elif mode is ReentrantBackwardMode.DW:
            targets = self.parameters
            parameter_grads = None
        else:
            targets = ()
            parameter_grads = (None,) * len(self.parameters)
        gradients = run_backward(
            output_tensors,
            sensitivity_tensors,
            keep_graph,
            False,
            targets,
            allow_unreachable=True,
            accumulate_grad=mode is ReentrantBackwardMode.NORMAL,
        )
        if gradients is None:
            gradients = tuple(getattr(tensor, "grad", None) for tensor in self.inputs)
        if mode is ReentrantBackwardMode.DW:
            input_grads = (None,) * self.input_count
            parameter_grads = tuple(gradients[:len(self.parameters)])
        else:
            input_grads = tuple(gradients[:self.input_count])
        if not keep_graph:
            self.release()
        return input_grads, tuple(parameter_grads)

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


class _ExecutionState:
    """Run one reentrant exclude call during forward or replay."""

    def __init__(self, exclude_cache: _checkpoint_exclude._ExcludeCache, recomputing: bool) -> None:
        """Initialize forward or replay state."""
        self.exclude_cache = exclude_cache
        self.recomputing = recomputing

    def __call__(
        self,
        wrapper: Any,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        """Execute one shared exclusion wrapper inside the reentrant phase."""
        module = wrapper._ckpt_wrapped_module  # pylint: disable=protected-access
        parameters = tuple(module.trainable_params()) if hasattr(module, "trainable_params") else ()
        if self.recomputing:
            entry = self.exclude_cache.pop(id(wrapper))
            return _apply_exclude_replay(entry, args, kwargs, parameters)

        prepared = _detach_inputs(args, kwargs)
        input_infos = (
            ()
            if wrapper.preserve_inputs
            else _checkpoint_exclude._capture_recompute_inputs_from_leaves(
                prepared.tensor_inputs
            )
        )
        input_bindings = []
        pack_token = _checkpoint_exclude._ACTIVE_PACK_STATE.set(
            _checkpoint_exclude._PackState(input_infos, input_bindings)
        )
        try:
            with _checkpoint_exclude._saved_tensors_context():
                tape = _ReplayGraph.record(
                    module,
                    prepared.args,
                    prepared.kwargs,
                    parameters,
                    input_tensors=prepared.tensors,
                )
        finally:
            _checkpoint_exclude._ACTIVE_PACK_STATE.reset(pack_token)
        rng_state = get_rng_state() if wrapper.save_rng_state else None
        self.exclude_cache.save(
            id(wrapper),
            _checkpoint_exclude._ExcludeCacheEntry(
                output=tape.output,
                input_bindings=input_bindings,
                parameter_count=len(parameters),
                rng_state=rng_state,
                tape=tape,
            ),
        )
        return tape.output


@contextlib.contextmanager
def _execution_state_ctx(state: _ExecutionState) -> Iterator[None]:
    """Install a checkpoint execution phase in the current dynamic scope."""
    with _checkpoint_exclude.reentrant_exclude_runner_ctx(state):
        yield


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


@lru_cache(maxsize=1)
def _get_exclude_replay_function() -> type:
    """Create the private MindSpore custom autograd bridge lazily."""
    class _ExcludeReplayFunction(_Function):
        """Return a cached output and connect it to replay inputs in backward."""

        @staticmethod
        def forward(
            ctx: Any,
            entry: _checkpoint_exclude._ExcludeCacheEntry,
            input_count: int,
            *tensors: Any,
        ) -> Any:
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
            """Route the bridge backward through the original excluded graph.

            Args:
                ctx: Autograd context created in forward.
                grad_outputs: Gradients of the cached outputs.
            """
            session = _CURRENT_SESSION.get()
            keep_graph = session is not None and session.retain_graph
            sensitivity = grad_outputs[0] if len(grad_outputs) == 1 else grad_outputs
            input_grads, parameter_grads = ctx.entry.tape.backward(sensitivity, keep_graph)
            if len(input_grads) != ctx.input_count:
                raise RuntimeError(
                    "Checkpoint exclusion input count changed between original forward and replay"
                )
            return (None, None, *input_grads, *parameter_grads)

    return _ExcludeReplayFunction


def _apply_exclude_replay(
    entry: _checkpoint_exclude._ExcludeCacheEntry,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    parameters: Sequence[Any],
) -> Any:
    """Apply an autograd bridge from current replay tensors to cached output."""
    if entry.tape is None:
        raise RuntimeError("Reentrant checkpoint exclusion entry is missing its local backward graph")
    if entry.rng_state is not None:
        set_rng_state(entry.rng_state)
    _checkpoint_exclude._materialize_recompute_inputs(entry, args, kwargs)
    input_tensors = tuple(_tensor_leaves((args, kwargs)))
    if len(input_tensors) != entry.tape.input_count:
        raise RuntimeError(
            "Checkpoint exclusion Tensor input count changed between original forward and replay: "
            f"expected {entry.tape.input_count}, but got {len(input_tensors)}"
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


class ReentrantCheckpointExcludeWrapper(_checkpoint_exclude.CheckpointExcludeWrapper):
    """Configure the shared exclusion wrapper for reentrant replay."""

    def __init__(self, module: Callable[..., Any], save_rng_state: bool = True) -> None:
        """Initialize a reentrant checkpoint exclusion wrapper."""
        super().__init__(module)
        self.save_rng_state = save_rng_state


class _ReentrantCheckpointCall:
    """Own the state and retained graphs for one checkpoint invocation."""

    def __init__(
        self,
        module: Callable[..., Any],
        parameters: Sequence[Any],
        context_fn: Optional[Callable[[], Tuple[Any, Any]]],
        save_rng_state: bool,
    ) -> None:
        """Initialize one checkpoint invocation."""
        self._wrapped_module = module
        self.parameters = tuple(parameters)
        self.context_fn = context_fn
        self.save_rng_state = save_rng_state
        self._exclude_cache = _checkpoint_exclude._ExcludeCache()
        self._original_args: Optional[Tuple[Any, ...]] = None
        self._original_kwargs: Optional[Dict[str, Any]] = None
        self._rng_state = None
        self._amp_strategy = None
        self._forward_contexts: Tuple[Any, ...] = ()
        self._recompute_contexts: Tuple[Any, ...] = ()
        self._replay_graphs: Dict[Any, _ReplayGraph] = {}
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
        execution_state = _ExecutionState(self._exclude_cache, recomputing=False)
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
        execution_state = _ExecutionState(self._exclude_cache, recomputing=True)
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

    def _record_replay(self) -> _ReplayGraph:
        """Replay the checkpoint function once and retain its PyNative graph."""
        if self._original_args is None or self._original_kwargs is None:
            raise RuntimeError("Cannot replay a reentrant checkpoint before its original forward")
        replay_inputs = _detach_inputs(self._original_args, self._original_kwargs)
        with self._replay_environment():
            return _ReplayGraph.record(
                self._wrapped_module,
                replay_inputs.args,
                replay_inputs.kwargs,
                self.parameters,
                input_tensors=replay_inputs.tensors,
            )

    def recompute(self, session_id: Any) -> None:
        """Pre-fire replay and retain its graph under *session_id*.

        Args:
            session_id: Identifier used by the later backward traversal.
        """
        with self._lock:
            if session_id not in self._replay_graphs:
                self._replay_graphs[session_id] = self._record_replay()
        with _SESSION_LOCK:
            _SESSION_CALLS[session_id].add(self)

    def _get_replay_graph(self, replay_key: Any) -> _ReplayGraph:
        """Get or lazily record the replay graph for one backward session."""
        with self._lock:
            replay_graph = self._replay_graphs.get(replay_key)
            if replay_graph is None:
                replay_graph = self._record_replay()
                self._replay_graphs[replay_key] = replay_graph
        if replay_key is not _DEFAULT_REPLAY_KEY:
            with _SESSION_LOCK:
                _SESSION_CALLS[replay_key].add(self)
        return replay_graph

    def backward(self, sensitivity: Any) -> Tuple[Tuple[Any, ...], Tuple[Any, ...]]:
        """Replay if needed and calculate checkpoint gradients.

        Args:
            sensitivity: Output gradients supplied to the custom backward.
        """
        session = _CURRENT_SESSION.get()
        replay_key = session.session_id if session is not None else _DEFAULT_REPLAY_KEY
        retain_graph = session is not None and session.retain_graph
        replay_graph = self._get_replay_graph(replay_key)
        input_grads, parameter_grads = replay_graph.backward(sensitivity, keep_graph=retain_graph)
        if not retain_graph:
            self._finish_replay(replay_key)
        return input_grads, parameter_grads

    def _finish_replay(self, replay_key: Any) -> None:
        """Drop replay and excluded graphs after their terminal consumer."""
        with self._lock:
            self._replay_graphs.pop(replay_key, None)
        self._exclude_cache.clear(release_retained=True)
        self._original_args = None
        self._original_kwargs = None

    def clear_recompute_session(self, session_id: Any) -> None:
        """Release replay state retained by one schedule session.

        Args:
            session_id: Identifier of the replay session to clear.
        """
        with self._lock:
            replay_graph = self._replay_graphs.pop(session_id, None)
        if replay_graph is not None:
            replay_graph.release()
        self._exclude_cache.clear(release_retained=True)
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
            ctx.call = call
            ctx.input_count = input_count
            ctx.parameter_count = len(tensors) - input_count
            del tensors
            return call.run_forward(args, kwargs)

        @staticmethod
        def backward(ctx: Any, *grad_outputs: Any) -> Tuple[Any, ...]:
            """Replay the checkpoint and return its gradients.

            Args:
                ctx: Autograd context created in forward.
                grad_outputs: Gradients of checkpoint outputs.
            """
            sensitivity = grad_outputs[0] if len(grad_outputs) == 1 else grad_outputs
            input_grads, parameter_grads = ctx.call.backward(sensitivity)
            if len(input_grads) != ctx.input_count:
                raise RuntimeError(
                    "Checkpoint Tensor input count changed between original forward and replay"
                )
            return (None, None, None, None, *input_grads, *parameter_grads)

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
        parameters = (
            _collect_callable_parameters(self._wrapped_module, args, kwargs)
            if should_expose_reentrant_parameters()
            else ()
        )
        call = _ReentrantCheckpointCall(
            self._wrapped_module,
            parameters,
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
            *call.parameters,
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
