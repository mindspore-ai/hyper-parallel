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
"""Hook-free reentrant activation checkpoint with excluded regions for Torch."""
import contextlib
from functools import lru_cache
import threading
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import torch

from . import checkpoint_exclude as _checkpoint_exclude
from hyper_parallel.core.activation_memory.recompute_state import create_recompute_contexts
from hyper_parallel.core.activation_memory.reentrant_exclude import (
    ReentrantExecutionState,
    reentrant_execution_state_ctx,
)
from hyper_parallel.core.activation_memory.reentrant_session import (
    clear_reentrant_recompute_session,
    collect_reentrant_call,
    get_current_reentrant_session,
    register_reentrant_session_call,
    reentrant_recompute_handle_collector_ctx,
    reentrant_recompute_session_ctx,
)
from hyper_parallel.core.activation_memory.reentrant_utils import (
    AutocastState,
    AutogradTape,
    RngState,
    autocast_context,
    capture_autocast_state,
    capture_rng_state,
    check_backward_validity,
    detach_tree,
    restore_rng_state,
    tensor_leaves,
    validate_output,
)
from hyper_parallel.core.activation_memory.wrapper import ActivationWrapper, CheckpointExcludeWrapper


_DEFAULT_REPLAY_KEY = object()


class ReentrantCheckpointExcludeWrapper(CheckpointExcludeWrapper):
    """Configure the shared exclusion wrapper for reentrant replay."""

    def __init__(self, module: Callable[..., Any], save_rng_state: bool = True) -> None:
        """Initialize a reentrant checkpoint exclusion wrapper."""
        super().__init__(module, save_output=True)
        self.save_rng_state = save_rng_state


def reentrant_checkpoint_exclude_wrapper(
    module: Callable[..., Any],
    save_rng_state: bool = True,
) -> ReentrantCheckpointExcludeWrapper:
    """Wrap a callable so its region is not executed during reentrant replay."""
    return ReentrantCheckpointExcludeWrapper(module, save_rng_state=save_rng_state)


@contextlib.contextmanager
def _stack_contexts(contexts: Tuple[Any, ...]) -> Iterator[None]:
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


class _ReentrantCheckpointCall:
    """Own the state and retained graphs for one checkpoint invocation."""

    def __init__(
        self,
        module: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        context_fn: Optional[Callable[[], Tuple[Any, Any]]],
        save_rng_state: bool,
    ) -> None:
        """Initialize one checkpoint invocation."""
        self._wrapped_module = module
        self.context_fn = context_fn
        self.save_rng_state = save_rng_state
        self._exclude_cache = _checkpoint_exclude._ExcludeCache()
        self._original_args = args
        self._original_kwargs = dict(kwargs)
        self._rng_state: Optional[RngState] = capture_rng_state((args, kwargs)) if save_rng_state else None
        self._autocast_state: AutocastState = capture_autocast_state((args, kwargs))
        self._forward_contexts, self._recompute_contexts = _create_phase_contexts(context_fn)
        self._replay_tapes: Dict[Any, AutogradTape] = {}
        self._lock = threading.RLock()

    @staticmethod
    def validate_inputs(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        """Validate native reentrant-style top-level Tensor inputs."""
        has_tensor = False
        for value in args + tuple(kwargs.values()):
            if torch.is_tensor(value):
                has_tensor = True
                continue
            if tensor_leaves(value):
                raise ValueError(
                    "reentrant checkpoint does not yet support Tensor inputs nested in tuple/list/dict values"
                )
        if not has_tensor:
            raise ValueError("reentrant checkpoint requires at least one Tensor input")
        check_backward_validity(args, kwargs)

    def run_forward(self) -> Any:
        """Execute the original checkpoint forward without recording its graph."""
        execution_state = ReentrantExecutionState(self._exclude_cache, recomputing=False)
        with _stack_contexts(self._forward_contexts), reentrant_execution_state_ctx(execution_state), torch.no_grad():
            output = self._wrapped_module(*self._original_args, **self._original_kwargs)
        validate_output(output)
        collect_reentrant_call(self)
        return output

    @contextlib.contextmanager
    def _replay_environment(self) -> Iterator[None]:
        """Restore RNG, autocast, and recompute contexts around replay."""
        current_rng_state = None
        if self.save_rng_state:
            current_rng_state = capture_rng_state((self._original_args, self._original_kwargs))
            restore_rng_state(self._rng_state)
        execution_state = ReentrantExecutionState(self._exclude_cache, recomputing=True)
        try:
            with (
                _stack_contexts(self._recompute_contexts),
                reentrant_execution_state_ctx(execution_state),
                autocast_context(self._autocast_state),
            ):
                yield
        finally:
            restore_rng_state(current_rng_state)

    def _record_replay(self) -> AutogradTape:
        """Replay the checkpoint function once and retain its eager graph."""
        if self._original_args is None or self._original_kwargs is None:
            raise RuntimeError("Cannot replay a reentrant checkpoint after its state was released")
        replay_args = detach_tree(self._original_args)
        replay_kwargs = detach_tree(self._original_kwargs)
        with self._replay_environment():
            return AutogradTape.record(self._wrapped_module, replay_args, replay_kwargs)

    def recompute(self, session_id: Any) -> None:
        """Pre-fire replay and retain its graph under *session_id*."""
        with self._lock:
            if session_id not in self._replay_tapes:
                self._replay_tapes[session_id] = self._record_replay()
        register_reentrant_session_call(session_id, self)

    def _get_replay_tape(self, replay_key: Any) -> AutogradTape:
        """Get or lazily record the replay tape for one backward session."""
        with self._lock:
            tape = self._replay_tapes.get(replay_key)
            if tape is None:
                tape = self._record_replay()
                self._replay_tapes[replay_key] = tape
        if replay_key is not _DEFAULT_REPLAY_KEY:
            register_reentrant_session_call(replay_key, self)
        return tape

    def backward(self, sensitivity: Any) -> Tuple[Any, ...]:
        """Replay if needed and calculate checkpoint input gradients."""
        session = get_current_reentrant_session()
        replay_key = session.session_id if session is not None else _DEFAULT_REPLAY_KEY
        retain_graph = session is not None and session.retain_graph
        tape = self._get_replay_tape(replay_key)
        input_grads = tape.backward(sensitivity, keep_graph=retain_graph)
        if not retain_graph:
            self._finish_replay(replay_key)
        return input_grads

    def _finish_replay(self, replay_key: Any) -> None:
        """Drop replay and excluded graphs after their terminal consumer."""
        with self._lock:
            self._replay_tapes.pop(replay_key, None)
        self._exclude_cache.clear(release_retained=True)
        self._original_args = None
        self._original_kwargs = None

    def clear_recompute_session(self, session_id: Any) -> None:
        """Release replay state retained by one schedule session."""
        with self._lock:
            tape = self._replay_tapes.pop(session_id, None)
        if tape is not None:
            tape.release()
        self._exclude_cache.clear(release_retained=True)
        self._original_args = None
        self._original_kwargs = None


@lru_cache(maxsize=1)
def _get_reentrant_checkpoint_function() -> type:
    """Create the outer custom autograd checkpoint node lazily."""
    class _ReentrantCheckpointFunction(torch.autograd.Function):
        """Run forward without a graph and replay it from custom backward."""

        @staticmethod
        def forward(ctx: Any, call: _ReentrantCheckpointCall, input_count: int, *tensors: Any) -> Any:
            """Execute and retain one original checkpoint invocation."""
            del tensors
            ctx.call = call
            ctx.input_count = input_count
            return call.run_forward()

        @staticmethod
        def backward(ctx: Any, *grad_outputs: Any) -> Tuple[Any, ...]:
            """Replay the checkpoint and return its input gradients."""
            sensitivity = grad_outputs[0] if len(grad_outputs) == 1 else grad_outputs
            input_grads = ctx.call.backward(sensitivity)
            if len(input_grads) != ctx.input_count:
                raise RuntimeError(
                    "Checkpoint Tensor input count changed between original forward and replay"
                )
            return (None, None, *input_grads)

    return _ReentrantCheckpointFunction


class ReentrantCheckpointWrapper(ActivationWrapper):
    """Wrap a Torch Module or callable with hook-free reentrant checkpointing."""

    def __init__(
        self,
        module: Callable[..., Any],
        context_fn: Optional[Callable[[], Tuple[Any, Any]]] = None,
        save_rng_state: bool = True,
    ) -> None:
        """Initialize a reentrant checkpoint wrapper."""
        if not callable(module):
            raise ValueError("module must be a Torch Module or callable")
        super().__init__(module)
        self.context_fn = context_fn
        self.save_rng_state = save_rng_state

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Create and execute one per-invocation reentrant checkpoint call."""
        if not torch.is_grad_enabled():
            return self._wrapped_module(*args, **kwargs)  # pylint: disable=not-callable
        _ReentrantCheckpointCall.validate_inputs(args, kwargs)
        call = _ReentrantCheckpointCall(
            self._wrapped_module,
            args,
            kwargs,
            context_fn=self.context_fn,
            save_rng_state=self.save_rng_state,
        )
        input_tensors = tuple(tensor_leaves((args, kwargs)))
        checkpoint_function = _get_reentrant_checkpoint_function()
        return checkpoint_function.apply(call, len(input_tensors), *input_tensors)


def reentrant_checkpoint_wrapper(
    module: Callable[..., Any],
    context_fn: Optional[Callable[[], Tuple[Any, Any]]] = None,
    save_rng_state: bool = True,
) -> ReentrantCheckpointWrapper:
    """Wrap *module* with the experimental hook-free reentrant checkpoint."""
    return ReentrantCheckpointWrapper(module, context_fn=context_fn, save_rng_state=save_rng_state)


__all__ = [
    "ReentrantCheckpointExcludeWrapper",
    "ReentrantCheckpointWrapper",
    "clear_reentrant_recompute_session",
    "reentrant_checkpoint_exclude_wrapper",
    "reentrant_checkpoint_wrapper",
    "reentrant_recompute_handle_collector_ctx",
    "reentrant_recompute_session_ctx",
]
