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
"""Excluded regions for hook-free reentrant activation checkpointing."""
import contextlib
import contextvars
from functools import lru_cache
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import torch

from . import checkpoint_exclude as _checkpoint_exclude
from hyper_parallel.core.activation_memory.reentrant_session import get_current_reentrant_session
from hyper_parallel.core.activation_memory.reentrant_utils import (
    AutogradTape,
    detach_tree,
    tensor_leaves,
)


class ReentrantExecutionState:
    """Expose one checkpoint invocation and phase to nested wrappers."""

    def __init__(self, exclude_cache: _checkpoint_exclude._ExcludeCache, recomputing: bool) -> None:
        """Initialize forward or replay state."""
        self.exclude_cache = exclude_cache
        self.recomputing = recomputing


_CURRENT_EXECUTION_STATE: contextvars.ContextVar[Optional[ReentrantExecutionState]] = contextvars.ContextVar(
    "hyper_parallel_torch_reentrant_execution_state",
    default=None,
)


@contextlib.contextmanager
def reentrant_execution_state_ctx(state: ReentrantExecutionState) -> Iterator[None]:
    """Install a checkpoint execution phase in the current dynamic scope."""
    token = _CURRENT_EXECUTION_STATE.set(state)
    try:
        yield
    finally:
        _CURRENT_EXECUTION_STATE.reset(token)


def get_current_execution_state() -> Optional[ReentrantExecutionState]:
    """Return the active reentrant checkpoint execution state, if any."""
    return _CURRENT_EXECUTION_STATE.get()


@lru_cache(maxsize=1)
def _get_exclude_replay_function() -> type:
    """Create the private Torch custom autograd bridge lazily."""
    class _ExcludeReplayFunction(torch.autograd.Function):
        """Return a cached output and connect it to replay inputs in backward."""

        @staticmethod
        def forward(
            ctx: Any,
            entry: _checkpoint_exclude._ExcludeCacheEntry,
            input_count: int,
            *tensors: Any,
        ) -> Any:
            """Return the cached output without executing the excluded module."""
            del tensors
            ctx.entry = entry
            ctx.input_count = input_count
            return detach_tree(entry.output, requires_grad=False)

        @staticmethod
        def backward(ctx: Any, *grad_outputs: Any) -> Tuple[Any, ...]:
            """Route the bridge backward through the original excluded graph."""
            session = get_current_reentrant_session()
            keep_graph = session is not None and session.retain_graph
            sensitivity = grad_outputs[0] if len(grad_outputs) == 1 else grad_outputs
            input_grads = ctx.entry.tape.backward(sensitivity, keep_graph)
            if len(input_grads) != ctx.input_count:
                raise RuntimeError(
                    "Checkpoint exclusion input count changed between original forward and replay"
                )
            return (None, None, *input_grads)

    return _ExcludeReplayFunction


def _apply_exclude_replay(
    entry: _checkpoint_exclude._ExcludeCacheEntry,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    """Apply an autograd bridge from current replay tensors to cached output."""
    if entry.tape is None:
        raise RuntimeError("Reentrant checkpoint exclusion entry is missing its local backward graph")
    _checkpoint_exclude._restore_rng_state(entry.rng_state)
    _checkpoint_exclude._materialize_recompute_inputs(entry, args, kwargs)
    input_tensors = tuple(tensor_leaves((args, kwargs)))
    if len(input_tensors) != entry.tape.input_count:
        raise RuntimeError(
            "Checkpoint exclusion Tensor input count changed between original forward and replay: "
            f"expected {entry.tape.input_count}, but got {len(input_tensors)}"
        )
    if not any(tensor.requires_grad for tensor in input_tensors):
        return detach_tree(entry.output, requires_grad=False)
    replay_function = _get_exclude_replay_function()
    return replay_function.apply(entry, len(input_tensors), *input_tensors)


def run_reentrant_exclude(
    wrapper_id: int,
    module: Callable[..., Any],
    state: ReentrantExecutionState,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    *,
    save_rng_state: bool,
    save_output: bool,
) -> Any:
    """Record one excluded call or bridge its replay to the retained local graph."""
    if not save_output:
        raise ValueError("Reentrant checkpoint exclusion requires save_output=True")
    if state.recomputing:
        entry = state.exclude_cache.pop(wrapper_id)
        return _apply_exclude_replay(entry, args, kwargs)

    detached_args = detach_tree(args, requires_grad=True)
    detached_kwargs = detach_tree(kwargs, requires_grad=True)
    input_infos = _checkpoint_exclude._capture_recompute_inputs(
        state.exclude_cache,
        detached_args,
        detached_kwargs,
    )
    input_bindings = []
    pack_token = _checkpoint_exclude._ACTIVE_PACK_STATE.set(
        _checkpoint_exclude._PackState(input_infos, input_bindings)
    )
    try:
        with _checkpoint_exclude._saved_tensors_context():
            tape = AutogradTape.record(module, detached_args, detached_kwargs)
    finally:
        _checkpoint_exclude._ACTIVE_PACK_STATE.reset(pack_token)
    rng_state = _checkpoint_exclude._capture_rng_state(
        detached_args,
        detached_kwargs,
        tape.output,
        save_rng_state,
    )
    state.exclude_cache.save(
        wrapper_id,
        _checkpoint_exclude._ExcludeCacheEntry(
            tape.output,
            input_bindings,
            rng_state=rng_state,
            tape=tape,
        ),
    )
    return tape.output


__all__ = [
    "ReentrantExecutionState",
    "get_current_execution_state",
    "reentrant_execution_state_ctx",
    "run_reentrant_exclude",
]
