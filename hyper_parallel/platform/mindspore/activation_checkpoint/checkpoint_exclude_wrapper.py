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
"""MindSpore wrapper for regions that should be saved instead of recomputed."""
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import mindspore as ms
from mindspore.common._grad_function import _Function

from hyper_parallel.core.activation_checkpoint.recompute_state import get_recompute_state
from hyper_parallel.platform.mindspore.activation_checkpoint.activation_swap import ActivationWrapper


_RECOMPUTE_INPUT_HANDLE_KEY = "hyper_parallel_recompute_input_handle"
_SAVE_OUTPUT_SOURCE_KEY = "hyper_parallel_save_output_source"
_InputPath = Tuple[Tuple[str, Any], ...]
_TensorInput = Tuple[_InputPath, Any]


class _RecomputedInputHandle:
    """Defer one excluded-region saved input until checkpoint replay."""

    def __init__(self) -> None:
        """Initialize an unused and unresolved handle."""
        self.used = False
        self._tensor = None

    def mark_used(self) -> None:
        """Record that an exclude operation saved this input for backward."""
        self.used = True

    def materialize(self, tensor: Any) -> None:
        """Bind the handle to the matching input produced during replay."""
        self._tensor = tensor

    def get_recomputed_tensor(self) -> Any:
        """Return the replay-produced input for backward."""
        if self._tensor is None:
            raise RuntimeError("Checkpoint-excluded input was requested before recomputation")
        return self._tensor


@dataclass(frozen=True)
class _InputBinding:
    """Map one input path to its deferred saved-tensor handle."""

    path: _InputPath
    handle: _RecomputedInputHandle


@dataclass
class _ExcludeCacheEntry:
    """Store one excluded call's replay value and deferred input bindings."""

    output: Any
    input_bindings: List[_InputBinding]
    output_tensor_count: int = 1


class _ExcludeCache:
    """Store excluded-region call entries for one checkpoint invocation."""

    def __init__(self) -> None:
        """Initialize an empty per-checkpoint output cache."""
        self._entries: Dict[int, Deque[_ExcludeCacheEntry]] = defaultdict(deque)

    def save(self, wrapper_id: int, entry: _ExcludeCacheEntry) -> None:
        """Save one call entry produced by a checkpoint-excluded region."""
        self._entries[wrapper_id].append(entry)

    def pop(self, wrapper_id: int) -> _ExcludeCacheEntry:
        """Return the matching forward call entry during recomputation."""
        entries = self._entries.get(wrapper_id)
        if not entries:
            raise RuntimeError("No cached forward output is available for this checkpoint exclusion wrapper")
        entry = entries.popleft()
        if not entries:
            self._entries.pop(wrapper_id)
        return entry

    def clear(self) -> None:
        """Release outputs not consumed because recomputation stopped early."""
        self._entries.clear()


def _pack_saved_tensor(tensor: Any) -> Any:
    """Return a deferred input handle or detached tensor data."""
    handle = tensor._get_user_data(_RECOMPUTE_INPUT_HANDLE_KEY)  # pylint: disable=protected-access
    if isinstance(handle, _RecomputedInputHandle):
        handle.mark_used()
        return handle
    return tensor.data


def _unpack_saved_tensor(tensor: Any) -> Any:
    """Restore the saved tensor for backward."""
    if isinstance(tensor, _RecomputedInputHandle):
        return tensor.get_recomputed_tensor()
    return tensor


def _saved_tensors_context() -> Any:
    """Create an inner hook that stores real tensors instead of placeholders."""
    return ms.saved_tensors_hooks(_pack_saved_tensor, _unpack_saved_tensor)


_EXCLUDE_CACHE_KEY = object()


def _append_tensor_inputs(
    value: Any,
    path: _InputPath,
    leaves: List[_TensorInput],
) -> None:
    """Append tensor leaves without creating a self-referential local function."""
    if isinstance(value, ms.Tensor):
        leaves.append((path, value))
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _append_tensor_inputs(item, path + (("index", index),), leaves)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _append_tensor_inputs(item, path + (("key", key),), leaves)


def _collect_tensor_inputs(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> List[_TensorInput]:
    """Return tensor leaves and self-describing paths from one excluded-region call."""
    leaves = []

    for index, arg in enumerate(args):
        _append_tensor_inputs(arg, (("arg", index),), leaves)
    for key, value in kwargs.items():
        _append_tensor_inputs(value, (("kwarg", key),), leaves)
    return leaves


def _mark_recompute_inputs(
    invocation_id: object,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Tuple[List[_InputBinding], List[Tuple[Any, Any]]]:
    """Attach deferred handles to inputs that replay reproduces."""
    bindings = []
    previous_handles = []
    seen_tensor_ids = set()
    try:
        for path, tensor in _collect_tensor_inputs(args, kwargs):
            if (
                isinstance(tensor, ms.Parameter)
                or id(tensor) in seen_tensor_ids
                or tensor._get_user_data(_SAVE_OUTPUT_SOURCE_KEY) is invocation_id  # pylint: disable=protected-access
            ):
                continue
            handle = _RecomputedInputHandle()
            previous = tensor._get_user_data(_RECOMPUTE_INPUT_HANDLE_KEY)  # pylint: disable=protected-access
            previous_handles.append((tensor, previous))
            tensor._set_user_data(_RECOMPUTE_INPUT_HANDLE_KEY, handle)  # pylint: disable=protected-access
            bindings.append(_InputBinding(path, handle))
            seen_tensor_ids.add(id(tensor))
    except BaseException:
        _restore_recompute_inputs(previous_handles)
        raise
    return bindings, previous_handles


def _restore_recompute_inputs(previous_handles: List[Tuple[Any, Any]]) -> None:
    """Restore Tensor user data overwritten for one excluded call."""
    for tensor, previous in previous_handles:
        tensor._set_user_data(_RECOMPUTE_INPUT_HANDLE_KEY, previous)  # pylint: disable=protected-access


def _resolve_input(args: Tuple[Any, ...], kwargs: Dict[str, Any], path: _InputPath) -> Any:
    """Resolve one replay input from its forward argument path."""
    root_kind, root_key = path[0]
    value = args[root_key] if root_kind == "arg" else kwargs[root_key]
    for _, token_value in path[1:]:
        value = value[token_value]
    return value


def _materialize_recompute_inputs(
    entry: _ExcludeCacheEntry,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> None:
    """Bind used input handles to tensors produced during checkpoint replay."""
    for binding in entry.input_bindings:
        if not binding.handle.used:
            continue
        tensor = _resolve_input(args, kwargs, binding.path)
        if not isinstance(tensor, ms.Tensor):
            raise RuntimeError(
                "Checkpoint replay did not reproduce a tensor input required by a checkpoint-excluded region"
            )
        binding.handle.materialize(tensor.data)


def _has_used_input(input_bindings: List[_InputBinding]) -> bool:
    """Return whether the excluded call saved any marked input."""
    return any(binding.handle.used for binding in input_bindings)


@lru_cache(maxsize=1)
def _get_replay_placeholder() -> Any:
    """Create a zero-element placeholder returned by an elided SAVE replay."""
    return ms.Tensor([], dtype=ms.float32)


def _make_replay_placeholder_output(tensor_count: int) -> Any:
    """Create one placeholder leaf for each forward output tensor."""
    if tensor_count == 0:
        return ()
    placeholder = _get_replay_placeholder()
    if tensor_count == 1:
        return placeholder
    return (placeholder,) * tensor_count


@lru_cache(maxsize=1)
def _get_recompute_trigger() -> Any:
    """Create the differentiable zero-element input used by recompute boundaries."""
    tensor = ms.Tensor([], dtype=ms.float32)
    tensor.requires_grad_()
    return tensor


class _RecomputeBoundary(_Function):
    """Trigger the outer checkpoint hook before excluded-region backward."""

    @staticmethod
    def forward(ctx: Any, tensor: Any, trigger: Any) -> Any:
        """Save one zero-element outer-hook dependency and return the tensor unchanged."""
        ctx.save_for_backward(trigger)
        return tensor

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Tuple[Any, None]:
        """Trigger dependency unpack and pass the gradient through."""
        _ = ctx.saved_tensors
        return grad_output, None


def _finalize_save_outputs(
    output: Any,
    add_recompute_boundary: bool,
    invocation_id: Optional[object],
    tensor_leaf_count: Optional[List[int]] = None,
) -> Any:
    """Apply the required boundary and SAVE provenance to output tensor leaves."""
    if isinstance(output, ms.Tensor):
        if tensor_leaf_count is not None:
            tensor_leaf_count[0] += 1
        if add_recompute_boundary:
            output = _RecomputeBoundary.apply(output, _get_recompute_trigger())
        if invocation_id is not None:
            output._set_user_data(_SAVE_OUTPUT_SOURCE_KEY, invocation_id)  # pylint: disable=protected-access
        return output
    if isinstance(output, list):
        return [
            _finalize_save_outputs(item, add_recompute_boundary, invocation_id, tensor_leaf_count)
            for item in output
        ]
    if isinstance(output, tuple):
        items = [
            _finalize_save_outputs(item, add_recompute_boundary, invocation_id, tensor_leaf_count)
            for item in output
        ]
        if hasattr(output, "_fields"):
            return type(output)(*items)
        return tuple(items)
    if isinstance(output, dict):
        return type(output)(
            (key, _finalize_save_outputs(value, add_recompute_boundary, invocation_id, tensor_leaf_count))
            for key, value in output.items()
        )
    return output


class CheckpointExcludeWrapper(ActivationWrapper):
    """Exclude a callable region from checkpoint recomputation."""

    def __init__(self, module: Callable[..., Any], *, save_output: bool = True) -> None:
        """Initialize a checkpoint exclusion wrapper for a MindSpore Cell or function."""
        if not callable(module):
            raise ValueError("module must be a MindSpore Cell or callable")
        if not isinstance(save_output, bool):
            raise ValueError(f"save_output must be a bool, got {type(save_output).__name__}")
        super().__init__(module, track_overlaps=False)
        self.save_output = save_output

    def construct(self, *args: Any, **kwargs: Any) -> Any:
        """Execute normally outside recompute and return the cached output in recompute."""
        state = get_recompute_state()
        if state is None:
            return self._ckpt_wrapped_module(*args, **kwargs)
        cache = state.get_resource(_EXCLUDE_CACHE_KEY, _ExcludeCache)
        if state.is_recomputing:
            entry = cache.pop(id(self))
            _materialize_recompute_inputs(entry, args, kwargs)
            output = (
                entry.output
                if self.save_output
                else _make_replay_placeholder_output(entry.output_tensor_count)
            )
            return _finalize_save_outputs(output, _has_used_input(entry.input_bindings), None)

        input_bindings, previous_handles = _mark_recompute_inputs(state.invocation_id, args, kwargs)
        try:
            with _saved_tensors_context():
                output = self._ckpt_wrapped_module(*args, **kwargs)
        finally:
            _restore_recompute_inputs(previous_handles)
        needs_recompute_boundary = _has_used_input(input_bindings)
        tensor_leaf_count = None if self.save_output else [0]
        finalized_output = _finalize_save_outputs(
            output,
            needs_recompute_boundary,
            state.invocation_id,
            tensor_leaf_count,
        )
        replay_output = output if self.save_output else None
        output_tensor_count = 1 if tensor_leaf_count is None else tensor_leaf_count[0]
        cache.save(id(self), _ExcludeCacheEntry(replay_output, input_bindings, output_tensor_count))
        return finalized_output


def checkpoint_exclude_wrapper(
    module: Callable[..., Any],
    *,
    save_output: bool = True,
) -> CheckpointExcludeWrapper:
    """Wrap a MindSpore Cell or function so its region is not recomputed.

    Args:
        module: MindSpore Cell or callable to execute only during the original
            checkpoint forward pass.
        save_output: Whether to retain the region output for checkpoint replay.
            Set this to ``False`` only when the output is passed directly as one
            argument to another checkpoint exclusion wrapper. Default: ``True``.

    Returns:
        A wrapper that saves the callable's autograd tensors and reuses its
        forward output while replaying a non-reentrant checkpoint.

    Note:
        This feature requires MindSpore PyNative mode and a surrounding
        HyperParallel checkpoint configured with ``use_reentrant=False``.
        Nested checkpoint exclusion wrappers are not supported.
    """
    return CheckpointExcludeWrapper(module, save_output=save_output)
