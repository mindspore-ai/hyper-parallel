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
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any, Callable, Deque, Dict, List, Tuple

import mindspore as ms
from mindspore.common._grad_function import _Function

from hyper_parallel.core.activation_checkpoint.recompute_state import get_recompute_state
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.platform.mindspore.activation_checkpoint.activation_swap import ActivationWrapper


_InputPath = Tuple[Tuple[str, Any], ...]
_TensorInput = Tuple[_InputPath, Any]


@dataclass(frozen=True)
class _TensorMetadata:
    """Alias-relevant metadata captured while an excluded call is running."""

    tensor_id: int
    storage_ptr: int
    storage_nbytes: int
    dtype: Any
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    storage_offset: int
    version: int
    itemsize: int
    numel: int
    is_contiguous: bool

    @cached_property
    def storage_key(self) -> Tuple[Any, ...]:
        """Return the backing-buffer revision that alias matching compares."""
        return (self.storage_ptr, self.storage_nbytes, self.dtype, self.version)

    @cached_property
    def layout_key(self) -> Tuple[Any, ...]:
        """Return the dtype/layout identity used to validate a rebuilt alias."""
        return (self.dtype, self.shape, self.stride, self.version, self.is_contiguous)


@dataclass(frozen=True)
class _InputInfo:
    """Describe one canonical local input without retaining its tensor/storage."""

    path: _InputPath
    metadata: _TensorMetadata


@dataclass(frozen=True)
class _ViewRecipe:
    """Rebuild one saved input alias from its matching replay input."""

    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    relative_offset: int
    dtype: Any
    base_shape: Tuple[int, ...]
    base_stride: Tuple[int, ...]
    base_version: int
    base_is_contiguous: bool
    exact_input: bool

    @cached_property
    def base_layout_key(self) -> Tuple[Any, ...]:
        """Return the replay-input layout identity this recipe was built from."""
        return (self.dtype, self.base_shape, self.base_stride, self.base_version, self.base_is_contiguous)


class _RecomputedInputHandle:
    """Defer one excluded-region saved alias until checkpoint replay."""

    def __init__(self, recipe: _ViewRecipe) -> None:
        """Initialize an unused and unresolved handle."""
        self.recipe = recipe
        self._tensor = None

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
    """Map one replay input path to a deferred saved-alias handle."""

    path: _InputPath
    handle: _RecomputedInputHandle


@dataclass
class _PackState:
    """Inputs and matched saved aliases for the active excluded call."""

    inputs: Tuple[_InputInfo, ...]
    bindings: List[_InputBinding]


_ACTIVE_PACK_STATE: ContextVar[Any] = ContextVar(
    "hyper_parallel_checkpoint_exclude_pack_state", default=None
)


@dataclass
class _ExcludeCacheEntry:
    """Store one excluded call's output and deferred input bindings."""

    output: Any
    input_bindings: List[_InputBinding]


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


def _canonical_input_tensor(tensor: Any) -> Any:
    """Return the local tensor carried by a DTensor wrapper."""
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def _tensor_metadata(tensor: Any) -> Any:
    """Return storage/layout metadata, or ``None`` when aliasing is unavailable."""
    if not isinstance(tensor, ms.Tensor) or tensor.numel() == 0:
        return None
    try:
        storage = tensor.untyped_storage()
        storage_ptr = storage.data_ptr()
        storage_nbytes = storage.size()
        itemsize = tensor.itemsize
        if storage_ptr == 0 or storage_nbytes == 0 or itemsize <= 0:
            return None
        return _TensorMetadata(
            tensor_id=id(tensor),
            storage_ptr=storage_ptr,
            storage_nbytes=storage_nbytes,
            dtype=tensor.dtype,
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            storage_offset=tensor.storage_offset(),
            version=tensor._version,  # pylint: disable=protected-access
            itemsize=itemsize,
            numel=tensor.numel(),
            is_contiguous=tensor.is_contiguous(),
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _storage_span(metadata: _TensorMetadata) -> Any:
    """Return inclusive element offsets touched by a non-negative-stride tensor."""
    if metadata.numel == 0 or any(step < 0 for step in metadata.stride):
        return None
    max_offset = metadata.storage_offset
    for size, step in zip(metadata.shape, metadata.stride):
        if size == 0:
            return None
        max_offset += (size - 1) * step
    if metadata.storage_offset < 0 or (max_offset + 1) * metadata.itemsize > metadata.storage_nbytes:
        return None
    return metadata.storage_offset, max_offset


def _exact_layout(saved: _TensorMetadata, base: _TensorMetadata) -> bool:
    """Return whether two tensors have identical dtype and logical layout."""
    return (
        saved.dtype == base.dtype
        and saved.shape == base.shape
        and saved.stride == base.stride
        and saved.storage_offset == base.storage_offset
    )


def _is_non_overlapping_and_dense(metadata: _TensorMetadata) -> bool:
    """Return whether a non-contiguous layout still covers every element exactly once.

    A contiguous tensor is only the special case whose strides already descend
    consecutively; ``transpose`` and ``permute`` are non-contiguous yet still pack
    the same storage densely, so a view of one stays rebuildable through the
    storage-based ``set_`` recipe.  Layouts with holes (a step-2 slice), expanded
    (zero-stride) or overlapping tensors, and negative strides are rejected, since
    replay only reproduces the base's logical elements, not the bytes in a gap.
    """
    expected_stride = 1
    for stride, size in sorted(
        (stride, size)
        for size, stride in zip(metadata.shape, metadata.stride)
        if size > 1
    ):
        if stride != expected_stride:
            return False
        expected_stride *= size
    return True


def _make_view_recipe(saved: _TensorMetadata, base: _TensorMetadata) -> Any:
    """Return a safe exact-input/view recipe, or ``None`` for a real save."""
    if saved.storage_key != base.storage_key:
        return None

    exact_input = _exact_layout(saved, base)
    if not exact_input:
        # Deriving a view needs a hole-free base: replay reproduces only the
        # base's logical elements, so a view reaching into a gap of a strided
        # base (e.g. a step-2 slice) would read storage replay never rebuilt.
        if not base.is_contiguous and not _is_non_overlapping_and_dense(base):
            return None
        saved_span = _storage_span(saved)
        base_span = _storage_span(base)
        if saved_span is None or base_span is None:
            return None
        if saved_span[0] < base_span[0] or saved_span[1] > base_span[1]:
            return None

    return _ViewRecipe(
        shape=saved.shape,
        stride=saved.stride,
        relative_offset=saved.storage_offset - base.storage_offset,
        dtype=saved.dtype,
        base_shape=base.shape,
        base_stride=base.stride,
        base_version=base.version,
        base_is_contiguous=base.is_contiguous,
        exact_input=exact_input,
    )


def _match_recompute_input(tensor: Any, inputs: Tuple[_InputInfo, ...]) -> Any:
    """Match a saved tensor to an excluded-call input by storage and layout."""
    saved = _tensor_metadata(tensor)
    if saved is None:
        return None

    exact_identity = []
    exact_layout = []
    views = []
    for input_info in inputs:
        recipe = _make_view_recipe(saved, input_info.metadata)
        if recipe is None:
            continue
        candidate = (input_info, recipe)
        if recipe.exact_input and saved.tensor_id == input_info.metadata.tensor_id:
            exact_identity.append(candidate)
        elif recipe.exact_input:
            exact_layout.append(candidate)
        else:
            views.append(candidate)

    if exact_identity:
        return exact_identity[0]
    if exact_layout:
        return exact_layout[0]
    if views:
        # Prefer the most specific containing input when several aliases share
        # a flat backing allocation.
        return min(views, key=lambda item: item[0].metadata.numel)
    return None


def _pack_saved_tensor(tensor: Any) -> Any:
    """Return a replayable input-alias handle or detached tensor data."""
    pack_state = _ACTIVE_PACK_STATE.get()
    if pack_state is not None:
        matched = _match_recompute_input(tensor, pack_state.inputs)
        if matched is not None:
            input_info, recipe = matched
            handle = _RecomputedInputHandle(recipe)
            pack_state.bindings.append(_InputBinding(input_info.path, handle))
            return handle
    return tensor.detach()


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


def _capture_recompute_inputs(
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Tuple[_InputInfo, ...]:
    """Capture canonical local input storage/layout without retaining tensors."""
    inputs = []
    seen_tensor_ids = set()
    for path, tensor in _collect_tensor_inputs(args, kwargs):
        if isinstance(tensor, ms.Parameter):
            continue
        canonical = _canonical_input_tensor(tensor)
        if id(canonical) in seen_tensor_ids:
            continue
        metadata = _tensor_metadata(canonical)
        if metadata is None:
            continue
        inputs.append(_InputInfo(path, metadata))
        seen_tensor_ids.add(id(canonical))
    return tuple(inputs)


def _resolve_input(args: Tuple[Any, ...], kwargs: Dict[str, Any], path: _InputPath) -> Any:
    """Resolve one replay input from its forward argument path."""
    root_kind, root_key = path[0]
    value = args[root_key] if root_kind == "arg" else kwargs[root_key]
    for _, token_value in path[1:]:
        value = value[token_value]
    return value


def _rebuild_saved_alias(tensor: Any, recipe: _ViewRecipe) -> Any:
    """Rebuild a detached saved input/view from one replay-produced base."""
    canonical = _canonical_input_tensor(tensor)
    metadata = _tensor_metadata(canonical)
    if metadata is None:
        raise RuntimeError("Checkpoint replay input does not expose usable storage metadata")
    if metadata.layout_key != recipe.base_layout_key:
        raise RuntimeError(
            "Checkpoint replay input layout/version changed for a checkpoint-excluded saved tensor"
        )

    if recipe.exact_input:
        return canonical.detach()

    view_offset = metadata.storage_offset + recipe.relative_offset
    if any(step < 0 for step in recipe.stride):
        raise RuntimeError("Checkpoint-excluded saved tensor has unsupported negative stride")
    max_offset = view_offset
    numel = 1
    for size, step in zip(recipe.shape, recipe.stride):
        if size == 0:
            raise RuntimeError("Checkpoint-excluded empty saved views should have been saved normally")
        numel *= size
        max_offset += (size - 1) * step
    if (
        numel == 0
        or view_offset < 0
        or (max_offset + 1) * metadata.itemsize > metadata.storage_nbytes
    ):
        raise RuntimeError("Checkpoint-excluded saved view is outside the replay input storage")

    restored = canonical.new_empty((0,))
    restored.set_(canonical.untyped_storage(), view_offset, recipe.shape, recipe.stride)
    return restored


def _materialize_recompute_inputs(
    entry: _ExcludeCacheEntry,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> None:
    """Bind saved-alias handles to tensors rebuilt from checkpoint replay inputs."""
    for binding in entry.input_bindings:
        tensor = _resolve_input(args, kwargs, binding.path)
        if not isinstance(tensor, ms.Tensor):
            raise RuntimeError(
                "Checkpoint replay did not reproduce a tensor input required by a checkpoint-excluded region"
            )
        binding.handle.materialize(_rebuild_saved_alias(tensor, binding.handle.recipe))


def _has_used_input(input_bindings: List[_InputBinding]) -> bool:
    """Return whether the excluded call saved any marked input."""
    return bool(input_bindings)


@lru_cache(maxsize=1)
def _get_recompute_trigger() -> Any:
    """Lazily create the reusable zero-element recomputation trigger."""
    return ms.Tensor([], dtype=ms.float32)


class _RecomputeBoundary(_Function):
    """Trigger the outer checkpoint hook before excluded-region backward."""

    @staticmethod
    def forward(ctx: Any, tensor: Any) -> Any:
        """Save one zero-element outer-hook dependency and return the tensor unchanged."""
        ctx.save_for_backward(_get_recompute_trigger())
        return tensor

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        """Trigger dependency unpack and pass the gradient through."""
        _ = ctx.saved_tensors
        return grad_output


def _apply_recompute_boundary(output: Any, input_bindings: List[_InputBinding]) -> Any:
    """Wrap output tensor leaves when deferred inputs require checkpoint replay."""
    if not _has_used_input(input_bindings):
        return output

    if isinstance(output, ms.Tensor):
        return _RecomputeBoundary.apply(output)
    if isinstance(output, list):
        return [_apply_recompute_boundary(item, input_bindings) for item in output]
    if isinstance(output, tuple):
        items = [_apply_recompute_boundary(item, input_bindings) for item in output]
        if hasattr(output, "_fields"):
            return type(output)(*items)
        return tuple(items)
    if isinstance(output, dict):
        return type(output)((key, _apply_recompute_boundary(value, input_bindings)) for key, value in output.items())
    return output


class CheckpointExcludeWrapper(ActivationWrapper):
    """Exclude a callable region from checkpoint recomputation."""

    def __init__(self, module: Callable[..., Any]) -> None:
        """Initialize a checkpoint exclusion wrapper for a MindSpore Cell or function."""
        if not callable(module):
            raise ValueError("module must be a MindSpore Cell or callable")
        super().__init__(module, track_overlaps=False)

    def construct(self, *args: Any, **kwargs: Any) -> Any:
        """Execute normally outside recompute and return the cached output in recompute."""
        state = get_recompute_state()
        if state is None:
            return self._ckpt_wrapped_module(*args, **kwargs)
        cache = state.get_resource(_EXCLUDE_CACHE_KEY, _ExcludeCache)
        if state.is_recomputing:
            entry = cache.pop(id(self))
            _materialize_recompute_inputs(entry, args, kwargs)
            return _apply_recompute_boundary(entry.output, entry.input_bindings)

        input_infos = _capture_recompute_inputs(args, kwargs)
        input_bindings = []
        pack_token = _ACTIVE_PACK_STATE.set(_PackState(input_infos, input_bindings))
        try:
            with _saved_tensors_context():
                output = self._ckpt_wrapped_module(*args, **kwargs)
        finally:
            _ACTIVE_PACK_STATE.reset(pack_token)
        cache.save(id(self), _ExcludeCacheEntry(output, input_bindings))
        return _apply_recompute_boundary(output, input_bindings)


def checkpoint_exclude_wrapper(module: Callable[..., Any]) -> CheckpointExcludeWrapper:
    """Wrap a MindSpore Cell or function so its region is not recomputed.

    Args:
        module: MindSpore Cell or callable to execute only during the original
            checkpoint forward pass.

    Returns:
        A wrapper that saves the callable's autograd tensors and reuses its
        forward output while replaying a non-reentrant checkpoint.

    Note:
        This feature requires MindSpore PyNative mode and a surrounding
        HyperParallel checkpoint configured with ``use_reentrant=False``.
        Nested checkpoint exclusion wrappers are not supported.
        The excluded callable must not mutate its tensor inputs in place. The
        backward-compatibility layer may deliberately preserve version counters,
        so not every in-place write can be detected from ``Tensor._version``.
    """
    return CheckpointExcludeWrapper(module)
