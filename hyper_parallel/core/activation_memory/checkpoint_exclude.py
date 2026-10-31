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
"""Saved-tensor helpers for regions excluded from checkpoint recomputation."""
import weakref
from collections import defaultdict, deque
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import torch


_InputPath = Tuple[Tuple[str, Any], ...]
_TensorInput = Tuple[_InputPath, Any]


@dataclass(frozen=True)
class _TensorMetadata:
    """Alias-relevant metadata captured while an excluded call is running."""

    tensor_id: int
    storage_ref: Any
    storage_nbytes: int
    dtype: torch.dtype
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    storage_offset: int
    version: int
    itemsize: int
    numel: int
    is_contiguous: bool

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
    dtype: torch.dtype
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
        if self._tensor._version != self.recipe.base_version:  # pylint: disable=protected-access
            raise RuntimeError(
                "Checkpoint-excluded saved tensor was modified in-place after replay materialization"
            )
        return self._tensor


@dataclass(frozen=True)
class _SavedTensorData:
    """Hold one detached saved tensor with its save-time version counter."""

    tensor: torch.Tensor
    version: int


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
    "hyper_parallel_checkpoint_exclude_pack_state",
    default=None,
)


@dataclass(frozen=True)
class _DeviceRNGState:
    """One accelerator device's RNG state captured at a SAVE boundary."""

    device: torch.device
    state: Any


@dataclass(frozen=True)
class _RNGStateSnapshot:
    """CPU and accelerator RNG states captured after one excluded call."""

    cpu_state: Any
    device_states: Tuple[_DeviceRNGState, ...]


@dataclass
class _ExcludeCacheEntry:
    """Store one excluded call's replay value, deferred inputs, and RNG state."""

    output: Any
    input_bindings: List[_InputBinding]
    output_tensor_count: int = 1
    rng_state: Optional[_RNGStateSnapshot] = None
    tape: Any = None

    def release(self) -> None:
        """Release cached output and any retained local backward graph."""
        self.output = None
        self.input_bindings.clear()
        if self.tape is not None:
            self.tape.release()


class _ExcludeCache:
    """Store excluded-region call entries for one checkpoint invocation."""

    def __init__(self) -> None:
        """Initialize an empty per-checkpoint output cache."""
        self._entries: Dict[int, Deque[_ExcludeCacheEntry]] = defaultdict(deque)
        self._retained_entries: List[_ExcludeCacheEntry] = []
        self._save_output_storages: Set[Any] = set()

    def save(self, wrapper_id: int, entry: _ExcludeCacheEntry) -> None:
        """Save one call entry produced by a checkpoint-excluded region."""
        self._entries[wrapper_id].append(entry)
        if entry.tape is not None:
            self._retained_entries.append(entry)

    def pop(self, wrapper_id: int) -> _ExcludeCacheEntry:
        """Return the matching forward call entry during recomputation."""
        entries = self._entries.get(wrapper_id)
        if not entries:
            raise RuntimeError("No cached forward output is available for this checkpoint exclusion wrapper")
        entry = entries.popleft()
        if not entries:
            self._entries.pop(wrapper_id)
        return entry

    def clear(self, release_retained: bool = False) -> None:
        """Release queued outputs and optionally graphs consumed during replay."""
        released = set()
        for entries in self._entries.values():
            for entry in entries:
                if id(entry) in released:
                    continue
                entry.release()
                released.add(id(entry))
        self._entries.clear()
        self._save_output_storages.clear()
        if release_retained:
            for entry in self._retained_entries:
                if id(entry) in released:
                    continue
                entry.release()
                released.add(id(entry))
            self._retained_entries.clear()

    def register_save_outputs(self, output: Any) -> None:
        """Record storage produced by this invocation's SAVE regions."""
        for tensor in _iter_tensor_leaves(output):
            metadata = _tensor_metadata(_canonical_input_tensor(tensor))
            if metadata is not None:
                self._save_output_storages.add(metadata.storage_ref)

    def is_save_output(self, tensor: Any) -> bool:
        """Return whether a tensor aliases an output from this invocation."""
        metadata = _tensor_metadata(_canonical_input_tensor(tensor))
        return metadata is not None and metadata.storage_ref in self._save_output_storages

    def contains_save_output_storage(self, storage_ref: Any) -> bool:
        """Return whether one captured input aliases a SAVE output."""
        return storage_ref in self._save_output_storages


@lru_cache(maxsize=1)
def _get_dtensor_type() -> Optional[type]:
    """Return the local DTensor type without importing it during module import."""
    try:
        from hyper_parallel.core.dtensor.dtensor_base import DTensorBase  # pylint: disable=C0415
    except ImportError:
        return None
    return DTensorBase


def _is_dtensor(tensor: Any) -> bool:
    """Return whether a value is a distributed tensor wrapper."""
    dtensor_type = _get_dtensor_type()
    return dtensor_type is not None and isinstance(tensor, dtensor_type)


def _canonical_input_tensor(tensor: Any) -> Any:
    """Return the local tensor carried by a DTensor wrapper."""
    if _is_dtensor(tensor):
        return tensor.to_local()
    return tensor


def _tensor_metadata(tensor: Any) -> Optional[_TensorMetadata]:
    """Return storage/layout metadata, or ``None`` when aliasing is unavailable."""
    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return None
    try:
        storage = tensor.untyped_storage()
        storage_nbytes = storage.size()
        itemsize = tensor.element_size()
        if storage_nbytes == 0 or itemsize <= 0:
            return None
        return _TensorMetadata(
            tensor_id=id(tensor),
            storage_ref=weakref.ref(storage),
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


def _storage_span(metadata: _TensorMetadata) -> Optional[Tuple[int, int]]:
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
    """Return whether a non-contiguous layout covers every element exactly once."""
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


def _make_exact_recipe(base: _TensorMetadata) -> _ViewRecipe:
    """Build a replay recipe for an unchanged exact input."""
    return _ViewRecipe(
        shape=base.shape,
        stride=base.stride,
        relative_offset=0,
        dtype=base.dtype,
        base_shape=base.shape,
        base_stride=base.stride,
        base_version=base.version,
        base_is_contiguous=base.is_contiguous,
        exact_input=True,
    )


def _make_view_recipe(saved: _TensorMetadata, base: _TensorMetadata) -> Optional[_ViewRecipe]:
    """Return a safe exact-input/view recipe, or ``None`` for a real save."""
    base_storage = base.storage_ref()
    if base_storage is None or base_storage is not saved.storage_ref():
        return None
    if saved.dtype != base.dtype or saved.version != base.version:
        return None

    exact_input = _exact_layout(saved, base)
    if not exact_input:
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


def _match_recompute_input(
    tensor: torch.Tensor,
    inputs: Tuple[_InputInfo, ...],
) -> Optional[Tuple[_InputInfo, _ViewRecipe]]:
    """Match a saved tensor to an excluded-call input by storage and layout."""
    if _is_dtensor(tensor):
        return None
    for input_info in inputs:
        base = input_info.metadata
        if id(tensor) != base.tensor_id:
            continue
        if (
            tensor.dtype != base.dtype
            or tuple(tensor.shape) != base.shape
            or tuple(tensor.stride()) != base.stride
            or tensor.storage_offset() != base.storage_offset
            or tensor._version != base.version  # pylint: disable=protected-access
        ):
            return None
        return input_info, _make_exact_recipe(base)

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
        return min(views, key=lambda item: item[0].metadata.numel)
    return None


def _pack_saved_tensor(tensor: Any) -> Any:
    """Return a replayable input-alias handle or detached tensor data."""
    pack_state = _ACTIVE_PACK_STATE.get()
    if pack_state is not None:
        if (
            isinstance(tensor, torch.nn.Parameter)
            or isinstance(tensor._base, torch.nn.Parameter)  # pylint: disable=protected-access
        ):
            return _SavedTensorData(tensor.detach(), tensor._version)  # pylint: disable=protected-access
        matched = _match_recompute_input(tensor, pack_state.inputs)
        if matched is not None:
            input_info, recipe = matched
            handle = _RecomputedInputHandle(recipe)
            pack_state.bindings.append(_InputBinding(input_info.path, handle))
            return handle
    return _SavedTensorData(tensor.detach(), tensor._version)  # pylint: disable=protected-access


def _unpack_saved_tensor(value: Any) -> Any:
    """Restore the saved tensor for backward."""
    if isinstance(value, _RecomputedInputHandle):
        return value.get_recomputed_tensor()
    if isinstance(value, _SavedTensorData):
        if value.tensor._version != value.version:  # pylint: disable=protected-access
            raise RuntimeError("Checkpoint-excluded saved tensor was modified in-place after it was saved")
        return value.tensor
    return value


def _saved_tensors_context() -> Any:
    """Create an inner hook that stores real tensors instead of outer holders."""
    return torch.autograd.graph.saved_tensors_hooks(_pack_saved_tensor, _unpack_saved_tensor)


_EXCLUDE_CACHE_KEY = object()


def _append_tensor_inputs(value: Any, path: _InputPath, leaves: List[_TensorInput]) -> None:
    """Append tensor leaves without creating a self-referential local function."""
    if isinstance(value, torch.Tensor):
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


def _iter_tensor_leaves(value: Any) -> List[torch.Tensor]:
    """Return tensor leaves from one structured excluded-region value."""
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        leaves = []
        for item in value:
            leaves.extend(_iter_tensor_leaves(item))
        return leaves
    if isinstance(value, dict):
        leaves = []
        for item in value.values():
            leaves.extend(_iter_tensor_leaves(item))
        return leaves
    return []


def _capture_recompute_inputs(
    cache: _ExcludeCache,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Tuple[_InputInfo, ...]:
    """Capture canonical local input storage/layout without retaining tensors."""
    inputs = []
    seen_tensor_ids = set()
    for path, tensor in _collect_tensor_inputs(args, kwargs):
        if isinstance(tensor, torch.nn.Parameter):
            continue
        canonical = _canonical_input_tensor(tensor)
        if id(canonical) in seen_tensor_ids:
            continue
        metadata = _tensor_metadata(canonical)
        if metadata is None or cache.contains_save_output_storage(metadata.storage_ref):
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
    max_offset = view_offset
    numel = 1
    for size, step in zip(recipe.shape, recipe.stride):
        if size == 0:
            raise RuntimeError("Checkpoint-excluded empty saved views should have been saved normally")
        numel *= size
        max_offset += (size - 1) * step
    if numel == 0 or view_offset < 0 or (max_offset + 1) * metadata.itemsize > metadata.storage_nbytes:
        raise RuntimeError("Checkpoint-excluded saved view is outside the replay input storage")

    return canonical.as_strided(recipe.shape, recipe.stride, view_offset).detach()


def _materialize_recompute_inputs(
    entry: _ExcludeCacheEntry,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> None:
    """Bind saved-alias handles to tensors rebuilt from checkpoint replay inputs."""
    for binding in entry.input_bindings:
        tensor = _resolve_input(args, kwargs, binding.path)
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(
                "Checkpoint replay did not reproduce a tensor input required by a checkpoint-excluded region"
            )
        binding.handle.materialize(_rebuild_saved_alias(tensor, binding.handle.recipe))


def _has_used_input(input_bindings: List[_InputBinding]) -> bool:
    """Return whether the excluded call saved any replayable input alias."""
    return bool(input_bindings)


def _collect_rng_devices(args: Tuple[Any, ...], kwargs: Dict[str, Any], output: Any) -> Tuple[torch.device, ...]:
    """Return distinct accelerator devices touched by one excluded call."""
    devices = []
    seen = set()
    tensors = _iter_tensor_leaves((args, kwargs))
    tensors.extend(_iter_tensor_leaves(output))
    for tensor in tensors:
        device = _canonical_input_tensor(tensor).device
        if device.type in {"cpu", "meta"}:
            continue
        key = (device.type, device.index)
        if key in seen:
            continue
        devices.append(device)
        seen.add(key)
    return tuple(devices)


def _get_device_rng_state(device: torch.device) -> Any:
    """Read one accelerator RNG state when its backend exposes the API."""
    device_module = getattr(torch, device.type, None)
    get_rng_state = getattr(device_module, "get_rng_state", None)
    if not callable(get_rng_state):
        return None
    try:
        return get_rng_state(device)
    except TypeError:
        return get_rng_state()


def _set_device_rng_state(device: torch.device, state: Any) -> None:
    """Restore one accelerator RNG state when its backend exposes the API."""
    device_module = getattr(torch, device.type, None)
    set_rng_state = getattr(device_module, "set_rng_state", None)
    if not callable(set_rng_state):
        return
    try:
        set_rng_state(state, device)
    except TypeError:
        set_rng_state(state)


def _capture_rng_state(
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    output: Any,
    enabled: bool,
) -> Optional[_RNGStateSnapshot]:
    """Snapshot RNG at a SAVE exit so replay can reproduce its state advancement."""
    if not enabled:
        return None
    device_states = []
    for device in _collect_rng_devices(args, kwargs, output):
        state = _get_device_rng_state(device)
        if state is not None:
            device_states.append(_DeviceRNGState(device, state))
    return _RNGStateSnapshot(torch.get_rng_state(), tuple(device_states))


def _restore_rng_state(snapshot: Optional[_RNGStateSnapshot]) -> None:
    """Restore the RNG state that the skipped SAVE body would have produced."""
    if snapshot is None:
        return
    torch.set_rng_state(snapshot.cpu_state)
    for device_state in snapshot.device_states:
        _set_device_rng_state(device_state.device, device_state.state)


@lru_cache(maxsize=1)
def _get_replay_placeholder() -> Any:
    """Create a zero-element placeholder returned by an elided SAVE replay."""
    return torch.empty(0, device="cpu")


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
    return torch.empty(0, device="cpu", requires_grad=True)


@lru_cache(maxsize=1)
def _get_recompute_boundary() -> Any:
    """Create the autograd Function used to trigger outer checkpoint replay."""
    class _RecomputeBoundary(torch.autograd.Function):
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

    return _RecomputeBoundary


def _finalize_save_outputs_impl(
    output: Any,
    add_recompute_boundary: bool,
    tensor_leaf_count: Optional[List[int]],
    tensor_type: type,
) -> Any:
    """Recursively finalize SAVE output leaves."""
    if isinstance(output, tensor_type):
        if tensor_leaf_count is not None:
            tensor_leaf_count[0] += 1
        if add_recompute_boundary:
            output = _get_recompute_boundary().apply(output, _get_recompute_trigger())
        return output
    if isinstance(output, list):
        return [
            _finalize_save_outputs_impl(item, add_recompute_boundary, tensor_leaf_count, tensor_type)
            for item in output
        ]
    if isinstance(output, tuple):
        items = [
            _finalize_save_outputs_impl(item, add_recompute_boundary, tensor_leaf_count, tensor_type)
            for item in output
        ]
        if hasattr(output, "_fields"):
            return type(output)(*items)
        return tuple(items)
    if isinstance(output, dict):
        return type(output)(
            (
                key,
                _finalize_save_outputs_impl(value, add_recompute_boundary, tensor_leaf_count, tensor_type),
            )
            for key, value in output.items()
        )
    return output


def _finalize_save_outputs(
    output: Any,
    add_recompute_boundary: bool,
    tensor_leaf_count: Optional[List[int]] = None,
) -> Any:
    """Apply the required replay boundary and count output tensor leaves."""
    return _finalize_save_outputs_impl(
        output,
        add_recompute_boundary,
        tensor_leaf_count,
        torch.Tensor,
    )
