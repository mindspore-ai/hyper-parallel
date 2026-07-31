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
"""Eager non-reentrant activation checkpointing for the Torch backend.

The saved-tensor hook algorithm is adapted from ``torch.utils.checkpoint`` in
PyTorch release/2.9. Hyper owns the scheduling and session extensions here so
the implementation can run consistently on PyTorch 2.6, 2.7, and 2.9 without
patching the installed framework.
"""
import contextlib
import contextvars
import threading
import uuid
import warnings
import weakref
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, Generator, Iterator, List, Optional, Tuple

import torch
from torch.utils._pytree import tree_map
from torch.utils.checkpoint import DefaultDeviceType
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.utils.checkpoint import set_checkpoint_early_stop


_DEFAULT_DETERMINISM_MODE = "default"
_RECOMPUTE_COLLECTOR = contextvars.ContextVar("hyper_recompute_collector", default=None)
_RECOMPUTE_SESSION = contextvars.ContextVar("hyper_recompute_session", default=None)
_SESSION_FRAMES: DefaultDict[Any, weakref.WeakSet] = defaultdict(weakref.WeakSet)
_SESSION_FRAMES_LOCK = threading.RLock()


class CheckpointError(RuntimeError):
    """Raised when checkpoint forward and recomputation are inconsistent."""


class _Handle:
    """Identity key for one recomputed saved tensor."""


class _Holder:
    """Saved-tensor placeholder containing handles keyed by recompute session."""

    def __init__(self) -> None:
        """Initialize an empty per-session handle mapping."""
        self.handles: Dict[Any, Optional[_Handle]] = {}


class _StopRecomputationError(Exception):
    """Internal control-flow exception used by early-stop recomputation."""


class _SessionActivation:
    """Control-plane state shared by checkpoint frames in one session scope."""

    def __init__(self, session_id: Any, retain_on_unpack: bool) -> None:
        """Initialize one scoped session activation."""
        self.session_id = session_id
        self.retain_on_unpack = retain_on_unpack
        self.frames: weakref.WeakSet = weakref.WeakSet()


class _NoopSaveInputs(torch.autograd.Function):
    """Save checkpoint inputs without adding a meaningful forward operation."""

    @staticmethod
    def forward(*args: Any) -> Any:
        """Return a dummy output whose grad function retains checkpoint inputs."""
        del args
        return torch.empty((0,))

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
        """Save tensor inputs while retaining non-tensor input structure."""
        del output
        tensor_pairs = [(index, value) for index, value in enumerate(inputs) if isinstance(value, torch.Tensor)]
        tensor_indices, tensors = zip(*tensor_pairs)
        index_to_saved_index = {input_index: saved_index for saved_index, input_index in enumerate(tensor_indices)}
        stored_args = [None if isinstance(value, torch.Tensor) else value for value in inputs]

        def get_args(saved_tensors: Tuple[Any, ...]) -> List[Any]:
            """Reconstruct the original checkpoint arguments."""
            restored_args = [
                saved_tensors[index_to_saved_index[index]] if index in tensor_indices else value
                for index, value in enumerate(stored_args)
            ]
            return restored_args[1:]

        ctx.get_args = get_args
        ctx.save_for_backward(*tensors)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> None:
        """Reject direct backward through the internal input saver."""
        del ctx, grad_outputs
        raise CheckpointError("The internal checkpoint input saver must not be backwarded directly.")


class _CheckpointFrame:
    """State shared by one checkpoint forward and its recomputations."""

    def __init__(self, recompute_fn: Callable, early_stop: bool, metadata_fn: Optional[Callable]) -> None:
        """Initialize frame state captured by the saved-tensor hooks."""
        self.recompute_fn = recompute_fn
        self.input_saver = None
        self.weak_holders: List[weakref.ReferenceType] = []
        self.recomputed: DefaultDict[Any, weakref.WeakKeyDictionary] = defaultdict(weakref.WeakKeyDictionary)
        self.recomp_counter: DefaultDict[Any, int] = defaultdict(int)
        self.is_recomputed: DefaultDict[Any, bool] = defaultdict(bool)
        self.early_stop = early_stop
        self.metadata_fn = metadata_fn
        self.x_metadatas: List[Any] = []
        self.forward_completed = False
        self.ignore_saved_mismatch = False
        self.active_session: Optional[_SessionActivation] = None

    def check_recomputed_tensors_match(self, session_id: Any) -> None:
        """Validate saved-tensor count and metadata after recomputation."""
        if self.ignore_saved_mismatch:
            return
        if len(self.weak_holders) != self.recomp_counter[session_id]:
            raise CheckpointError(
                "Hyper checkpoint saved a different number of tensors during forward and recomputation. "
                f"Forward saved {len(self.weak_holders)} tensors, but recomputation saved "
                f"{self.recomp_counter[session_id]} tensors."
            )

        mismatches = []
        for index, weak_holder in enumerate(self.weak_holders):
            holder = weak_holder()
            if holder is None:
                continue
            handle = holder.handles.get(session_id)
            _internal_assert(handle is not None, "Missing recomputed tensor handle during metadata validation.")
            _internal_assert(
                handle in self.recomputed[session_id],
                "Missing recomputed tensor during metadata validation.",
            )
            recomputed_tensor = self.recomputed[session_id][handle]
            recomputed_metadata = self.metadata_fn(recomputed_tensor)
            if self.x_metadatas[index] != recomputed_metadata:
                mismatches.append((index, self.x_metadatas[index], recomputed_metadata))

        if mismatches:
            details = "\n".join(
                f"tensor {index}: forward={forward_metadata}, recompute={recomputed_metadata}"
                for index, forward_metadata, recomputed_metadata in mismatches
            )
            raise CheckpointError(
                "Hyper checkpoint detected different tensor metadata during recomputation:\n" + details
            )

    def clear_session(self, session_id: Any) -> None:
        """Release all tensors and handles associated with one session."""
        for weak_holder in self.weak_holders:
            holder = weak_holder()
            if holder is not None:
                holder.handles.pop(session_id, None)
        self.recomputed.pop(session_id, None)
        self.recomp_counter.pop(session_id, None)
        self.is_recomputed.pop(session_id, None)


def _bind_session_activation(frame: _CheckpointFrame, activation: _SessionActivation) -> None:
    """Bind one activation to a frame outside the unpack hot path."""
    if frame.active_session is activation:
        return
    if frame.active_session is not None:
        raise CheckpointError("Concurrent recompute sessions on the same checkpoint frame are not supported.")
    frame.active_session = activation
    activation.frames.add(frame)


def _register_session_frame(
    frame: _CheckpointFrame,
    session_id: Any,
    activation: Optional[_SessionActivation] = None,
) -> None:
    """Register a frame for cleanup and bind its current activation when present."""
    with _SESSION_FRAMES_LOCK:
        _SESSION_FRAMES[session_id].add(frame)
        if activation is not None:
            _internal_assert(activation.session_id == session_id, "Session activation key does not match its frame.")
            _bind_session_activation(frame, activation)


def _activate_registered_frames(activation: _SessionActivation) -> None:
    """Install an activation on every frame already registered for its session."""
    with _SESSION_FRAMES_LOCK:
        for frame in list(_SESSION_FRAMES.get(activation.session_id, ())):
            _bind_session_activation(frame, activation)


def _deactivate_session(activation: _SessionActivation) -> None:
    """Remove one activation from every frame bound at context entry."""
    with _SESSION_FRAMES_LOCK:
        for frame in list(activation.frames):
            if frame.active_session is activation:
                frame.active_session = None
        activation.frames.clear()


def _internal_assert(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointError(message)


def _noop_context_fn() -> Tuple[contextlib.AbstractContextManager, contextlib.AbstractContextManager]:
    return contextlib.nullcontext(), contextlib.nullcontext()


def _default_metadata_fn(tensor: Any) -> Dict[str, Any]:
    return {"shape": tensor.shape, "dtype": tensor.dtype, "device": tensor.device}


def _infer_device_type(*args: Any) -> str:
    """Return the preferred non-CPU device type found in checkpoint inputs."""
    device_types = []

    def add_device_type(value: Any) -> None:
        """Record one non-CPU tensor device type."""
        if isinstance(value, torch.Tensor) and value.device.type != "cpu":
            device_types.append(value.device.type)

    tree_map(add_device_type, args)
    device_types_set = set(device_types)
    if len(device_types_set) > 1:
        warnings.warn(
            "Hyper checkpoint received tensors on multiple non-CPU device types. RNG state is preserved only for "
            "one device type; CUDA is preferred when present.",
            stacklevel=3,
        )
    if not device_types:
        return DefaultDeviceType.get_device_type()
    if "cuda" in device_types_set:
        return "cuda"
    return device_types[0]


def _get_device_module(device_type: str) -> Any:
    if device_type == "meta":
        return torch.device("meta")
    return getattr(torch, device_type)


def _get_device_states(device_type: str, *args: Any) -> Tuple[List[int], List[Any]]:
    """Capture RNG states for non-CPU input devices of the requested type."""
    device_ids = []

    def add_device_id(value: Any) -> None:
        """Record one non-CPU tensor device index."""
        if isinstance(value, torch.Tensor) and value.device.type not in {"cpu", "meta"}:
            device_ids.append(value.get_device())

    tree_map(add_device_id, args)
    device_module = _get_device_module(device_type)
    states = []
    for device_id in device_ids:
        with device_module.device(device_id):
            states.append(device_module.get_rng_state())
    return device_ids, states


def _set_device_states(device_type: str, devices: List[int], states: List[Any]) -> None:
    if device_type == "meta":
        return
    device_module = _get_device_module(device_type)
    for device, state in zip(devices, states):
        with device_module.device(device):
            device_module.set_rng_state(state)


def _get_autocast_kwargs(device_type: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Return active autocast settings for the selected device and CPU."""
    device_kwargs = None
    if torch.amp.is_autocast_available(device_type):
        device_kwargs = {
            "enabled": torch.is_autocast_enabled(device_type),
            "dtype": torch.get_autocast_dtype(device_type),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
    cpu_kwargs = {
        "enabled": torch.is_autocast_enabled("cpu"),
        "dtype": torch.get_autocast_dtype("cpu"),
        "cache_enabled": torch.is_autocast_cache_enabled(),
    }
    return device_kwargs, cpu_kwargs


def _create_recomputation_hooks(frame: _CheckpointFrame, session_id: Any) -> Any:
    """Create saved-tensor hooks that retain tensors from one recomputation."""
    frame_ref = weakref.ref(frame)

    def pack_hook(tensor: Any) -> Any:
        """Store recomputed tensors in their forward holders."""
        tensor = tensor.detach() if tensor.requires_grad else tensor
        target_frame = frame_ref()
        _internal_assert(target_frame is not None, "Checkpoint frame was released during recomputation.")
        recompute_index = target_frame.recomp_counter[session_id]
        target_frame.recomp_counter[session_id] += 1

        if recompute_index >= len(target_frame.weak_holders):
            if not target_frame.early_stop and not target_frame.forward_completed:
                target_frame.ignore_saved_mismatch = True
                return tensor
            raise CheckpointError(
                "Hyper checkpoint tried to save more tensors during recomputation than during forward."
            )

        holder = target_frame.weak_holders[recompute_index]()
        if holder is not None:
            _internal_assert(
                holder.handles.get(session_id) is None,
                "A recomputed tensor handle already exists for this session.",
            )
            handle = _Handle()
            holder.handles[session_id] = handle
            target_frame.recomputed[session_id][handle] = tensor

        if target_frame.early_stop and target_frame.recomp_counter[session_id] == len(target_frame.weak_holders):
            raise _StopRecomputationError
        return tensor

    def unpack_hook(tensor: Any) -> Any:
        """Return tensors saved by operations inside the recomputation."""
        return tensor

    return torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook)


# PyTorch exposes this tracing guard only as a private decorator.
@torch._disable_dynamo  # pylint: disable=protected-access
def _run_fn_with_dynamo_disabled(function: Callable, *args: Any, **kwargs: Any) -> Any:
    """Run recomputation without tracing the saved-tensor unpack hook with Dynamo."""
    return function(*args, **kwargs)


def _run_recomputation(frame: _CheckpointFrame, session_id: Any) -> None:
    """Run and validate a frame recomputation for the given session."""
    if frame.is_recomputed[session_id]:
        return

    activation = frame.active_session
    if activation is not None:
        _internal_assert(activation.session_id == session_id, "Active session key does not match recomputation key.")
    previous_activation = _RECOMPUTE_SESSION.get()
    token = None
    if activation is not None and previous_activation is not activation:
        token = _RECOMPUTE_SESSION.set(activation)
    try:
        input_context = frame.input_saver.grad_fn
        args = input_context.get_args(input_context.saved_tensors)
        try:
            with _create_recomputation_hooks(frame, session_id), torch.autograd.enable_grad():
                _run_fn_with_dynamo_disabled(frame.recompute_fn, *args)
        except _StopRecomputationError:
            pass
    finally:
        if token is not None:
            _RECOMPUTE_SESSION.reset(token)
    frame.is_recomputed[session_id] = True
    frame.check_recomputed_tensors_match(session_id)


def _create_checkpoint_hooks(frame: _CheckpointFrame) -> Any:
    """Create hooks that lazily recompute tensors saved during forward."""
    def pack_hook(tensor: Any) -> _Holder:
        """Replace a forward saved tensor with an opaque holder."""
        holder = _Holder()
        frame.weak_holders.append(weakref.ref(holder))
        if frame.metadata_fn is not None:
            with torch.no_grad():
                frame.x_metadatas.append(frame.metadata_fn(tensor))
        return holder

    def unpack_hook(holder: _Holder) -> Any:
        """Return the corresponding tensor from lazy or prefired recomputation."""
        activation = frame.active_session
        if activation is not None:
            session_id = activation.session_id
            retain_on_unpack = activation.retain_on_unpack
        else:
            session_id = torch._C._current_graph_task_id()  # pylint: disable=W0212
            if session_id == -1:
                session_id = int(uuid.uuid4())
            retain_on_unpack = False

        _run_recomputation(frame, session_id)
        _internal_assert(session_id in holder.handles, "No recomputed tensor was saved for this checkpoint value.")
        handle = holder.handles[session_id]
        if handle is None:
            raise CheckpointError("A checkpoint tensor was unpacked more than once in the same recompute session.")
        _internal_assert(handle in frame.recomputed[session_id], "The recomputed tensor has already been released.")
        tensor = frame.recomputed[session_id][handle]
        if not retain_on_unpack:
            holder.handles[session_id] = None
        return tensor

    return torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook)


def _is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    return bool(compiler is not None and compiler.is_compiling())


def _native_checkpoint(
    function: Callable,
    *args: Any,
    context_fn: Callable,
    preserve_rng_state: bool,
    determinism_check: str,
    debug: bool,
    early_stop: bool,
    **kwargs: Any,
) -> Any:
    """Use the public native API for compile, adapting 2.6/2.7 early-stop."""
    with set_checkpoint_early_stop(early_stop):
        return torch_checkpoint(
            function,
            *args,
            use_reentrant=False,
            context_fn=context_fn,
            preserve_rng_state=preserve_rng_state,
            determinism_check=determinism_check,
            debug=debug,
            **kwargs,
        )


def _checkpoint_without_reentrant_generator(
    function: Callable,
    preserve_rng_state: bool,
    context_fn: Callable,
    determinism_check: str,
    early_stop: bool,
    *args: Any,
    **kwargs: Any,
) -> Generator[None, None, None]:
    """Set up eager checkpoint state around the caller's forward execution."""
    metadata_functions = {_DEFAULT_DETERMINISM_MODE: _default_metadata_fn, "none": lambda tensor: None}
    if determinism_check not in metadata_functions:
        raise ValueError(
            f"determinism_check must be one of {list(metadata_functions)}, but got {determinism_check!r}."
        )
    metadata_fn = metadata_functions[determinism_check]

    device_type = _infer_device_type(*args)
    device_module = _get_device_module(device_type)
    contexts = context_fn()
    if not isinstance(contexts, tuple) or len(contexts) != 2:
        raise ValueError("context_fn must return a (forward_context, recompute_context) tuple.")
    forward_context, recompute_context = contexts
    device_autocast_kwargs, cpu_autocast_kwargs = _get_autocast_kwargs(device_type)

    had_device_in_forward = False
    forward_devices: List[int] = []
    forward_device_states: List[Any] = []
    forward_cpu_state = None
    if preserve_rng_state:
        forward_cpu_state = torch.get_rng_state()
        if getattr(device_module, "_initialized", False):
            had_device_in_forward = True
            forward_devices, forward_device_states = _get_device_states(device_type, *args)

    def recompute_fn(*inputs: Any) -> None:
        """Restore execution state and rerun the checkpointed function."""
        function_kwargs, *function_args = inputs
        rng_devices = forward_devices if preserve_rng_state and had_device_in_forward else []
        with torch.random.fork_rng(
            devices=rng_devices,
            enabled=preserve_rng_state,
            device_type=device_type,
        ):
            if preserve_rng_state:
                torch.set_rng_state(forward_cpu_state)
                if had_device_in_forward:
                    _set_device_states(device_type, forward_devices, forward_device_states)

            device_autocast_context = contextlib.nullcontext()
            if device_autocast_kwargs is not None:
                device_autocast_context = torch.amp.autocast(device_type=device_type, **device_autocast_kwargs)
            with device_autocast_context, torch.amp.autocast("cpu", **cpu_autocast_kwargs), recompute_context:
                function(*function_args, **function_kwargs)

    frame = _CheckpointFrame(recompute_fn, early_stop, metadata_fn)
    dummy = torch.empty((0,), requires_grad=True)
    frame.input_saver = _NoopSaveInputs.apply(dummy, kwargs, *args)

    if frame.input_saver.grad_fn is None:
        yield
        return

    activation = _RECOMPUTE_SESSION.get()
    if activation is not None:
        raise CheckpointError("Nested checkpoint is not supported during scheduled recomputation.")

    collector = _RECOMPUTE_COLLECTOR.get()
    if collector is not None:
        collector.append(frame)
    try:
        with _create_checkpoint_hooks(frame), forward_context:
            yield
        frame.forward_completed = True

        if getattr(device_module, "_initialized", False) and preserve_rng_state and not had_device_in_forward:
            raise RuntimeError(
                "The device state was initialized inside a Hyper checkpoint forward, so its initial RNG state "
                "could not be preserved. Initialize the device before entering checkpoint."
            )
    except BaseException:
        if collector is not None and frame in collector:
            collector.remove(frame)
        raise


def checkpoint(
    function: Callable,
    *args: Any,
    use_reentrant: bool = False,
    context_fn: Callable = _noop_context_fn,
    preserve_rng_state: bool = True,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    early_stop: bool = True,
    **kwargs: Any,
) -> Any:
    """Run Hyper's non-reentrant checkpoint implementation.

    Eager execution always uses this implementation. Compile execution falls
    back to PyTorch's public non-reentrant checkpoint API.
    """
    if use_reentrant is not False:
        raise ValueError("Hyper checkpoint only supports use_reentrant=False.")
    if not isinstance(early_stop, bool):
        raise ValueError(f"early_stop must be bool, but got {type(early_stop).__name__}.")
    if not isinstance(preserve_rng_state, bool):
        raise ValueError(
            f"preserve_rng_state must be bool, but got {type(preserve_rng_state).__name__}."
        )
    if not callable(context_fn):
        raise ValueError("context_fn must be callable.")
    if _is_compiling():
        return _native_checkpoint(
            function,
            *args,
            context_fn=context_fn,
            preserve_rng_state=preserve_rng_state,
            determinism_check=determinism_check,
            debug=debug,
            early_stop=early_stop,
            **kwargs,
        )
    if debug:
        raise ValueError("debug=True is not supported by Hyper eager checkpoint yet.")

    generator = _checkpoint_without_reentrant_generator(
        function,
        preserve_rng_state,
        context_fn,
        determinism_check,
        early_stop,
        *args,
        **kwargs,
    )
    next(generator)
    try:
        result = function(*args, **kwargs)
    except BaseException:
        generator.close()
        raise
    try:
        next(generator)
    except StopIteration:
        return result
    generator.close()
    raise CheckpointError("The internal checkpoint generator yielded more than once.")


@contextlib.contextmanager
def recompute_handle_collector_ctx() -> Iterator[List[Any]]:
    """Collect opaque checkpoint handles created in this context."""
    handles = []
    token = _RECOMPUTE_COLLECTOR.set(handles)
    try:
        yield handles
    finally:
        _RECOMPUTE_COLLECTOR.reset(token)


def recompute_handle(handle: Any, session_id: Any) -> None:
    """Run one collected checkpoint recomputation ahead of backward."""
    if not isinstance(handle, _CheckpointFrame):
        raise ValueError("handle must be produced by recompute_handle_collector_ctx().")
    _validate_session_id(session_id)
    activation = _RECOMPUTE_SESSION.get()
    if (
        activation is not None
        and activation.session_id == session_id
        and activation.retain_on_unpack
    ):
        _register_session_frame(handle, session_id, activation)
        _run_recomputation(handle, session_id)
        return
    if activation is not None:
        raise CheckpointError("recompute_handle cannot enter another active recompute session.")

    _register_session_frame(handle, session_id)
    with recompute_session_ctx(session_id=session_id, retain_on_unpack=True):
        _run_recomputation(handle, session_id)


def _validate_session_id(session_id: Any) -> None:
    if session_id is None:
        raise ValueError("session_id must not be None.")
    try:
        hash(session_id)
    except TypeError as error:
        raise ValueError("session_id must be hashable.") from error


@contextlib.contextmanager
def recompute_session_ctx(session_id: Any, retain_on_unpack: bool = False) -> Iterator[Any]:
    """Select the key and retention policy used by checkpoint unpack hooks."""
    _validate_session_id(session_id)
    if not isinstance(retain_on_unpack, bool):
        raise ValueError(f"retain_on_unpack must be bool, but got {type(retain_on_unpack).__name__}.")
    if _RECOMPUTE_SESSION.get() is not None:
        raise CheckpointError("Nested recompute session contexts are not supported.")
    activation = _SessionActivation(session_id, retain_on_unpack)
    token = _RECOMPUTE_SESSION.set(activation)
    try:
        _activate_registered_frames(activation)
        yield session_id
    finally:
        try:
            _deactivate_session(activation)
        finally:
            _RECOMPUTE_SESSION.reset(token)


def clear_recompute_session(session_id: Any) -> None:
    """Release retained recomputation data for a session; repeated calls are safe."""
    _validate_session_id(session_id)
    with _SESSION_FRAMES_LOCK:
        registered_frames = _SESSION_FRAMES.pop(session_id, None)
        frames = list(registered_frames) if registered_frames is not None else []
    for frame in frames:
        frame.clear_session(session_id)
