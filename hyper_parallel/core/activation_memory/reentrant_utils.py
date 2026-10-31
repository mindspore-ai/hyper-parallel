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
"""Shared helpers for hook-free reentrant activation checkpointing."""
import contextlib
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.checkpoint import DefaultDeviceType


@dataclass(frozen=True)
class RngState:
    """Store CPU and initialized accelerator default-generator states."""

    cpu_state: Any
    device_states: Dict[Tuple[str, int], Any]


@dataclass(frozen=True)
class AutocastState:
    """Autocast settings captured during the original checkpoint forward."""

    device_type: str
    device_kwargs: Optional[Dict[str, Any]]
    cpu_kwargs: Dict[str, Any]


def is_tensor(value: Any) -> bool:
    """Return whether *value* is a Torch Tensor."""
    return torch.is_tensor(value)


def tensor_leaves(value: Any) -> List[Any]:
    """Collect Tensor leaves from supported Python containers."""
    if is_tensor(value):
        return [value]
    leaves = []
    if isinstance(value, (tuple, list)):
        for item in value:
            leaves.extend(tensor_leaves(item))
    elif isinstance(value, dict):
        for item in value.values():
            leaves.extend(tensor_leaves(item))
    return leaves


def map_tensors(function: Callable[[Any], Any], value: Any) -> Any:
    """Apply *function* to Tensor leaves while preserving container types."""
    if is_tensor(value):
        return function(value)
    if isinstance(value, list):
        return [map_tensors(function, item) for item in value]
    if isinstance(value, tuple):
        items = [map_tensors(function, item) for item in value]
        if hasattr(value, "_fields"):
            return type(value)(*items)
        return tuple(items)
    if isinstance(value, dict):
        return type(value)((key, map_tensors(function, item)) for key, item in value.items())
    return value


def detach_tree(value: Any, requires_grad: Optional[bool] = None) -> Any:
    """Detach Tensor leaves and optionally override differentiable leaves."""
    def detach(tensor: Any) -> Any:
        detached = tensor.detach()
        differentiable = detached.is_floating_point() or detached.is_complex()
        if differentiable:
            detached.requires_grad_(tensor.requires_grad if requires_grad is None else requires_grad)
        return detached

    return map_tensors(detach, value)


def validate_output(output: Any) -> None:
    """Validate the output structure supported by the custom bridge."""
    if is_tensor(output):
        return
    if isinstance(output, (tuple, list)) and output and all(is_tensor(item) for item in output):
        return
    raise ValueError(
        "reentrant checkpoint currently requires a Tensor or a non-empty flat tuple/list of Tensors output"
    )


def normalize_sensitivities(output: Any, sensitivity: Any) -> Tuple[Any, ...]:
    """Flatten sensitivities and materialize zeros for unused outputs."""
    output_tensors = tuple(tensor_leaves(output))
    if is_tensor(output):
        sensitivity_tensors = (sensitivity,)
    elif isinstance(sensitivity, (tuple, list)):
        sensitivity_tensors = tuple(sensitivity)
    else:
        sensitivity_tensors = tuple(tensor_leaves(sensitivity))
    if len(output_tensors) != len(sensitivity_tensors):
        raise RuntimeError(
            "The number of reentrant checkpoint output sensitivities does not match its Tensor outputs: "
            f"expected {len(output_tensors)}, but got {len(sensitivity_tensors)}"
        )
    return tuple(
        torch.zeros_like(tensor) if grad is None else grad
        for tensor, grad in zip(output_tensors, sensitivity_tensors)
    )


def differentiable_mask(tensors: Sequence[Any]) -> Tuple[bool, ...]:
    """Return which tensors can participate in autograd."""
    return tuple(tensor.requires_grad and (tensor.is_floating_point() or tensor.is_complex()) for tensor in tensors)


def check_backward_validity(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    """Warn when native-style reentrant checkpointing cannot create a grad edge."""
    if not any(tensor.requires_grad for tensor in tensor_leaves((args, kwargs))):
        warnings.warn("None of the inputs have requires_grad=True. Gradients will be None", stacklevel=3)


def infer_device_type(values: Any) -> str:
    """Return the preferred non-CPU device type found in checkpoint inputs."""
    device_types = [tensor.device.type for tensor in tensor_leaves(values) if tensor.device.type != "cpu"]
    device_types_set = set(device_types)
    if len(device_types_set) > 1:
        warnings.warn(
            "Hyper reentrant checkpoint received tensors on multiple non-CPU device types. RNG state is "
            "preserved for every initialized default generator, while autocast follows CUDA when present "
            "and otherwise the first device type encountered.",
            stacklevel=3,
        )
    if not device_types:
        return DefaultDeviceType.get_device_type()
    if "cuda" in device_types_set:
        return "cuda"
    return device_types[0]


def capture_rng_state(values: Any) -> RngState:
    """Capture default RNG states for CPU and Tensor devices in *values*."""
    device_states = {}
    devices = {
        (tensor.device.type, tensor.device.index or 0)
        for tensor in tensor_leaves(values)
        if tensor.device.type != "cpu"
    }
    for device_type, device_index in devices:
        device_module = getattr(torch, device_type, None)
        is_initialized = getattr(device_module, "is_initialized", None)
        if device_module is None or (callable(is_initialized) and not is_initialized()):
            continue
        device_states[(device_type, device_index)] = device_module.get_rng_state(device_index)
    return RngState(torch.get_rng_state(), device_states)


def restore_rng_state(state: Optional[RngState]) -> None:
    """Restore a state captured by :func:`capture_rng_state`."""
    if state is None:
        return
    torch.set_rng_state(state.cpu_state)
    for (device_type, device_index), device_state in state.device_states.items():
        getattr(torch, device_type).set_rng_state(device_state, device_index)


def capture_autocast_state(values: Any) -> AutocastState:
    """Capture CPU and active-device autocast settings."""
    device_type = infer_device_type(values)
    device_kwargs = None
    if torch.amp.is_autocast_available(device_type):
        device_kwargs = {
            "enabled": torch.is_autocast_enabled(device_type),
            "dtype": torch.get_autocast_dtype(device_type),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }
    return AutocastState(
        device_type=device_type,
        device_kwargs=device_kwargs,
        cpu_kwargs={
            "enabled": torch.is_autocast_enabled("cpu"),
            "dtype": torch.get_autocast_dtype("cpu"),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        },
    )


@contextlib.contextmanager
def autocast_context(state: AutocastState) -> Iterator[None]:
    """Restore captured autocast settings around replay."""
    device_context = contextlib.nullcontext()
    if state.device_kwargs is not None:
        device_context = torch.amp.autocast(device_type=state.device_type, **state.device_kwargs)
    with device_context, torch.amp.autocast("cpu", **state.cpu_kwargs):
        yield


class AutogradTape:
    """Retain one eager autograd graph and expose a backward helper."""

    def __init__(self, output: Any, inputs: Sequence[Any]) -> None:
        """Initialize a tape from recorded output and gradient targets."""
        validate_output(output)
        self.output = output
        self.input_count = len(inputs)
        self._input_mask = differentiable_mask(inputs)
        self._input_grads = [None] * self.input_count
        self._released = False
        for index, tensor in enumerate(inputs):
            if self._input_mask[index]:
                tensor.register_hook(self._make_input_grad_hook(index))

    def _make_input_grad_hook(self, index: int) -> Callable[[Any], Any]:
        """Create a hook that captures one replay input gradient."""
        def hook(grad: Any) -> Any:
            self._input_grads[index] = grad
            return grad

        return hook

    @classmethod
    def record(cls, function: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> "AutogradTape":
        """Execute *function* with eager grad recording enabled."""
        with torch.enable_grad():
            output = function(*args, **kwargs)
        return cls(output, tensor_leaves((args, kwargs)))

    def backward(self, sensitivity: Any, keep_graph: bool) -> Tuple[Any, ...]:
        """Backward through the retained graph and return input gradients."""
        if self._released:
            raise RuntimeError("The retained reentrant checkpoint graph has already been released")
        outputs = tuple(tensor_leaves(self.output))
        grad_outputs = normalize_sensitivities(self.output, sensitivity)
        outputs_with_grad = []
        grad_outputs_with_grad = []
        for output, grad_output in zip(outputs, grad_outputs):
            if output.requires_grad:
                outputs_with_grad.append(output)
                grad_outputs_with_grad.append(grad_output)
        if not outputs_with_grad:
            raise RuntimeError("none of output has requires_grad=True, this checkpoint is not necessary")
        self._input_grads = [None] * self.input_count
        torch.autograd.backward(outputs_with_grad, grad_outputs_with_grad, retain_graph=keep_graph)
        input_grads = tuple(grad if differentiable else None for grad, differentiable in zip(
            self._input_grads,
            self._input_mask,
        ))
        if not keep_graph:
            self.release()
        return input_grads

    def release(self) -> None:
        """Release all Python references keeping the eager graph alive."""
        self.output = None
        self._input_grads = []
        self._input_mask = ()
        self.input_count = 0
        self._released = True
