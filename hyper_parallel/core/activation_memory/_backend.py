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
"""PyTorch device primitives used by the activation-swap implementation."""

__all__ = [
    "Tensor",
    "get_device_handle",
    "new_stream",
    "get_stream_context",
    "get_current_stream",
    "new_event",
    "no_grad",
    "preserve_version_counter",
    "cat",
    "empty_like",
    "tree_map",
    "get_element_size",
    "alloc_tensor_buffer",
    "register_forward_pre_hook",
    "register_full_backward_hook",
    "register_full_backward_pre_hook",
]

import torch
from torch import Tensor


def get_device_handle(device_type: str = "npu"):
    """Return the torch device module (e.g. ``torch.npu`` or ``torch.cuda``)."""
    try:
        return getattr(torch, device_type)
    except AttributeError as e:
        raise RuntimeError(f"expect got device handle: 'torch.{device_type}' failed.") from e


def new_stream():
    """Create a new device stream on the current accelerator."""
    return get_device_handle().Stream()


def get_stream_context():
    """Return the stream context manager (``torch.npu.stream`` / ``torch.cuda.stream``)."""
    return get_device_handle().stream


def get_current_stream():
    """Return the current device stream."""
    return get_device_handle().current_stream()


def new_event():
    """Create a new device event on the current accelerator."""
    return get_device_handle().Event()


def no_grad():
    """Return ``torch.no_grad()``."""
    return torch.no_grad()


def preserve_version_counter(tensor):
    """Temporarily keep the tensor version counter unchanged across an in-place write."""
    return torch.autograd._unsafe_preserve_version_counter(tensor)  # pylint: disable=W0212


def cat(tensors, dim=0):
    """Concatenate tensors along *dim*."""
    return torch.cat(tensors, dim=dim)


def empty_like(tensor, *, dtype=None, device=None, pin_memory=False):
    """Allocate an uninitialized tensor shaped like *tensor*."""
    return torch.empty_like(tensor, dtype=dtype, device=device, pin_memory=pin_memory)


def tree_map(fn, tree):
    """Apply *fn* to every leaf of *tree* and rebuild the same structure."""
    return torch.utils._pytree.tree_map(fn, tree)  # pylint: disable=W0212


def get_element_size(tensor) -> int:
    """Return the size in bytes of one element of *tensor*."""
    return tensor.element_size()


def alloc_tensor_buffer(numel: int, dtype, device="cpu", pin_memory: bool = False):
    """Allocate an uninitialized 1-D tensor buffer."""
    if pin_memory:
        return torch.empty(numel, dtype=dtype, device="cpu", pin_memory=True)
    return torch.empty(numel, dtype=dtype, device=device)


def register_forward_pre_hook(module, hook, prepend=False, with_kwargs=False):
    """Register a forward pre-hook on *module*, ignoring *prepend*."""
    del prepend
    return module.register_forward_pre_hook(hook, with_kwargs=with_kwargs)


def register_full_backward_hook(module, hook, prepend=False):
    """Register a full backward hook on *module*, ignoring *prepend*."""
    del prepend
    return module.register_full_backward_hook(hook)


def register_full_backward_pre_hook(module, hook, prepend=False):
    """Register a full backward pre-hook on *module*, ignoring *prepend*."""
    del prepend
    return module.register_full_backward_pre_hook(hook)
