# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

# Adapted from
# hyper_parallel/platform/torch/activation_checkpoint/activation_swap.py
# adapted for MindSpore Cell API.
# ============================================================================
"""Activation Swap Wrapper implementation for MindSpore."""
import enum
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Optional, Callable, Any

import mindspore as ms
from mindspore import Tensor
from mindspore.common.parameter import Parameter
from mindspore.nn import Cell

from hyper_parallel.core.activation_checkpoint.swap import Storage, SwapManager, SwapTensor


_CKPT_WRAPPED_MODULE = "_ckpt_wrapped_module"
_CKPT_PREFIX = _CKPT_WRAPPED_MODULE + "."


class ActivationWrapper(Cell, ABC):
    """
    Base class for Activation Checkpoint Wrapper in MindSpore.

    Wraps a :class:`mindspore.nn.Cell` and forwards attribute lookups,
    parameter iteration, and indexing to the inner cell.  Concrete
    sub-classes must implement :meth:`construct`.

    Not meant to be instantiated directly.
    """

    def __init__(self, module: Cell):
        super().__init__(auto_prefix=False)
        self._ckpt_wrapped_module = module
        self._wrapped_param_names = {
            id(param): param.name for _, param in module.parameters_and_names()
        }

    @abstractmethod
    def construct(self, *args, **kwargs):
        raise ValueError("Subclasses should implement construct().")

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to the wrapped cell.

        .. warning::
            Do **not** call ``super().__getattr__(name)`` here.
            MindSpore's ``Cell.__init__`` calls ``hasattr(self, "bprop")`` at
            line 252 of ``cell.py`` *after* ``_cells`` is initialised as an
            empty ``OrderedDict`` but *before* ``ActivationWrapper.__init__``
            has registered ``_ckpt_wrapped_module`` into ``_cells``.  The
            PyTorch ``nn.Module.__init__`` is pure Python and never calls
            ``hasattr`` on ``self``, so this issue does not arise there.

            Using ``super().__getattr__`` here would raise ``AttributeError``
            (``_ckpt_wrapped_module`` not yet in ``_cells``), the fallback
            ``getattr(self._ckpt_wrapped_module, name)`` would access
            ``self._ckpt_wrapped_module`` — triggering another
            ``__getattr__("_ckpt_wrapped_module")`` — and the cycle repeats
            as infinite recursion.

            Instead we replicate ``Cell.__getattr__``'s own dict-probe logic
            and fall through to the wrapped module only when it is already
            registered.
        """
        for attr_dict in ('_params', '_buffers', '_cells', '_params_list'):
            d = self.__dict__.get(attr_dict)
            if d is not None and name in d:
                return d[name]
        cells = self.__dict__.get('_cells', {})
        wrapped = cells.get(_CKPT_WRAPPED_MODULE)
        if wrapped is not None:
            return getattr(wrapped, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the wrapped cell is a SequentialCell."""
        return self._ckpt_wrapped_module.__getitem__(key)  # type: ignore[operator]

    def parameters_and_names(
        self,
        name_prefix: str = '',
        expand: bool = True,
    ) -> Iterator[tuple[str, Parameter]]:
        """
        Override :meth:`parameters_and_names` to strip the wrapper prefix.

        Removes all occurrences of ``_ckpt_wrapped_module.`` from parameter
        names so that a checkpoint saved from this wrapper is compatible with
        the unwrapped cell.

        Args:
            name_prefix (str): Prefix prepended to every parameter name.
            expand (bool): Whether to recursively expand sub-cells.

        Yields:
            tuple[str, Parameter]: ``(name, parameter)`` pairs with the
            wrapper prefix removed.
        """
        for param_name, param in super().parameters_and_names(name_prefix, expand):
            yield param_name.replace(_CKPT_PREFIX, ""), param

    def update_parameters_name(self, prefix='', recurse=True):
        """
        Update wrapped parameter names without collapsing existing full paths.

        When a wrapper replaces an already-registered child cell, the wrapped
        parameters usually already have globally unique names such as
        ``0.attn.qkv.weight``. MindSpore will still call
        ``wrapper.update_parameters_name("attn.")`` during reassignment; if we
        blindly apply that prefix again through the wrapper view, those names
        are rewritten to ``attn.qkv.weight`` and collide across layers.

        For parameters that already contain the requested child prefix in their
        existing full name, keep the current name unchanged. For fresh
        standalone modules that only have local names like ``qkv.weight``,
        synthesize the prefixed name as usual.
        """
        if prefix is None:
            prefix = ''
        for local_name, param in self._ckpt_wrapped_module.parameters_and_names(expand=recurse):
            original_name = self._wrapped_param_names.get(id(param), param.name)
            if prefix and (original_name.startswith(prefix) or f".{prefix}" in original_name):
                new_name = original_name
            elif prefix:
                new_name = prefix + local_name
            else:
                new_name = local_name
            if new_name != param.name:
                param.is_init = False
                param.name = new_name
            self._wrapped_param_names[id(param)] = new_name


class ActivationPolicy(enum.Enum):
    """Per-tensor swap policy returned by a user-supplied policy function."""
    SAVE = 0   # keep tensor on device (do not swap)
    SWAP = 1   # offload tensor to host memory


def base_check_fn(tensor: Any) -> bool:
    """
    Basic eligibility check: returns ``True`` when *tensor* may be offloaded.

    Skips:

    * Non-tensor objects.
    * :class:`~mindspore.common.parameter.Parameter` objects.
    * Empty tensors (zero elements).

    Args:
        tensor: The value to test.

    Returns:
        bool: ``True`` if the tensor is eligible for CPU offloading.
    """
    if not isinstance(tensor, Tensor):
        return False
    if isinstance(tensor, Parameter):
        return False
    if tensor.size == 0:
        return False
    return True


class AsyncSaveOnCpu(ms.saved_tensors_hooks):
    """
    Context manager to offload tensors to CPU during forward pass.
    """
    def __init__(self, policy_fn=None) -> None:
        self.add_to_storage = False
        self.storage = Storage()
        self.count_idx = 0
        self.pack_count = 0
        self.unpack_count = 0
        self.policy_fn = policy_fn

        def pack_to_cpu(tensor: ms.Tensor):
            # skip ineligible tensors
            if not base_check_fn(tensor):
                return tensor

            if (policy_fn is not None) and (policy_fn(tensor)==ActivationPolicy.SAVE):
                return tensor

            if not self.add_to_storage:
                group_name = SwapManager().get_current_group_name()
                SwapManager().add_storage(group_name, self.storage)
                self.add_to_storage = True
            self.storage.swap_storage[self.count_idx].append(SwapTensor(tensor))
            idx = self.count_idx
            self.count_idx += 1
            self.pack_count += 1
            return idx

        def unpack_from_cpu(idx) -> ms.Tensor:
            if isinstance(idx, ms.Tensor):
                return idx

            swap_tensor = self.storage.swap_storage[idx].pop(0)
            tensor = swap_tensor.get_val()
            self.unpack_count += 1
            if self.unpack_count == self.pack_count:
                self.storage = None
            return tensor

        super().__init__(pack_to_cpu, unpack_from_cpu)


class SwapWrapper(ActivationWrapper):
    """
    MindSpore counterpart of :class:`~hyper_parallel.platform.torch
    .activation_checkpoint.activation_swap.SwapWrapper`.

    Wraps a :class:`~mindspore.nn.Cell` and applies async activation swap
    during the forward pass via the platform's ``async_save_on_cpu`` context
    manager.  Falls back to a no-op context when that context is not yet
    available on the current platform.

    Args:
        mod (Cell): The cell whose intermediate activations should be swapped.
        policy_fn (callable, optional): Per-tensor swap policy; see
            :class:`AsyncSaveOnCpu`.

    Example:
        >>> from hyper_parallel.platform.mindspore.activation_checkpoint import swap_wrapper
        >>> model.layers[i].attn = swap_wrapper(model.layers[i].attn, policy_fn)
    """

    def __init__(self, mod: Cell, policy_fn: Optional[Callable] = None):
        super().__init__(mod)
        self.policy_fn = policy_fn

    def construct(self, *args, **kwargs):
        with AsyncSaveOnCpu(policy_fn=self.policy_fn):
            return self._ckpt_wrapped_module(*args, **kwargs)


def swap_wrapper(module: Cell, policy_fn: Optional[Callable] = None) -> SwapWrapper:
    """
    Wrap *module* with async activation swap.
    """
    return SwapWrapper(module, policy_fn)
