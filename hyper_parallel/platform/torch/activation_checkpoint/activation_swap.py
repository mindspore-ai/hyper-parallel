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

# Adapted from https://github.com/pytorch/pytorch/blob/release/2.6/torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py
# enhanced with activation swap functionality.
# ============================================================================
"""Activation Swap implementation for PyTorch."""
# pylint: disable=W0212, W0613

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Optional, Callable, Any, Union
import warnings
import torch
from torch import nn
from torch.distributed.utils import _replace_by_prefix
from hyper_parallel.core.activation_checkpoint.activation_checkpoint import CheckpointPolicy
from hyper_parallel.core.activation_checkpoint.swap import SwapManager, SwapTensor, Storage


_SWAP_WRAPPED_MODULE = "_swap_wrapped_module"
_SWAP_PREFIX = _SWAP_WRAPPED_MODULE + "."


class FuncModule(nn.Module):
    """
    Thin :class:`~torch.nn.Module` adapter that wraps a plain callable.

    Allows ordinary Python functions (or any callable without Module
    parameters) to be passed to :func:`swap_wrapper` and
    :func:`~hyper_parallel.core.activation_checkpoint.checkpoint_wrapper`
    in place of an :class:`~torch.nn.Module`.
    The wrapped function is stored as ``_fn`` and invoked in
    :meth:`forward`; the module has no trainable parameters.

    Args:
        fn (callable): The function to wrap.

    Example:
        >>> wrapped = swap_wrapper(lambda x: x * 2)
    """

    def __init__(self, fn: Callable):
        super().__init__()
        self._fn = fn

    def forward(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def base_check_fn(tensor) -> bool:
    """
    Basic check to determine if a tensor is eligible for offloading.
    - Skip Parameters and their views.
    - Skip empty storage tensors.
    """
    if isinstance(tensor._base, torch.nn.parameter.Parameter) or isinstance(tensor, torch.nn.parameter.Parameter):  # pylint: disable=W0212
        return False
    if tensor.untyped_storage().size() == 0:
        return False
    return True


class AsyncSaveOnCpu(torch.autograd.graph.saved_tensors_hooks):
    """
    Context manager to offload tensors to CPU during forward pass.
    """
    def __init__(self, policy_fn=None, group_swap: bool = False) -> None:
        self.add_to_storage = False
        self.storage = Storage()
        self.count_idx = 0
        self.policy_fn = policy_fn

        # Cache per-context-manager state once to avoid per-tensor singleton lookups.
        swap_manager = SwapManager()

        def pack_to_cpu(tensor: torch.Tensor):
            if not base_check_fn(tensor):
                return tensor
            if (policy_fn is not None) and (policy_fn(tensor) == CheckpointPolicy.MUST_SAVE):
                return tensor
            group_name = swap_manager.get_current_group_name()
            if not self.add_to_storage:
                swap_manager.add_storage(group_name, self.storage)
                self.add_to_storage = True
            funcname = f"{group_name}::{tensor.shape}"
            self.storage[self.count_idx].append(
                SwapTensor(tensor, funcname, group_swap=group_swap)
            )
            self.count_idx += 1
            return tensor

        def unpack_from_cpu(tensor) -> torch.Tensor:
            if self.storage is not None:
                self.storage.clear()
                self.storage = None
            return tensor

        super().__init__(pack_to_cpu, unpack_from_cpu)


class ActivationWrapper(torch.nn.Module, ABC):
    """
    Base class for Activation Swap.

    Not meant to be instantiated directly.
    """

    def __init__(self, module: Union[nn.Module, Callable]):
        if callable(module) and not isinstance(module, nn.Module):
            module = FuncModule(module)
        super().__init__()
        self._swap_wrapped_module = module
        # state_dict post hook to remove prefix to allow loading into a
        # non-swap wrapped module.
        self._register_state_dict_hook(self._post_state_dict_hook)
        # load_state_dict pre-hook to allow loading back into
        # swap-wrapped module.
        self.register_load_state_dict_pre_hook(self._pre_load_state_dict_hook)

    @property
    def _wrapped_module(self):
        return self._swap_wrapped_module

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise ValueError("Subclasses should implement forward().")

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self._swap_wrapped_module, name)

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is a nn.Sequential."""
        return self._swap_wrapped_module.__getitem__(key)  # type: ignore[operator]

    def named_parameters(
        self,
        *args,
        **kwargs,
    ) -> Iterator[tuple[str, torch.nn.Parameter]]:
        """
        Override :meth:`named_parameters()` to intercept parameter names.

        remove all occurrences of ``_SWAP_PREFIX``.
        """
        for param_name, param in super().named_parameters(*args, **kwargs):
            yield param_name.replace(_SWAP_PREFIX, ""), param

    @staticmethod
    def _post_state_dict_hook(
        module: nn.Module,  # pylint: disable=W0613
        state_dict: dict[str, Any],
        prefix: str,
        *args: Any,  # pylint: disable=W0613
    ) -> dict[str, Any]:
        """
        _post_state_dict_hook() is called after the state_dict() of this FSDP module is executed.

        For ``swap_wrapper``, it will strip swap-wrapped module prefix,
        so that this module can be loaded into non-swapped modules.
        It would still be able to be loaded into swap-wrapped modules as this class,
        adds the prefix back before loading the state_dict.
        """
        _replace_by_prefix(state_dict, f"{prefix}{_SWAP_PREFIX}", prefix)
        return state_dict

    @staticmethod
    def _pre_load_state_dict_hook(
        module: nn.Module,
        state_dict: dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> None:
        """
        ``_pre_state_dict_hook` is called before ``self._load_from_state_dict()`` is called.

        For ``swap_wrapper``, it will add back the module
        prefix so that non-swapped modules can be loaded into
        swap_wrapper modules properly.
        """
        _replace_by_prefix(state_dict, prefix, prefix + f"{_SWAP_PREFIX}")


class SwapWrapper(ActivationWrapper):
    """
    Customize an nn.Module wrapper class to add an AsyncSaveOnCpu context manager for the target model.
    """
    def __init__(
        self,
        mod: Union[nn.Module, Callable],
        policy_fn: Optional[Callable] = None,
        group_swap: bool = False,
    ):
        super().__init__(mod)
        self.policy_fn = policy_fn
        self.group_swap = group_swap

    def forward(self, *args, **kwargs):
        with AsyncSaveOnCpu(policy_fn=self.policy_fn, group_swap=self.group_swap):
            return self._swap_wrapped_module(*args, **kwargs)


def swap_wrapper(
    module: Union[nn.Module, Callable],
    policy_fn: Optional[Callable] = None,
    group_swap: bool = False,
) -> SwapWrapper:
    return SwapWrapper(module, policy_fn, group_swap)


def swap_tensor_wrapper(target, tag: Optional[str] = None, group_swap: bool = False):
    """Register selected tensors into the current swap group.

    This helper is intended to be used inside a forward path that already
    participates in the existing swap scheduling managed by ``SwapManager``.
    It preserves the input structure and returns the original tensors.
    """
    swap_manager = SwapManager()
    group_name = swap_manager.get_current_group_name()
    if not group_name:
        warnings.warn(
            f"Tensor {tag} cannot be swapped, for its group is unregistered."
        )
        return target
    if swap_manager.is_last_group(group_name):
        return target

    storage = Storage()
    count_idx = 0

    def _register_tensor(tensor):
        nonlocal count_idx
        if not base_check_fn(tensor):
            return tensor

        tensor_tag = tag or f"{group_name}_swap_tensor"
        funcname = f"{tensor_tag}::{tuple(tensor.shape)}"
        storage[count_idx].append(SwapTensor(tensor, funcname, group_swap=group_swap))
        count_idx += 1
        return tensor

    wrapped = torch.utils._pytree.tree_map(  # pylint: disable=protected-access
        lambda x: _register_tensor(x) if isinstance(x, torch.Tensor) else x,
        target,
    )
    if count_idx > 0:
        _manager.add_storage(group_name, storage)
    return wrapped
