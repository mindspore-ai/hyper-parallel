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

# Adapted from
# https://github.com/pytorch/pytorch/blob/release/2.6/torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py
# enhanced with activation swap functionality.
# ============================================================================
"""Activation Swap implementation for PyTorch."""
# pylint: disable=W0212, W0613

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Optional, Callable, Any, Union
import types
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
        """Invoke the wrapped callable with the given arguments."""
        return self._fn(*args, **kwargs)


def _is_callable_exempt_from_overlap_check(callable_obj: Callable) -> bool:
    """Return True for callables that cannot be reliably overlap-tracked by object marks."""
    return isinstance(callable_obj, (types.FunctionType, types.BuiltinFunctionType, types.MethodType))


def _iter_wrappable_callable_attrs(module: nn.Module) -> Iterator[tuple[str, Callable]]:
    """Yield public per-instance callable attributes not registered as child modules.

    Plain functions, builtins and bound methods are skipped: these are stateless
    module-level utilities shared by reference across many modules (e.g.
    ``self.act = F.gelu`` repeated in every layer).  They are never standalone
    checkpoint regions, and marking a shared function's ``_is_wrapped`` flag
    would both mutate a global object and falsely flag every sibling module that
    references the same function as an overlapping wrap.  Only per-instance
    callables participate in overlap tracking.
    """
    for attr_name, attr_value in vars(module).items():
        if attr_name.startswith("_") or isinstance(attr_value, nn.Module):
            continue
        if _is_callable_exempt_from_overlap_check(attr_value):
            continue
        if callable(attr_value):
            yield attr_name, attr_value


def _mark_wrapped(obj: Any) -> None:
    try:
        obj._is_wrapped = True  # pylint: disable=W0212
    except (AttributeError, TypeError):
        pass


def _get_wrapped_callable(module: nn.Module) -> Optional[Callable]:
    wrapped_module = getattr(module, _SWAP_WRAPPED_MODULE, None)
    if isinstance(wrapped_module, FuncModule):
        return getattr(wrapped_module, "_fn", None)
    if isinstance(module, FuncModule):
        return getattr(module, "_fn", None)
    return None


def _raise_callable_already_wrapped(callable_obj: Callable) -> None:
    warnings.warn(
        f"Callable '{callable_obj.__class__.__name__}' is already wrapped. "
        "Wrapping overlapping module regions is not allowed."
    )


def _check_callable_attr_not_wrapped(owner: nn.Module, attr_name: str, attr_value: Callable) -> None:
    del owner, attr_name
    if getattr(attr_value, '_is_wrapped', False):
        _raise_callable_already_wrapped(attr_value)


def _check_and_mark_callable(callable_obj: Callable) -> None:
    if _is_callable_exempt_from_overlap_check(callable_obj):
        return
    if getattr(callable_obj, '_is_wrapped', False):
        warnings.warn(
            f"Callable '{callable_obj.__class__.__name__}' or one of its ancestors is already wrapped. "
            "Wrapping overlapping module regions is not allowed."
        )
    _mark_wrapped(callable_obj)


def _check_and_mark_wrapped(module: nn.Module) -> None:
    """Validate no wrapping overlap, then mark module and all descendants as wrapped."""
    if getattr(module, '_is_wrapped', False):
        warnings.warn(
            f"Module '{module.__class__.__name__}' or one of its ancestors is already wrapped. "
            "Wrapping overlapping module regions is not allowed."
        )
    for submodule in module.modules():
        if submodule is module:
            continue
        wrapped_callable = _get_wrapped_callable(submodule)
        if wrapped_callable is not None and _is_callable_exempt_from_overlap_check(wrapped_callable):
            continue
        if getattr(submodule, '_is_wrapped', False):
            if wrapped_callable is not None:
                _raise_callable_already_wrapped(wrapped_callable)
            warnings.warn(
                f"Submodule '{getattr(submodule, '_swap_wrapped_module', submodule).__class__.__name__}' of "
                f"'{module.__class__.__name__}' is already wrapped. "
                "Wrapping overlapping module regions is not allowed."
            )
    for submodule in module.modules():
        for attr_name, attr_value in _iter_wrappable_callable_attrs(submodule):
            _check_callable_attr_not_wrapped(submodule, attr_name, attr_value)
    for submodule in module.modules():
        _mark_wrapped(submodule)
        for _, attr_value in _iter_wrappable_callable_attrs(submodule):
            _mark_wrapped(attr_value)


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
            if policy_fn is not None:
                if policy_fn(tensor) == CheckpointPolicy.MUST_SAVE:
                    return tensor
                if policy_fn(tensor) != CheckpointPolicy.MUST_SWAP:
                    raise RuntimeError(f"Swap :set an invalid policy {policy_fn(tensor)}")
            group_name = swap_manager.get_current_group_name()
            if not group_name:
                return tensor
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

    def __init__(self, module: Union[nn.Module, Callable], *, track_overlaps: bool = True):
        """Initialize a wrapper and optionally participate in overlap tracking."""
        if callable(module) and not isinstance(module, nn.Module):
            if track_overlaps:
                _check_and_mark_callable(module)
            module = FuncModule(module)
            if track_overlaps:
                _mark_wrapped(module)
        elif track_overlaps:
            _check_and_mark_wrapped(module)
        super().__init__()
        self._swap_wrapped_module = module
        self._is_wrapped = track_overlaps
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
        """Run the wrapped module's forward pass with activation swapping. Must be implemented by subclasses."""
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

    def named_modules(
        self,
        memo: Optional[set[nn.Module]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, nn.Module]]:
        """
        Yield wrapped-module children without exposing the internal wrapper prefix.

        PyTorch parent modules implement ``named_parameters(recurse=True)`` by
        iterating ``named_modules()`` and reading each module's direct
        ``_parameters``. They do not call child modules' ``named_parameters()``
        overrides. Exposing the wrapped module under the wrapper's own prefix
        keeps root-module traversals aligned with ``state_dict()`` keys.

        Args:
            memo (Optional[set[nn.Module]], optional): A memo set to avoid infinite recursion. Default: ``None``.
            prefix (str, optional): A prefix to prepend to all module names. Default: ``""``.
            remove_duplicate (bool, optional): Whether to remove duplicate modules. Default: ``True``.

        Returns:
            Iterator[tuple[str, nn.Module]] An iterator of (name, module) pairs.
        """
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
        yield from self._swap_wrapped_module.named_modules(
            memo=memo,
            prefix=prefix,
            remove_duplicate=remove_duplicate,
        )

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
        """Run the wrapped module inside an AsyncSaveOnCpu context for activation swapping."""
        with AsyncSaveOnCpu(policy_fn=self.policy_fn, group_swap=self.group_swap):
            return self._swap_wrapped_module(*args, **kwargs)


def swap_wrapper(
    module: Union[nn.Module, Callable],
    policy_fn: Optional[Callable] = None,
    group_swap: bool = False,
) -> SwapWrapper:
    """Wrap a module or callable with activation swap functionality."""
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
        swap_manager.add_storage(group_name, storage)
    return wrapped
