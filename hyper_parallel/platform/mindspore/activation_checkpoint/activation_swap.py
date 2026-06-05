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
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Optional, Callable, Any, Union
import types
import warnings
import mindspore as ms
from mindspore import Tensor
from mindspore.common.parameter import Parameter
from mindspore.nn import Cell


_CKPT_WRAPPED_MODULE = "_ckpt_wrapped_module"


def _strip_ckpt_wrapped_module_prefix(name: str) -> str:
    """Remove the wrapper cell segment from a dotted MindSpore cell name."""
    return ".".join(part for part in name.split(".") if part != _CKPT_WRAPPED_MODULE)


class FuncCell(Cell):
    """
    Thin :class:`~mindspore.nn.Cell` adapter that wraps a plain callable.

    Allows ordinary Python functions (or any callable without Cell
    parameters) to be passed to :func:`checkpoint_wrapper` and
    :func:`swap_wrapper` in place of a :class:`~mindspore.nn.Cell`.
    The wrapped function is stored as ``_fn`` and invoked in
    :meth:`construct`; the cell has no trainable parameters.

    Args:
        fn (callable): The function to wrap.

    Example:
        >>> wrapped = checkpoint_wrapper(lambda x: x * 2)
    """

    def __init__(self, fn: Callable):
        super().__init__()
        self._fn = fn

    def construct(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def _iter_wrappable_callable_attrs(module: Cell) -> Iterator[tuple[str, Callable]]:
    """Yield public per-instance callable attributes not registered as child cells.

    Plain functions, builtins and bound methods are skipped: these are stateless
    module-level utilities shared by reference across many modules (e.g.
    ``self.reshape = mint.reshape`` / ``self.cast = ops.cast`` repeated in every
    layer).  They are never standalone checkpoint regions, and marking a shared
    function's ``_is_wrapped`` flag would both mutate a global object and falsely
    flag every sibling module that references the same function as an overlapping
    wrap.  Only per-instance callables (e.g. MindSpore ``Primitive`` objects)
    participate in overlap tracking.
    """
    for attr_name, attr_value in vars(module).items():
        if attr_name.startswith("_") or isinstance(attr_value, Cell):
            continue
        if isinstance(attr_value, (types.FunctionType, types.BuiltinFunctionType, types.MethodType)):
            continue
        if callable(attr_value):
            yield attr_name, attr_value


def _mark_wrapped(obj: Any) -> None:
    try:
        obj._is_wrapped = True  # pylint: disable=W0212
    except (AttributeError, TypeError):
        pass


def _get_wrapped_callable(cell: Cell) -> Optional[Callable]:
    wrapped_module = getattr(cell, _CKPT_WRAPPED_MODULE, None)
    if isinstance(wrapped_module, FuncCell):
        return getattr(wrapped_module, "_fn", None)
    return None


def _raise_callable_already_wrapped(callable_obj: Callable) -> None:
    raise ValueError(
        f"Callable '{callable_obj.__class__.__name__}' is already wrapped. "
        "Wrapping overlapping module regions is not allowed."
    )


def _check_callable_attr_not_wrapped(owner: Cell, attr_name: str, attr_value: Callable) -> None:
    del owner, attr_name
    if getattr(attr_value, '_is_wrapped', False):
        _raise_callable_already_wrapped(attr_value)


def _check_and_mark_callable(callable_obj: Callable) -> None:
    if getattr(callable_obj, '_is_wrapped', False):
        raise ValueError(
            f"Callable '{callable_obj.__class__.__name__}' or one of its ancestors is already wrapped. "
            "Wrapping overlapping module regions is not allowed."
        )
    _mark_wrapped(callable_obj)


def _check_and_mark_wrapped(module: Cell) -> None:
    """Validate no wrapping overlap, then mark module and all descendants as wrapped.

    Raises:
        ValueError: If ``module`` or any of its descendants is already wrapped.
    """
    if getattr(module, '_is_wrapped', False):
        raise ValueError(
            f"Module '{module.__class__.__name__}' or one of its ancestors is already wrapped. "
            "Wrapping overlapping module regions is not allowed."
        )
    for _, submodule in module.cells_and_names():
        if submodule is module:
            continue
        if getattr(submodule, '_is_wrapped', False):
            wrapped_callable = _get_wrapped_callable(submodule)
            if wrapped_callable is not None:
                _raise_callable_already_wrapped(wrapped_callable)
            raise ValueError(
                f"Submodule '{getattr(submodule, '_ckpt_wrapped_module', submodule).__class__.__name__}' of "
                f"'{module.__class__.__name__}' is already wrapped. "
                "Wrapping overlapping module regions is not allowed."
            )
    for _, submodule in module.cells_and_names():
        for attr_name, attr_value in _iter_wrappable_callable_attrs(submodule):
            _check_callable_attr_not_wrapped(submodule, attr_name, attr_value)
    for _, submodule in module.cells_and_names():
        _mark_wrapped(submodule)
        for _, attr_value in _iter_wrappable_callable_attrs(submodule):
            _mark_wrapped(attr_value)


class ActivationWrapper(Cell, ABC):
    """
    Base class for Activation Checkpoint Wrapper in MindSpore.

    Wraps a :class:`mindspore.nn.Cell` and forwards attribute lookups,
    parameter iteration, and indexing to the inner cell.  Concrete
    sub-classes must implement :meth:`construct`.

    Not meant to be instantiated directly.
    """

    def __init__(self, module: Union[Cell, Callable]):
        if callable(module) and not isinstance(module, Cell):
            _check_and_mark_callable(module)
            module = FuncCell(module)
            _mark_wrapped(module)
        else:
            _check_and_mark_wrapped(module)
        super().__init__(auto_prefix=False)
        self._ckpt_wrapped_module = module
        self._is_wrapped = True
        self._wrapped_param_names = {
            id(param): param.name for _, param in module.parameters_and_names()
        }

    @property
    def _wrapped_module(self):
        return self._ckpt_wrapped_module

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

    @property
    def unwrap_cell(self) -> Cell:
        """Recursively return the innermost wrapped cell."""
        return self._ckpt_wrapped_module

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the wrapped cell is a SequentialCell."""
        return self._ckpt_wrapped_module.__getitem__(key)  # type: ignore[operator]

    def cells_and_names(self, cells=None, name_prefix=''):
        """
        Return wrapped cells without exposing the wrapper storage prefix.

        MindSpore registers ``_ckpt_wrapped_module`` as a real child cell, so
        the default :meth:`Cell.cells_and_names` would expose names such as
        ``layer._ckpt_wrapped_module.attn``.  Strip that implementation detail
        so downstream code sees the same names as it would for the unwrapped
        model.
        """
        for cell_name, cell in super().cells_and_names(cells, name_prefix):
            yield _strip_ckpt_wrapped_module_prefix(cell_name), cell

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
            yield _strip_ckpt_wrapped_module_prefix(param_name), param

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
    if tensor.param_info is not None:
        return False
    if tensor.untyped_storage().size() == 0:
        return False
    return True


def _normalize_device(device: str) -> str:
    if ":" in device:
        return device.split(":", maxsplit=1)[0]
    return device


class AsyncSaveOnCpu(ms.saved_tensors_hooks):
    """
    Context manager to offload tensors to CPU during forward pass.
    """
    def __init__(self, policy_fn=None, group_swap: bool = False) -> None:
        # pylint: disable=C0415
        from hyper_parallel.core.activation_checkpoint.activation_checkpoint import CheckpointPolicy
        from hyper_parallel.core.activation_checkpoint.swap import Storage, SwapManager, SwapTensor
        self.add_to_storage = False
        self.storage = Storage()
        self.count_idx = 0
        self.policy_fn = policy_fn

        # Cache per-context-manager state once to avoid per-tensor singleton lookups.
        swap_manager = SwapManager()
        def pack_to_cpu(tensor: ms.Tensor):
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

        def unpack_from_cpu(tensor) -> ms.Tensor:
            if self.storage is not None:
                self.storage.clear()
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

    def __init__(
        self,
        mod: Union[Cell, Callable],
        policy_fn: Optional[Callable] = None,
        group_swap: bool = False,
    ):
        super().__init__(mod)
        self.policy_fn = policy_fn
        self.group_swap = group_swap

    def construct(self, *args, **kwargs):
        with AsyncSaveOnCpu(policy_fn=self.policy_fn, group_swap=self.group_swap):
            return self._ckpt_wrapped_module(*args, **kwargs)


def swap_wrapper(
    module: Union[Cell, Callable],
    policy_fn: Optional[Callable] = None,
    group_swap: bool = False,
) -> SwapWrapper:
    """
    Wrap *module* with async activation swap.

    Args:
        module (Cell or callable): The cell or plain function to wrap.
            If a plain callable is passed it is automatically wrapped in a
            :class:`FuncCell` before being stored.
        policy_fn (callable, optional): Per-tensor swap policy; see
            :class:`AsyncSaveOnCpu`.

    Returns:
        SwapWrapper: The wrapped cell with activation swap enabled.
    """
    return SwapWrapper(module, policy_fn, group_swap)


def swap_tensor_wrapper(target, tag: Optional[str] = None, group_swap: bool = False):
    """Register selected tensors into the current swap group.

    This helper is intended to be used inside a forward path that already
    participates in the existing swap scheduling managed by ``SwapManager``.
    It preserves the input structure and returns the original tensors.
    """
    # pylint: disable=C0415
    from hyper_parallel.core.activation_checkpoint.swap import Storage, SwapManager, SwapTensor
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

    def _apply(x):
        nonlocal count_idx
        if isinstance(x, Tensor) and base_check_fn(x):
            tensor_tag = tag or f"{group_name}_swap_tensor"
            funcname = f"{tensor_tag}::{tuple(x.shape)}"
            storage[count_idx].append(SwapTensor(x, funcname, group_swap=group_swap))
            count_idx += 1
        return x

    def _map(tree):
        if isinstance(tree, dict):
            return type(tree)((k, _map(v)) for k, v in tree.items())
        if isinstance(tree, tuple):
            return tuple(_map(v) for v in tree)
        if isinstance(tree, list):
            return [_map(v) for v in tree]
        return _apply(tree)

    wrapped = _map(target)
    if count_idx > 0:
        swap_manager.add_storage(group_name, storage)
    return wrapped
