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
# ============================================================================
"""dtensor"""
import copy as cp
import inspect
import warnings
from typing import Any, Callable, Optional, Sequence, Set, Tuple, Union

import numpy as np

from hyper_parallel.core.dtensor.device_mesh import _mesh_resources
from hyper_parallel.core.dtensor.layout import Layout, DeviceMesh, _get_slice_tensor_by_layout
from hyper_parallel.core.dtensor.placement_types import Placement, Replicate
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.platform import PlatformType
from hyper_parallel.core.utils import compute_local_shape_and_global_offset

platform = get_platform()
DTensorBase = platform.DTensorBase
Tensor = platform.Tensor


class SkipDTensorDispatch():
    """Context manager that disables DTensor op dispatch for the enclosed block.

    Args:
        no_skip: Optional set of op callables or canonical op name strings that
            should still be dispatched through DTensor even within this context.
            All other ops bypass DTensor dispatch and operate on local tensors.

    Example:
        >>> import torch
        >>> with SkipDTensorDispatch(no_skip={torch.zeros_like}):
        ...     # zeros_like still goes through DTensor dispatch;
        ...     # everything else uses the local tensor path.
        ...     result = torch.zeros_like(dtensor)
    """

    def __init__(self, no_skip: Optional[Set] = None):
        self._no_skip_names: Set[str] = set()
        if no_skip:
            for op in no_skip:
                if isinstance(op, str):
                    self._no_skip_names.add(op)
                else:
                    self._no_skip_names.add(platform.get_op_name(op))

    def __enter__(self):
        # pylint: disable=C0415
        from hyper_parallel.core.shard._op_dispatch import disable_dtensor_dispatch, add_no_skip_ops
        disable_dtensor_dispatch()
        if self._no_skip_names:
            add_no_skip_ops(self._no_skip_names)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # pylint: disable=C0415
        from hyper_parallel.core.shard._op_dispatch import enable_dtensor_dispatch, remove_no_skip_ops
        if self._no_skip_names:
            remove_no_skip_ops(self._no_skip_names)
        enable_dtensor_dispatch()


# Cache for _build_layout to avoid redundant Layout computations
# Key: (device_mesh.to_hash(), tuple(placements), tensor_dim)
# Value: Layout
_LAYOUT_CACHE = {}


def _is_alias_placements(placements) -> bool:
    """
    Check if placements use alias strings rather than Placement objects.

    Alias placements use mesh dimension names (strings) to specify
    the sharding strategy, e.g., ("dp", "tp") or (("dp", "tp"), "None").
    All elements must be strings or tuples of strings for the sequence
    to be recognized as alias-style.

    Args:
        placements: A sequence of placement specifications.

    Returns:
        bool: True if all elements are alias strings or tuples of strings.
    """
    if len(placements) == 0:
        return False
    for p in placements:
        if isinstance(p, str):
            continue
        if isinstance(p, tuple) and len(p) > 0 and all(isinstance(x, str) for x in p):
            continue
        return False
    return True


def _build_layout(
        device_mesh: DeviceMesh,
        placements: Union[Sequence[Placement], Sequence[Union[str, Tuple[str, ...]]]],
        tensor_dim: int
) -> Layout:
    """
    Build Layout from device_mesh and placements.

    This function uses a cache to avoid redundant Layout computations
    for the same (device_mesh, placements, tensor_dim) combination.

    Args:
        device_mesh: The device mesh describing the device topology.
        placements: Supports two styles:
            - Placement objects (Shard, Replicate, etc.)
            - Alias strings ("dp", "None", ("dp", "tp"), etc.), length must
              equal the number of tensor dimensions (``tensor_dim``).
        tensor_dim: Number of dimensions in the tensor.

    Returns:
        Layout: The built layout object.

    Raises:
        ValueError: If alias placements length does not match tensor dimensions.
    """
    mesh_key = device_mesh.to_hash()
    placements_key = tuple(placements)
    cache_key = (mesh_key, placements_key, tensor_dim)

    if cache_key in _LAYOUT_CACHE:
        return _LAYOUT_CACHE[cache_key]

    layout = Layout.from_device_mesh(device_mesh)

    if _is_alias_placements(placements):
        if len(placements) != tensor_dim:
            raise ValueError(
                f"Alias placements length ({len(placements)}) must equal "
                f"tensor dimensions ({tensor_dim})."
            )
        result = layout(*placements)
    else:
        result = layout(placements)
        result.placement_to_tensor_map(tensor_dim)

    _LAYOUT_CACHE[cache_key] = result

    return result


def _is_broadcastable(src_shape: Sequence[int], dst_shape: Sequence[int]) -> bool:
    """Return True iff ``src_shape`` is broadcastable to ``dst_shape``.

    Standard NumPy / PyTorch right-aligned broadcast rule: ``src`` cannot
    have more dimensions than ``dst``; each right-aligned dimension pair
    must be equal, or ``src``'s dimension must be 1.
    """
    src_shape = tuple(src_shape)
    dst_shape = tuple(dst_shape)
    if len(src_shape) > len(dst_shape):
        return False
    for i in range(1, len(src_shape) + 1):
        s, d = src_shape[-i], dst_shape[-i]
        if s not in (d, 1):
            return False
    return True


class DTensor(DTensorBase):
    """
    DTensor - Distributed Tensor

    A DTensor represents a tensor that is distributed across multiple devices
    according to a DeviceMesh and placement specifications.

    Args:
        local_tensor (Tensor): The local tensor shard on this device.
        device_mesh (DeviceMesh): The device mesh describing the device topology.
        placements: The placement strategy. Supports two styles:
            - Placement objects (e.g., ``[Shard(0), Replicate()]``).
            - Alias strings (e.g., ``("dp", "None")`` or
              ``(("dp", "tp"), "None")``), length must equal the number of
              tensor dimensions.

    Example:
        >>> mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
        >>> local_tensor = Tensor(np.ones((4, 4)))
        >>> # Placement style
        >>> dtensor = DTensor.from_local(local_tensor, mesh, [Shard(0), Replicate()])
        >>> # Alias style — length matches tensor dims
        >>> dtensor = DTensor.from_local(local_tensor, mesh, ("dp", "None"))
    """
    _local_tensor: Tensor
    _device_mesh: DeviceMesh
    _placements: Sequence[Placement]

    def __init_data__(
        self,
        local_tensor: Tensor,
        device_mesh: DeviceMesh,
        placements: Union[Sequence[Placement], Sequence[Union[str, Tuple[str, ...]]]]
    ):
        self._local_tensor = local_tensor
        self._device_mesh = device_mesh
        self._layout = _build_layout(
            device_mesh, placements, len(local_tensor.shape)
        )
        self._placements = tuple(self._layout.placements)

    @property
    def device_mesh(self) -> DeviceMesh:
        """The device mesh of this DTensor."""
        return self._device_mesh

    @property
    def placements(self) -> Sequence[Placement]:
        """The placements of this DTensor."""
        return self._placements

    @property
    def layout(self) -> Layout:
        """Internal layout for redistribution (for backward compatibility)."""
        if not hasattr(self, '_layout'):
            return None
        return self._layout

    @staticmethod
    def from_local(
        local_tensor: Tensor,
        device_mesh: DeviceMesh,
        placements: Union[Sequence[Placement], Sequence[Union[str, Tuple[str, ...]]]]
    ) -> 'DTensor':
        """
        Create a DTensor from a local tensor with device mesh and placements.

        Args:
            local_tensor (Tensor): The local tensor shard on this device.
            device_mesh (DeviceMesh): The device mesh describing the device topology.
            placements: The placement strategy. Supports two styles:
                - Placement objects (e.g., ``[Shard(0), Replicate()]``).
                - Alias strings (e.g., ``("dp", "None")`` or
                  ``(("dp", "tp"), "None")``), length must equal the number
                  of tensor dimensions.

        Returns:
            DTensor: A new DTensor instance.

        Example:
            >>> mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
            >>> local_tensor = Tensor(np.ones((4, 4)))
            >>> dtensor = DTensor.from_local(local_tensor, mesh, [Shard(0), Replicate()])
            >>> dtensor = DTensor.from_local(local_tensor, mesh, ("dp", "None"))
        """
        return DTensor(local_tensor, device_mesh, placements)

    def _alias_placements(self) -> Sequence[Placement]:
        """Return alias_placements from layout, falling back to _placements."""
        if hasattr(self, '_layout') and self._layout:
            return self._layout.alias_placements
        return self._placements

    def _from_converted_local(self, local_tensor: Tensor) -> 'DTensor':
        """Rebuild converted DTensor data without preserving Parameter identity."""
        cls = DTensor if isinstance(self, platform.Parameter) else self.__class__
        return cls(local_tensor, device_mesh=self._device_mesh,
                   placements=self._alias_placements())

    def to(self, *args, **kwargs):
        """Move the DTensor to a different device or dtype.

        Delegates to the underlying local tensor's ``to`` method and
        reconstructs a DTensor preserving device_mesh and placements.

        Args:
            *args (tuple): Arguments passed to the underlying tensor's ``to``
                method (e.g., device or dtype).
            **kwargs (dict): Keyword arguments for the tensor conversion
                (e.g., dtype, device, non_blocking).

        Returns:
            DTensor: A new DTensor with the converted local tensor.
        """
        new_local = self._local_tensor.to(*args, **kwargs)
        return self._from_converted_local(new_local)

    def float(self):
        """Convert the DTensor to float dtype.

        Returns:
            DTensor: A new DTensor with float32 local tensor.
        """
        new_local = self._local_tensor.float()
        return self._from_converted_local(new_local)

    def to_local(self) -> Tensor:
        """
        Convert DTensor to local tensor.

        Returns:
            Tensor: The local tensor shard on this device.
        """
        return self._local_tensor

    def copy_(self, src: "DTensor", non_blocking: bool = False) -> "DTensor":
        """In-place copy of ``src`` into this DTensor's local shard.

        Delegates to ``Tensor.copy_`` on the underlying local tensors.
        Follows standard ``Tensor.copy_`` semantics: version counter is
        bumped and autograd edges are created when grad is enabled.

        Constraints on ``src``:
            * must be a ``DTensor`` on the same ``DeviceMesh`` as ``self``;
            * its placements must equal ``self.placements``, OR
              ``src._local_tensor.numel() == 1`` (single-element broadcast);
            * its local shape must equal or be broadcastable to
              ``self._local_tensor.shape``.

        No redistribute / implicit slicing is performed; src dtype is cast
        to self dtype in-place.

        Args:
            src (DTensor): Source DTensor satisfying the constraints above.
            non_blocking (bool): Forwarded to the underlying ``copy_``.

        Returns:
            DTensor: ``self``.

        Raises:
            TypeError:  if ``src`` is not a ``DTensor``.
            ValueError: if mesh, placement, or shape constraint is violated.
        """
        if not isinstance(src, DTensor):
            raise TypeError(
                f"For DTensor.copy_, src should be a DTensor, but got {type(src).__name__}."
            )
        if src._device_mesh is not self._device_mesh:
            raise ValueError(
                f"For DTensor.copy_, src and self should share the same DeviceMesh, "
                f"but got src._device_mesh={src._device_mesh!r}, "
                f"self._device_mesh={self._device_mesh!r}."
            )

        placement_eq = tuple(src._placements) == tuple(self._placements)
        shape_eq = src._local_tensor.shape == self._local_tensor.shape
        src_is_scalar = src._local_tensor.numel() == 1

        if not placement_eq and not src_is_scalar:
            raise ValueError(
                f"For DTensor.copy_, src.placements should equal self.placements "
                f"or src.numel() should be 1, but got "
                f"src.placements={src._placements}, "
                f"self.placements={self._placements}, "
                f"src.numel()={src._local_tensor.numel()}."
            )
        if not shape_eq and not src_is_scalar and not _is_broadcastable(
            src._local_tensor.shape, self._local_tensor.shape
        ):
            raise ValueError(
                f"For DTensor.copy_, src local shape should be broadcastable to "
                f"self local shape, but got "
                f"src.shape={tuple(src._local_tensor.shape)}, "
                f"self.shape={tuple(self._local_tensor.shape)}."
            )

        self._local_tensor.copy_(src._local_tensor, non_blocking=non_blocking)
        return self

    def zero_(self) -> "DTensor":
        """In-place fill with zeros. Returns ``self``."""
        self._local_tensor.zero_()
        return self

    def fill_(self, value) -> "DTensor":
        """In-place fill with ``value``. Returns ``self``."""
        self._local_tensor.fill_(value)
        return self

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        The global shape of this DTensor.

        Returns:
            Tuple[int, ...]: The global tensor shape.
        """
        return self._layout.get_global_shape(self._local_tensor.shape)

    def size(self, dim=None):
        """Return the global shape, consistent with .shape.

        Without ``dim`` returns a tuple matching ``self.shape``.
        With ``dim`` returns the size of that dimension.
        """
        global_shape = self.shape
        if dim is not None:
            return global_shape[dim]
        return global_shape

    def numel(self) -> int:
        """Return the number of elements in this DTensor."""
        return int(np.prod(self.shape))

    @property
    def local_shape(self) -> Tuple[int, ...]:
        """
        The local shape of this DTensor on this device.

        Returns:
            Tuple[int, ...]: The local tensor shape.
        """
        return self._local_tensor.shape

    def redistribute(
        self,
        device_mesh: DeviceMesh,
        placements: Union[Sequence[Placement], Sequence[Union[str, Tuple[str, ...]]]]
    ) -> 'DTensor':
        """
        Redistribute this DTensor to a new device mesh and placements.

        Args:
            device_mesh (DeviceMesh): The target device mesh.
            placements: The target placements. Supports Placement objects
                or alias strings.

        Returns:
            DTensor: A new DTensor with the specified distribution.

        Example:
            >>> new_dtensor = dtensor.redistribute(mesh, [Replicate(), Shard(1)])
            >>> new_dtensor = dtensor.redistribute(mesh, ("None", "tp"))
        """
        # Build dst_layout from device_mesh and placements
        dst_layout = _build_layout(
            device_mesh, placements, len(self._local_tensor.shape)
        )

        # pylint: disable=C0415
        from hyper_parallel.core.dtensor.tensor_redistribution import _tensor_redistribution
        out = _tensor_redistribution.redistribution(self, dst_layout)
        return out

    def reduce_partial(self) -> 'DTensor':
        """
        Reduce partial sharding state for this DTensor.

        Returns:
            DTensor: A new DTensor with partial state reduced.
        """
        if not self._layout:
            return self
        to_layout = cp.deepcopy(self._layout)
        to_layout.reset_partial()
        # pylint: disable=C0415
        from hyper_parallel.core.dtensor.tensor_redistribution import _tensor_redistribution
        out = _tensor_redistribution.reduce_partial(self, to_layout)
        return out

    def full_tensor(self) -> Tensor:
        """
        Return the full tensor of this DTensor.

        Returns:
            Tensor: A Tensor object that represents the full tensor of this DTensor.
                    The returned tensor contains the complete data gathered from
                    all ranks.

        Note:
            This operation involves communication across all ranks in the DeviceMesh,
            which may be expensive for large tensors. Use with caution in
            performance-critical code paths.

        Example:
            >>> # Assume dtensor is sharded across multiple devices
            >>> local_tensor = dtensor.to_local()  # Returns only the local shard
            >>> full_tensor = dtensor.full_tensor()  # Returns the complete tensor
        """
        if not self._layout:
            return self._local_tensor

        # Create a fully replicated layout
        replicated_layout = cp.deepcopy(self._layout)

        # Set all placements to Replicate and convert to tensor_map
        replicated_placements = [Replicate()] * len(replicated_layout.mesh_shape)
        replicated_layout.set_placements(replicated_placements)
        replicated_layout.placement_to_tensor_map(len(self._local_tensor.shape))

        # Clear partial status from original layout since Replicate has no partial
        replicated_layout.reset_partial()

        # Redistribute to the replicated layout and return local tensor
        # pylint: disable=C0415
        from hyper_parallel.core.dtensor.tensor_redistribution import _tensor_redistribution
        out = _tensor_redistribution.redistribution(self, replicated_layout)
        return out.to_local()


def distribute_tensor(
    tensor: Tensor,
    device_mesh: DeviceMesh,
    placements: Union[Sequence[Placement], Sequence[Union[str, Tuple[str, ...]]]]
) -> DTensor:
    """
    Distribute a global tensor to the device mesh according to the placements.

    Args:
        tensor (Tensor): The global tensor to be distributed. All ranks
            should have the same tensor data.
        device_mesh (DeviceMesh): The device mesh describing the device topology.
        placements: The placement strategy. Supports two styles:
            - Placement objects (e.g., ``[Shard(0), Replicate()]``).
            - Alias strings (e.g., ``("dp", "None")`` or
              ``(("dp", "tp"), "None")``), length must equal the number of
              tensor dimensions.

    Returns:
        DTensor: A new DTensor with the local shard on each rank.

    Note:
        This method assumes all ranks have the same global tensor. It slices
        the tensor locally without communication. If ranks have different
        data, use `from_local` instead.

    Example:
        >>> mesh = init_device_mesh(device_type="npu", mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
        >>> global_tensor = Tensor(np.arange(16).reshape(4, 4))
        >>> dtensor = distribute_tensor(global_tensor, mesh, [Shard(0), Replicate()])
        >>> dtensor = distribute_tensor(global_tensor, mesh, ("dp", "None"))
    """
    layout = _build_layout(device_mesh, placements, len(tensor.shape))
    local_tensor = _get_slice_tensor_by_layout(tensor, layout)
    return DTensor(local_tensor, device_mesh, layout.alias_placements)


def _distribute_module_param_source(param: Any) -> Tensor:
    """Tensor data used as the global tensor for :func:`distribute_tensor` (PyTorch uses ``param.data``)."""
    if hasattr(param, "data"):
        return param.data
    return platform.get_param_local_data(param)


def _distribute_module_new_parameter(key: str, dtensor: DTensor, requires_grad: bool) -> Any:
    """Build a framework :class:`Parameter` holding *dtensor* (Torch vs MindSpore kwargs differ)."""
    if platform.platform_type == PlatformType.MINDSPORE:
        return platform.Parameter(dtensor, name=key, requires_grad=requires_grad)
    return platform.Parameter(dtensor, requires_grad=requires_grad)


def _distribute_module_set_param(module: Any, key: str, new_param: Any) -> None:
    """Register or assign a parameter on *module* (``nn.Module`` or MindSpore ``Cell``)."""
    if hasattr(module, "register_parameter"):
        module.register_parameter(key, new_param)
        return
    if hasattr(module, "_params"):
        module._params[key] = new_param
        if hasattr(module, "_params_list"):
            module._params_list[key] = new_param
        if key in module.__dict__:
            module.__dict__[key] = new_param
        return
    raise TypeError(
        f"distribute_module expects nn.Module-like objects with register_parameter or _params; "
        f"got {type(module)}."
    )


def _distribute_module_iter_params(module: Any) -> list:
    """Return ``[(name, param), ...]`` for direct parameters (``_parameters`` or ``_params``)."""
    if hasattr(module, "_parameters"):
        return list(module._parameters.items())
    if hasattr(module, "_params"):
        return list(module._params.items())
    return []


def _distribute_module_iter_buffers(module: Any) -> list:
    """Return ``[(name, buffer), ...]`` if the module has ``_buffers`` (PyTorch ``nn.Module``)."""
    if hasattr(module, "_buffers"):
        return list(module._buffers.items())
    return []


def _distribute_module_named_modules(module: Any):
    """``nn.Module.named_modules`` or MindSpore ``Cell.cells_and_names`` (submodule FQNs)."""
    if hasattr(module, "named_modules"):
        return module.named_modules()
    if hasattr(module, "cells_and_names"):
        return module.cells_and_names()
    raise TypeError(
        f"distribute_module expects module-like objects with named_modules or cells_and_names; "
        f"got {type(module)}."
    )


def _replicate_submodule_params_buffers(
    sub_mod: Any,
    device_mesh: DeviceMesh,
    *,
    module_prefix: str = "",
) -> None:
    """Convert plain params/buffers on *sub_mod* to fully replicated :class:`DTensor`."""
    full_replicate = [Replicate()] * device_mesh.ndim
    for key, param in _distribute_module_iter_params(sub_mod):
        if param is None or isinstance(param, DTensorBase):
            continue
        src = _distribute_module_param_source(param)
        requires_grad = bool(getattr(param, "requires_grad", True))
        dt = distribute_tensor(src, device_mesh, full_replicate)
        param_name = f"{module_prefix}.{key}" if module_prefix else key
        new_param = _distribute_module_new_parameter(param_name, dt, requires_grad)
        _distribute_module_set_param(sub_mod, key, new_param)
    for key, buffer in _distribute_module_iter_buffers(sub_mod):
        if buffer is None or isinstance(buffer, DTensorBase):
            continue
        sub_mod._buffers[key] = distribute_tensor(buffer, device_mesh, full_replicate)


def _distribute_module_run_partition_and_replicate(
    module: Any,
    device_mesh: DeviceMesh,
    partition_fn: Optional[Callable[[str, Any, DeviceMesh], None]],
) -> None:
    """Call optional ``partition_fn`` per ``named_modules`` and replicate remaining tensors."""
    if partition_fn is None:
        for mod_name, submod in _distribute_module_named_modules(module):
            _replicate_submodule_params_buffers(submod, device_mesh, module_prefix=mod_name)
        return
    for mod_name, submod in _distribute_module_named_modules(module):
        partition_fn(mod_name, submod, device_mesh)
        _replicate_submodule_params_buffers(submod, device_mesh, module_prefix=mod_name)


def _distribute_module_register_input_fn(
    module: Any,
    device_mesh: DeviceMesh,
    input_fn: Callable[..., Any],
) -> None:
    """Register *input_fn* as a forward pre-hook on *module* (2- or 3-arg, PyTorch-compatible)."""
    num_args = len(inspect.signature(input_fn).parameters)
    if num_args == 2:
        warnings.warn(
            "Deprecating input_fn that takes two arguments (inputs, device_mesh), "
            "please use input_fn that takes in (module, inputs, device_mesh) instead!",
            FutureWarning,
            stacklevel=3,
        )
        module.register_forward_pre_hook(
            lambda _, inputs: input_fn(inputs, device_mesh)
        )
    elif num_args == 3:
        module.register_forward_pre_hook(
            lambda mod, inputs: input_fn(mod, inputs, device_mesh)
        )
    else:
        raise ValueError(
            f"input_fn should take in 2 or 3 arguments, but got {num_args} arguments!"
        )


def _distribute_module_register_output_fn(
    module: Any,
    device_mesh: DeviceMesh,
    output_fn: Callable[..., Any],
) -> None:
    """Register *output_fn* as a forward hook on *module* (2- or 3-arg, PyTorch-compatible)."""
    num_args = len(inspect.signature(output_fn).parameters)
    if num_args == 2:
        warnings.warn(
            "Deprecating output_fn that takes two arguments (outputs, device_mesh), "
            "please use output_fn that takes in (module, outputs, device_mesh) instead!",
            FutureWarning,
            stacklevel=3,
        )
        module.register_forward_hook(
            lambda mod, inputs, outputs: output_fn(outputs, device_mesh)
        )
    elif num_args == 3:
        module.register_forward_hook(
            lambda mod, inputs, outputs: output_fn(mod, outputs, device_mesh)
        )
    else:
        raise ValueError(
            f"output_fn should take in 2 or 3 arguments, but got {num_args} arguments!"
        )


def distribute_module(
    module: Any,
    device_mesh: Optional[DeviceMesh] = None,
    partition_fn: Optional[Callable[[str, Any, DeviceMesh], None]] = None,
    input_fn: Optional[Callable[..., Any]] = None,
    output_fn: Optional[Callable[..., Any]] = None,
) -> Any:
    """PyTorch ``distribute_module`` parity: shard/replicate params and optional I/O hooks.

    Unsharded parameters and buffers become fully replicated :class:`DTensor` after
    ``partition_fn``. ``input_fn`` / ``output_fn`` attach only to the root *module*.

    Args:
        module: Root ``nn.Module`` or MindSpore ``Cell`` with compatible APIs.
        device_mesh: Placement mesh; if ``None``, uses ``_mesh_resources.get_current_mesh()``.
        partition_fn: Per ``named_modules`` callback before replicate pass; ``None`` replicates all.
        input_fn: ``(module, inputs, mesh)`` or deprecated ``(inputs, mesh)`` pre-hook.
        output_fn: ``(module, outputs, mesh)`` or deprecated ``(outputs, mesh)`` forward hook.

    Returns:
        *module* in place, with distributed tensors where applied.

    Raises:
        RuntimeError: If called twice on the same *module*.
        ValueError: If ``input_fn`` / ``output_fn`` arity is not 2 or 3.

    Note:
        XLA / ``torch_xla`` is not supported; strided device :class:`DTensor` only.
    """
    if getattr(module, "_distribute_module_applied", False):
        raise RuntimeError(
            "distribute_module should only be called once on a module, "
            "but it has already been called on this module!"
        )
    device_mesh = device_mesh or _mesh_resources.get_current_mesh()
    _distribute_module_run_partition_and_replicate(module, device_mesh, partition_fn)
    if input_fn is not None:
        _distribute_module_register_input_fn(module, device_mesh, input_fn)
    if output_fn is not None:
        _distribute_module_register_output_fn(module, device_mesh, output_fn)
    module._distribute_module_applied = True
    return module


def _dtensor_init_helper(
        init_op,
        size,
        device_mesh,
        placements,
        **kwargs,
) -> DTensor:
    """
        Helper function to create and initialize a distributed tensor.

        Args:
            size: Shape of the tensor.
            dtype: Data type of the tensor.
            device: Target device for the tensor.
            requires_grad: Whether the tensor requires gradient.

        Returns:
            DTensor: The initialized distributed tensor.
    """
    # get local tensor shape
    local_shape = compute_local_shape_and_global_offset(
        size, device_mesh, placements
    )

    # initialize the local tensor
    if init_op is platform.full:
        fill_value = kwargs.pop("fill_value", 0)
        local_tensor = init_op(local_shape, fill_value, **kwargs)
    else:
        local_tensor = init_op(local_shape, **kwargs)

    return DTensor.from_local(
            local_tensor,
            device_mesh,
            placements,
    )


def ones(
    size,
    device_mesh,
    placements,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with the scalar value 1, with the shape defined
    by the variable argument ``size``.

    Args:
        size (Union[tuple[int], list[int], int, Tensor]): The specified shape of output tensor. Only positive integer or
            tuple or Tensor containing positive integers are allowed. If it is a Tensor,
            it must be a 0-D or 1-D Tensor with int32 or int64 dtypes.

    Keyword args:
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    ones_ = platform.ones
    return _dtensor_init_helper(
        ones_,
        size,
        device_mesh=device_mesh,
        placements=placements,
    )


def empty(
    size,
    device_mesh,
    placements,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with uninitialized data. The shape of the :class:`DTensor`
    is defined by the variable argument ``size``.

    Args:
        size (Union[tuple[int], list[int], int]): The specified shape of output tensor. Can be variable numbers of
            positive integers or tuple or list containing positive integers.

    Keyword args:
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    empty_ = platform.empty
    return _dtensor_init_helper(
        empty_,
        size,
        device_mesh=device_mesh,
        placements=placements,
    )


def full(
    size,
    fill_value,
    *,
    device_mesh,
    placements,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with ``fill_value`` according to ``device_mesh`` and
    ``placements``, with the shape defined by the argument ``size``.

    Args:
        size (Union[tuple[int], list[int]]): The specified shape of output tensor.
        fill_value (Union[numbers.Number, Tensor]): Value to fill the returned tensor. It can be a scalar number, a 0-D
            Tensor, or a 1-D Tensor with only one element.

    Keyword args:
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks.
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    full_ = platform.full
    return _dtensor_init_helper(
        full_,
        size,
        fill_value=fill_value,
        device_mesh=device_mesh,
        placements=placements,
    )


def zeros(
    size,
    device_mesh,
    placements,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with the scalar value 0.

    Args:
        size (Union[tuple[int], list[int], int, Tensor]): The specified shape of output tensor. Only positive integer or
        tuple or Tensor containing positive integers are allowed. If it is a Tensor,
            it must be a 0-D or 1-D Tensor with int32 or int64 dtypes.
    Keyword args:
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``

    Returns:
        A :class:`DTensor` object on each rank
    """
    zeros_ = platform.zeros
    return _dtensor_init_helper(
        zeros_,
        size,
        device_mesh=device_mesh,
        placements=placements,
    )
