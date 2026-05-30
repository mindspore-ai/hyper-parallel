# Copyright 2025 Huawei Technologies Co., Ltd
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
"""torch dtensor base"""
from typing import Tuple, Dict, Any, Optional
import torch
from torch import Tensor


class DTensorBase(Tensor):
    """torch dtensor base"""

    def __new__(cls, local_tensor, device_mesh=None, placements=None):
        """
        Create a new DTensorBase instance.

        Args:
            local_tensor: The local tensor shard or another DTensorBase instance.
            device_mesh: The device mesh describing the device topology.
            placements: The placement strategy for each mesh dimension.
        """
        if isinstance(local_tensor, DTensorBase):
            # Copy from existing DTensorBase — use alias_placements to preserve multi-axis ordering
            t = Tensor._make_subclass(cls, local_tensor._local_tensor, local_tensor._local_tensor.requires_grad)
            copy_placements = local_tensor.layout.alias_placements if local_tensor.layout else local_tensor.placements
            t.__init_data__(local_tensor._local_tensor, local_tensor.device_mesh, copy_placements)
            return t

        if device_mesh is None:
            raise ValueError("device_mesh is None, must provide a DeviceMesh instance")
        if placements is None:
            raise ValueError("placements is None, must provide placements")

        # Create Tensor subclass instance, sharing local_tensor's underlying storage
        t = Tensor._make_subclass(cls, local_tensor, local_tensor.requires_grad)
        t.__init_data__(local_tensor, device_mesh, placements)
        return t

    # pylint: disable=W0613
    @classmethod
    def __torch_function__(
        cls,
        func: torch._C._FunctionBase,
        types: Tuple[type, ...],
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Override PyTorch's __torch_function__ to intercept tensor operations.

        This method dispatches operations through the distributed operator dispatcher
        to handle DTensor-specific layout inference and redistribution.

        Args:
            func (torch._C._FunctionBase): The PyTorch function being called.
            types (Tuple[type, ...]): The types of tensors involved in the operation.
            args (Tuple[Any, ...]): Positional arguments passed to the function.
            kwargs (Optional[Dict[str, Any]]): Keyword arguments passed to the function.

        Returns:
            Any: The result of the dispatched operation, typically a DTensor or tuple of DTensors.
        """
        kwargs = kwargs or {}
        # pylint: disable=C0415
        from hyper_parallel.core.shard._op_dispatch import _OP_DISPATCHER
        out = _OP_DISPATCHER.dispatch(func, args, kwargs)
        return out

    @property
    def grad(self) -> Optional[Tensor]:
        """
        Get the gradient tensor of the local tensor.

        Returns:
            Optional[Tensor]: The gradient tensor, or None if no gradient is set.
        """
        return self._local_tensor.grad

    @grad.setter
    def grad(self, value: Optional[Tensor]) -> None:
        """
        Set the gradient tensor for the local tensor.

        Args:
            value (Optional[Tensor]): The gradient tensor to set, or None to clear.
        """
        self._local_tensor.grad = value

    @property
    def requires_grad(self) -> bool:
        """
        Check if gradient computation is enabled for this tensor.

        Returns:
            bool: True if gradients should be computed for this tensor.
        """
        return self._local_tensor.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        """
        Enable or disable gradient computation for this tensor.

        Args:
            value (bool): True to enable gradient computation, False to disable.
        """
        self._local_tensor.requires_grad_(value)
        # Sync DTensor wrapper's requires_grad
        super().requires_grad_(value)

    def requires_grad_(self, requires_grad: bool = True):
        """
        Enable or disable gradient computation in-place.

        Args:
            requires_grad (bool): True to enable gradient computation. Default: True.

        Returns:
            DTensorBase: Self for method chaining.
        """
        self._local_tensor.requires_grad_(requires_grad)
        super().requires_grad_(requires_grad)
        return self

    @property
    def grad_fn(self) -> Optional[torch.autograd.Function]:
        """
        Get the gradient function that created this tensor.

        Returns:
            Optional[torch.autograd.Function]: The gradient function, or None if not applicable.
        """
        return self._local_tensor.grad_fn

    def grad_zero_(self):
        """
        Zero out the gradient tensor in-place.

        Returns:
            DTensorBase: Self for method chaining.
        """
        if self._local_tensor.grad is not None:
            self._local_tensor.grad.zero_()
        return self

    def detach(self):
        """
        Create a detached DTensor that does not require gradient.

        Returns:
            DTensorBase: A new DTensor with the same data but detached from the computation graph.
        """
        detached_local = self._local_tensor.detach()
        return self.__class__(detached_local, device_mesh=self._device_mesh, placements=self._alias_placements())

    def detach_(self):
        """
        Detach this tensor from the computation graph in-place.

        Returns:
            DTensorBase: Self for method chaining.
        """
        self._local_tensor.detach_()
        super().detach_()
        return self

    # ====================== Computation graph related overrides ======================
    @property
    def is_leaf(self) -> bool:
        """
        Check if this tensor is a leaf node in the computation graph.

        Returns:
            bool: True if this is a leaf tensor (created by user, not by any operation).
        """
        return self._local_tensor.is_leaf

    @property
    def retains_grad(self) -> bool:
        """
        Check if this tensor retains its gradient during backward pass.

        Returns:
            bool: True if gradients are retained for non-leaf tensors.
        """
        return self._local_tensor.retains_grad

    @retains_grad.setter
    def retains_grad(self, value: bool) -> None:
        """
        Enable or disable gradient retention for this tensor.

        Args:
            value (bool): True to enable gradient retention.
        """
        self._local_tensor.retains_grad_(value)

    def backward(self, gradient=None, retain_graph=None, create_graph=False) -> None:
        """
        Compute the gradients for this tensor.

        Args:
            gradient (Optional[Tensor]): The gradient of the loss w.r.t. this tensor.
            retain_graph (Optional[bool]): Whether to retain the computation graph.
            create_graph (bool): Whether to create a graph of the gradient computation.
        """
        self._local_tensor.backward(gradient, retain_graph, create_graph)

    # ====================== Metadata related overrides (sync with local_tensor) ======================
    @property
    def device(self) -> torch.device:
        """
        Get the device on which this tensor is stored.

        Returns:
            torch.device: The device object (e.g., 'cuda:0', 'cpu').
        """
        return self._local_tensor.device

    @property
    # pylint: disable=C2801
    def data(self):
        return Tensor.data.__get__(self, type(self))

    @data.setter
    # pylint: disable=C2801
    def data(self, value):
        local_value = value.to_local() if isinstance(value, DTensorBase) else value
        Tensor.data.__set__(self, local_value)
        Tensor.data.__set__(self._local_tensor, local_value)

    @property
    def dtype(self) -> torch.dtype:
        """
        Get the data type of this tensor.

        Returns:
            torch.dtype: The data type (e.g., torch.float32, torch.int64).
        """
        return self._local_tensor.dtype

    @property
    def shape(self) -> torch.Size:
        """
        Get the shape of this tensor.

        Returns:
            torch.Size: The shape of the tensor.
        """
        return self._local_tensor.shape

    def type(self, dtype=None, non_blocking=False):
        """
        Convert this tensor to the specified dtype.

        Args:
            dtype (Optional[torch.dtype]): The target dtype. If None, returns the current type string.
            non_blocking (bool): Whether to perform the operation asynchronously. Default: False.

        Returns:
            Union[str, DTensorBase]: The type string if dtype is None, otherwise a new DTensor.
        """
        if dtype is None:
            return self._local_tensor.type()
        new_local = self._local_tensor.to(dtype=dtype, non_blocking=non_blocking)
        return self.__class__(new_local, device_mesh=self._device_mesh, placements=self._alias_placements())

    def size(self, dim: Optional[int] = None):
        """
        Get the size of this tensor.

        Args:
            dim (Optional[int]): The dimension to query. If None, returns the full shape.

        Returns:
            Union[torch.Size, int]: The shape or size along a specific dimension.
        """
        return self._local_tensor.size(dim)

    @property
    def ndim(self) -> int:
        """
        Get the number of dimensions of this tensor.

        Returns:
            int: The number of dimensions.
        """
        return self._local_tensor.ndim

    def data_ptr(self) -> int:
        """
        Get the pointer to the data storage of the local tensor.

        Returns:
            int: The memory address of the tensor's data.
        """
        # Force return local_tensor's data pointer (ensure address consistency)
        return self._local_tensor.data_ptr()

    def numel(self) -> int:
        """
        Get the total number of elements in this tensor.

        Returns:
            int: The total number of elements.
        """
        return self._local_tensor.numel()

    # ====================== Data operation overrides (sync storage + fix in-place ops) ======================
    def zero_(self):
        """Set tensor zeros"""
        if self._local_tensor.requires_grad and self._local_tensor.is_leaf:
            # Create new tensor + rebind DTensor (ensure storage sharing)
            new_local = torch.zeros_like(self._local_tensor, requires_grad=True)
            # Key: sync DTensor wrapper's storage to new local_tensor
            super().copy_(new_local)  # sync underlying data
            self._local_tensor = new_local  # replace internal attribute
        else:
            self._local_tensor.zero_()
            super().zero_()  # sync wrapper's in-place zero
        return self

    def copy_(self, src: Tensor, non_blocking: bool = False):
        """Copy data from src tensor"""
        if self._local_tensor.requires_grad and self._local_tensor.is_leaf:
            new_local = src.to(self._local_tensor.device, non_blocking=non_blocking).detach().clone()
            new_local.requires_grad = self._local_tensor.requires_grad
            super().copy_(new_local)
            self._local_tensor = new_local
        else:
            self._local_tensor.copy_(src, non_blocking=non_blocking)
            super().copy_(src, non_blocking=non_blocking)
        return self

    def fill_(self, value):
        """Fill tensor with value"""
        if self._local_tensor.requires_grad and self._local_tensor.is_leaf:
            # Step 1: Create new tensor (non-in-place)
            new_local = torch.full_like(
                self._local_tensor,
                fill_value=value,
                requires_grad=True,
                device=self._local_tensor.device
            )
            # Step 2: Sync DTensor wrapper's underlying storage to new local_tensor
            super().copy_(new_local)  # Key: make DTensor wrapper point to new address
            # Step 3: Replace internal local_tensor (ensure attribute consistency)
            self._local_tensor = new_local
        else:
            # Non-leaf tensor: direct in-place fill + sync wrapper
            self._local_tensor.fill_(value)
            super().fill_(value)  # sync DTensor wrapper's fill
        return self

    # ====================== Auxiliary print ======================
    def _alias_placements(self):
        """Return alias_placements from layout, falling back to _placements."""
        if hasattr(self, '_layout') and self._layout is not None:
            return self._layout.alias_placements
        return self._placements

    def to(self, *args, **kwargs):
        """Move the DTensor to a different device or dtype.

        This method overrides the base Tensor.to() to properly reconstruct
        a DTensor with device_mesh and placements preserved. Uses _make_subclass
        to avoid issues with Parameter subclasses that don't accept extra kwargs.

        Args:
            *args: Arguments passed to the underlying tensor's to() method.
            **kwargs: Keyword arguments for the tensor conversion.

        Returns:
            DTensorBase: A new DTensor with the converted local tensor.
        """
        new_local = self._local_tensor.to(*args, **kwargs)
        new_dt = Tensor._make_subclass(type(self), new_local, new_local.requires_grad)
        new_dt.__init_data__(new_local, self._device_mesh, self._alias_placements())
        return new_dt

    def __repr__(self) -> str:
        return (
            f"DTensor(\n"
            f"  local_tensor={self._local_tensor},\n"
            f"  device_mesh={self._device_mesh},\n"
            f"  placements={self._placements},\n"
            f"  layout={getattr(self, '_layout', None)},\n"
            f"  device={self.device},\n"
            f"  dtype={self.dtype},\n"
            f"  requires_grad={self.requires_grad},\n"
            f"  grad={self.grad},\n"
            f"  is_leaf={self.is_leaf},\n"
            f"  data_ptr={self.data_ptr()}\n"
            f")"
        )
