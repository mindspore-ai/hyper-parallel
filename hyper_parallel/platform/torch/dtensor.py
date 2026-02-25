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
            # Copy from existing DTensorBase
            t = Tensor._make_subclass(cls, local_tensor._local_tensor, local_tensor._local_tensor.requires_grad)
            t.__init_data__(local_tensor._local_tensor, local_tensor.device_mesh, local_tensor.placements)
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
        kwargs = kwargs or {}
        # pylint: disable=C0415
        from hyper_parallel.core.shard._op_dispatch import _OP_DISPATCHER
        out = _OP_DISPATCHER.dispatch(func, args, kwargs)
        return out

    def to(self, device):
        """set tensor to device"""
        target_device = torch.device(device) if isinstance(device, str) else device
        src_local = self._local_tensor

        if src_local.device.type == "meta":
            new_local = torch.empty_like(
                src_local,
                device=target_device,
                requires_grad=src_local.requires_grad
            )
        else:
            new_local = src_local.to(target_device)

        return self.__class__(new_local, device_mesh=self._device_mesh, placements=self._placements)

    def float(self):
        """convert tensor to float dtype"""
        local_tensor = self._local_tensor
        new_local = local_tensor.float()
        return self.__class__(new_local, device_mesh=self._device_mesh, placements=self._placements)

    @property
    def grad(self) -> Optional[Tensor]:
        return self._local_tensor.grad

    @grad.setter
    def grad(self, value: Optional[Tensor]) -> None:
        self._local_tensor.grad = value

    @property
    def requires_grad(self) -> bool:
        return self._local_tensor.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self._local_tensor.requires_grad_(value)
        # Sync DTensor wrapper's requires_grad
        super().requires_grad_(value)

    def requires_grad_(self, requires_grad: bool = True):
        self._local_tensor.requires_grad_(requires_grad)
        super().requires_grad_(requires_grad)
        return self

    @property
    def grad_fn(self) -> Optional[torch.autograd.Function]:
        return self._local_tensor.grad_fn

    def grad_zero_(self):
        if self._local_tensor.grad is not None:
            self._local_tensor.grad.zero_()
        return self

    def detach(self):
        detached_local = self._local_tensor.detach()
        return self.__class__(detached_local, device_mesh=self._device_mesh, placements=self._placements)

    def detach_(self):
        self._local_tensor.detach_()
        super().detach_()
        return self

    # ====================== Computation graph related overrides ======================
    @property
    def is_leaf(self) -> bool:
        return self._local_tensor.is_leaf

    @property
    def retains_grad(self) -> bool:
        return self._local_tensor.retains_grad

    @retains_grad.setter
    def retains_grad(self, value: bool) -> None:
        self._local_tensor.retains_grad_(value)

    def backward(self, gradient=None, retain_graph=None, create_graph=False) -> None:
        self._local_tensor.backward(gradient, retain_graph, create_graph)

    # ====================== Metadata related overrides (sync with local_tensor) ======================
    @property
    def device(self) -> torch.device:
        return self._local_tensor.device

    @property
    def dtype(self) -> torch.dtype:
        return self._local_tensor.dtype

    @property
    def shape(self) -> torch.Size:
        return self._local_tensor.shape

    def type(self, dtype=None, non_blocking=False):
        if dtype is None:
            return self._local_tensor.type()
        new_local = self._local_tensor.to(dtype=dtype, non_blocking=non_blocking)
        return self.__class__(new_local, device_mesh=self._device_mesh, placements=self._placements)

    def size(self, dim: Optional[int] = None):
        return self._local_tensor.size(dim)

    @property
    def ndim(self) -> int:
        return self._local_tensor.ndim

    def data_ptr(self) -> int:
        # Force return local_tensor's data pointer (ensure address consistency)
        return self._local_tensor.data_ptr()

    def numel(self) -> int:
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
