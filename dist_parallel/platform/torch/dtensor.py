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
import copy as cp
import torch
from torch import Tensor
from typing import Tuple, Dict, Any, Optional, Callable

# ====================== 1. 适配 Ascend 设备的辅助判断 ======================
def is_ascend_available() -> bool:
    """判断 Ascend 设备是否可用（PyTorch Ascend 版以 npu 标识）"""
    try:
        #return False
        return torch.npu.is_available()
    except AttributeError:
        try:
            return torch.device("ascend").type == "ascend"
        except:
            return False

def get_ascend_device() -> torch.device:
    """获取 Ascend 设备对象（优先 npu:0，兼容 ascend）"""
    if is_ascend_available():
        try:
            return torch.device("npu:0")
        except:
            return torch.device("ascend")
    raise RuntimeError("Ascend device is not available!")


# ====================== 3. 完整的 DTensor 实现（同步存储 + 修复所有问题） ======================
class DTensorBase(Tensor):

    def __new__(cls, local_tensor, layout=None):
        if not layout:
            raise ValueError("Layout is None, must provide a Layout instance")

        # 创建 Tensor 子类实例，共享 local_tensor 的底层存储
        t = Tensor._make_subclass(cls, local_tensor, local_tensor.requires_grad)
        t.__init_data__(local_tensor, layout)
        return t

    @classmethod
    def __torch_function__(
        cls,
        func: torch._C._FunctionBase,
        types: Tuple[type, ...],
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        kwargs = kwargs or {}
        from dist_parallel.tensor_parallel._op_dispatch import _OP_DISPATCHER
        out = _OP_DISPATCHER.dispatch(func, args, kwargs)
        return out

    def to(self, device: str | torch.device):
        """兼容 Meta → 实际设备，返回新 DTensor（自动同步存储）"""
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

        # 重新创建 DTensor，保证新实例与 new_local 共享存储
        return DTensor(new_local, layout=self._layout)

    # ====================== 梯度相关重写 ======================
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
        # 同步 DTensor 外壳的 requires_grad
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
        return DTensor(detached_local, layout=self._layout)

    def detach_(self):
        self._local_tensor.detach_()
        super().detach_()
        return self

    # ====================== 计算图相关重写 ======================
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

    # ====================== 元数据相关重写（保证与 local_tensor 同步） ======================
    @property
    def device(self) -> torch.device:
        return self._local_tensor.device

    @property
    def dtype(self) -> torch.dtype:
        return self._local_tensor.dtype

    @property
    def shape(self) -> torch.Size:
        return self._local_tensor.shape

    def size(self, dim: Optional[int] = None) -> torch.Size | int:
        return self._local_tensor.size(dim)

    @property
    def ndim(self) -> int:
        return self._local_tensor.ndim

    def data_ptr(self) -> int:
        # 强制返回 local_tensor 的数据指针（保证地址一致）
        return self._local_tensor.data_ptr()

    def numel(self) -> int:
        return self._local_tensor.numel()

    # ====================== 数据操作重写（同步存储 + 修复原地操作） ======================
    def zero_(self):
        """原地置零：同步 DTensor 外壳与 local_tensor 存储"""
        if self._local_tensor.requires_grad and self._local_tensor.is_leaf:
            # 方案1：创建新张量 + 重新绑定 DTensor（保证存储共享）
            new_local = torch.zeros_like(self._local_tensor, requires_grad=True)
            # 关键：将 DTensor 外壳的存储同步到新 local_tensor
            super().copy_(new_local)  # 同步底层数据
            self._local_tensor = new_local  # 替换内部属性
        else:
            self._local_tensor.zero_()
            super().zero_()  # 同步外壳的原地置零
        return self

    def copy_(self, src: Tensor, non_blocking: bool = False):
        """原地复制：同步外壳与 local_tensor"""
        if self._local_tensor.requires_grad and self._local_tensor.is_leaf:
            new_local = src.to(self._local_tensor.device, non_blocking=non_blocking).detach().clone()
            new_local.requires_grad = self._local_tensor.requires_grad
            super().copy_(new_local)
            self._local_tensor = new_local
        else:
            self._local_tensor.copy_(src, non_blocking=non_blocking)
            super().copy_(src, non_blocking=non_blocking)
        return self

    def fill_(self, value: float | int):
        """原地填充：核心修复 + 同步存储"""
        if self._local_tensor.requires_grad and self._local_tensor.is_leaf:
            # 步骤1：创建新张量（非原地）
            new_local = torch.full_like(
                self._local_tensor,
                fill_value=value,
                requires_grad=True,
                device=self._local_tensor.device
            )
            # 步骤2：同步 DTensor 外壳的底层存储到新 local_tensor
            super().copy_(new_local)  # 关键：让 DTensor 外壳指向新地址
            # 步骤3：替换内部的 local_tensor（保证属性一致）
            self._local_tensor = new_local
        else:
            # 非叶子张量：直接原地填充 + 同步外壳
            self._local_tensor.fill_(value)
            super().fill_(value)  # 同步 DTensor 外壳的填充
        return self

    # ====================== 辅助打印 ======================
    def __repr__(self) -> str:
        return (
            f"DTensor(\n"
            f"  local_tensor={self._local_tensor},\n"
            f"  layout={self._layout},\n"
            f"  device={self.device},\n"
            f"  dtype={self.dtype},\n"
            f"  requires_grad={self.requires_grad},\n"
            f"  grad={self.grad},\n"
            f"  is_leaf={self.is_leaf},\n"
            f"  data_ptr={self.data_ptr()}\n"  # 打印数据指针，验证地址一致
            f")"
        )
