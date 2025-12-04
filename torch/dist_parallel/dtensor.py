import copy as cp
import torch
from torch._ops import OpOverload, OpOverloadPacket
from torch import Tensor
from typing import Tuple, Dict, Any, Optional, Callable
from .layout import Layout
from .spmd._op_dispatch import OpDispatcher
from .tensor_redistribution import _tensor_redistribution

_OP_DISPATCHER = OpDispatcher()

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
class DTensor(Tensor):
    _local_tensor: Tensor
    _layout: Layout

    def __new__(cls, local_tensor: Tensor, layout: Layout = None) -> "DTensor":
        if not layout:
            raise ValueError("Layout is None, must provide a Layout instance")

        # 创建 Tensor 子类实例，共享 local_tensor 的底层存储
        t = Tensor._make_subclass(cls, local_tensor, local_tensor.requires_grad)
        t._local_tensor = local_tensor
        t._layout = layout
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
        out = _OP_DISPATCHER.dispatch(func, args, kwargs)
        return out

    @property
    def layout(self):
        """Sharding state for dtensor"""
        if not hasattr(self, '_layout'):
            return None
        return self._layout

    @staticmethod
    def from_local(local_tensor: Tensor, layout: Layout):
        d_tensor =  DTensor(local_tensor, layout)
        return d_tensor

    def to_local(self):
        """covert global_tensor to local_tensor"""
        return self._local_tensor

    @property
    def shape(self):
        return self._layout.get_global_shape(self._local_tensor.shape)

    @property
    def local_shape(self):
        return self._local_tensor.shape

    def redistribute(self, dst_layout):
        """
        Redistribute dtensor to destination layout.
        """
        out = _tensor_redistribution.redistribution(self, dst_layout)
        return out

    def reduce_partial(self):
        """
        Reduce partial sharding state for dtensor.

        """
        if not self.layout:
            return self
        to_layout = cp.deepcopy(self.layout)
        to_layout.reset_partial()
        out = _tensor_redistribution.reduce_partial(self, to_layout)
        return out
    
    
    def to(self, device: str | torch.device) -> "DTensor":
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

    def requires_grad_(self, requires_grad: bool = True) -> "DTensor":
        self._local_tensor.requires_grad_(requires_grad)
        super().requires_grad_(requires_grad)
        return self

    @property
    def grad_fn(self) -> Optional[torch.autograd.Function]:
        return self._local_tensor.grad_fn

    def grad_zero_(self) -> "DTensor":
        if self._local_tensor.grad is not None:
            self._local_tensor.grad.zero_()
        return self

    def detach(self) -> "DTensor":
        detached_local = self._local_tensor.detach()
        return DTensor(detached_local, layout=self._layout)

    def detach_(self) -> "DTensor":
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
    def zero_(self) -> "DTensor":
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

    def copy_(self, src: Tensor, non_blocking: bool = False) -> "DTensor":
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

    def fill_(self, value: float | int) -> "DTensor":
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

