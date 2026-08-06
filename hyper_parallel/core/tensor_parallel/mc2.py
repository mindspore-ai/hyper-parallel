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
"""MC2 (matmul + communication) fused linear primitives for tensor parallelism.

PyTorch / Ascend exposes fused kernels via ``torch_npu.npu_all_gather_base_mm`` and
``torch_npu.npu_mm_reduce_scatter_base``. These kernels do not ship a complete
autograd formula for every training path, so this module wraps each fused forward
with a custom backward so they can be used inside a trainable TP + SP model.

The two kernels are mathematical duals (aligned with MindFormers MC2):

* ``all_gather_matmul``     forward : ``Y = AllGather_m(X) @ W^T``
                            backward: ``dX = matmul_reduce_scatter(dY, W)``
                                      ``dW = dY^T @ AllGather_m(X)``

* ``matmul_reduce_scatter`` forward : ``Y = ReduceScatter_m(X @ W^T)``
                            backward: ``dX, dY_full = all_gather_matmul(dY, W)``
                                      ``dW = AllGather_m(dY)^T @ X``

Both forward and backward ``dX`` paths use fused Ascend MC2 kernels. Callers must
satisfy kernel constraints (notably contraction dim ``k ∈ [256, 65535)``); for
column-parallel backward that ``k`` is local ``out_features`` (``n_local``).

All autograd functions operate on **local** (non-DTensor) 2-D tensors.
``W`` uses the native ``nn.Linear`` layout ``(out, in)``.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.platform import get_platform

platform = get_platform()

__all__ = [
    "get_hcomm_info",
    "AllGatherMatmulFunction",
    "MatmulReduceScatterFunction",
    "MC2Linear",
]


def get_hcomm_info(group: Any) -> str:
    """Resolve Ascend HCCL communicator name for a torch ProcessGroup.

    Args:
        group (Any): ``torch.distributed.ProcessGroup`` for the TP mesh axis.

    Returns:
        HCCL communicator handle name expected by ``torch_npu`` MC2 kernels.
    """
    rank = torch.distributed.get_rank(group)
    if torch.__version__ > "2.0":
        global_rank = torch.distributed.get_global_rank(group, rank)
        # torch.distributed ProcessGroup exposes HCCL via a private backend API.
        return group._get_backend(torch.device("npu")).get_hccl_comm_name(  # pylint: disable=protected-access
            global_rank
        )
    return group.get_hccl_comm_name(rank)


def _require_torch_npu():
    try:
        import torch_npu  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError(
            "MC2 fused kernels require torch_npu "
            "(npu_all_gather_base_mm / npu_mm_reduce_scatter_base)."
        ) from exc
    return torch_npu


def _normalize_sequence_dim(sequence_dim: int, ndim_leading: int) -> int:
    """Resolve a possibly-negative sequence dim against leading-rank count."""
    seq_dim = sequence_dim
    if seq_dim < 0:
        seq_dim += ndim_leading
    if seq_dim < 0 or seq_dim >= ndim_leading:
        raise RuntimeError(
            f"MC2Linear sequence_dim={sequence_dim} is out of range "
            f"for leading rank {ndim_leading}."
        )
    return seq_dim


def _move_dim_to_front(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Permute ``dim`` to axis 0 so fused AG/RS on flattened dim-0 is SP-correct."""
    if dim == 0:
        return tensor
    order = (dim,) + tuple(i for i in range(tensor.dim()) if i != dim)
    return tensor.permute(*order).contiguous()


def _move_front_to_dim(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Inverse of :func:`_move_dim_to_front`."""
    if dim == 0:
        return tensor
    # front -> dim: [1..dim] + [0] + [dim+1..]
    order = tuple(range(1, dim + 1)) + (0,) + tuple(range(dim + 1, tensor.dim()))
    return tensor.permute(*order).contiguous()


class AllGatherMatmulFunction(platform.Function):
    """Column-parallel fused all-gather + matmul with custom backward.

    Forward (local view): ``out = AllGather_m(x) @ w^T``.
    """

    @staticmethod
    def forward(ctx, x, w, group, world_size, bias):  # pylint: disable=arguments-differ
        """Run fused all-gather + matmul and stash tensors for backward."""
        torch_npu = _require_torch_npu()
        hcom = get_hcomm_info(group)
        # x2 = w.T -> physical (k, n_local); kernel computes AG(x) @ x2
        out, gathered = torch_npu.npu_all_gather_base_mm(
            x,
            w.t(),
            hcom,
            world_size,
            bias=None,
            gather_index=0,
            gather_output=True,
        )
        if bias is not None:
            out = out + bias
        ctx.save_for_backward(gathered, w)
        ctx.group = group
        ctx.world_size = world_size
        ctx.hcom = hcom
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, grad_out):  # pylint: disable=arguments-differ
        """Gradient: dx via fused matmul_reduce_scatter, dw via gathered-input matmul.

        Matches MindFormers: ``dX = matmul_reduce_scatter(dY, W)``. The fused
        kernel contracts over ``n_local`` (``W.shape[0]``), which must be in
        ``[256, 65535)``.
        """
        torch_npu = _require_torch_npu()
        gathered, w = ctx.saved_tensors
        # input (m_full, n_local) @ x2 (n_local, k) -> RS_m -> (m_local, k)
        grad_x = torch_npu.npu_mm_reduce_scatter_base(
            grad_out.contiguous(),
            w,
            ctx.hcom,
            ctx.world_size,
            reduce_op="sum",
            bias=None,
        )
        grad_w = grad_out.t().matmul(gathered)
        grad_bias = grad_out.sum(dim=0) if ctx.has_bias else None
        return grad_x, grad_w, None, None, grad_bias


class MatmulReduceScatterFunction(platform.Function):
    """Row-parallel fused matmul + reduce-scatter with custom backward.

    Forward (local view): ``out = ReduceScatter_m(x @ w^T)``.
    """

    @staticmethod
    def forward(ctx, x, w, group, world_size, bias):  # pylint: disable=arguments-differ
        """Run fused matmul + reduce-scatter and stash tensors for backward."""
        torch_npu = _require_torch_npu()
        hcom = get_hcomm_info(group)
        out = torch_npu.npu_mm_reduce_scatter_base(
            x,
            w.t(),
            hcom,
            world_size,
            reduce_op="sum",
            bias=None,
        )
        if bias is not None:
            out = out + bias
        ctx.save_for_backward(x, w)
        ctx.group = group
        ctx.world_size = world_size
        ctx.hcom = hcom
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, grad_out):  # pylint: disable=arguments-differ
        """Gradient: dx via fused all-gather+matmul, dw via gathered-grad matmul."""
        torch_npu = _require_torch_npu()
        x, w = ctx.saved_tensors
        # AG(dY) @ W : pass W (n, k) so the kernel uses transposed-x2 semantics.
        grad_x, grad_out_full = torch_npu.npu_all_gather_base_mm(
            grad_out,
            w,
            ctx.hcom,
            ctx.world_size,
            bias=None,
            gather_index=0,
            gather_output=True,
        )
        grad_w = grad_out_full.t().matmul(x)
        grad_bias = grad_out_full.sum(dim=0) if ctx.has_bias else None
        return grad_x, grad_w, None, None, grad_bias


class MC2Linear(nn.Linear):
    """``nn.Linear`` that uses fused matmul + TP communication kernels."""

    def configure_mc2(
        self,
        mode: str,
        group: Any,
        world_size: int,
        sequence_dim: int = 0,
    ) -> None:
        """Configure the fused collective used by this layer.

        Args:
            mode (str): ``\"all_gather\"`` (column) or ``\"reduce_scatter\"`` (row).
            group (Any): TP process group.
            world_size (int): Size of ``group``.
            sequence_dim (int, optional): Tensor dim that carries sequence sharding
                under SP. Default: ``0``.
        """
        if mode not in ("all_gather", "reduce_scatter"):
            raise ValueError(
                "For MC2Linear.configure_mc2, mode should be 'all_gather' or "
                f"'reduce_scatter', but got {mode}."
            )
        self.mc2_mode = mode
        self.mc2_group = group
        self.mc2_world_size = world_size
        self.mc2_sequence_dim = sequence_dim

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "MC2Linear":
        """Convert a Linear in place while preserving parameters and module state."""
        if not isinstance(linear, nn.Linear):
            raise TypeError(
                f"MC2Linear can only replace nn.Linear, but got {type(linear).__name__}."
            )
        linear.__class__ = cls
        return linear

    def _mc2_forward(self, input_: DTensor, weight: DTensor) -> DTensor:
        """Run the configured fused kernel on local tensors.

        Ascend MC2 kernels all-gather / reduce-scatter the **flattened dim-0**
        of a 2-D activation. That is only layout-correct when the SP sequence
        axis is the outermost leading dim. For ``sequence_dim != 0`` (e.g.
        batch-first ``[B, S, H]`` with ``Shard(1)``), move the sequence dim to
        front before flatten, then restore after the fused op.
        """
        input_local = input_.to_local()
        weight_local = weight.to_local()
        leading_global = tuple(int(s) for s in input_.shape[:-1])
        seq_dim = _normalize_sequence_dim(self.mc2_sequence_dim, len(leading_global))

        # [..., S_local_or_full, ..., H] -> [S_*, *other_leading, H] -> 2-D
        x_seq_first = _move_dim_to_front(input_local, seq_dim)
        other_leading = tuple(x_seq_first.shape[1:-1])
        input_2d = x_seq_first.reshape(-1, x_seq_first.shape[-1])

        bias_local = None
        if self.bias is not None:
            bias = self.bias
            bias_local = bias.to_local() if isinstance(bias, DTensor) else bias

        if self.mc2_mode == "all_gather":
            output_2d = AllGatherMatmulFunction.apply(
                input_2d,
                weight_local,
                self.mc2_group,
                self.mc2_world_size,
                bias_local,
            )
            # AG expands the sequence dim; other leading dims stay local sizes.
            seq_global = leading_global[seq_dim]
            output = output_2d.reshape(seq_global, *other_leading, output_2d.shape[-1])
            output = _move_front_to_dim(output, seq_dim)
            return DTensor.from_local(output, input_.device_mesh, (Shard(-1),))

        seq_global = leading_global[seq_dim]
        if seq_global % self.mc2_world_size != 0:
            raise RuntimeError(
                f"MC2Linear reduce_scatter requires sequence dim {seq_dim} "
                f"(size {seq_global}) divisible by world_size "
                f"{self.mc2_world_size}."
            )
        output_2d = MatmulReduceScatterFunction.apply(
            input_2d,
            weight_local,
            self.mc2_group,
            self.mc2_world_size,
            bias_local,
        )
        seq_local = seq_global // self.mc2_world_size
        output = output_2d.reshape(seq_local, *other_leading, output_2d.shape[-1])
        output = _move_front_to_dim(output, seq_dim)
        return DTensor.from_local(output, input_.device_mesh, (Shard(seq_dim),))

    def forward(self, input_: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward using MC2 for DTensor inputs and ``nn.Linear`` otherwise."""
        if not isinstance(input_, DTensor):
            return super().forward(input_)
        if not hasattr(self, "mc2_mode"):
            raise RuntimeError(
                "MC2Linear must be configured by an MC2 parallel style before use."
            )

        if weight is None:
            weight = self.weight
        if not isinstance(weight, DTensor):
            raise TypeError(
                "MC2Linear expects a DTensor weight after tensor-parallel sharding, "
                f"but got {type(weight).__name__}."
            )
        return self._mc2_forward(input_, weight)
