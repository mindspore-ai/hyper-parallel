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
"""Expert-major grouped matrix multiplication adapters."""

from __future__ import annotations

from typing import Any, Optional, Protocol, Union

import torch

from ..factory import _select_for_call
from ..registry import OpRegistry
from ..spec import OpSpec
from .._dependency import require_package, require_symbol


class GroupedMatmulFn(Protocol):
    """Multiply each expert-major token slice by its assigned expert matrix.

    Cumulative group ends partition the input rows. Repeated ends represent
    empty experts. There is no bias or token reordering. The native reference
    retains gradient connections for unused expert matrices.
    """

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        """Multiply expert-major rows without a bias.

        Args:
            x: Token rows [tokens, in_features], already grouped by expert.
            weight: Expert matrices [experts, in_features, out_features].
            group_list: Nondecreasing cumulative row ends [experts]; the final end equals tokens.

        Returns:
            Rows [tokens, out_features] in the original expert-major order.
        """


GROUPED_MATMUL = OpSpec[GroupedMatmulFn](config_key='grouped_matmul')


def grouped_matmul(x: torch.Tensor, weight: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
    """Multiply expert-major rows without a bias.

    Args:
        x: Token rows [tokens, in_features], already grouped by expert.
        weight: Expert matrices [experts, in_features, out_features].
        group_list: Nondecreasing cumulative row ends [experts]; the final end equals tokens.

    Returns:
        Rows [tokens, out_features] in the original expert-major order.

    Notes:
        Multiply each expert-major token slice by its assigned expert matrix.

        Cumulative group ends partition the input rows. Repeated ends represent
        empty experts. There is no bias or token reordering. The native reference
        retains gradient connections for unused expert matrices.
    """
    operation = _select_for_call(GROUPED_MATMUL, device=x.device)
    return operation(x, weight, group_list)


def _native(x: torch.Tensor, weight: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
    # One small matmul per expert keeps the reference path independent of
    # grouped kernels without gathering an entire expert weight per token.
    """Multiply token rows by their expert matrices, retaining empty-expert gradients."""
    ends = group_list.tolist()
    starts = [0, *ends[:-1]]
    if len(ends) != weight.shape[0] or (ends[-1] if ends else 0) != x.shape[0]:
        raise ValueError("Grouped matmul boundaries must cover every token and contain one entry per expert")
    if any(start > end for start, end in zip(starts, ends)):
        raise ValueError("Grouped matmul boundaries must be nondecreasing")
    if not ends:
        return x @ weight.sum(dim=0)
    return torch.cat([x[start:end] @ expert for start, end, expert in zip(starts, ends, weight.unbind())])


GroupList = Optional[Union[torch.Tensor, list[int], tuple[int, ...]]]


class GMMFunction(torch.autograd.Function):
    """Autograd bridge for NPU grouped matrix multiplication."""

    @staticmethod
    def forward(
        ctx: Any,
        original_weight: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        group_args: tuple,
    ) -> torch.Tensor:
        """Run grouped matrix multiplication and save its backward state."""
        torch_npu = require_package("torch_npu")
        group_list, group_type, gemm_fusion, group_list_type, group_list_data_type = group_args

        npu_bias = [bias] if bias is not None else []

        outputs = torch_npu.npu_grouped_matmul(
            x=[x],
            weight=[weight],
            bias=npu_bias,
            group_list=group_list,
            split_item=3,
            group_type=group_type,
            group_list_type=group_list_type
        )

        if group_list_data_type == 0:
            ctx.save_for_backward(x, weight)
            ctx.group_list = group_list
        else:
            ctx.save_for_backward(x, weight, group_list)

        ctx.original_weight = original_weight
        ctx.gemm_fusion = gemm_fusion
        ctx.group_list_type = group_list_type
        ctx.group_list_data_type = group_list_data_type
        return outputs[0]

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> tuple:
        """Compute the input and grouped-weight gradients."""
        torch_npu = require_package("torch_npu")
        grad_outputs = grad_outputs.contiguous()

        if ctx.group_list_data_type == 0:
            x, weight = ctx.saved_tensors
            group_list = ctx.group_list
        else:
            x, weight, group_list = ctx.saved_tensors

        original_weight = ctx.original_weight

        if isinstance(group_list, (list, tuple)):
            group_list_tensor = torch.tensor(group_list, device=x.device, dtype=torch.int64)
        else:
            group_list_tensor = group_list

        dx_list = torch_npu.npu_grouped_matmul(
            x=[grad_outputs],
            weight=[weight.transpose(-1, -2)],
            bias=[],
            group_list=group_list_tensor,
            split_item=3,
            group_type=0,
            group_list_type=ctx.group_list_type
        )
        dx = dx_list[0]

        grad_weight = None
        use_fusion = False
        if ctx.gemm_fusion:
            if hasattr(original_weight, 'main_grad'):
                if original_weight.main_grad is not None:
                    use_fusion = True

        if use_fusion:
            n_size = original_weight.main_grad.shape[-1]
            main_grad_view = original_weight.main_grad.view(-1, n_size)

            torch_npu.npu_grouped_matmul_add_(
                main_grad_view,
                x,
                grad_outputs,
                group_list_tensor,
                transpose_x=True,
                transpose_weight=False,
                group_type=2
            )

            if getattr(weight, 'zero_out_wgrad', False):
                grad_weight = torch.zeros_like(weight)
            else:
                grad_weight = torch.empty_like(weight)

            original_weight.grad_added_to_main_grad = True
        else:
            numel = x.nelement()
            if (
                not torch.compiler.is_compiling()
                and isinstance(numel, int)
                and numel == 0
            ):
                grad_weight = torch.zeros_like(weight)
            else:
                grad_weight = torch_npu.npu_grouped_matmul(
                    x=[x.transpose(-1, -2)],
                    weight=[grad_outputs],
                    bias=[],
                    group_list=group_list_tensor,
                    split_item=2,
                    group_type=2,
                    group_list_type=ctx.group_list_type
                )[0]

        return None, dx, grad_weight, None, None


def npu_grouped_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    group_list: GroupList = None,
    group_type: int = 0,
    group_list_type: int = 0,
    gemm_fusion: bool = False,
    original_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply NPU grouped matrix multiplication.

    Args:
        x: Expert-major input tensor.
        weight: Grouped weight tensor.
        bias: Optional grouped bias tensor.
        group_list: Cumulative group boundaries as a tensor or integer sequence.
        group_type: Grouping axis accepted by ``npu_grouped_matmul``.
        group_list_type: Group-list representation accepted by the NPU kernel.
        gemm_fusion: Whether to accumulate the weight gradient into ``main_grad``.
        original_weight: Parameter that owns ``main_grad``. Defaults to ``weight``.

    Returns:
        The grouped matrix multiplication output.
    """
    group_list_data_type = 1 if isinstance(group_list, torch.Tensor) else 0
    group_args = (group_list, group_type, gemm_fusion, group_list_type, group_list_data_type)

    if original_weight is None:
        original_weight = weight

    return GMMFunction.apply(original_weight, x, weight, bias, group_args)


def _load_native() -> GroupedMatmulFn:
    return _native


def _load_npu() -> GroupedMatmulFn:
    """Load the NPU grouped matmul adapter with cumulative row boundaries."""
    require_symbol(require_package("torch_npu"), "npu_grouped_matmul")

    def run(x: torch.Tensor, weight: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        """Translate cumulative row boundaries for the selected grouped matmul kernel."""
        return npu_grouped_matmul(
            x, weight, bias=None, group_list=group_list.to(torch.int64), group_type=0, group_list_type=0,
        )

    return run


def _load_cuda() -> GroupedMatmulFn:
    require_symbol(torch, "_grouped_mm")

    def run(x: torch.Tensor, weight: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        """Translate cumulative row boundaries for the selected grouped matmul kernel."""
        reason = check_torch_grouped_mm(x)
        if reason is not None:
            raise ValueError(reason)
        return torch._grouped_mm(x, weight, offs=group_list.to(torch.int32))  # pylint: disable=protected-access

    return run


def check_torch_grouped_mm(x: torch.Tensor) -> str | None:
    """Check CUDA requirements shared by grouped matrix multiplication users."""
    if x.dtype != torch.bfloat16:
        return "torch_grouped_mm requires BF16 inputs"
    if torch.cuda.get_device_capability(x.device)[0] < 8:
        return "torch_grouped_mm requires compute capability 8.0 or later"
    return None


def register_grouped_matmul(registry: OpRegistry) -> None:
    """Register one grouped GEMM over local expert-major token rows."""
    registry.register_family("grouped_matmul", specs=(GROUPED_MATMUL,))
    registry.register_impl(GROUPED_MATMUL, name="native", devices=("cuda", "cpu", "npu"), load=_load_native)
    registry.register_impl(GROUPED_MATMUL, name="torch_npu", devices=("npu",), load=_load_npu)
    registry.register_impl(
        GROUPED_MATMUL, name="torch_grouped_mm", devices=("cuda",), load=_load_cuda,
    )
