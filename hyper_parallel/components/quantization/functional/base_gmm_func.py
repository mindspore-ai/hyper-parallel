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
"""Shared native grouped-linear autograd flow.

The grouped-linear lifecycle is common to the native formats currently using
it: prepare the forward operands, execute GMM, retain only the views required
by backward, prepare ``grad_output``, and execute dgrad/wgrad. Format-specific
strategies provide operand preparation and backend calls. A future fake
strategy may reuse this boundary after its tensor-save/release semantics are
generalized; this module deliberately contains no fake-quantization code.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.quantization.tensor import (
    QuantizedTensor,
    QuantizedTensorStorage,
)


class GroupedLinear(ABC):
    """Own the common grouped-linear flow and expose format-specific hooks.

    The weight contract is fixed to the DeepSeek-V3 live layout
    ``[E, O, K]`` (contracting ``K`` at ``-1``): the adapter passes the packed
    source parameters through unchanged.  ``forward`` transposes each weight
    once, inside the flow, to the GMM-ready ``[E, K, N]`` that strategies and
    operators consume; ``backward`` transposes the wgrad result back to
    ``[E, O, K]``.
    """

    FORMAT_NAME: str

    @abstractmethod
    def normalize_group_list(
        self,
        group_list: torch.Tensor,
        group_list_type: int,
    ) -> tuple[torch.Tensor, int]:
        """Adapt grouped metadata to this backend's fixed contract.

        The adapter may hand per-expert token counts (``group_list_type=1``)
        or cumulative offsets (``group_list_type=0``). MXFP8 accepts either
        representation; W4A8 normalizes counts to cumulative offsets and
        leaves existing offsets unchanged.
        """

    @abstractmethod
    def output_features(self, weight: torch.Tensor) -> int:
        """Return the output width ``O`` in the live ``[E, O, K]`` layout."""

    @abstractmethod
    def weight_quantization_directions(
        self,
        needs_grad_input: bool,
    ) -> tuple[bool, bool]:
        """Return the row/column weight views needed for forward and dgrad."""

    @abstractmethod
    def retain_weight_backward(
        self,
        weight_quant: QuantizedTensorStorage,
        needs_grad_input: bool,
    ) -> None:
        """Release the forward-only weight view after the forward GMM.

        Keeps the direction dgrad consumes: row-wise for MXFP8 (``rowwise``
        kept), transposed column-wise for W4A8 (``colwise`` kept).  The
        input/gradient views are released inline in the template instead,
        because no native format varies them.
        """

    @abstractmethod
    def quantize_input(
        self,
        inputs: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
        group_list: torch.Tensor,
        group_list_type: int,
    ) -> QuantizedTensor:
        """Create the format-specific activation representation."""

    @abstractmethod
    def quantize_weight(
        self,
        weight: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
    ) -> QuantizedTensor:
        """Create the format-specific weight representation."""

    @abstractmethod
    def quantize_grad_output(
        self,
        grad_output: torch.Tensor,
        *,
        rowwise: bool,
        colwise: bool,
        group_list: torch.Tensor,
        group_list_type: int,
    ) -> QuantizedTensor:
        """Create the format-specific gradient-output representation."""

    @abstractmethod
    def grouped_matmul(
        self,
        left: QuantizedTensor,
        right: QuantizedTensor,
        *,
        layout: str,
        group_list: torch.Tensor,
        group_type: int,
        group_list_type: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Execute one format-specific grouped matrix multiplication."""

    def forward(
        self,
        ctx: torch.autograd.function.FunctionCtx,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        group_list: torch.Tensor,
        group_list_type: int,
    ) -> torch.Tensor:
        """Run the complete forward lifecycle using this strategy's hooks."""

        # 共享校验门禁——live [E,O,K] 几何/group 不变量，
        # 错误前缀取 FORMAT_NAME
        validate_grouped_linear_inputs(
            inputs,
            weight,
            group_list,
            group_list_type,
            weight_input_dim=-1,
            name=self.FORMAT_NAME,
        )
        # hook: group 元数据归一化——W4A8 覆写为 counts→offsets
        effective_group_list, effective_group_list_type = self.normalize_group_list(
            group_list,
            group_list_type,
        )
        ctx.input_shape = inputs.shape
        ctx.input_dtype = inputs.dtype
        ctx.input_device = inputs.device
        ctx.weight_shape = weight.shape
        ctx.weight_dtype = weight.dtype
        ctx.weight_device = weight.device
        ctx.group_list = effective_group_list
        ctx.group_list_type = effective_group_list_type
        ctx.strategy = self
        ctx.empty_input = inputs.shape[0] == 0
        if ctx.empty_input:
            # abstract: 给出输出宽 O，空输入也能产出 (0,O)
            return inputs.new_empty((0, self.output_features(weight)))

        # The adapter hands live ``[E, O, K]`` weights; transpose once to the
        # GMM-ready ``[E, K, N]`` that quantizers and operators consume.
        gmm_weight = weight.transpose(-2, -1).contiguous()
        needs_grad_input = inputs.requires_grad
        needs_grad_weight = weight.requires_grad
        # abstract: 量化激活（行视图给 forward，列视图给 wgrad）
        input_quant = self.quantize_input(
            inputs,
            rowwise=True,
            colwise=needs_grad_weight,
            group_list=effective_group_list,
            group_list_type=effective_group_list_type,
        )
        # hook: 决定要构建的 weight 视图——MXFP8 列向 fwd、W4A8 行向 fwd
        weight_rowwise, weight_colwise = self.weight_quantization_directions(
            needs_grad_input
        )
        # abstract: 量化 GMM-ready [E,K,N] weight
        weight_quant = self.quantize_weight(
            gmm_weight,
            rowwise=weight_rowwise,
            colwise=weight_colwise,
        )
        # abstract: 执行前向 NN GMM，产出真实输出
        output = self.grouped_matmul(
            input_quant,
            weight_quant,
            layout="NN",
            group_list=effective_group_list,
            group_type=0,
            group_list_type=effective_group_list_type,
            output_dtype=inputs.dtype,
        )

        ctx.input_quant = input_quant if needs_grad_weight else None
        ctx.weight_quant = weight_quant if needs_grad_input else None
        # The forward view is spent; keep only the direction backward needs.
        input_quant.update_usage(rowwise=False, colwise=needs_grad_weight)
        # hook: 只保留 dgrad 需要的 weight 方向——W4A8 覆写
        self.retain_weight_backward(weight_quant, needs_grad_input)
        return output

    def backward(
        self,
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run the complete dgrad/wgrad lifecycle using this strategy's hooks."""

        needs_grad_input = ctx.needs_input_grad[0]
        needs_grad_weight = ctx.needs_input_grad[1]
        if ctx.empty_input:
            grad_input = (
                torch.zeros(
                    ctx.input_shape,
                    dtype=ctx.input_dtype,
                    device=ctx.input_device,
                )
                if needs_grad_input
                else None
            )
            grad_weight = (
                torch.zeros(
                    ctx.weight_shape,
                    dtype=ctx.weight_dtype,
                    device=ctx.weight_device,
                )
                if needs_grad_weight
                else None
            )
            return grad_input, grad_weight

        # abstract: 量化 grad_output，保留 backward 需要的两个方向
        grad_quant = self.quantize_grad_output(
            grad_output,
            rowwise=needs_grad_input,
            colwise=needs_grad_weight,
            group_list=ctx.group_list,
            group_list_type=ctx.group_list_type,
        )
        grad_input = None
        grad_weight = None
        if needs_grad_input:
            # abstract: dgrad——grad_output × 保留的 weight（NT）
            grad_input = self.grouped_matmul(
                grad_quant,
                ctx.weight_quant,
                layout="NT",
                group_list=ctx.group_list,
                group_type=0,
                group_list_type=ctx.group_list_type,
                output_dtype=ctx.input_dtype,
            )
        if needs_grad_weight:
            # abstract: wgrad——保留的 input × grad_output（TN，得 [E,K,O]）
            grad_weight = self.grouped_matmul(
                ctx.input_quant,
                grad_quant,
                layout="TN",
                group_list=ctx.group_list,
                group_type=2,
                group_list_type=ctx.group_list_type,
                output_dtype=ctx.weight_dtype,
            )
            # The TN wgrad is GMM-shaped ``[E, K, O]``; transpose back to the
            # live ``[E, O, K]`` parameter layout.
            grad_weight = grad_weight.transpose(-2, -1).contiguous()

        grad_quant.update_usage(rowwise=False, colwise=False)
        ctx.input_quant = None
        ctx.weight_quant = None
        return grad_input, grad_weight


def validate_grouped_linear_inputs(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int,
    *,
    weight_input_dim: int,
    name: str,
) -> None:
    """Validate shape and grouping invariants shared by native formats."""

    if inputs.ndim != 2:
        raise ValueError(
            f"{name} grouped linear inputs must be two-dimensional, "
            f"got shape {tuple(inputs.shape)}."
        )
    if weight.ndim != 3:
        raise ValueError(
            f"{name} grouped linear weight must be three-dimensional, "
            f"got shape {tuple(weight.shape)}."
        )
    if inputs.shape[-1] != weight.shape[weight_input_dim]:
        raise ValueError(
            f"{name} grouped linear contracting dimensions differ: "
            f"inputs={inputs.shape[-1]}, weight={weight.shape[weight_input_dim]}."
        )
    if not isinstance(group_list, torch.Tensor) or group_list.ndim != 1:
        raise ValueError(f"{name} grouped linear group_list must be one-dimensional.")
    if group_list.shape[0] != weight.shape[0]:
        raise ValueError(
            f"{name} grouped linear requires one group per expert: "
            f"groups={group_list.shape[0]}, experts={weight.shape[0]}."
        )
    if group_list_type not in (0, 1):
        raise ValueError(
            f"{name} grouped linear group_list_type must be 0 or 1, "
            f"but got {group_list_type}."
        )


class _GroupedLinearFunction(torch.autograd.Function):
    """Bridge PyTorch autograd callbacks to the strategy selected for the call.

    Model adapters pass the strategy instance into ``apply`` as an argument;
    inside the bridge the strategy's own forward/backward lifecycle runs.
    Because ``Function`` forward/backward are static methods, instance state
    (quantizer, tile size, ...) can only reach the flow through the ``apply``
    arguments, so the strategy is carried on ``ctx`` for backward.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        group_list: torch.Tensor,
        strategy: GroupedLinear,
        group_list_type: int,
    ) -> torch.Tensor:
        """Delegate the complete forward lifecycle to the strategy."""

        return strategy.forward(
            ctx,
            inputs,
            weight,
            group_list,
            group_list_type,
        )

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        None,
        None,
        None,
    ]:
        """Delegate the complete backward lifecycle to the strategy."""

        grad_input, grad_weight = ctx.strategy.backward(ctx, grad_output)
        return grad_input, grad_weight, None, None, None


__all__ = [
    "GroupedLinear",
    "_GroupedLinearFunction",
    "validate_grouped_linear_inputs",
]
