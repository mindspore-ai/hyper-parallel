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
"""Shared grouped-linear autograd flow for native and fake-QAT strategies.

The lifecycle prepares forward operands, executes GMM, retains only the views
required by backward, prepares ``grad_output``, and executes dgrad/wgrad.
Format-specific strategies provide quantization, their forward weight
direction, group normalization, and backend calls. The shared template derives
the dgrad direction and owns weight-view cleanup.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.quantization.tensor import QuantizedTensor
from hyper_parallel.components.quantization.config import (
    GMM_GROUP_K,
    GMM_GROUP_M,
    GROUPED_NDIM,
    GROUP_LIST_COUNTS,
    GROUP_LIST_OFFSETS,
    MATRIX_NDIM,
)
from hyper_parallel.components.quantization.functional._saved_quantized import (
    restore_quantized_operands,
    save_quantized_operands,
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
    CAST_ROUTED_OUTPUT_TO_INPUT_DTYPE = False

    def _validate_inputs(
        self,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        group_list: torch.Tensor,
        group_list_type: int,
    ) -> None:
        """Validate the common live-weight and group metadata contract."""

        validate_grouped_linear_inputs(
            inputs,
            weight,
            group_list,
            group_list_type,
            weight_input_dim=-1,
            name=self.FORMAT_NAME,
        )

    @abstractmethod
    def normalize_group_list(
        self,
        group_list: torch.Tensor,
        group_list_type: int,
    ) -> tuple[torch.Tensor, int]:
        """Adapt grouped metadata to this backend's fixed contract.

        ``group_list_type=1`` means ``group_list`` contains per-expert token
        counts; ``group_list_type=0`` means it contains cumulative expert
        offsets. The returned tensor and type must describe the same
        representation. MXFP8 and HiFloat8 preserve either representation,
        while W4A8 strategies convert counts to cumulative offsets and return
        type 0 because their grouped-matmul path uses the offsets contract.
        The shared lifecycle passes this pair through forward and saves it for
        backward, so callers must not assume the returned type is always 0.
        """

    @abstractmethod
    def output_features(self, weight: torch.Tensor) -> int:
        """Return the output width ``O`` in the live ``[E, O, K]`` layout."""

    @property
    @abstractmethod
    def weight_forward_is_rowwise(self) -> bool:
        """Declare the weight view required by this strategy's forward NN GMM.

        Return ``True`` when the forward kernel consumes the row-wise weight
        view, or ``False`` when it consumes the column-wise view. This is a
        backend/kernel layout contract, not a user-facing quantization option:
        the shared lifecycle uses it to build the forward view and, when an
        input gradient is needed, the opposite view for the dgrad NT GMM. After
        the forward call, it releases the forward view and retains only the
        view required by backward.

        The current strategies use these layouts:

        * MXFP8 and HiFloat8 use the column-wise view for forward NN GMM.
        * Native W4A8 uses the row-wise view for forward NN GMM.
        * Fake W4A8 uses the column-wise view with block size 32, matching its
          ordinary MXFP8 GMM weight layout.

        Each strategy must declare its layout explicitly because no single
        default is valid for all supported kernels; an incorrect view can make
        the quantized weights incompatible with the selected GMM kernel.

        Returns:
            Whether the forward NN GMM consumes the row-wise weight view.
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

        # Shared validation gate for live [E, O, K] geometry and group
        # invariants. Errors use FORMAT_NAME as their prefix.
        self._validate_inputs(
            inputs,
            weight,
            group_list,
            group_list_type,
        )
        # Prepare the strategy-specific representation once. Keep the tensor
        # and its type paired: W4A8 converts counts to offsets/type 0, whereas
        # MXFP8 and HiFloat8 can preserve counts/type 1.
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
        # Backward must interpret the saved group list exactly as forward did.
        ctx.group_list_type = effective_group_list_type
        ctx.strategy = self
        ctx.empty_input = inputs.shape[0] == 0
        if ctx.empty_input:
            # Return the output width O even when the input is empty.
            ctx.save_for_backward(effective_group_list)
            return inputs.new_empty((0, self.output_features(weight)))

        # The adapter hands live ``[E, O, K]`` weights; transpose once to the
        # GMM-ready ``[E, K, N]`` that quantizers and operators consume.
        gmm_weight = weight.transpose(-2, -1).contiguous()
        needs_grad_input = inputs.requires_grad
        needs_grad_weight = weight.requires_grad
        # Quantize activations: rowwise for forward and colwise for wgrad.
        input_quant = self.quantize_input(
            inputs,
            rowwise=True,
            colwise=needs_grad_weight,
            group_list=effective_group_list,
            group_list_type=effective_group_list_type,
        )
        # The forward NN and dgrad NT kernels consume opposite weight views.
        forward_rowwise = self.weight_forward_is_rowwise
        # Always build the forward view; build the opposite dgrad view only
        # when the input gradient needs the weight.
        if forward_rowwise:
            weight_rowwise, weight_colwise = True, needs_grad_input
        else:
            weight_rowwise, weight_colwise = needs_grad_input, True
        # Quantize the GMM-ready [E, K, N] weight.
        weight_quant = self.quantize_weight(
            gmm_weight,
            rowwise=weight_rowwise,
            colwise=weight_colwise,
        )
        # Run the forward NN GMM and produce the output.
        output = self.grouped_matmul(
            input_quant,
            weight_quant,
            layout="NN",
            group_list=effective_group_list,
            group_type=GMM_GROUP_M,
            group_list_type=effective_group_list_type,
            output_dtype=inputs.dtype,
        )
        # Some CANN grouped-matmul kernels promote their result to float32
        # even when ``output_dtype`` is supplied.  The routed expert contract
        # is the input activation dtype; enforce it here so the next model
        # operator (for example BF16 RMSNorm) receives a supported pair.
        if output.dtype != inputs.dtype:
            output = output.to(dtype=inputs.dtype)

        # The forward view is spent; keep only the direction backward needs.
        input_quant.update_usage(rowwise=False, colwise=needs_grad_weight)
        # Dgrad consumes the opposite view. Release both views when it is unused.
        if not needs_grad_input:
            weight_quant.update_usage(rowwise=False, colwise=False)
        elif forward_rowwise:
            weight_quant.update_usage(rowwise=False, colwise=True)
        else:
            weight_quant.update_usage(rowwise=True, colwise=False)
        operands = (
            input_quant if needs_grad_weight else None,
            weight_quant if needs_grad_input else None,
        )
        ctx.quantized_operands_saved = all(
            operand is None or isinstance(operand, QuantizedTensor)
            for operand in operands
        )
        if ctx.quantized_operands_saved:
            save_quantized_operands(ctx, *operands, effective_group_list)
            ctx.input_quant = None
            ctx.weight_quant = None
        else:
            ctx.save_for_backward(effective_group_list)
            ctx.input_quant, ctx.weight_quant = operands
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
            (group_list,) = ctx.saved_tensors
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

        if ctx.quantized_operands_saved:
            group_list, input_quant, weight_quant = restore_quantized_operands(ctx)
        else:
            (group_list,) = ctx.saved_tensors
            input_quant = ctx.input_quant
            weight_quant = ctx.weight_quant
        # Quantize grad_output with the views required by backward.
        grad_quant = self.quantize_grad_output(
            grad_output,
            rowwise=needs_grad_input,
            colwise=needs_grad_weight,
            group_list=group_list,
            group_list_type=ctx.group_list_type,
        )
        grad_input = None
        grad_weight = None
        if needs_grad_input:
            # dgrad: grad_output × retained weight (NT).
            grad_input = self.grouped_matmul(
                grad_quant,
                weight_quant,
                layout="NT",
                group_list=group_list,
                group_type=GMM_GROUP_M,
                group_list_type=ctx.group_list_type,
                output_dtype=ctx.input_dtype,
            )
        if needs_grad_weight:
            # wgrad: retained input × grad_output (TN), producing [E, K, O].
            grad_weight = self.grouped_matmul(
                input_quant,
                grad_quant,
                layout="TN",
                group_list=group_list,
                group_type=GMM_GROUP_K,
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

    if inputs.ndim != MATRIX_NDIM:
        raise ValueError(
            f"{name} grouped linear inputs must be two-dimensional, "
            f"got shape {tuple(inputs.shape)}."
        )
    if weight.ndim != GROUPED_NDIM:
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
    if group_list_type not in (GROUP_LIST_OFFSETS, GROUP_LIST_COUNTS):
        raise ValueError(
            f"{name} grouped linear group_list_type must be "
            f"{GROUP_LIST_OFFSETS} or {GROUP_LIST_COUNTS}, "
            f"but got {group_list_type}."
        )


# PyTorch's Function base exposes variadic callbacks; this bridge intentionally
# specializes them for the grouped-linear ``apply`` contract.
class _GroupedLinearFunction(torch.autograd.Function):  # pylint: disable=abstract-method,arguments-differ
    """Bridge PyTorch autograd callbacks to the strategy selected for the call.

    Model adapters pass the strategy instance into ``apply`` as an argument;
    inside the bridge the strategy's own forward/backward lifecycle runs.
    Because ``Function`` forward/backward are static methods, instance state
    (quantizer, tile size, ...) can only reach the flow through the ``apply``
    arguments, so the strategy is carried on ``ctx`` for backward.
    """

    @staticmethod
    def forward(  # pylint: disable=arguments-differ
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
    def backward(  # pylint: disable=arguments-differ
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
