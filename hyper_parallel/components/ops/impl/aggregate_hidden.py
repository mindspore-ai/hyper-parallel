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
"""Masked causal hidden-state aggregation without bias or a residual connection."""

from __future__ import annotations

from typing import Any, Protocol, Tuple

import torch
from torch.nn import functional as F

from .._dependency import require_package, require_symbol
from ..factory import _select_for_call
from ..registry import OpRegistry
from ..spec import OpSpec


class AggregateHiddenFn(Protocol):
    """Apply a masked causal depthwise filter without bias or residual addition.

    For output position t, sum input[t - K + 1 + k] * weight[k] over
    valid historical positions. The final tap multiplies the current token.
    Left padding is zero. The output mask does not remove historical inputs.
    Native FP16/BF16 products and sums use FP32 before restoring the input dtype.
    """

    def __call__(
        self, input_tensor: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """Map [S, B, H] states, [K, H] taps and a Boolean [B, S] output mask to [S, B, H].

        Args:
            input_tensor: Floating states [sequence, batch, hidden].
            weight: Filter taps [window, hidden], matching input device and dtype.
            mask: Boolean output-position mask [batch, sequence]; false positions become zero.

        Returns:
            Filtered states with the input shape, device and dtype; input and filter remain differentiable.
        """


def aggregate_hidden(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply masked causal depthwise aggregation using the configured backend.

    Args:
        input_tensor: Floating states [sequence, batch, hidden].
        weight: Filter taps [window, hidden], matching input device and dtype.
        mask: Boolean output-position mask [batch, sequence]; false positions become zero.

    Returns:
        Filtered states with the input shape, device and dtype; input and filter remain differentiable.

    Raises:
        ValueError: Shapes, dtypes or devices violate the aggregation contract.

    Notes:
        Apply a masked causal depthwise filter without bias or residual addition.

        For output position t, sum input[t - K + 1 + k] * weight[k] over
        valid historical positions. The final tap multiplies the current token.
        Left padding is zero. The output mask does not remove historical inputs.
        Native FP16/BF16 products and sums use FP32 before restoring the input dtype.
    """
    if input_tensor.ndim != 3 or weight.ndim != 2 or mask.ndim != 2:
        raise ValueError("aggregate_hidden requires input [S, B, H], weight [K, H] and mask [B, S]")
    sequence, batch, hidden = input_tensor.shape
    if hidden == 0 or weight.shape[0] == 0 or weight.shape[1] != hidden or mask.shape != (batch, sequence):
        raise ValueError("aggregate_hidden requires matching hidden/mask dimensions and a nonempty filter")
    if not input_tensor.is_floating_point() or weight.dtype != input_tensor.dtype or mask.dtype != torch.bool:
        raise ValueError("aggregate_hidden requires matching floating input/weight dtypes and a Boolean mask")
    if weight.device != input_tensor.device or mask.device != input_tensor.device:
        raise ValueError("aggregate_hidden requires input, weight and mask on the same device")
    return _select_for_call(AGGREGATE_HIDDEN, device=input_tensor.device)(input_tensor, weight, mask)


AGGREGATE_HIDDEN = OpSpec[AggregateHiddenFn](config_key='aggregate_hidden')


def native_aggregate_hidden(
    input_tensor: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Convolve with left zero padding; tap K-1 multiplies the current token."""
    if not input_tensor.numel():
        return input_tensor + weight[:0].sum().to(input_tensor.dtype)
    accumulation_dtype = (torch.float32 if input_tensor.dtype in (torch.float16, torch.bfloat16)
                          else input_tensor.dtype)
    values = input_tensor.permute(1, 2, 0).to(accumulation_dtype)
    kernels = weight.transpose(0, 1).unsqueeze(1).to(accumulation_dtype)
    mixed = torch.conv1d(F.pad(values, (weight.shape[0] - 1, 0)), kernels, groups=input_tensor.shape[-1])
    mixed = mixed.permute(2, 0, 1).to(input_tensor.dtype)
    return mixed * mask.transpose(0, 1).unsqueeze(-1)


class _AggregateHidden(torch.autograd.Function):
    """Autograd bridge for the aggregate-hidden custom operator."""

    @staticmethod
    def forward(
        ctx: Any,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the aggregate-hidden forward operator."""
        reason = _check_omni(input_tensor)
        if reason is not None:
            raise ValueError(reason)
        if not input_tensor.numel():
            raise ValueError("Omni aggregation requires nonempty states")
        ctx.save_for_backward(input_tensor, weight, mask)
        return torch.ops.custom.npu_aggregate_hidden(input_tensor, weight, mask=mask)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """Run the aggregate-hidden backward operator."""
        input_tensor, weight, mask = ctx.saved_tensors
        grad_input, grad_weight = torch.ops.custom.npu_aggregate_hidden_grad(
            grad_output.contiguous(),
            input_tensor,
            weight,
            mask=mask,
        )
        return grad_input, grad_weight, None


def load_omni_aggregate_hidden() -> AggregateHiddenFn:
    """Load the optional Omni autograd bridge for aggregation."""
    require_package("omni_training_custom_ops")
    require_symbol(torch.ops.custom, "npu_aggregate_hidden")
    require_symbol(torch.ops.custom, "npu_aggregate_hidden_grad")
    return _AggregateHidden.apply


def _check_omni(input_tensor: torch.Tensor) -> str | None:
    if input_tensor.dtype != torch.bfloat16:
        return "Omni aggregate_hidden requires BF16 inputs and weights"
    return None


def register_aggregate_hidden(registry: OpRegistry) -> None:
    """Register portable aggregation and the optional Ascend Omni implementation."""
    registry.register_family("aggregate_hidden", specs=(AGGREGATE_HIDDEN,))
    registry.register_impl(AGGREGATE_HIDDEN, name="native", devices=("cpu", "cuda", "npu"),
                           load=lambda: native_aggregate_hidden)
    registry.register_impl(AGGREGATE_HIDDEN, name="omni", devices=("npu",), load=load_omni_aggregate_hidden)
