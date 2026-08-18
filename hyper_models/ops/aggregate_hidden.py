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
"""Masked hidden-state aggregation function."""

from typing import Any, Tuple

import torch  # pylint: disable=forbidden-backend-import
import omni_training_custom_ops  # noqa: F401  # pylint: disable=unused-import


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


def aggregate_hidden(input_tensor: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Aggregate masked hidden states with the NPU custom operator.

    Args:
        input_tensor: Hidden states in ``[sequence, batch, hidden]`` layout.
        weight: Depthwise aggregation weights in ``[window, hidden]`` layout.
        mask: Boolean validity mask in ``[batch, sequence]`` layout.

    Returns:
        Aggregated hidden states.
    """
    return _AggregateHidden.apply(input_tensor, weight, mask)
