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
"""First-order weighted SwiGLU VJP without reconstructing the FC2 output."""
from __future__ import annotations

import torch


def _project(values, weight, groups):
    return torch.ops.npu.npu_grouped_matmul(
        [values], [weight], bias=[], group_list=groups, split_item=3, group_type=0, group_list_type=0,
    )[0]


def _accumulate(buffer, left, right, groups):
    if buffer is not None:
        torch.ops.npu.npu_grouped_matmul_add_(
            buffer.view(-1, buffer.shape[-1]), left, right, groups,
            transpose_x=True, transpose_weight=False, group_type=2, group_list_type=0,
        )


@torch.no_grad()
def grouped_swiglu_vjp(
    inputs: torch.Tensor, probabilities: torch.Tensor, cotangent: torch.Tensor,
    gate_up_weight: torch.Tensor, down_weight: torch.Tensor, counts: torch.Tensor,
    grad_gate_up: torch.Tensor | None, grad_down: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dX/dp and add expert gradients to private FP32 accumulators.

    Args:
        inputs: Expert-major BF16 NPU rows [R,H].
        probabilities: BF16 scores [R], with the original score-after cast.
        cotangent: Unweighted output gradient [R,H].
        gate_up_weight: Packed compute tensor [E,2I,H], not a new Parameter.
        down_weight: Existing expert down weight [E,H,I].
        counts: Rows per local expert [E].
        grad_gate_up: Optional contiguous FP32 [E,2I,H] accumulation buffer.
        grad_down: Optional contiguous FP32 [E,H,I] accumulation buffer.

    Returns:
        Input and score gradients. Reassociation can change BF16 rounding;
        first-order numerical parity must be checked against the original graph.
    """
    if not inputs.shape[0]:
        return torch.empty_like(inputs), torch.empty_like(probabilities)
    # Import only for the NPU path; CPU users need neither torch_npu nor Triton.
    from .npu_moe_probability_vjp import probability_vjp  # pylint: disable=C0415

    groups = counts.to(torch.int64).cumsum(0)
    gate_up = _project(inputs, gate_up_weight.transpose(1, 2), groups)
    activated = torch.ops.npu.npu_swiglu(gate_up, dim=-1)
    unweighted = _project(cotangent.contiguous(), down_weight, groups)
    grad_probability, grad_activation = probability_vjp(
        unweighted, activated, probabilities, scale_activation=grad_down is not None,
    )
    del unweighted
    _accumulate(grad_down, cotangent, activated, groups)
    del activated
    grad_hidden = torch.ops.npu.npu_swiglu_backward(grad_activation, gate_up, dim=-1)
    del grad_activation, gate_up
    _accumulate(grad_gate_up, grad_hidden, inputs, groups)
    grad_input = _project(grad_hidden, gate_up_weight, groups)
    return grad_input, grad_probability
