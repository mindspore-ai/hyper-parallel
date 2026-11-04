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
"""Offline numerical oracle for active expert ranges and borrowed weights.

This implementation intentionally materializes intermediate tensors. It is the
comparison oracle for the native phases, not a claim of fused-kernel performance.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional


@dataclass(frozen=True)
class ReferenceState:
    """Backward activations only; no borrowed home or replica weight references."""

    x: torch.Tensor
    preact: torch.Tensor
    boundaries: torch.Tensor


def grouped_linear(x: torch.Tensor, weights: Sequence[torch.Tensor], boundaries: torch.Tensor,
                   output_width: int) -> torch.Tensor:
    """Multiply nonempty expert ranges by independently owned matrix views."""
    if not weights:
        if x.shape[0] or boundaries.numel():
            raise ValueError("empty weights require an empty expert batch")
        return x.new_empty((0, output_width))
    if len(weights) != boundaries.numel():
        raise ValueError("each active expert requires one weight reference")
    if x.device.type == "npu":
        return torch.ops.npu.npu_grouped_matmul(
            [x], list(weights), bias=[], group_list=boundaries,
            split_item=3, group_type=0, group_list_type=0,
        )[0]
    outputs, start = [], 0
    for end, weight in zip(boundaries.tolist(), weights):
        outputs.append(x[start:end] @ weight)
        start = end
    return torch.cat(outputs)


def grouped_gradient(x: torch.Tensor, dy: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    """Accumulate BF16 expert partials in FP32 until the complete owner merge."""
    if not boundaries.numel():
        return torch.empty((0, x.shape[1], dy.shape[1]), device=x.device,
                           dtype=torch.float32 if x.dtype == torch.bfloat16 else x.dtype)
    if x.device.type == "npu":
        result = torch.zeros((boundaries.numel() * x.shape[1], dy.shape[1]),
                             dtype=torch.float32, device=x.device)
        torch.ops.npu.npu_grouped_matmul_add_(result, x, dy.contiguous(), boundaries,
                                            transpose_x=True, transpose_weight=False, group_type=2)
        return result.reshape(boundaries.numel(), x.shape[1], dy.shape[1])
    outputs, start = [], 0
    for end in boundaries.tolist():
        outputs.append(x[start:end].float().transpose(0, 1) @ dy[start:end].float()
                       if x.dtype == torch.bfloat16 else x[start:end].transpose(0, 1) @ dy[start:end])
        start = end
    return torch.stack(outputs)


def swiglu(preact: torch.Tensor) -> torch.Tensor:
    """Use the NPU baseline's activation rounding when running on device."""
    if preact.device.type == "npu":
        return torch.ops.npu.npu_swiglu(preact, dim=-1)
    gate, up = preact.chunk(2, dim=-1)
    return functional.silu(gate) * up


def swiglu_gradient(dy: torch.Tensor, preact: torch.Tensor) -> torch.Tensor:
    """Apply the same activation derivative used by the unfused baseline."""
    if preact.device.type == "npu":
        return torch.ops.npu.npu_swiglu_backward(dy.contiguous(), preact, dim=-1)
    gate, up = preact.chunk(2, dim=-1)
    return torch.cat((torch.ops.aten.silu_backward(dy * up, gate), dy * functional.silu(gate)), dim=-1)


@torch.no_grad()
def forward_values(x: torch.Tensor, gate: Sequence[torch.Tensor], down: Sequence[torch.Tensor],
                   boundaries: torch.Tensor, intermediate: int) -> tuple[torch.Tensor, ReferenceState]:
    """Evaluate two projections and retain X/Z, with weights borrowed for this call."""
    z = grouped_linear(x, gate, boundaries, 2 * intermediate)
    v = grouped_linear(swiglu(z), down, boundaries, x.shape[1])
    return v, ReferenceState(x, z, boundaries)


@torch.no_grad()
def backward_values(dv: torch.Tensor, state: ReferenceState, gate: Sequence[torch.Tensor],
                    down: Sequence[torch.Tensor], on_gradient: Callable[[str, torch.Tensor], None] | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run B_down, B_gate, then B_x; publish each role's dW at its ready point."""
    x, z, boundaries = state.x, state.preact, state.boundaries
    da = grouped_linear(dv, [weight.transpose(0, 1) for weight in down], boundaries, z.shape[1] // 2)
    dd = grouped_gradient(swiglu(z), dv, boundaries)
    if on_gradient is not None:
        on_gradient("down", dd)
    dz = swiglu_gradient(da, z)
    del da
    dg = grouped_gradient(x, dz, boundaries)
    if on_gradient is not None:
        on_gradient("gate", dg)
    dx = grouped_linear(dz, [weight.transpose(0, 1) for weight in gate], boundaries, x.shape[1])
    return dx, dg, dd


def combine(values: torch.Tensor, order: torch.Tensor, scores: torch.Tensor, tokens: int) -> torch.Tensor:
    """Preserve ordinary EP's elementwise product rounding before aggregation."""
    rows = order // scores.shape[-1]
    selected = scores.reshape(-1).index_select(0, order)
    weighted = values * selected.unsqueeze(-1)
    result = weighted.new_zeros((tokens, values.shape[-1]))
    result.index_add_(0, rows, weighted)
    return result


def combine_gradient(dy: torch.Tensor, values: torch.Tensor, order: torch.Tensor,
                     scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return selection gradients and score derivatives in original top-k order."""
    selected_dy = dy.index_select(0, order // scores.shape[-1])
    selected_scores = scores.reshape(-1).index_select(0, order)
    dv = (selected_dy * selected_scores.unsqueeze(-1)).to(values.dtype)
    ds = torch.empty_like(scores).reshape(-1)
    ds[order] = (selected_dy * values).sum(-1).to(scores.dtype)
    return dv, ds.reshape_as(scores)
