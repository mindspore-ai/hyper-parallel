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
"""Ascend grouped-GEMM implementation for packed SwiGLU experts."""

from typing import Any


_NPU_GROUPED_MATMUL = None


def _get_npu_grouped_matmul() -> Any:
    """Build and cache the autograd wrapper after the Torch backend is selected."""
    global _NPU_GROUPED_MATMUL  # pylint: disable=global-statement
    if _NPU_GROUPED_MATMUL is not None:
        return _NPU_GROUPED_MATMUL

    import torch  # pylint: disable=C0415
    import torch_npu  # pylint: disable=C0415

    class _NPUGroupedMatmul(torch.autograd.Function):  # pylint: disable=abstract-method
        """Autograd wrapper around ``torch_npu.npu_grouped_matmul``."""

        @staticmethod
        def forward(ctx: Any, inputs: Any, weight: Any, group_list: Any) -> Any:
            """Run one grouped matrix multiplication and retain backward inputs."""
            ctx.save_for_backward(inputs, weight, group_list)
            return torch_npu.npu_grouped_matmul(
                x=[inputs],
                weight=[weight],
                bias=[],
                group_list=group_list,
                split_item=3,
                group_type=0,
                group_list_type=0,
            )[0]

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, Any, None]:
            """Compute grouped input and weight gradients."""
            inputs, weight, group_list = ctx.saved_tensors
            grad_output = grad_output.contiguous()
            grad_inputs = torch_npu.npu_grouped_matmul(
                x=[grad_output],
                weight=[weight.transpose(-1, -2)],
                bias=[],
                group_list=group_list,
                split_item=3,
                group_type=0,
                group_list_type=0,
            )[0]
            if inputs.numel() == 0:
                grad_weight = torch.zeros_like(weight)
            else:
                grad_weight = torch_npu.npu_grouped_matmul(
                    x=[inputs.transpose(0, 1)],
                    weight=[grad_output],
                    bias=[],
                    group_list=group_list,
                    split_item=2,
                    group_type=2,
                    group_list_type=0,
                )[0]
            return grad_inputs, grad_weight, None

    _NPU_GROUPED_MATMUL = _NPUGroupedMatmul
    return _NPU_GROUPED_MATMUL


def npu_grouped_swiglu(
    inputs: Any,
    gate_up_weight: Any,
    down_weight: Any,
    tokens_per_expert: Any,
) -> Any:
    """Run packed local SwiGLU experts with two grouped GEMMs.

    Args:
        inputs: Expert-major routed tokens with shape ``[T, H]``.
        gate_up_weight: Packed projection weights with shape ``[E, 2I, H]``.
        down_weight: Down projection weights with shape ``[E, H, I]``.
        tokens_per_expert: Token count for each local expert, shape ``[E]``.

    Returns:
        Expert-major output with shape ``[T, H]``.

    Raises:
        ValueError: If called for a non-NPU tensor or incompatible shapes.
    """
    if inputs.device.type != "npu":
        raise ValueError("npu_grouped_swiglu requires inputs on an Ascend NPU")
    if gate_up_weight.ndim != 3 or down_weight.ndim != 3:
        raise ValueError("grouped SwiGLU weights must be three-dimensional")
    if gate_up_weight.shape[0] != tokens_per_expert.numel():
        raise ValueError("tokens_per_expert must contain one count per local expert")

    import torch  # pylint: disable=C0415
    import torch_npu  # pylint: disable=C0415

    grouped_matmul = _get_npu_grouped_matmul()
    group_list = torch.cumsum(tokens_per_expert.to(torch.int64), dim=0)
    gate_up = grouped_matmul.apply(
        inputs,
        gate_up_weight.transpose(1, 2),
        group_list,
    )
    intermediate = torch_npu.npu_swiglu(gate_up, dim=-1)
    return grouped_matmul.apply(
        intermediate,
        down_weight.transpose(1, 2),
        group_list,
    )
