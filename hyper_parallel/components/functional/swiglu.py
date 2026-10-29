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
"""NPU SwiGLU function."""

import torch  # pylint: disable=forbidden-backend-import
import torch_npu


def swiglu(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Apply NPU-accelerated SwiGLU.

    Args:
        x: Input tensor containing the gate and up projections.
        dim: Dimension along which the input is split.

    Returns:
        The fused SwiGLU output.
    """
    return torch_npu.npu_swiglu(x, dim=dim)


def swiglu_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Apply the explicit NPU SwiGLU backward operator.

    Args:
        grad_output: Gradient of the fused SwiGLU output.
        x: Packed Gate/Up input used by the forward operator.
        dim: Dimension along which ``x`` is split.

    Returns:
        Gradient with respect to the packed input ``x``.
    """
    return torch_npu.npu_swiglu_backward(grad_output, x, dim=dim)
