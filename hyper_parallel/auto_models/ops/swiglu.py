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
