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
"""NPU grouped matrix multiplication function."""

from typing import Optional, Union

import torch

from hyper_parallel.components.ops import grouped_matmul as selected_grouped_matmul
from hyper_parallel.components.ops.impl.grouped_matmul import GMMFunction, npu_grouped_matmul

GroupList = Optional[Union[torch.Tensor, list[int], tuple[int, ...]]]

__all__ = ["GMMFunction", "GroupList", "grouped_matmul"]


def grouped_matmul(
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
    if bias is None and group_type == 0 and group_list_type == 0 and not gemm_fusion:
        if group_list is not None:
            ends = torch.as_tensor(group_list, device=x.device, dtype=torch.int64)
            return selected_grouped_matmul(x, weight, ends)
    return npu_grouped_matmul(
        x, weight, bias, group_list, group_type, group_list_type, gemm_fusion, original_weight,
    )
