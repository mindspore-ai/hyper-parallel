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
"""All local leaves in one native Level 3 BMM submission."""
import torch

from .plan import PikPlan
from .reduction import fixed_tree


def batched_leaves(x: torch.Tensor, weight: torch.Tensor, *, plan: PikPlan,
                   local_leaves: int) -> torch.Tensor:
    """Return BF16 [L,M,N] leaves; strided views avoid explicit input/weight packing."""
    rows, local_k = x.shape
    outputs = weight.shape[0]
    leaf_k = local_k // local_leaves
    left = x.reshape(rows, local_leaves, leaf_k).permute(1, 0, 2)
    right = weight.reshape(outputs, local_leaves, leaf_k).permute(1, 2, 0)
    if plan.leaf_layout == "packed":
        left, right = left.contiguous(), right.contiguous()
    return torch.bmm(left, right)


def local_gemm(x: torch.Tensor, weight: torch.Tensor, *, plan: PikPlan,
               local_leaves: int) -> torch.Tensor:
    """Compute one FP32 subtree without rounding at the local-rank boundary."""
    return fixed_tree(batched_leaves(x, weight, plan=plan, local_leaves=local_leaves))
