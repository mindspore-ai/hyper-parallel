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
"""Transport-only HCCL with a fixed Triton tree at the receiving rank."""
from typing import Any

import torch
import torch.distributed as dist

from .reduction import fixed_tree


def combine_partials(partial: torch.Tensor, group: Any, world: int, rank: int, *,
                     scatter: bool, transport: str) -> torch.Tensor:
    """Combine FP32 rank subtrees; HCCL never performs the forward SUM."""
    if world == 1:
        return partial
    rows, columns = partial.shape
    if transport == "all_gather":
        received = torch.empty((world * rows, columns), dtype=torch.float32, device=partial.device)
        dist.all_gather_into_tensor(received, partial.contiguous(), group=group)
        root = fixed_tree(received.view(world, rows, columns))
        return root.narrow(0, rank * (rows // world), rows // world) if scatter else root
    received = torch.empty((world, rows // world, columns), dtype=torch.float32, device=partial.device)
    dist.all_to_all_single(received.view(-1), partial.contiguous().view(-1), group=group)
    owned = fixed_tree(received)
    if scatter:
        return owned
    result = torch.empty_like(partial)
    dist.all_gather_into_tensor(result, owned.contiguous(), group=group)
    return result


def gather_sequence_gradient(gradient: torch.Tensor, group: Any, world: int,
                             sequence_dim: int) -> torch.Tensor:
    """Gather disjoint SP gradient rows, without a floating-point reduction."""
    ordered = gradient.movedim(sequence_dim, 0).contiguous()
    shape = (ordered.shape[0] * world, *ordered.shape[1:])
    received = torch.empty(shape, dtype=gradient.dtype, device=gradient.device)
    dist.all_gather_into_tensor(received, ordered, group=group)
    return received.movedim(0, sequence_dim)
