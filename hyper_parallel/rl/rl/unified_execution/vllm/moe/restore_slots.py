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
"""Restore native expert outputs to the training router's unweighted top-k slots."""

import torch
import triton
import triton.language as tl


@triton.jit
def _restore_slots_kernel(values, ids, inverse, output, first: tl.constexpr, count: tl.constexpr,
                          hidden: tl.constexpr, block: tl.constexpr):
    """Gather local expert rows into router slots and zero slots owned by other ranks."""
    slot = tl.program_id(0)
    columns = tl.program_id(1) * block + tl.arange(0, block)
    expert = tl.load(ids + slot)
    owned = (expert >= first) & (expert < first + count)

    # Only the owner has a valid inverse index. Combine the ownership mask,
    # inverse gather and zero fill in one pass before the EP collective.
    row = tl.where(owned, tl.load(inverse + slot), 0)
    value = tl.load(values + row.to(tl.int64) * hidden + columns,
                    mask=owned & (columns < hidden), other=0)
    tl.store(output + slot.to(tl.int64) * hidden + columns, value, mask=columns < hidden)


def restore_slots(values: torch.Tensor, ids: torch.Tensor, inverse: torch.Tensor,
                  first: int, count: int) -> torch.Tensor:
    """Return [sources × tokens, top_k, hidden] with one contributing EP owner per slot."""
    hidden = values.shape[-1]
    slots = values.new_empty((*ids.shape, hidden))
    # Narrow tiles expose more independent work for single-token decode.
    block = min(triton.next_power_of_2(hidden), 256 if ids.numel() < 64 else 2048)
    grid = (ids.numel(), triton.cdiv(hidden, block))
    _restore_slots_kernel[grid](values, ids, inverse, slots, first, count, hidden, block)
    return slots
