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
"""Triton-Ascend receive map construction, imported only after backend detection."""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows"])
def _write_indices(ends, offsets, output, rows, SEGMENTS: tl.constexpr,
                   STEPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    valid = row < rows
    low = tl.full((BLOCK,), 0, tl.int32)
    high = tl.full((BLOCK,), SEGMENTS, tl.int32)
    for _ in range(STEPS):
        middle = (low + high) // 2
        end = tl.load(ends + middle, valid & (middle < SEGMENTS), other=0)
        right = (middle < SEGMENTS) & (end <= row)
        low = tl.where(right, middle + 1, low)
        high = tl.where(right, high, middle)
    shift = tl.load(offsets + low, valid & (low < SEGMENTS), other=0)
    tl.store(output + row, row + shift, valid)


def generate_receive_indices(source_offsets: torch.Tensor, counts: torch.Tensor,
                             total_received: int) -> torch.Tensor:
    """Write a map directly, without expanding each segment's offset first.

    Args:
        source_offsets: Contiguous integer source starts in expert-major order.
        counts: Contiguous integer lengths in expert-major order.
        total_received: Positive receive-buffer row count equal to counts.sum().

    Returns:
        Long tensor mapping expert-major destination rows to rank-major sources.
    """
    # A single segment is an identity map. Avoid a degenerate binary search,
    # which also triggers a compiler assertion with some Ascend toolchains.
    if counts.numel() == 1:
        return torch.arange(total_received, device=counts.device, dtype=torch.int64)
    ends = counts.cumsum(0)
    offsets = source_offsets - (ends - counts)
    output = torch.empty(total_received, device=counts.device, dtype=torch.int64)
    segments = counts.numel()
    # Upper-bound search skips empty segments, including repeated end offsets.
    # Rows are not specialized, avoiding recompilation for changing route counts.
    _write_indices[(triton.cdiv(total_received, 256),)](
        ends, offsets, output, total_received, segments, segments.bit_length(), 256,
    )
    return output
