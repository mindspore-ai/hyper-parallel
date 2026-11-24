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
"""Integer row placement shared by DSA and CSA2 Indexer return routes."""

import torch
import triton
import triton.language as tl


@triton.jit
def _restore_topk_rows(
    selected, received, local_sources, local_destinations, remote_destinations, output,
    LOCAL: tl.constexpr, REMOTE: tl.constexpr, WIDTH: tl.constexpr,
    ROWS: tl.constexpr, COLUMNS: tl.constexpr, LOCAL_TILES: tl.constexpr,
):
    program = tl.program_id(0)
    column = tl.arange(0, COLUMNS)
    if program < LOCAL_TILES:
        row = program.to(tl.int64) * ROWS + tl.arange(0, ROWS)
        source = tl.load(local_sources + row, row < LOCAL, other=0)
        destination = tl.load(local_destinations + row, row < LOCAL, other=0)
        mask = (row[:, None] < LOCAL) & (column[None, :] < WIDTH)
        value = tl.load(selected + source[:, None] * WIDTH + column[None, :], mask, other=0)
        tl.store(output + destination[:, None] * WIDTH + column[None, :], value, mask)
    else:
        row = (program - LOCAL_TILES).to(tl.int64) * ROWS + tl.arange(0, ROWS)
        destination = tl.load(remote_destinations + row, row < REMOTE, other=0)
        mask = (row[:, None] < REMOTE) & (column[None, :] < WIDTH)
        value = tl.load(received + row[:, None] * WIDTH + column[None, :], mask, other=0)
        tl.store(output + destination[:, None] * WIDTH + column[None, :], value, mask)


def restore_topk_rows(
    selected: torch.Tensor, received: torch.Tensor, indices: tuple[torch.Tensor, ...], output: torch.Tensor,
) -> torch.Tensor:
    """Launch a disjoint local/remote row placement after boundary validation."""
    result = output
    # Copy int64 IDs as paired words; no vector arithmetic may narrow the IDs.
    if output.dtype == torch.int64:
        selected, received, output = (value.view(torch.int32) for value in (selected, received, output))
    local, remote = indices[0].numel(), indices[2].numel()
    width = output.shape[1]
    columns = triton.next_power_of_2(width)
    rows = min(32, max(1, 4096 // columns))
    local_tiles = triton.cdiv(local, rows)
    _restore_topk_rows[(local_tiles + triton.cdiv(remote, rows),)](
        selected, received, *indices, output, local, remote, width, rows, columns, local_tiles,
    )
    return result
