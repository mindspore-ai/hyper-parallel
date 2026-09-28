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
"""RD pointwise kernels; matrix multiplication stays on the existing path."""
import triton
import triton.language as tl


@triton.jit
def kda_rd_finish_round_kernel(
    product: tl.tensor, previous: tl.tensor, matrix: tl.tensor,
    row_count: tl.constexpr, block_rows: tl.constexpr, save_matrix: tl.constexpr,
) -> None:
    """Add the previous state and write a compact transfer for later rounds.

    Args:
        product: Fresh contiguous FP32 packed S/M product pointer.
        previous: Previous packed S/M pointer; only S is read.
        matrix: Compact M output pointer, unused when save_matrix is false.
        row_count: Number of flattened matrix rows.
        block_rows: Compile-time row tile size.
        save_matrix: Whether a later round consumes compact M.
    """
    rows = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    cols = tl.arange(0, 128)
    offsets = rows[:, None] * 256 + cols[None, :]
    left = tl.load(product + offsets, rows[:, None] < row_count, 0)
    old = tl.load(previous + offsets, rows[:, None] < row_count, 0)
    tl.store(product + offsets, left + old, rows[:, None] < row_count)
    if save_matrix:
        right = tl.load(product + offsets + 128, rows[:, None] < row_count, 0)
        tl.store(matrix + rows[:, None] * 128 + cols[None, :], right, rows[:, None] < row_count)


@triton.jit
def kda_rd_finish_to_state_kernel(
    product: tl.tensor, previous: tl.tensor, matrix: tl.tensor, state: tl.tensor,
    row_count: tl.constexpr, save_matrix: tl.constexpr,
) -> None:
    """Also write the compact state consumed by the final communication stage.

    Args:
        product: Fresh contiguous FP32 packed S/M product pointer.
        previous: Previous packed S/M pointer; only S is read.
        matrix: Compact M output pointer, unused when save_matrix is false.
        state: Compact S output pointer.
        row_count: Number of flattened matrix rows.
        save_matrix: Whether a later round consumes compact M.
    """
    rows = tl.program_id(0) * 64 + tl.arange(0, 64)
    cols = tl.arange(0, 128)
    packed = rows[:, None] * 256 + cols[None, :]
    compact = rows[:, None] * 128 + cols[None, :]
    left = tl.load(product + packed, rows[:, None] < row_count, 0)
    old = tl.load(previous + packed, rows[:, None] < row_count, 0)
    result = left + old
    tl.store(product + packed, result, rows[:, None] < row_count)
    tl.store(state + compact, result, rows[:, None] < row_count)
    if save_matrix:
        right = tl.load(product + packed + 128, rows[:, None] < row_count, 0)
        tl.store(matrix + compact, right, rows[:, None] < row_count)
