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
# Portions copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li.
# FLA's MIT-licensed CP merge recurrence is adapted to separate saved M and gathered G.
# See FLA_LICENSE in this directory.
"""FLA-style one-kernel prefix/suffix with separate transfer storage and fused transpose."""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["count", "rank"])
def kda_cached_affine_merge_kernel(
    output: tl.tensor, states: tl.tensor, matrices: tl.tensor, count: tl.tensor, rank: tl.tensor,
    S_RANK: tl.constexpr, S_HEAD: tl.constexpr, S_ROW: tl.constexpr,
    M_RANK: tl.constexpr, M_HEAD: tl.constexpr, M_ROW: tl.constexpr,
    FORWARD: tl.constexpr, BV: tl.constexpr,
) -> None:
    """Retain the FP32 affine recurrence while avoiding repeated host GEMM launches.

    Args:
        output: Contiguous FP32 boundary output pointer.
        states: Chronological affine offsets [P,B,H,128,128].
        matrices: Matching chronological FP32 transfer matrices.
        count: Number of chronological offsets to merge.
        rank: Chronological rank within the CP group.
        S_RANK: Element stride between source-rank state slabs.
        S_HEAD: Element stride between flattened batch/head state slabs.
        S_ROW: Element stride between key rows in state storage.
        M_RANK: Element stride between source-rank transfer slabs.
        M_HEAD: Element stride between flattened batch/head transfer slabs.
        M_ROW: Element stride between rows in M.
        FORWARD: Compile-time selection of prefix or transposed suffix recurrence.
        BV: Value-column tile width.
    """
    column, head = tl.program_id(0), tl.program_id(1)
    rows = tl.arange(0, 128)
    cols = column*BV + tl.arange(0, BV)
    value = tl.zeros((128, BV), tl.float32)
    for index in range(count):
        source = index if FORWARD else rank+count-index
        offset = states + source*S_RANK + head*S_HEAD
        state = tl.load(offset + rows[:, None]*S_ROW + cols[None, :])
        offset_m = matrices + source*M_RANK + head*M_HEAD
        if FORWARD:
            offsets_m = rows[:, None]*M_ROW + rows[None, :]
        else:
            offsets_m = rows[None, :]*M_ROW + rows[:, None]
        matrix = tl.load(offset_m+offsets_m)
        value = tl.dot(matrix.to(tl.float32), value) + state.to(tl.float32)
    tl.store(output+head*16384+rows[:, None]*128+cols[None, :], value)
