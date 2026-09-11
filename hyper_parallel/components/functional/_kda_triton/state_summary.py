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
# The affine state decomposition follows the MIT-licensed implementation in
# flash-linear-attention/fla/ops/cp/chunk_delta_h.py.

# pylint: disable=invalid-name,missing-public-type-hints

"""Fixed-shape Triton-Ascend kernel for a packed KDA state summary."""

import triton
import triton.language as tl


@triton.jit
def _exp2(value):
    """Evaluate base-two exponentiation in FP32."""
    return tl.math.exp2(value.to(tl.float32))


@triton.jit(do_not_specialize=["T"])
def kda_split_state_summary_kernel(
    key,
    w,
    u,
    gate,
    state_ext,
    transition,
    T,
    H: tl.constexpr,
    H_TOTAL: tl.constexpr,
    HEAD_OFFSET: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    OUTPUT_MODE: tl.constexpr = 0,
    W_HEAD_FIRST: tl.constexpr = False,
):
    """Build ``S_ext`` and ``M`` directly without a packed-output copy."""
    i_col = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H + HEAD_OFFSET

    stride_k = H_TOTAL * K
    stride_w = K if W_HEAD_FIRST else stride_k
    stride_v = H_TOTAL * V
    key += (i_b * T * H_TOTAL + i_h) * K
    if W_HEAD_FIRST:
        w += (i_b * H_TOTAL + i_h) * T * K
    else:
        w += (i_b * T * H_TOTAL + i_h) * K
    u += (i_b * T * H_TOTAL + i_h) * V
    gate += (i_b * T * H_TOTAL + i_h) * K
    state_ext += i_bh * K * V
    transition += i_bh * K * K

    col = i_col * BV + tl.arange(0, BV)
    row1 = tl.arange(0, 64)
    row2 = 64 + tl.arange(0, 64)
    if OUTPUT_MODE == 1:
        is_transition = False
        transition_col = col
    elif OUTPUT_MODE == 2:
        is_transition = True
        transition_col = col
    else:
        is_transition = i_col * BV >= V
        transition_col = col - V
    state1 = tl.where(
        is_transition & (row1[:, None] == transition_col[None, :]),
        1.0,
        0.0,
    ).to(tl.float32)
    state2 = tl.where(
        is_transition & (row2[:, None] == transition_col[None, :]),
        1.0,
        0.0,
    ).to(tl.float32)

    num_chunks = tl.cdiv(T, BT)
    for chunk_idx in range(num_chunks):
        w1_ptr = tl.make_block_ptr(
            w,
            (T, K),
            (stride_w, 1),
            (chunk_idx * BT, 0),
            (BT, 64),
            (1, 0),
        )
        w2_ptr = tl.make_block_ptr(
            w,
            (T, K),
            (stride_w, 1),
            (chunk_idx * BT, 64),
            (BT, 64),
            (1, 0),
        )
        w1 = tl.load(w1_ptr, boundary_check=(0, 1))
        w2 = tl.load(w2_ptr, boundary_check=(0, 1))
        value_new = tl.dot(w1, state1.to(w1.dtype))
        value_new += tl.dot(w2, state2.to(w2.dtype))

        if OUTPUT_MODE == 2:
            value_new = -value_new
        else:
            u_col_offset = tl.where(is_transition, 0, i_col * BV)
            u_ptr = tl.make_block_ptr(
                u,
                (T, V),
                (stride_v, 1),
                (chunk_idx * BT, u_col_offset),
                (BT, BV),
                (1, 0),
            )
            u_value = tl.load(
                u_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
            )
            value_new = tl.where(is_transition, 0.0, u_value) - value_new
        value_new = value_new.to(key.dtype.element_ty)

        last_idx = min((chunk_idx + 1) * BT, T) - 1
        gate_last_ptr = gate + last_idx * H_TOTAL * K
        gate_last1 = tl.load(gate_last_ptr + row1, mask=row1 < K, other=0.0)
        gate_last2 = tl.load(gate_last_ptr + row2, mask=row2 < K, other=0.0)
        state1 *= _exp2(gate_last1)[:, None]
        state2 *= _exp2(gate_last2)[:, None]

        key1_ptr = tl.make_block_ptr(
            key,
            (K, T),
            (1, stride_k),
            (0, chunk_idx * BT),
            (64, BT),
            (0, 1),
        )
        key2_ptr = tl.make_block_ptr(
            key,
            (K, T),
            (1, stride_k),
            (64, chunk_idx * BT),
            (64, BT),
            (0, 1),
        )
        key1 = tl.load(key1_ptr, boundary_check=(0, 1))
        key2 = tl.load(key2_ptr, boundary_check=(0, 1))
        state1 += tl.dot(key1, value_new)
        state2 += tl.dot(key2, value_new)

    state_mask = (~is_transition) & (col[None, :] < V)
    transition_mask = is_transition & (transition_col[None, :] < K)
    tl.store(
        state_ext + row1[:, None] * V + col[None, :],
        state1,
        mask=state_mask,
    )
    tl.store(
        state_ext + row2[:, None] * V + col[None, :],
        state2,
        mask=state_mask,
    )
    tl.store(
        transition + row1[:, None] * K + transition_col[None, :],
        state1,
        mask=transition_mask,
    )
    tl.store(
        transition + row2[:, None] * K + transition_col[None, :],
        state2,
        mask=transition_mask,
    )


@triton.jit(do_not_specialize=["T"])
def kda_state_grad_ext_kernel(
    query,
    key,
    w,
    gate,
    grad_output,
    grad_value,
    grad_state_ext,
    scale,
    T,
    H: tl.constexpr,
    H_TOTAL: tl.constexpr,
    HEAD_OFFSET: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    """Build the local-output contribution to the incoming state gradient."""
    i_col = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H + HEAD_OFFSET

    stride_k = H_TOTAL * K
    stride_v = H_TOTAL * V
    query += (i_b * T * H_TOTAL + i_h) * K
    key += (i_b * T * H_TOTAL + i_h) * K
    w += (i_b * T * H_TOTAL + i_h) * K
    gate += (i_b * T * H_TOTAL + i_h) * K
    grad_output += (i_b * T * H_TOTAL + i_h) * V
    grad_value += (i_b * T * H_TOTAL + i_h) * V
    grad_state_ext += i_bh * K * V

    row1 = tl.arange(0, 64)
    row2 = 64 + tl.arange(0, 64)
    state1 = tl.zeros([64, BV], dtype=tl.float32)
    state2 = tl.zeros([64, BV], dtype=tl.float32)

    num_chunks = tl.cdiv(T, BT)
    for reverse_idx in range(num_chunks):
        chunk_idx = num_chunks - 1 - reverse_idx

        key1_ptr = tl.make_block_ptr(
            key,
            (T, K),
            (stride_k, 1),
            (chunk_idx * BT, 0),
            (BT, 64),
            (1, 0),
        )
        key2_ptr = tl.make_block_ptr(
            key,
            (T, K),
            (stride_k, 1),
            (chunk_idx * BT, 64),
            (BT, 64),
            (1, 0),
        )
        key1 = tl.load(key1_ptr, boundary_check=(0, 1))
        key2 = tl.load(key2_ptr, boundary_check=(0, 1))
        value_grad = tl.dot(key1, state1.to(key1.dtype))
        value_grad += tl.dot(key2, state2.to(key2.dtype))

        grad_value_ptr = tl.make_block_ptr(
            grad_value,
            (T, V),
            (stride_v, 1),
            (chunk_idx * BT, i_col * BV),
            (BT, BV),
            (1, 0),
        )
        value_grad += tl.load(grad_value_ptr, boundary_check=(0, 1))

        last_idx = min((chunk_idx + 1) * BT, T) - 1
        gate_last_ptr = gate + last_idx * H_TOTAL * K
        gate_last1 = tl.load(
            gate_last_ptr + row1,
            mask=row1 < K,
            other=0.0,
        )
        gate_last2 = tl.load(
            gate_last_ptr + row2,
            mask=row2 < K,
            other=0.0,
        )
        state1 *= _exp2(gate_last1)[:, None]
        state2 *= _exp2(gate_last2)[:, None]

        output_grad_ptr = tl.make_block_ptr(
            grad_output,
            (T, V),
            (stride_v, 1),
            (chunk_idx * BT, i_col * BV),
            (BT, BV),
            (1, 0),
        )
        output_grad = tl.load(output_grad_ptr, boundary_check=(0, 1))

        query1_ptr = tl.make_block_ptr(
            query,
            (K, T),
            (1, stride_k),
            (0, chunk_idx * BT),
            (64, BT),
            (0, 1),
        )
        query2_ptr = tl.make_block_ptr(
            query,
            (K, T),
            (1, stride_k),
            (64, chunk_idx * BT),
            (64, BT),
            (0, 1),
        )
        w1_ptr = tl.make_block_ptr(
            w,
            (K, T),
            (1, stride_k),
            (0, chunk_idx * BT),
            (64, BT),
            (0, 1),
        )
        w2_ptr = tl.make_block_ptr(
            w,
            (K, T),
            (1, stride_k),
            (64, chunk_idx * BT),
            (64, BT),
            (0, 1),
        )
        query1 = tl.load(query1_ptr, boundary_check=(0, 1))
        query2 = tl.load(query2_ptr, boundary_check=(0, 1))
        w1 = tl.load(w1_ptr, boundary_check=(0, 1))
        w2 = tl.load(w2_ptr, boundary_check=(0, 1))
        state1 += tl.dot(query1, output_grad.to(query1.dtype)) * scale
        state1 -= tl.dot(w1, value_grad.to(w1.dtype))
        state2 += tl.dot(query2, output_grad.to(query2.dtype)) * scale
        state2 -= tl.dot(w2, value_grad.to(w2.dtype))

    output1_ptr = tl.make_block_ptr(
        grad_state_ext,
        (K, V),
        (V, 1),
        (0, i_col * BV),
        (64, BV),
        (1, 0),
    )
    output2_ptr = tl.make_block_ptr(
        grad_state_ext,
        (K, V),
        (V, 1),
        (64, i_col * BV),
        (64, BV),
        (1, 0),
    )
    tl.store(
        output1_ptr,
        state1.to(output1_ptr.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        output2_ptr,
        state2.to(output2_ptr.dtype.element_ty),
        boundary_check=(0, 1),
    )


__all__ = [
    "kda_split_state_summary_kernel",
    "kda_state_grad_ext_kernel",
]
