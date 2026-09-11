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
# -*- coding: utf-8 -*-
#
# The state-summary decomposition follows the MIT-licensed context-parallel
# implementation in flash-linear-attention/fla/ops/cp/chunk_delta_h.py.

# pylint: disable=missing-public-type-hints,invalid-name

"""Fixed-shape Triton-Ascend kernels for GDN state summaries."""

import triton
import triton.language as tl

from .utils import get_autotune_config


@triton.autotune(
    configs=get_autotune_config(
        multibuffer_list=(True, False),
        set_workspace_multibuffer_list=(2, 4),
        tile_mix_vector_loop_num_list=(2,),
        tile_mix_cube_loop_num_list=(2,),
    ),
    key=["H", "K", "V", "BT", "BV"],
)
@triton.jit(do_not_specialize=["T"])
def gdn_packed_state_summary_kernel(
    k,
    w,
    u,
    g,
    packed_summary,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
):
    """Build the local affine state transition and extension in one buffer."""
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    stride_k = H * K
    stride_v = H * V
    k += (i_b * T * H + i_h) * K
    w += (i_b * T * H + i_h) * K
    u += (i_b * T * H + i_h) * V
    g += i_b * T * H + i_h
    packed_summary += i_bh * K * (V + K)

    col = tl.arange(0, BV)
    row1 = tl.arange(0, 64)
    row2 = 64 + tl.arange(0, 64)
    is_transition = i_v * BV >= V
    transition_col = i_v * BV - V + col
    b_h1 = tl.where(
        is_transition & (row1[:, None] == transition_col[None, :]), 1.0, 0.0
    ).to(tl.float32)
    b_h2 = tl.where(
        is_transition & (row2[:, None] == transition_col[None, :]), 1.0, 0.0
    ).to(tl.float32)

    for i_t in range(NT):
        p_w1 = tl.make_block_ptr(
            w, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        p_w2 = tl.make_block_ptr(
            w, (T, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0)
        )
        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        b_w2 = tl.load(p_w2, boundary_check=(0, 1))
        b_v = tl.dot(b_w1, b_h1.to(b_w1.dtype))
        b_v += tl.dot(b_w2, b_h2.to(b_w2.dtype))

        p_u = tl.make_block_ptr(
            u,
            (T, V),
            (stride_v, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_u, boundary_check=(0, 1)) - b_v

        last_idx = min((i_t + 1) * BT, T) - 1
        token = i_t * BT + tl.arange(0, BT)
        mask = token < T
        b_g_last = tl.load(g + last_idx * H).to(tl.float32)
        b_g = tl.load(g + token * H, mask=mask, other=0.0).to(tl.float32)
        b_v *= tl.where(mask, tl.exp(b_g_last - b_g), 0.0)[:, None]
        decay = tl.exp(b_g_last)
        b_h1 *= decay
        b_h2 *= decay
        b_v = b_v.to(k.dtype.element_ty)

        p_k1 = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        p_k2 = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
        )
        b_h1 += tl.dot(tl.load(p_k1, boundary_check=(0, 1)), b_v)
        b_h2 += tl.dot(tl.load(p_k2, boundary_check=(0, 1)), b_v)

    p_out1 = tl.make_block_ptr(
        packed_summary,
        (K, V + K),
        (V + K, 1),
        (0, i_v * BV),
        (64, BV),
        (1, 0),
    )
    p_out2 = tl.make_block_ptr(
        packed_summary,
        (K, V + K),
        (V + K, 1),
        (64, i_v * BV),
        (64, BV),
        (1, 0),
    )
    tl.store(p_out1, b_h1, boundary_check=(0, 1))
    tl.store(p_out2, b_h2, boundary_check=(0, 1))


@triton.autotune(
    configs=get_autotune_config(
        multibuffer_list=(True, False),
        set_workspace_multibuffer_list=(2, 4),
        tile_mix_vector_loop_num_list=(2,),
        tile_mix_cube_loop_num_list=(2,),
    ),
    key=["H", "K", "V", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def gdn_state_grad_ext_kernel(
    q,
    k,
    w,
    g,
    do,
    dv,
    grad_state_ext,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
):
    """Build the local-loss contribution to the incoming state gradient."""
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    stride_k = H * K
    stride_v = H * V
    q += (i_b * T * H + i_h) * K
    k += (i_b * T * H + i_h) * K
    w += (i_b * T * H + i_h) * K
    g += i_b * T * H + i_h
    do += (i_b * T * H + i_h) * V
    dv += (i_b * T * H + i_h) * V
    grad_state_ext += i_bh * K * V

    b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
    b_dh2 = tl.zeros([64, BV], dtype=tl.float32)

    for reverse_idx in range(NT):
        i_t = NT - 1 - reverse_idx
        last_idx = min((i_t + 1) * BT, T) - 1
        token = i_t * BT + tl.arange(0, BT)
        mask = token < T
        b_g_last = tl.load(g + last_idx * H).to(tl.float32)
        b_g = tl.load(g + token * H, mask=mask, other=0.0).to(tl.float32)

        p_k1 = tl.make_block_ptr(
            k, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        p_k2 = tl.make_block_ptr(
            k, (T, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0)
        )
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))
        b_k2 = tl.load(p_k2, boundary_check=(0, 1))
        b_dv = tl.dot(b_k1, b_dh1.to(b_k1.dtype))
        b_dv += tl.dot(b_k2, b_dh2.to(b_k2.dtype))
        b_dv *= tl.where(mask, tl.exp(b_g_last - b_g), 0.0)[:, None]

        p_dv = tl.make_block_ptr(
            dv,
            (T, V),
            (stride_v, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_dv += tl.load(p_dv, boundary_check=(0, 1))

        p_do = tl.make_block_ptr(
            do,
            (T, V),
            (stride_v, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_do = tl.load(p_do, boundary_check=(0, 1))
        decay = tl.exp(b_g_last)
        b_dh1 *= decay
        b_dh2 *= decay

        p_q1 = tl.make_block_ptr(
            q, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        p_q2 = tl.make_block_ptr(
            q, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
        )
        p_w1 = tl.make_block_ptr(
            w, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        p_w2 = tl.make_block_ptr(
            w, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
        )
        b_q1 = tl.load(p_q1, boundary_check=(0, 1))
        b_q2 = tl.load(p_q2, boundary_check=(0, 1))
        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        b_w2 = tl.load(p_w2, boundary_check=(0, 1))
        gate = tl.exp(b_g)[None, :]
        b_q1 *= gate
        b_q2 *= gate
        b_dh1 += tl.dot(b_q1, b_do.to(b_q1.dtype)) * scale
        b_dh1 -= tl.dot(b_w1, b_dv.to(b_w1.dtype))
        b_dh2 += tl.dot(b_q2, b_do.to(b_q2.dtype)) * scale
        b_dh2 -= tl.dot(b_w2, b_dv.to(b_w2.dtype))

    p_out1 = tl.make_block_ptr(
        grad_state_ext, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
    )
    p_out2 = tl.make_block_ptr(
        grad_state_ext, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
    )
    tl.store(p_out1, b_dh1, boundary_check=(0, 1))
    tl.store(p_out2, b_dh2, boundary_check=(0, 1))


__all__ = ["gdn_packed_state_summary_kernel", "gdn_state_grad_ext_kernel"]
