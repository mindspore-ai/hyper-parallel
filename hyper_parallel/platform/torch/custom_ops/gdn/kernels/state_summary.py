# -*- coding: utf-8 -*-
# Copyright 2026 Huawei Technologies Co., Ltd
#
# The state-summary decomposition follows the MIT-licensed context-parallel
# implementation in flash-linear-attention/fla/ops/cp/chunk_delta_h.py.

"""Fixed-shape Triton-Ascend kernels for GDN state summaries."""

import torch
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
def _gdn_packed_state_summary_kernel(
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
            w,
            (T, K),
            (stride_k, 1),
            (i_t * BT, 0),
            (BT, 64),
            (1, 0),
        )
        p_w2 = tl.make_block_ptr(
            w,
            (T, K),
            (stride_k, 1),
            (i_t * BT, 64),
            (BT, 64),
            (1, 0),
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
            k,
            (K, T),
            (1, stride_k),
            (0, i_t * BT),
            (64, BT),
            (0, 1),
        )
        p_k2 = tl.make_block_ptr(
            k,
            (K, T),
            (1, stride_k),
            (64, i_t * BT),
            (64, BT),
            (0, 1),
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
    key=["H", "K", "V", "BT", "BV", "SEGMENTS"],
)
@triton.jit(do_not_specialize=["T"])
def _gdn_segment_state_summary_kernel(
    k,
    w,
    u,
    g,
    packed_segments,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    SEGMENTS: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_bhs = tl.program_id(1)
    i_segment = i_bhs % SEGMENTS
    i_bh = i_bhs // SEGMENTS
    i_b = i_bh // H
    i_h = i_bh % H

    stride_k = H * K
    stride_v = H * V
    k += (i_b * T * H + i_h) * K
    w += (i_b * T * H + i_h) * K
    u += (i_b * T * H + i_h) * V
    g += i_b * T * H + i_h
    packed_segments += i_bhs * K * (V + K)

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

    chunks_per_segment: tl.constexpr = NT // SEGMENTS
    for segment_idx in range(chunks_per_segment):
        i_t = i_segment * chunks_per_segment + segment_idx
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
        packed_segments,
        (K, V + K),
        (V + K, 1),
        (0, i_v * BV),
        (64, BV),
        (1, 0),
    )
    p_out2 = tl.make_block_ptr(
        packed_segments,
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
        multibuffer_list=(False,),
        set_workspace_multibuffer_list=(2,),
        tile_mix_vector_loop_num_list=(2,),
        tile_mix_cube_loop_num_list=(2,),
    ),
    key=["H", "K", "V", "BT", "IS_TRANSITION"],
)
@triton.jit(do_not_specialize=["T"])
def _gdn_prepare_state_summary_kernel(
    k,
    v,
    beta,
    A,
    g,
    w,
    u,
    summary,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    IS_TRANSITION: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_b = i_bh // H
    i_h = i_bh % H

    stride_k = H * K
    stride_v = H * V
    k += (i_b * T * H + i_h) * K
    v += (i_b * T * H + i_h) * V
    beta += i_b * T * H + i_h
    A += (i_b * T * H + i_h) * BT
    g += i_b * T * H + i_h
    w += (i_b * T * H + i_h) * K
    u += (i_b * T * H + i_h) * V
    summary += i_bh * K * BV

    col = tl.arange(0, BV)
    row1 = tl.arange(0, 64)
    row2 = 64 + tl.arange(0, 64)
    if IS_TRANSITION:
        b_h1 = tl.where(row1[:, None] == col[None, :], 1.0, 0.0).to(tl.float32)
        b_h2 = tl.where(row2[:, None] == col[None, :], 1.0, 0.0).to(tl.float32)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)

    for i_t in range(NT):
        p_A = tl.make_block_ptr(
            A,
            (T, BT),
            (H * BT, 1),
            (i_t * BT, 0),
            (BT, BT),
            (1, 0),
        )
        b_A = tl.load(p_A, boundary_check=(0, 1))
        token = i_t * BT + tl.arange(0, BT)
        mask = token < T
        b_beta = tl.load(beta + token * H, mask=mask, other=0.0)
        b_g = tl.load(g + token * H, mask=mask, other=0.0).to(tl.float32)

        if IS_TRANSITION:
            b_value = tl.zeros([BT, BV], dtype=tl.float32)
        else:
            p_v = tl.make_block_ptr(
                v,
                (T, V),
                (stride_v, 1),
                (i_t * BT, 0),
                (BT, BV),
                (1, 0),
            )
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_u = tl.dot(b_A, (b_v * b_beta[:, None]).to(b_v.dtype))
            p_u = tl.make_block_ptr(
                u,
                (T, V),
                (stride_v, 1),
                (i_t * BT, 0),
                (BT, BV),
                (1, 0),
            )
            tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))
            b_value = b_u.to(tl.float32)

        p_k1 = tl.make_block_ptr(
            k,
            (T, K),
            (stride_k, 1),
            (i_t * BT, 0),
            (BT, 64),
            (1, 0),
        )
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))
        b_k_beta1 = b_k1 * (b_beta * tl.exp(b_g))[:, None]
        b_w1 = tl.dot(b_A, b_k_beta1.to(b_k1.dtype))
        b_value -= tl.dot(b_w1, b_h1.to(b_w1.dtype))
        if not IS_TRANSITION:
            p_w1 = tl.make_block_ptr(
                w,
                (T, K),
                (stride_k, 1),
                (i_t * BT, 0),
                (BT, 64),
                (1, 0),
            )
            tl.store(p_w1, b_w1.to(p_w1.dtype.element_ty), boundary_check=(0, 1))

        p_k2 = tl.make_block_ptr(
            k,
            (T, K),
            (stride_k, 1),
            (i_t * BT, 64),
            (BT, 64),
            (1, 0),
        )
        b_k2 = tl.load(p_k2, boundary_check=(0, 1))
        b_k_beta2 = b_k2 * (b_beta * tl.exp(b_g))[:, None]
        b_w2 = tl.dot(b_A, b_k_beta2.to(b_k2.dtype))
        b_value -= tl.dot(b_w2, b_h2.to(b_w2.dtype))
        if not IS_TRANSITION:
            p_w2 = tl.make_block_ptr(
                w,
                (T, K),
                (stride_k, 1),
                (i_t * BT, 64),
                (BT, 64),
                (1, 0),
            )
            tl.store(p_w2, b_w2.to(p_w2.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + last_idx * H).to(tl.float32)
        b_value *= tl.where(mask, tl.exp(b_g_last - b_g), 0.0)[:, None]
        decay = tl.exp(b_g_last)
        b_h1 *= decay
        b_h2 *= decay
        b_value = b_value.to(k.dtype.element_ty)
        b_h1 += tl.dot(tl.trans(b_k1), b_value)
        b_h2 += tl.dot(tl.trans(b_k2), b_value)

    p_out1 = tl.make_block_ptr(
        summary,
        (K, BV),
        (BV, 1),
        (0, 0),
        (64, BV),
        (1, 0),
    )
    p_out2 = tl.make_block_ptr(
        summary,
        (K, BV),
        (BV, 1),
        (64, 0),
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
def _gdn_state_grad_ext_kernel(
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
            k,
            (T, K),
            (stride_k, 1),
            (i_t * BT, 0),
            (BT, 64),
            (1, 0),
        )
        p_k2 = tl.make_block_ptr(
            k,
            (T, K),
            (stride_k, 1),
            (i_t * BT, 64),
            (BT, 64),
            (1, 0),
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
            q,
            (K, T),
            (1, stride_k),
            (0, i_t * BT),
            (64, BT),
            (0, 1),
        )
        p_q2 = tl.make_block_ptr(
            q,
            (K, T),
            (1, stride_k),
            (64, i_t * BT),
            (64, BT),
            (0, 1),
        )
        p_w1 = tl.make_block_ptr(
            w,
            (K, T),
            (1, stride_k),
            (0, i_t * BT),
            (64, BT),
            (0, 1),
        )
        p_w2 = tl.make_block_ptr(
            w,
            (K, T),
            (1, stride_k),
            (64, i_t * BT),
            (64, BT),
            (0, 1),
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
        grad_state_ext,
        (K, V),
        (V, 1),
        (0, i_v * BV),
        (64, BV),
        (1, 0),
    )
    p_out2 = tl.make_block_ptr(
        grad_state_ext,
        (K, V),
        (V, 1),
        (64, i_v * BV),
        (64, BV),
        (1, 0),
    )
    tl.store(p_out1, b_dh1, boundary_check=(0, 1))
    tl.store(p_out2, b_dh2, boundary_check=(0, 1))


@triton.autotune(
    configs=get_autotune_config(
        multibuffer_list=(False,),
        set_workspace_multibuffer_list=(2,),
        tile_mix_vector_loop_num_list=(2,),
        tile_mix_cube_loop_num_list=(2,),
    ),
    key=["H", "K", "V", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def _gdn_bwd_prepare_grad_summary_kernel(
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
    i_bh = tl.program_id(0)
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
        token = i_t * BT + tl.arange(0, BT)
        mask = token < T
        b_g = tl.load(g + token * H, mask=mask, other=0.0).to(tl.float32)
        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = tl.load(g + last_idx * H).to(tl.float32)

        p_k1 = tl.make_block_ptr(
            k, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        p_k2 = tl.make_block_ptr(
            k, (T, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0)
        )
        p_q1 = tl.make_block_ptr(
            q, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        p_q2 = tl.make_block_ptr(
            q, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
        )
        b_k1 = tl.load(p_k1, boundary_check=(0, 1))
        b_k2 = tl.load(p_k2, boundary_check=(0, 1))
        b_q1 = tl.load(p_q1, boundary_check=(0, 1))
        b_q2 = tl.load(p_q2, boundary_check=(0, 1))

        p_do = tl.make_block_ptr(
            do, (T, V), (stride_v, 1), (i_t * BT, 0), (BT, BV), (1, 0)
        )
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_attn = tl.dot(b_k1, b_q1) + tl.dot(b_k2, b_q2)
        row = tl.arange(0, BT)
        causal = (row[:, None] <= row[None, :]) & (mask[:, None] & mask[None, :])
        b_attn = tl.where(
            causal,
            b_attn * tl.exp(b_g[None, :] - b_g[:, None]) * scale,
            0.0,
        ).to(b_do.dtype)
        b_dv_local = tl.dot(b_attn, b_do)
        p_dv = tl.make_block_ptr(
            dv, (T, V), (stride_v, 1), (i_t * BT, 0), (BT, BV), (1, 0)
        )
        tl.store(p_dv, b_dv_local.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

        b_dv_state = tl.dot(b_k1, b_dh1.to(b_k1.dtype))
        b_dv_state += tl.dot(b_k2, b_dh2.to(b_k2.dtype))
        b_dv_state *= tl.where(mask, tl.exp(b_g_last - b_g), 0.0)[:, None]
        b_dv_total = b_dv_state + b_dv_local

        decay = tl.exp(b_g_last)
        b_dh1 *= decay
        b_dh2 *= decay
        p_w1 = tl.make_block_ptr(
            w, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        p_w2 = tl.make_block_ptr(
            w, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
        )
        b_w1 = tl.load(p_w1, boundary_check=(0, 1))
        b_w2 = tl.load(p_w2, boundary_check=(0, 1))
        gate = tl.exp(b_g)[None, :]
        b_qg1 = (b_q1 * gate).to(b_q1.dtype)
        b_qg2 = (b_q2 * gate).to(b_q2.dtype)
        b_dh1 += tl.dot(b_qg1, b_do.to(b_qg1.dtype)) * scale
        b_dh1 -= tl.dot(b_w1, b_dv_total.to(b_w1.dtype))
        b_dh2 += tl.dot(b_qg2, b_do.to(b_qg2.dtype)) * scale
        b_dh2 -= tl.dot(b_w2, b_dv_total.to(b_w2.dtype))

    p_out1 = tl.make_block_ptr(
        grad_state_ext, (K, V), (V, 1), (0, 0), (64, BV), (1, 0)
    )
    p_out2 = tl.make_block_ptr(
        grad_state_ext, (K, V), (V, 1), (64, 0), (64, BV), (1, 0)
    )
    tl.store(p_out1, b_dh1, boundary_check=(0, 1))
    tl.store(p_out2, b_dh2, boundary_check=(0, 1))


@torch.compiler.disable
def chunk_gated_delta_rule_state_summary_fwd(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    *,
    chunk_size: int = 64,
    block_size: int = 128,
    segments: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``state_out = transition @ state_in + state_ext``.

    The first implementation intentionally supports the Qwen3.5 validation
    shape only. Inputs use the existing fused GDN layout ``[B, T, H, D]``.
    """
    if segments != 1:
        return chunk_gated_delta_rule_segmented_state_summary_fwd(
            k,
            w,
            u,
            g,
            segments=segments,
            chunk_size=chunk_size,
            block_size=block_size,
        )
    if k.ndim != 4 or w.ndim != 4 or u.ndim != 4 or g.ndim != 3:
        raise ValueError("GDN state summary expects k/w/u [B,T,H,D] and g [B,T,H].")
    batch, seq_len, heads, key_dim = k.shape
    value_dim = u.shape[-1]
    if key_dim != 128 or value_dim != 128 or chunk_size != 64:
        raise NotImplementedError(
            "The initial NPU GDN state-summary kernel supports K=V=128 and chunk_size=64."
        )
    if block_size not in (64, 128):
        raise ValueError(
            f"GDN state-summary block_size must be 64 or 128, got {block_size}."
        )
    if seq_len % chunk_size != 0:
        raise ValueError(
            f"GDN state-summary sequence length {seq_len} must be divisible by {chunk_size}."
        )
    if w.shape != k.shape or u.shape[:3] != k.shape[:3] or g.shape != k.shape[:3]:
        raise ValueError(
            f"Incompatible GDN state-summary shapes: k={tuple(k.shape)}, "
            f"w={tuple(w.shape)}, u={tuple(u.shape)}, g={tuple(g.shape)}."
        )

    k = k.contiguous()
    w = w.contiguous()
    u = u.contiguous()
    g = g.contiguous()
    num_chunks = seq_len // chunk_size
    packed_summary = torch.empty(
        batch,
        heads,
        key_dim,
        value_dim + key_dim,
        device=k.device,
        dtype=torch.float32,
    )
    _gdn_packed_state_summary_kernel[
        (triton.cdiv(value_dim + key_dim, block_size), batch * heads)
    ](
        k,
        w,
        u,
        g,
        packed_summary,
        seq_len,
        H=heads,
        K=key_dim,
        V=value_dim,
        BT=chunk_size,
        BV=block_size,
        NT=num_chunks,
    )
    state_ext = packed_summary[..., :value_dim].contiguous()
    transition = packed_summary[..., value_dim:].contiguous()
    return state_ext, transition


def _merge_gdn_state_summary_parts(
    state_ext: torch.Tensor,
    transition: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    while state_ext.shape[2] > 1:
        left_state = state_ext[:, :, 0::2]
        right_state = state_ext[:, :, 1::2]
        left_transition = transition[:, :, 0::2]
        right_transition = transition[:, :, 1::2]
        state_ext = torch.matmul(right_transition, left_state) + right_state
        transition = torch.matmul(right_transition, left_transition)
    return state_ext[:, :, 0].contiguous(), transition[:, :, 0].contiguous()


@torch.compiler.disable
def chunk_gated_delta_rule_segmented_state_summary_fwd(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    *,
    segments: int,
    chunk_size: int = 64,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build independent sequence-segment summaries and merge them as a tree."""
    if segments not in (1, 2, 4):
        raise ValueError(f"GDN state-summary segments must be 1, 2, or 4, got {segments}.")
    if segments == 1:
        return chunk_gated_delta_rule_state_summary_fwd(
            k,
            w,
            u,
            g,
            chunk_size=chunk_size,
            block_size=block_size,
            segments=1,
        )

    if k.ndim != 4 or w.ndim != 4 or u.ndim != 4 or g.ndim != 3:
        raise ValueError("GDN state summary expects k/w/u [B,T,H,D] and g [B,T,H].")
    batch, seq_len, heads, key_dim = k.shape
    value_dim = u.shape[-1]
    if key_dim != 128 or value_dim != 128 or chunk_size != 64:
        raise NotImplementedError(
            "The segmented NPU GDN state-summary kernel supports "
            "K=V=128 and chunk_size=64."
        )
    if block_size not in (64, 128):
        raise ValueError(
            f"GDN state-summary block_size must be 64 or 128, got {block_size}."
        )
    num_chunks = seq_len // chunk_size
    if seq_len % chunk_size != 0 or num_chunks % segments != 0:
        raise ValueError(
            f"GDN state-summary requires {seq_len=} to contain an integer number "
            f"of chunks per segment for {chunk_size=} and {segments=}."
        )
    if w.shape != k.shape or u.shape[:3] != k.shape[:3] or g.shape != k.shape[:3]:
        raise ValueError(
            f"Incompatible GDN state-summary shapes: k={tuple(k.shape)}, "
            f"w={tuple(w.shape)}, u={tuple(u.shape)}, g={tuple(g.shape)}."
        )

    k = k.contiguous()
    w = w.contiguous()
    u = u.contiguous()
    g = g.contiguous()
    packed_segments = torch.empty(
        batch,
        heads,
        segments,
        key_dim,
        value_dim + key_dim,
        device=k.device,
        dtype=torch.float32,
    )
    _gdn_segment_state_summary_kernel[
        (triton.cdiv(value_dim + key_dim, block_size), batch * heads * segments)
    ](
        k,
        w,
        u,
        g,
        packed_segments,
        seq_len,
        H=heads,
        K=key_dim,
        V=value_dim,
        BT=chunk_size,
        BV=block_size,
        NT=num_chunks,
        SEGMENTS=segments,
    )
    state_segments = packed_segments[..., :value_dim].contiguous()
    transition_segments = packed_segments[..., value_dim:].contiguous()
    state_ext, transition = _merge_gdn_state_summary_parts(
        state_segments, transition_segments
    )
    return state_ext, transition


@torch.compiler.disable
def chunk_gated_delta_rule_prepare_state_summary_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build WY tensors and the affine state summary in one scan.

    This experimental fixed-shape kernel fuses ``recompute_w_u_fwd`` with
    state-summary construction. It returns ``w``, ``u``, ``state_ext`` and
    ``transition`` in the same layouts as the separate helpers.
    """
    if k.ndim != 4 or v.ndim != 4 or beta.ndim != 3 or A.ndim != 4 or g.ndim != 3:
        raise ValueError(
            "Fused GDN prepare-summary expects k/v [B,T,H,D], "
            "beta/g [B,T,H], and A [B,T,H,BT]."
        )
    batch, seq_len, heads, key_dim = k.shape
    value_dim = v.shape[-1]
    expected_prefix = (batch, seq_len, heads)
    if key_dim != 128 or value_dim != 128 or chunk_size != 64:
        raise NotImplementedError(
            "The fused NPU GDN prepare-summary kernel supports "
            "K=V=128 and chunk_size=64."
        )
    if seq_len % chunk_size != 0:
        raise ValueError(
            f"GDN prepare-summary sequence length {seq_len} must be divisible "
            f"by {chunk_size}."
        )
    if (
        v.shape[:3] != expected_prefix
        or beta.shape != expected_prefix
        or g.shape != expected_prefix
        or A.shape != (*expected_prefix, chunk_size)
    ):
        raise ValueError(
            "Incompatible fused GDN prepare-summary shapes: "
            f"k={tuple(k.shape)}, v={tuple(v.shape)}, beta={tuple(beta.shape)}, "
            f"A={tuple(A.shape)}, g={tuple(g.shape)}."
        )

    k = k.contiguous()
    v = v.contiguous()
    beta = beta.contiguous()
    A = A.contiguous()
    g = g.contiguous()
    w = torch.empty_like(k)
    u = torch.empty_like(v)
    state_ext = torch.empty(
        batch,
        heads,
        key_dim,
        value_dim,
        device=k.device,
        dtype=torch.float32,
    )
    transition = torch.empty(
        batch,
        heads,
        key_dim,
        key_dim,
        device=k.device,
        dtype=torch.float32,
    )
    common_args = dict(
        T=seq_len,
        H=heads,
        K=key_dim,
        V=value_dim,
        BT=chunk_size,
        BV=128,
        NT=seq_len // chunk_size,
    )
    _gdn_prepare_state_summary_kernel[(batch * heads,)](
        k,
        v,
        beta,
        A,
        g,
        w,
        u,
        state_ext,
        IS_TRANSITION=False,
        **common_args,
    )
    _gdn_prepare_state_summary_kernel[(batch * heads,)](
        k,
        v,
        beta,
        A,
        g,
        w,
        u,
        transition,
        IS_TRANSITION=True,
        **common_args,
    )
    return w, u, state_ext, transition


@torch.compiler.disable
def chunk_gated_delta_rule_state_gradient_summary_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    grad_output: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    *,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Return the local-loss contribution to the incoming state gradient."""
    batch, seq_len, heads, key_dim = q.shape
    value_dim = grad_output.shape[-1]
    if key_dim != 128 or value_dim != 128 or chunk_size != 64:
        raise NotImplementedError(
            "The initial NPU GDN state-gradient summary supports "
            "K=V=128 and chunk_size=64."
        )
    if seq_len % chunk_size != 0:
        raise ValueError(
            f"GDN state-gradient sequence length {seq_len} must be divisible by "
            f"{chunk_size}."
        )
    expected_qk_shape = (batch, seq_len, heads, key_dim)
    expected_v_shape = (batch, seq_len, heads, value_dim)
    if (
        k.shape != expected_qk_shape
        or w.shape != expected_qk_shape
        or g.shape != expected_qk_shape[:3]
        or grad_output.shape != expected_v_shape
        or dv.shape != expected_v_shape
    ):
        raise ValueError(
            "Incompatible GDN state-gradient summary shapes: "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, w={tuple(w.shape)}, "
            f"g={tuple(g.shape)}, do={tuple(grad_output.shape)}, dv={tuple(dv.shape)}."
        )

    q = q.contiguous()
    k = k.contiguous()
    w = w.contiguous()
    g = g.contiguous()
    grad_output = grad_output.contiguous()
    dv = dv.contiguous()
    grad_state_ext = torch.empty(
        batch,
        heads,
        key_dim,
        value_dim,
        device=q.device,
        dtype=torch.float32,
    )
    _gdn_state_grad_ext_kernel[(1, batch * heads)](
        q,
        k,
        w,
        g,
        grad_output,
        dv,
        grad_state_ext,
        scale,
        seq_len,
        H=heads,
        K=key_dim,
        V=value_dim,
        BT=chunk_size,
        BV=128,
        NT=seq_len // chunk_size,
    )
    return grad_state_ext


@torch.compiler.disable
def chunk_gated_delta_rule_bwd_prepare_state_gradient_summary(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    grad_output: torch.Tensor,
    scale: float,
    *,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build local ``dv`` and its incoming-state gradient summary together."""
    if q.ndim != 4 or k.ndim != 4 or w.ndim != 4 or g.ndim != 3:
        raise ValueError(
            "Fused GDN backward prepare-summary expects q/k/w [B,T,H,K] "
            "and g [B,T,H]."
        )
    batch, seq_len, heads, key_dim = q.shape
    value_dim = grad_output.shape[-1]
    expected_qk_shape = (batch, seq_len, heads, key_dim)
    expected_v_shape = (batch, seq_len, heads, value_dim)
    if key_dim != 128 or value_dim != 128 or chunk_size != 64:
        raise NotImplementedError(
            "The fused NPU GDN backward prepare-summary kernel supports "
            "K=V=128 and chunk_size=64."
        )
    if seq_len % chunk_size != 0:
        raise ValueError(
            f"GDN backward prepare-summary sequence length {seq_len} must be "
            f"divisible by {chunk_size}."
        )
    if (
        k.shape != expected_qk_shape
        or w.shape != expected_qk_shape
        or g.shape != expected_qk_shape[:3]
        or grad_output.shape != expected_v_shape
    ):
        raise ValueError(
            "Incompatible fused GDN backward prepare-summary shapes: "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, w={tuple(w.shape)}, "
            f"g={tuple(g.shape)}, do={tuple(grad_output.shape)}."
        )

    q = q.contiguous()
    k = k.contiguous()
    w = w.contiguous()
    g = g.contiguous()
    grad_output = grad_output.contiguous()
    dv = torch.empty_like(grad_output)
    grad_state_ext = torch.empty(
        batch,
        heads,
        key_dim,
        value_dim,
        device=q.device,
        dtype=torch.float32,
    )
    _gdn_bwd_prepare_grad_summary_kernel[(batch * heads,)](
        q,
        k,
        w,
        g,
        grad_output,
        dv,
        grad_state_ext,
        scale,
        seq_len,
        H=heads,
        K=key_dim,
        V=value_dim,
        BT=chunk_size,
        BV=128,
        NT=seq_len // chunk_size,
    )
    return dv, grad_state_ext


def apply_gdn_state_summary(
    state_ext: torch.Tensor,
    transition: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> torch.Tensor:
    """Apply a local state summary in FP32."""
    if initial_state is None:
        return state_ext
    return torch.matmul(transition, initial_state.float()) + state_ext


def apply_gdn_state_gradient_summary(
    grad_state_ext: torch.Tensor,
    transition: torch.Tensor,
    grad_final_state: torch.Tensor | None,
) -> torch.Tensor:
    """Apply the adjoint affine summary to a gradient from the next rank."""
    if grad_final_state is None:
        return grad_state_ext
    return (
        torch.matmul(transition.transpose(-2, -1), grad_final_state.float())
        + grad_state_ext
    )


__all__ = [
    "apply_gdn_state_gradient_summary",
    "apply_gdn_state_summary",
    "chunk_gated_delta_rule_bwd_prepare_state_gradient_summary",
    "chunk_gated_delta_rule_prepare_state_summary_fwd",
    "chunk_gated_delta_rule_segmented_state_summary_fwd",
    "chunk_gated_delta_rule_state_summary_fwd",
    "chunk_gated_delta_rule_state_gradient_summary_bwd",
]
