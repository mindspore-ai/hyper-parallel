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
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the parent KDA directory.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Keep the imported kernel structure aligned with upstream FLA.
# pylint: disable=line-too-long,missing-public-type-hints,missing-public-docstring
# pylint: disable=missing-function-docstring,missing-module-docstring,unused-argument
# pylint: disable=unsupported-binary-operation,invalid-name,disallowed-name
# pylint: disable=consider-using-from-import,import-outside-toplevel,no-else-return
# pylint: disable=no-member,use-dict-literal,global-statement,unsubscriptable-object
# pylint: disable=invalid-unary-operand-type,nested-min-max,used-before-assignment

"""KDA token-output kernel adapted from FLA's GLA chunk output."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .utils import exp2, input_guard, prepare_chunk_indices


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def _chunk_kda_fwd_output_kernel(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        nt = tl.cdiv(T, BT)
        i_tg = (i_b * nt + i_t).to(tl.int64)
        bos = (i_b * T).to(tl.int64)

    causal_mask = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    q += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    v += (bos * HV + i_hv) * V
    o += (bos * HV + i_hv) * V
    h += (i_tg * HV + i_hv).to(tl.int64) * K * V
    A += (bos * HV + i_hv) * BT

    block_o = tl.zeros([BT, BV], dtype=tl.float32)
    offs_t = i_t * BT + tl.arange(0, BT)
    offs_v = i_v * BV + tl.arange(0, BV)
    offs_i = tl.arange(0, BT)
    mask_t = offs_t < T
    mask_v = offs_v < V
    mask_tv = mask_t[:, None] & mask_v[None, :]
    mask_a = mask_t[:, None] & (offs_i[None, :] < BT)
    for i_k in range(tl.cdiv(K, BK)):
        offs_k = i_k * BK + tl.arange(0, BK)
        mask_k = offs_k < K
        mask_qk = mask_t[:, None] & mask_k[None, :]
        ptr_q = q + offs_t[:, None] * (H * K) + offs_k[None, :]
        ptr_g = g + offs_t[:, None] * (HV * K) + offs_k[None, :]
        if STATE_V_FIRST:
            ptr_h = h + offs_v[:, None] * K + offs_k[None, :]
            mask_h = mask_v[:, None] & mask_k[None, :]
        else:
            ptr_h = h + offs_k[:, None] * V + offs_v[None, :]
            mask_h = mask_k[:, None] & mask_v[None, :]

        block_q = tl.load(ptr_q, mask=mask_qk, other=0.0)
        block_g = tl.load(ptr_g, mask=mask_qk, other=0.0).to(tl.float32)
        block_qg = (block_q * exp2(block_g)).to(block_q.dtype)
        block_h = tl.load(ptr_h, mask=mask_h, other=0.0)
        if STATE_V_FIRST:
            block_o += tl.dot(block_qg, tl.trans(block_h).to(block_qg.dtype))
        else:
            block_o += tl.dot(block_qg, block_h.to(block_qg.dtype))

    block_o *= scale
    ptr_v = v + offs_t[:, None] * (HV * V) + offs_v[None, :]
    ptr_o = o + offs_t[:, None] * (HV * V) + offs_v[None, :]
    ptr_a = A + offs_t[:, None] * (HV * BT) + offs_i[None, :]
    block_v = tl.load(ptr_v, mask=mask_tv, other=0.0)
    block_a = tl.load(ptr_a, mask=mask_a, other=0.0)
    block_a = tl.where(causal_mask, block_a, 0.0).to(block_v.dtype)
    block_o += tl.dot(block_a, block_v)
    tl.store(ptr_o, block_o.to(ptr_o.dtype.element_ty), mask=mask_tv)


@input_guard
def chunk_kda_fwd_output(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    attention: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    *,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Produce token outputs from prepared KDA chunks and recurrent states."""
    batch, sequence_length, num_query_heads, key_dim = q.shape
    num_value_heads, value_dim = v.shape[2:]
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None
        else None
    )
    num_chunks = (
        triton.cdiv(sequence_length, chunk_size)
        if chunk_indices is None
        else len(chunk_indices)
    )
    output = torch.zeros_like(v)
    grid = (triton.cdiv(value_dim, 128), num_chunks, batch * num_value_heads)
    _chunk_kda_fwd_output_kernel[grid](
        q=q,
        v=v,
        g=g,
        h=h,
        o=output,
        A=attention,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=sequence_length,
        H=num_query_heads,
        HV=num_value_heads,
        K=key_dim,
        V=value_dim,
        BT=chunk_size,
        BK=64,
        BV=128,
        STATE_V_FIRST=state_v_first,
    )
    return output


__all__ = ["chunk_kda_fwd_output"]
