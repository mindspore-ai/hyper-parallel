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
"""Inference Q/K normalization and half-rotation for packed Qwen3 grouped QKV."""

import torch
import triton
import triton.language as tl


@triton.jit
def _variance(input_ptr, variance_ptr, rows, groups: tl.constexpr, queries: tl.constexpr, dim: tl.constexpr,
              eps: tl.constexpr, block: tl.constexpr):
    """Write per-token/head RMS variance in the shared FP32 reduction order."""
    h = tl.program_id(0)
    t = tl.program_id(1) * block + tl.arange(0, block)
    d = tl.arange(0, dim)
    stride = groups * (queries + 2) * dim
    offset = (h // (queries + 1) * (queries + 2) + h % (queries + 1)) * dim
    x = tl.load(input_ptr + t[:, None] * stride + offset + d[None, :], t[:, None] < rows, other=0).to(tl.float32)
    var = tl.sum(x * x, axis=1) / dim + eps
    tl.store(variance_ptr + t * (groups * (queries + 1)) + h, var, t < rows)


@triton.jit
def _normalize_rotate(input_ptr, inverse_ptr, query_weight, key_weight, cos_ptr, sin_ptr, query_ptr, key_ptr, rows,
                      groups: tl.constexpr, queries: tl.constexpr, dim: tl.constexpr, block: tl.constexpr):
    """Apply affine RMSNorm and RoPE with the reference BF16 rounding points."""
    h = tl.program_id(0)
    group = h // (queries + 1)
    lane = h % (queries + 1)
    t = tl.program_id(1) * block + tl.arange(0, block)
    d = tl.arange(0, dim // 2)
    stride = groups * (queries + 2) * dim
    src = input_ptr + t[:, None] * stride + (group * (queries + 2) + lane) * dim + d[None, :]
    freq = t[:, None] * dim + d[None, :]
    valid = t[:, None] < rows
    inv = tl.load(inverse_ptr + t * (groups * (queries + 1)) + h, t < rows, other=0)
    if lane == queries:
        weight = key_weight
        dst = key_ptr + (t[:, None] * groups + group) * dim + d[None, :]
    else:
        weight = query_weight
        dst = query_ptr + (t[:, None] * (groups * queries) + group * queries + lane) * dim + d[None, :]

    # HF rounds the normalized value before multiplying by the affine weight.
    # RoPE then consumes those weighted BF16 values with FP32 products.
    a = (tl.load(src, valid, other=0).to(tl.float32) * inv[:, None]).to(tl.bfloat16)
    b = (tl.load(src + dim // 2, valid, other=0).to(tl.float32) * inv[:, None]).to(tl.bfloat16)
    a = (a.to(tl.float32) * tl.load(weight + d).to(tl.float32)[None, :]).to(tl.bfloat16).to(tl.float32)
    b = (b.to(tl.float32) * tl.load(weight + dim // 2 + d).to(tl.float32)[None, :]).to(tl.bfloat16).to(tl.float32)
    c = tl.load(cos_ptr + freq, valid, other=0).to(tl.float32)
    s = tl.load(sin_ptr + freq, valid, other=0).to(tl.float32)
    tl.store(dst, a * c - b * s, valid)
    c = tl.load(cos_ptr + freq + dim // 2, valid, other=0).to(tl.float32)
    s = tl.load(sin_ptr + freq + dim // 2, valid, other=0).to(tl.float32)
    tl.store(dst + dim // 2, b * c + a * s, valid)


@torch.library.custom_op("hyper_rl::qk_norm_rope", mutates_args=())
def qk_norm_rope(qkv: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor,
                 cos: torch.Tensor, sin: torch.Tensor, groups: int, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize and rotate Q/K from local BF16 grouped-QKV storage.

    qkv is [T, G * (Q+2) * D], where G is the local KV head count, Q is
    the query heads per group and D=128. The precomputed cos/sin are [T,D].
    Return contiguous [T,G*Q,D] and [T,G,D]. V stays in the caller's buffer.
    """
    rows, width = qkv.shape
    dim = q_weight.numel()
    q = width // (groups * dim) - 2
    query = qkv.new_empty((rows, groups * q, dim))
    key = qkv.new_empty((rows, groups, dim))
    if rows:
        block = min(16, triton.next_power_of_2(rows))
        grid = (groups * (q + 1), triton.cdiv(rows, block))
        var = qkv.new_empty((rows, groups * (q + 1)), dtype=torch.float32)
        _variance[grid](qkv, var, rows, groups, q, dim, eps, block, enable_fp_fusion=False)
        # The device Rsqrt and Triton rsqrt differ by an FP32 ulp on real operands.
        inv = torch.rsqrt(var)
        _normalize_rotate[grid](qkv, inv, q_weight, k_weight, cos, sin, query, key, rows,
                                groups, q, dim, block, enable_fp_fusion=False)
    return query, key


@qk_norm_rope.register_fake
def _fake(qkv, q_weight, k_weight, cos, sin, groups, eps):
    del k_weight, cos, sin, eps
    dim = q_weight.numel()
    return (qkv.new_empty((qkv.shape[0], qkv.shape[1] // dim - 2 * groups, dim)),
            qkv.new_empty((qkv.shape[0], groups, dim)))
