# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S3.3（2 进程）: NeMo 风格 (q,k,v) CP wrapper — CP=2 输出 vs 单卡参考。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyper_models.components.distributed.sharding_applier import (
    _wrap_cp_inner_attention,
)
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from tests.components.distributed.conftest import run_dist


class ToyNeMoAttention(nn.Module):
    """NeMo 风格：inner_attention 子模块 forward(q,k,v,is_causal=False)。"""

    class Inner(nn.Module):
        def forward(self, q, k, v, is_causal=False, attn_mask=None):
            return F.scaled_dot_product_attention(
                q, k, v, is_causal=is_causal, attn_mask=attn_mask)

    def __init__(self):
        super().__init__()
        self.inner_attention = self.Inner()

    def forward(self, q, k, v, is_causal=False, attn_mask=None):
        return self.inner_attention(q, k, v, is_causal=is_causal,
                                    attn_mask=attn_mask)


def _worker(rank, world_size):
    cp_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("cp",))
    torch.manual_seed(0)
    B, N, S, D = 2, 2, 8, 4
    full_q = torch.randn(B, N, S, D)
    full_k = torch.randn(B, N, S, D)
    full_v = torch.randn(B, N, S, D)
    chunk = S // world_size
    slc = slice(rank * chunk, (rank + 1) * chunk)

    for causal in (False, True):
        # 单卡参考：全序列 attention 后取本地 Q chunk 切片。
        # 注意不能用 F.sdpa(q_chunk, full_k, full_v, is_causal=True) 当参考——
        # torch 的 is_causal 在 q_len ≠ kv_len 时按左上角对齐（等价于假设
        # chunk 位于序列开头），对 rank>0 的 chunk 会错误掩码（G4）。
        ref_full = F.scaled_dot_product_attention(
            full_q, full_k, full_v, is_causal=causal)
        ref = ref_full[:, :, slc]

        attn = ToyNeMoAttention()
        _wrap_cp_inner_attention(attn, cp_mesh)
        out = attn(full_q[:, :, slc].contiguous(),
                   full_k[:, :, slc].contiguous(),
                   full_v[:, :, slc].contiguous(),
                   is_causal=causal)
        # N4（causal 时）：rank1 的 Q 全局位置 [S/2, S) 必须 attend 到 [0, 位置]
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_cp_qkv_wrapper_2proc():
    run_dist(2, _worker)
