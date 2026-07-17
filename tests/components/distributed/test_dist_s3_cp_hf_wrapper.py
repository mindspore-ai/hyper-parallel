# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S3.4（2 进程）: HF 原语拦截 CP wrapper — G4 causal + 拦截还原 + 双模式同源。"""

import torch
import torch.nn.functional as F

from hyper_models.components.distributed.sharding_applier import (
    _wrap_cp_inner_attention,
)
from hyper_models.components.distributed.sharding_config import (
    CP,
    ModuleShardingSpec,
    TP,
)
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Partial, Shard
from tests.components.distributed.conftest import (
    TinyConfig,
    TinyLlamaAttention,
    run_dist,
)


def _worker(rank, world_size):
    cp_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("cp",))
    full_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("cp",))
    cfg = TinyConfig()
    S = 8
    chunk = S // world_size
    slc = slice(rank * chunk, (rank + 1) * chunk)

    torch.manual_seed(1234)
    ref_attn = TinyLlamaAttention(cfg, causal=True).eval()
    torch.manual_seed(0)
    hidden = torch.randn(2, S, cfg.hidden_size)
    with torch.no_grad():
        ref = ref_attn(hidden)  # 单卡参考（全序列 causal）

    # ── G4 causal 用例：HF 风格 attention CP=2 vs 单卡 ──
    torch.manual_seed(1234)
    attn = TinyLlamaAttention(cfg, causal=True).eval()
    spec = ModuleShardingSpec(
        out_src={"output": {CP: Shard(1), TP: Partial()}},
    )
    _wrap_cp_inner_attention(attn, cp_mesh, spec=spec, mesh=full_mesh,
                             mesh_dim_names=("cp",))
    orig_sdpa = F.scaled_dot_product_attention
    with torch.no_grad():
        out = attn(hidden[:, slc].contiguous())
    # 拦截后 F.scaled_dot_product_attention 已还原
    assert F.scaled_dot_product_attention is orig_sdpa
    # N4：rank1（Q 全局位置 [S/2, S)）输出 == 单卡参考对应切片
    torch.testing.assert_close(out, ref[:, slc], rtol=1e-5, atol=1e-5)

    # ── 双模式同源断言：DTensor 输入（validate 路径）与 local 输入
    # （production 路径）经同一 wrapper 输出逐元素相等 ──
    torch.manual_seed(1234)
    attn2 = TinyLlamaAttention(cfg, causal=True).eval()
    _wrap_cp_inner_attention(attn2, cp_mesh, spec=spec, mesh=full_mesh,
                             mesh_dim_names=("cp",))
    hs_dt = DTensor.from_local(hidden[:, slc].contiguous(), full_mesh,
                               (Shard(1),))
    with torch.no_grad():
        out_dt = attn2(hs_dt)
        out_local = attn2(hidden[:, slc].contiguous())
    assert isinstance(out_dt, DTensor)
    # validate 出口按声明 out_src 重包装
    assert tuple(out_dt.placements)[0] == Shard(1)
    torch.testing.assert_close(out_dt.to_local(), out_local,
                               rtol=1e-6, atol=1e-6)


def test_cp_hf_wrapper_2proc():
    run_dist(2, _worker)
