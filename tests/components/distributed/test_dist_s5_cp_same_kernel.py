# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S5.2（2 进程）: D-01'' — CP wrapper 双模式共用，区域内计算 kernel 级一致。

同一 attention 模块、同一 wrapper：DTensor 输入（validate 路径）与 local 输入
（production 路径）输出逐元素相等（严格容差）。
"""

import torch

from hyper_models.components.distributed.sharding_applier import (
    _wrap_cp_inner_attention,
)
from hyper_models.components.distributed.sharding_config import (
    CP,
    ModuleShardingSpec,
)
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Shard
from tests.components.distributed.conftest import (
    TinyConfig,
    TinyLlamaAttention,
    run_dist,
)


def _worker(rank, world_size):
    cp_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("cp",))
    cfg = TinyConfig()
    S = 8
    chunk = S // world_size
    slc = slice(rank * chunk, (rank + 1) * chunk)
    torch.manual_seed(0)
    hidden = torch.randn(2, S, cfg.hidden_size)

    torch.manual_seed(1234)
    attn = TinyLlamaAttention(cfg, causal=True).eval()
    spec = ModuleShardingSpec(out_src={"output": {CP: Shard(1)}})
    _wrap_cp_inner_attention(attn, cp_mesh, spec=spec, mesh=cp_mesh,
                             mesh_dim_names=("cp",))

    hs_local = hidden[:, slc].contiguous()
    hs_dt = DTensor.from_local(hs_local.clone(), cp_mesh, (Shard(1),))
    with torch.no_grad():
        out_local = attn(hs_local)
        out_dt = attn(hs_dt)
    assert isinstance(out_dt, DTensor)
    # 严格容差：两模式跑同一份 wrapper、同一 all-gather、同一 SDPA
    torch.testing.assert_close(out_dt.to_local(), out_local,
                               rtol=1e-6, atol=1e-7)


def test_cp_same_kernel_dual_mode_2proc():
    run_dist(2, _worker)
