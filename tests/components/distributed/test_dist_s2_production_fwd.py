# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S2.6（2 进程）: _wrap_production_forward — tiny_llama attention+mlp TP=2 数值。"""

import torch

from hyper_models.components.distributed import ShardingPlanner, apply_sharding_plan
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from tests.components.distributed.conftest import (
    TinyConfig,
    TinyLlamaForCausalLM,
    run_dist,
)


def _build():
    torch.manual_seed(1234)
    return TinyLlamaForCausalLM(TinyConfig()).eval()


def _worker(rank, world_size):
    # 单卡参考
    ref_model = _build()
    torch.manual_seed(7)
    x = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        ref = ref_model(x)

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    model = _build()
    plan = ShardingPlanner().plan(model, mesh, tp_size=world_size)
    model, _ = apply_sharding_plan(model, plan, mesh, validate_mode=False)
    with torch.no_grad():
        out = model(x)
    # 逐 rank 断言（输出为完整 logits——lm_head out_dst=Replicate）
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_production_forward_tp2():
    run_dist(2, _worker)
