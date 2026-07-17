# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S4.1（2 进程）: MoE 参数分片 — expert EP 切片 + gate 全复制（无独立 EP _apply 入口）。"""

import torch

from hyper_models.components.distributed import ShardingPlanner, apply_sharding_plan
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from tests.components.distributed.conftest import (
    TinyConfig,
    TinyLlamaForCausalLM,
    run_dist,
)


def _worker(rank, world_size):
    torch.manual_seed(1234)
    ref = TinyLlamaForCausalLM(TinyConfig(num_experts=4)).eval()
    torch.manual_seed(1234)
    model = TinyLlamaForCausalLM(TinyConfig(num_experts=4)).eval()

    mesh = init_device_mesh("cpu", (1, world_size), mesh_dim_names=("tp", "ep"))
    plan = ShardingPlanner().plan(model, mesh, tp_size=1, ep_size=world_size)
    model, _ = apply_sharding_plan(model, plan, mesh)

    chunk = 4 // world_size
    slc = slice(rank * chunk, (rank + 1) * chunk)
    for i in (0, 1):
        moe = model.model.layers[i].mlp
        ref_moe = ref.model.layers[i].mlp
        # N7：rank1 只持 e2/e3——逐 rank 断言本地 expert 切片 == 全量对应段
        assert moe.experts.w1.shape[0] == chunk
        torch.testing.assert_close(moe.experts.w1.data, ref_moe.experts.w1.data[slc])
        torch.testing.assert_close(moe.experts.w2.data, ref_moe.experts.w2.data[slc])
        torch.testing.assert_close(moe.experts.w3.data, ref_moe.experts.w3.data[slc])
        # gate 两 rank 全复制一致
        torch.testing.assert_close(moe.gate.weight.data, ref_moe.gate.weight.data)


def test_ep_shard_2proc():
    run_dist(2, _worker)
