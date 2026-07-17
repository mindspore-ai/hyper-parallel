# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S6.2（D-10 TP-extend-EP，05 §6.4.8）: 扩展 EP 分布式用例。

TP-extend-EP 语义：ep_size 即扩展 EP 组大小；expert 权重仅在 expert 维
Shard(0)，每 rank 持 num_experts/ep_size 个完整 expert，无 AG/RS 对。
- mesh (dp=4, tp=2)，ep=4 → 扩展 EP 组 {0,1,2,3}/{4,5,6,7}（跨 2 个 TP 组
  × 2 个 dp rank，用户示例拓扑）：双模式 e2e vs 单卡。
"""

import torch

from hyper_models.components.distributed import ShardingPlanner, apply_sharding_plan
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from tests.components.distributed.conftest import (
    TinyBatchedMoEForCausalLM,
    TinyConfig,
    TinyHFNativeMoEForCausalLM,
    run_dist,
)


def _build():
    torch.manual_seed(1234)
    return TinyHFNativeMoEForCausalLM(TinyConfig(num_experts=4)).eval()


def _build_batched():
    torch.manual_seed(1234)
    return TinyBatchedMoEForCausalLM(TinyConfig(
        num_experts=4, architectures=["Qwen3MoeForCausalLM"])).eval()


def _worker_ep_extend_e2e(rank, world_size):
    """mesh (dp=4, tp=2)，ep=4 → 扩展 EP 组 {0,1,2,3}/{4,5,6,7}（含 TP rank，
    跨 dp 坐标）：双模式输出等价单卡。"""
    assert world_size == 8
    ref_model = _build()
    torch.manual_seed(7)
    input_ids = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        ref_out = ref_model(input_ids)

    mesh = init_device_mesh("cpu", (4, 2), mesh_dim_names=("dp", "tp"))
    for mode in ("production", "validate"):
        model = _build()
        plan = ShardingPlanner().plan(model, mesh, tp_size=2, ep_size=4)
        spec = plan.modules["model.layers.0.mlp"]
        assert spec._ep_size == 4   # ep_size 即扩展 EP 组大小
        model, _ = apply_sharding_plan(
            model, plan, mesh, validate_mode=(mode == "validate"))
        with torch.no_grad():
            out = model(input_ids)
        torch.testing.assert_close(out, ref_out, rtol=1e-5, atol=1e-5)


def test_ep_extend_unified_e2e_8proc():
    run_dist(8, _worker_ep_extend_e2e)


def _worker_batched_ep_extend_e2e(rank, world_size):
    """D-11 batched 布局 e2e：mesh (dp=4, tp=2)，ep=4（experts.gate_up_proj
    无需堆叠直接 Shard(0)，qwen3moe TopKRouter adapter）：双模式等价单卡。"""
    assert world_size == 8
    ref_model = _build_batched()
    torch.manual_seed(7)
    input_ids = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        ref_out = ref_model(input_ids)

    mesh = init_device_mesh("cpu", (4, 2), mesh_dim_names=("dp", "tp"))
    for mode in ("production", "validate"):
        model = _build_batched()
        plan = ShardingPlanner().plan(model, mesh, tp_size=2, ep_size=4)
        spec = plan.modules["model.layers.0.mlp"]
        assert spec._ep_size == 4
        assert spec._ep_stack == {}      # batched 天生 stacked，Phase A 无堆叠
        assert spec._moe_router == "qwen3moe"
        model, _ = apply_sharding_plan(
            model, plan, mesh, validate_mode=(mode == "validate"))
        with torch.no_grad():
            out = model(input_ids)
        torch.testing.assert_close(out, ref_out, rtol=1e-5, atol=1e-5)


def test_batched_ep_extend_e2e_8proc():
    run_dist(8, _worker_batched_ep_extend_e2e)
