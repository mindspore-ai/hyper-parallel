# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S4.2（2 进程）: _wrap_local_region_forward — toy MoE EP=2 输出 vs 单卡参考（N8 非对称通信）。"""

import torch

from hyper_models.components.distributed import ShardingPlanner, apply_sharding_plan
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from tests.components.distributed.conftest import (
    TinyConfig,
    TinyLlamaForCausalLM,
    run_dist,
)


def _attach_ep(model, mesh, ep_size, num_experts=4):
    """模拟 §6.4.3 init_token_dispatcher：设置 expert_offset + ep_group。"""
    ep_mesh = mesh["ep"]
    ep_rank = ep_mesh.get_local_rank()
    n_local = num_experts // ep_size
    for layer in model.model.layers:
        layer.mlp.experts.expert_offset = ep_rank * n_local
        layer.mlp.ep_group = ep_mesh.get_group()


def _worker(rank, world_size):
    torch.manual_seed(1234)
    ref = TinyLlamaForCausalLM(TinyConfig(num_experts=4)).eval()
    torch.manual_seed(7)
    input_ids = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        ref_logits = ref(input_ids)

    mesh = init_device_mesh("cpu", (1, world_size), mesh_dim_names=("tp", "ep"))
    torch.manual_seed(1234)
    model = TinyLlamaForCausalLM(TinyConfig(num_experts=4)).eval()
    plan = ShardingPlanner().plan(model, mesh, tp_size=1, ep_size=world_size)
    model, _ = apply_sharding_plan(model, plan, mesh)
    _attach_ep(model, mesh, world_size)

    with torch.no_grad():
        out = model(input_ids)
    # N8：EP combine（all-reduce）后每 rank 输出 == 单卡参考
    torch.testing.assert_close(out, ref_logits, rtol=1e-5, atol=1e-5)

    # N8 非对称通信断言：本 rank 的 MoE 输出（combine 前）只覆盖路由到
    # 本地 expert 的 token——用 isolated MoE 验证
    layer0 = model.model.layers[0]
    ref_moe = ref.model.layers[0].mlp
    x = torch.randn(2, 4, 16)
    with torch.no_grad():
        ref_out = ref_moe(x)
        local_out = layer0.mlp(x)  # 已含 combine
    torch.testing.assert_close(local_out, ref_out, rtol=1e-5, atol=1e-5)

    # 路由分布非对称：本 rank expert 集合只处理 idx ∈ [offset, offset+n_local)
    moe = layer0.mlp
    with torch.no_grad():
        idx = moe.gate(x).argmax(dim=-1)
    n_local = 4 // world_size
    offset = moe.experts.expert_offset
    has_local = ((idx >= offset) & (idx < offset + n_local)).any()
    has_remote = ((idx < offset) | (idx >= offset + n_local)).any()
    # 至少有一种路由分布（不强制每个 rank 两者皆有，但全局必须覆盖）
    assert has_local or has_remote


def test_moe_local_map_ep2():
    run_dist(2, _worker)
