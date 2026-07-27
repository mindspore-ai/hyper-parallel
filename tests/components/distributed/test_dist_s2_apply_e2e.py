# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S2.11（2 进程）: apply_sharding_plan 主入口 — TP=2 双模式端到端。"""

import torch
import torch.nn as nn

from hyper_models.components.distributed import ShardingPlanner, apply_sharding_plan
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.components.distributed.conftest import (
    TinyConfig,
    TinyLlamaForCausalLM,
    run_dist,
)


def _build():
    torch.manual_seed(1234)
    return TinyLlamaForCausalLM(TinyConfig()).eval()


def _worker(rank, world_size):
    ref_model = _build()
    torch.manual_seed(7)
    x = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        ref = ref_model(x)

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))

    outs = {}
    for mode in ("production", "validate"):
        model = _build()
        plan = ShardingPlanner().plan(model, mesh, tp_size=world_size)
        model, tp_grad_info = apply_sharding_plan(
            model, plan, mesh, validate_mode=(mode == "validate"))
        # 返回二元组结构
        assert isinstance(model, nn.Module)
        if mode == "production":
            assert tp_grad_info is not None
            # tp_grad_info 覆盖所有 spec 参数
            assert "model.layers.0.self_attn.q_proj.weight" in tp_grad_info
            assert tp_grad_info["model.layers.0.self_attn.q_proj.weight"][0] == Shard(0)
            assert tp_grad_info["model.layers.0.input_layernorm.weight"][0] == Replicate()
        else:
            assert tp_grad_info is None
        with torch.no_grad():
            outs[mode] = model(x)

    # 双模式端到端：production == validate == 单卡参考（逐 rank 断言）
    torch.testing.assert_close(outs["production"], ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(outs["validate"], ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(outs["production"], outs["validate"],
                               rtol=1e-5, atol=1e-5)


def _worker_list_model(rank, world_size):
    """list[nn.Module] 支持（PP 多 part 形式）。"""
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    model = _build()
    plan = ShardingPlanner().plan(model, mesh, tp_size=world_size)
    result, info = apply_sharding_plan([model], plan, mesh)
    assert isinstance(result, list) and len(result) == 1


def test_apply_e2e_dual_mode_tp2():
    run_dist(2, _worker)


def test_apply_list_model_2proc():
    run_dist(2, _worker_list_model)
