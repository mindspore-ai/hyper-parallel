# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S2.7（2 进程）: _wrap_validate_forward — 正确 plan 全 pass + 错误声明抛错。"""

import pytest
import torch

from hyper_models.components.distributed import ShardingPlanner, apply_sharding_plan
from hyper_models.components.distributed.sharding_config import (
    PlacementMismatchError,
    TP,
)
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


def _worker_pass(rank, world_size):
    ref_model = _build()
    torch.manual_seed(7)
    x = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        ref = ref_model(x)

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    model = _build()
    plan = ShardingPlanner().plan(model, mesh, tp_size=world_size)
    model, tp_grad_info = apply_sharding_plan(model, plan, mesh, validate_mode=True)
    assert tp_grad_info is None  # validate 模式无 tp_grad_info
    with torch.no_grad():
        out = model(x)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def _worker_mismatch(rank, world_size):
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    model = _build()
    plan = ShardingPlanner().plan(model, mesh, tp_size=world_size)
    # 故意把 out_src 声明为 Replicate（DTensor 传播实际产出 Partial）
    plan.modules["model.layers.0.self_attn"].out_src = {
        "output": {TP: Replicate()}}
    model, _ = apply_sharding_plan(model, plan, mesh, validate_mode=True)
    torch.manual_seed(7)
    x = torch.randint(0, 32, (2, 8))
    with pytest.raises(PlacementMismatchError) as exc:
        model(x)
    assert exc.value.stage.startswith("out_src")


def _worker_param_mismatch(rank, world_size):
    """参数声明错误（o_proj 应为 Shard(1) 误写 Shard(0)）→ dispatch 层
    layout 校验即拦截（ValueError）——比 out_src 校验更早暴露。"""
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    model = _build()
    plan = ShardingPlanner().plan(model, mesh, tp_size=world_size)
    plan.modules["model.layers.0.self_attn"].params["o_proj.weight"] = {TP: Shard(0)}
    model, _ = apply_sharding_plan(model, plan, mesh, validate_mode=True)
    torch.manual_seed(7)
    x = torch.randint(0, 32, (2, 8))
    with pytest.raises(ValueError, match="layout"):
        model(x)


def _worker_terminal_out_dst(rank, world_size):
    """terminal 模块（lm_head）的 out_dst 校验函数：声明与实际 DTensor 传播
    placement 不一致 → 抛 PlacementMismatchError(stage="out_dst")。

    注：端到端路径下 boundary.redistribute_outputs 以声明的 out_dst 为目标，
    产出恒等于声明——out_dst 校验是防御性的，此处直接对校验函数构造不一致。
    """
    from hyper_models.components.distributed.sharding_applier import (
        _validate_out_dst,
    )
    from hyper_models.components.distributed.sharding_config import (
        ModuleShardingSpec,
    )
    from hyper_parallel.core.dtensor.dtensor import DTensor

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    spec = ModuleShardingSpec(
        out_dst={"output": {TP: Shard(1)}},
    )
    spec._is_terminal = True
    dt = DTensor.from_local(torch.randn(2, 4), mesh, (Replicate(),))
    with pytest.raises(PlacementMismatchError) as exc:
        _validate_out_dst(dt, spec, ("tp",), "Linear")
    assert exc.value.stage.startswith("out_dst")

    # 一致 → 不抛
    spec2 = ModuleShardingSpec(out_dst={"output": {TP: Replicate()}})
    spec2._is_terminal = True
    _validate_out_dst(dt, spec2, ("tp",), "Linear")


def test_validate_forward_pass_tp2():
    run_dist(2, _worker_pass)


def test_validate_forward_out_src_mismatch_tp2():
    run_dist(2, _worker_mismatch)


def test_validate_forward_param_mismatch_tp2():
    run_dist(2, _worker_param_mismatch)


def test_validate_terminal_out_dst_tp2():
    run_dist(2, _worker_terminal_out_dst)
