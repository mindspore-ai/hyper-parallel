# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S3.5（2 进程）: _apply_phase_c CP 分支 — 双模式注入同一 wrapper。"""

import torch

from hyper_models.components.distributed import ShardingPlanner, apply_sharding_plan
from hyper_models.components.distributed.cp_utils import shard_batch_for_cp
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from tests.components.distributed.conftest import (
    TinyConfig,
    TinyLlamaForCausalLM,
    run_dist,
)


def _worker(rank, world_size):
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("cp",))
    cp_mesh = mesh["cp"]

    torch.manual_seed(1234)
    ref_model = TinyLlamaForCausalLM(TinyConfig(), causal=True).eval()
    torch.manual_seed(7)
    input_ids = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        ref = ref_model(input_ids)

    chunk = 8 // world_size
    slc = slice(rank * chunk, (rank + 1) * chunk)

    for mode in ("production", "validate"):
        torch.manual_seed(1234)
        model = TinyLlamaForCausalLM(TinyConfig(), causal=True).eval()
        # 保存原始 attention forward 引用，验证注入发生
        orig_fwd = model.model.layers[0].self_attn.forward
        plan = ShardingPlanner().plan(model, mesh, tp_size=1, cp_size=world_size)
        model, _ = apply_sharding_plan(model, plan, mesh,
                                       validate_mode=(mode == "validate"))
        attn = model.model.layers[0].self_attn
        # 两模式均注入了 CP wrapper（forward 已被替换）
        assert attn.forward != orig_fwd
        batch = shard_batch_for_cp({"input_ids": input_ids}, cp_mesh)
        with torch.no_grad():
            out = model(batch["input_ids"])
        # D-07：lm_head 输出为本 rank CP chunk 的 logits，逐 rank 对拍
        torch.testing.assert_close(out, ref[:, slc], rtol=1e-5, atol=1e-5)


def _worker_no_cp_no_inject(rank, world_size):
    """cp 轴 size=1 → planner 过滤后无 cp 维，不注入 CP wrapper。"""
    mesh = init_device_mesh("cpu", (1, world_size), mesh_dim_names=("cp", "tp"))
    torch.manual_seed(1234)
    model = TinyLlamaForCausalLM(TinyConfig()).eval()
    orig_fwd = model.model.layers[0].self_attn.forward
    plan = ShardingPlanner().plan(model, mesh, tp_size=world_size, cp_size=1)
    assert "cp" not in plan.mesh_dim_names
    apply_sharding_plan(model, plan, mesh)
    # 只被 production boundary 包装了一层（__wrapped__ 即原始 forward；
    # bound method 每次访问生成新对象，用 == 比较而非 is）。
    assert model.model.layers[0].self_attn.forward.__wrapped__ == orig_fwd


def test_phase_c_cp_dual_mode_2proc():
    run_dist(2, _worker)


def test_phase_c_cp_size1_no_inject():
    run_dist(2, _worker_no_cp_no_inject)
