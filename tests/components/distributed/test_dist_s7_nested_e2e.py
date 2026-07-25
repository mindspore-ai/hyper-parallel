# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S7.2（2/4 进程）: D-14 嵌套 spec 端到端 —— 双模式数值对拍（05 §13.7）。

覆盖矩阵行：
- N1：外层普通 boundary（model.layers.0，params={} 仅 I/O 契约）+ 内层标准
  边界，TP=2，validate/production 双模式对拍单卡；
- N2：外层 local_map + 内层 validate 孤岛，TP=2，双模式对拍；
- N7：N2 validate 模式下，外层 local region 内内层边界参数保持 DTensor
  （不变式 3 解包作用域排除，forward_pre_hook 直接断言）；
- N3：旗舰场景——根 spec（fqn ""，整个 LM 外层契约）+ 内层关键模块孤岛，
  TP=2×CP=2，双模式对拍（CP 组 all-gather 拼回全序列）；
- N4：外层 local_map + 内层 MoE local region（嵌套 local region 不嵌套
  包装），TP=2×EP=2，双模式对拍。
"""

import torch
import torch.distributed as dist
import torch.nn as nn

from hyper_models.components.distributed import ShardingPlanner, apply_sharding_plan
from hyper_models.components.distributed.cp_utils import shard_batch_for_cp
from hyper_models.components.distributed.sharding_config import (
    CP,
    TP,
    ModuleShardingSpec,
)
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.components.distributed.conftest import (
    TinyConfig,
    TinyHFNativeMoEForCausalLM,
    TinyLlamaForCausalLM,
    run_dist,
)


def _build(causal=False):
    torch.manual_seed(1234)
    return TinyLlamaForCausalLM(TinyConfig(), causal=causal).eval()


def _build_moe():
    torch.manual_seed(1234)
    return TinyHFNativeMoEForCausalLM(TinyConfig(num_experts=4)).eval()


def _block_spec(use_local_map):
    """外层 decoder layer 边界：identity I/O 契约，params={}。"""
    return ModuleShardingSpec(
        params={},
        in_src={"hidden_states": {TP: Shard(1), CP: Replicate()}},
        in_dst={"hidden_states": {TP: Shard(1), CP: Replicate()}},
        out_src={"output": {TP: Shard(1), CP: Replicate()}},
        out_dst={"output": {TP: Shard(1), CP: Replicate()}},
        use_local_map=use_local_map,
    )


# ── N1/N2/N7：外层 decoder layer + 内层标准边界（TP=2）────────────────────

def _worker_nested_block(rank, world_size, use_local_map):
    ref_model = _build()
    torch.manual_seed(7)
    x = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        ref = ref_model(x)

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))

    for mode in ("production", "validate"):
        model = _build()
        planner = ShardingPlanner(plan_overrides={
            "model.layers.0": _block_spec(use_local_map),
        })
        plan = planner.plan(model, mesh, tp_size=world_size)
        assert "model.layers.0.self_attn" in plan.modules  # 内层保留
        model, _ = apply_sharding_plan(
            model, plan, mesh, validate_mode=(mode == "validate"))

        # N7：validate + local_map 外层 → 内层边界参数在 forward 中保持
        # DTensor（外层 local region 解包排除内层子树）
        if mode == "validate" and use_local_map:
            seen = {}

            def _probe(module, args, kwargs=None):
                seen["q_proj_is_dtensor"] = isinstance(
                    module.q_proj.weight, DTensor)

            handle = model.model.layers[0].self_attn.register_forward_pre_hook(_probe)
            with torch.no_grad():
                out = model(x)
            handle.remove()
            assert seen.get("q_proj_is_dtensor") is True, (
                "inner boundary params must stay DTensor inside the outer "
                "local region (D-14 invariant 3)")
        else:
            with torch.no_grad():
                out = model(x)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_n1_nested_normal_boundary_tp2():
    run_dist(2, _worker_nested_block, args=(False,))


def test_n2n7_nested_outer_local_map_tp2():
    run_dist(2, _worker_nested_block, args=(True,))


# ── N3：根 spec（整个 LM 外层契约）+ 内层孤岛（TP=2×CP=2）──────────────────

def _worker_nested_root(rank, world_size):
    ref_model = _build(causal=True)
    torch.manual_seed(7)
    x = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        ref = ref_model(x)

    cp_size, tp_size = 2, 2
    mesh = init_device_mesh("cpu", (cp_size, tp_size),
                            mesh_dim_names=("cp", "tp"))
    cp_mesh = mesh["cp"]

    root_spec = ModuleShardingSpec(
        params={},
        use_local_map=True,     # 整模型范围太大，glue 代码不走 dispatch
        in_src={"input_ids": {TP: Replicate(), CP: Shard(1)}},
        in_dst={"input_ids": {TP: Replicate(), CP: Shard(1)}},
        out_src={"output": {TP: Replicate(), CP: Shard(1)}},   # D-07 本地 chunk
        out_dst={"output": {TP: Replicate(), CP: Shard(1)}},
    )

    for mode in ("production", "validate"):
        model = _build(causal=True)
        planner = ShardingPlanner(plan_overrides={"": root_spec})
        plan = planner.plan(model, mesh, tp_size=tp_size, cp_size=cp_size)
        assert "" in plan.modules
        assert "model.layers.0.self_attn" in plan.modules
        model, _ = apply_sharding_plan(
            model, plan, mesh, validate_mode=(mode == "validate"))

        local_x = shard_batch_for_cp({"input_ids": x}, cp_mesh)["input_ids"]
        with torch.no_grad():
            out_local = model(local_x)          # [B, S/cp, V]（D-07）

        gathered = [torch.empty_like(out_local) for _ in range(cp_size)]
        dist.all_gather(gathered, out_local, group=cp_mesh.get_group())
        out = torch.cat(gathered, dim=1)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_n3_nested_root_whole_lm_tp2cp2():
    run_dist(4, _worker_nested_root)


# ── N4：外层 local_map + 内层 MoE local region（TP=2×EP=2）─────────────────

def _worker_nested_moe(rank, world_size):
    ref_model = _build_moe()
    torch.manual_seed(7)
    x = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        ref = ref_model(x)

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))

    for mode in ("production", "validate"):
        model = _build_moe()
        planner = ShardingPlanner(plan_overrides={
            "model.layers.0": _block_spec(use_local_map=True),
        })
        plan = planner.plan(
            model, mesh, tp_size=world_size, ep_size=world_size)
        # 内层 MoE local region 与外层 local region 并存（各自成区）
        assert plan.modules["model.layers.0.mlp"]._ep_size == world_size
        model, _ = apply_sharding_plan(
            model, plan, mesh, validate_mode=(mode == "validate"))
        with torch.no_grad():
            out = model(x)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_n4_nested_outer_local_map_moe_tp2ep2():
    run_dist(2, _worker_nested_moe)
