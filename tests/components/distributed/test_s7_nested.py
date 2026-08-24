# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================

"""test_s7_nested.py: 核心套件合并文件。

来源: test_s7_nested_plan.py, test_dist_s7_nested_e2e.py
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from hyper_parallel.auto_models.components.distributed import (
    ShardingPlanner,
    apply_sharding_plan,
)
from hyper_parallel.auto_models.components.distributed.cp_utils import shard_batch_for_cp
from hyper_parallel.auto_models.components.distributed.sharding_config import (
    CP,
    ModuleShardingSpec,
    TP,
)
from hyper_parallel.auto_models.components.distributed.sharding_planner import ShardingPlanner
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import (
    Replicate,
    Shard,
)
from tests.components.distributed.conftest import (
    TinyConfig,
    TinyHFNativeMoEForCausalLM,
    TinyLlamaForCausalLM,
    cp_sdpa_hf_injection,
    ep_archetype_injection,
    run_dist,
)


# ==========================================================================
# 来源: test_s7_nested_plan.py
# S7.1: D-14 嵌套 spec 的 plan 期行为（05 §13.2/§13.3，单进程）。
# ==========================================================================

def _identity_block_spec():
    """外层容器边界的 identity I/O 契约（params={}）。"""
    return ModuleShardingSpec(
        params={},
        in_src={"hidden_states": {TP: Shard(1)}},
        in_dst={"hidden_states": {TP: Shard(1)}},
        out_src={TP: Shard(1)},
        out_dst={TP: Shard(1)},
    )


def test_terminal_is_last_boundary_only(tiny_llama, make_mesh):
    """D-14 后 _is_terminal 语义：仅 forward 顺序最后一个边界（lm_head）
    为 terminal，其余全部非 terminal（不再按 out_dst 被引用与否判定）。"""
    mesh = make_mesh((1,), ("tp",))
    plan = ShardingPlanner().plan(tiny_llama, mesh, tp_size=2)
    assert plan.modules["lm_head"]._is_terminal is True
    for fqn, spec in plan.modules.items():
        if fqn != "lm_head":
            assert spec._is_terminal is False, fqn


def test_terminal_with_nested_outer(tiny_llama, make_mesh):
    """嵌套外层 spec 不影响 terminal 判定：外层（model.layers.0）在 forward
    顺序中段 → 非 terminal；lm_head 仍 terminal。"""
    mesh = make_mesh((1,), ("tp",))
    planner = ShardingPlanner(
        plan_overrides={"model.layers.0": _identity_block_spec()})
    plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)
    assert plan.modules["model.layers.0"]._is_terminal is False
    assert plan.modules["lm_head"]._is_terminal is True


def test_root_spec_allowed(tiny_llama, make_mesh):
    """根 spec（fqn ""，整个 LM 外层契约）合法插入：与所有内层边界构成
    嵌套，params={} 不触发唯一归属冲突；forward 顺序最前 → 非 terminal。"""
    mesh = make_mesh((1,), ("tp",))
    root = ModuleShardingSpec(
        params={},
        in_src={"input_ids": {TP: Shard(1)}},
        in_dst={"input_ids": {TP: Shard(1)}},
        out_src={TP: Shard(1)},
        out_dst={TP: Shard(1)},
    )
    planner = ShardingPlanner(plan_overrides={"": root})
    plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)
    assert "" in plan.modules
    assert plan.modules[""]._is_terminal is False
    assert plan.modules["lm_head"]._is_terminal is True
    # 内层派生边界全部保留
    assert "model.layers.0.self_attn" in plan.modules
    assert "model.embed_tokens" in plan.modules


def test_outer_declares_intermediate_params(tiny_llama, make_mesh):
    """外层可声明不属任何内层边界子树的中间层参数（唯一归属不冲突）。

    构造：layers.0 内挂一个非边界旁路 Linear（planner 不会为其生成边界），
    外层 spec 声明其参数 → plan 成功且参数进入外层 spec。
    """
    mesh = make_mesh((1,), ("tp",))
    bypass = nn.Linear(16, 16, bias=False)
    tiny_llama.model.layers[0].bypass = bypass
    block = ModuleShardingSpec(
        params={"bypass.weight": {TP: Shard(0)}},   # 中间层参数，归外层
        in_src={"hidden_states": {TP: Shard(1)}},
        in_dst={"hidden_states": {TP: Shard(1)}},
        out_src={TP: Shard(1)},
        out_dst={TP: Shard(1)},
    )
    planner = ShardingPlanner(plan_overrides={"model.layers.0": block})
    plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)
    assert plan.modules["model.layers.0"].params["bypass.weight"][TP] == Shard(0)


# ==========================================================================
# 来源: test_dist_s7_nested_e2e.py
# S7.2（2/4 进程）: D-14 嵌套 spec 端到端 —— 双模式数值对拍（05 §13.7）。
# ==========================================================================

def _build(causal=False):
    torch.manual_seed(1234)
    return TinyLlamaForCausalLM(TinyConfig(), causal=causal).eval()


def _build_moe():
    torch.manual_seed(1234)
    return TinyHFNativeMoEForCausalLM(TinyConfig(num_experts=4)).eval()


def _block_spec(region_dispatch):
    """外层 decoder layer 边界：identity I/O 契约，params={}。"""
    return ModuleShardingSpec(
        params={},
        in_src={"hidden_states": {TP: Shard(1), CP: Replicate()}},
        in_dst={"hidden_states": {TP: Shard(1), CP: Replicate()}},
        out_src={"output": {TP: Shard(1), CP: Replicate()}},
        out_dst={"output": {TP: Shard(1), CP: Replicate()}},
        region_dispatch=region_dispatch,
    )


def _worker_nested_block(rank, world_size, region_dispatch):
    ref_model = _build()
    torch.manual_seed(7)
    x = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        ref = ref_model(x)

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))

    for mode in ("production", "validate"):
        model = _build()
        planner = ShardingPlanner(plan_overrides={
            "model.layers.0": _block_spec(region_dispatch),
        })
        plan = planner.plan(model, mesh, tp_size=world_size)
        assert "model.layers.0.self_attn" in plan.modules  # 内层保留
        model, _ = apply_sharding_plan(
            model, plan, mesh, validate_mode=(mode == "validate"))

        # N7：validate + region_dispatch=False 外层 → 内层边界参数在 forward
        # 中保持 DTensor（外层 local region 解包排除内层子树）
        if mode == "validate" and region_dispatch is False:
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
    run_dist(2, _worker_nested_block, args=(None,))


def test_n2n7_nested_outer_local_map_tp2():
    """外层 region_dispatch=False（整个 decoder layer 的 glue 代码不可
    dispatch）→ 外层走 local-region 骨架（D-14 嵌套解包排除）。"""
    run_dist(2, _worker_nested_block, args=(False,))


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
        region_dispatch=False,     # 整模型范围太大，glue 代码不走 dispatch
        in_src={"input_ids": {TP: Replicate(), CP: Shard(1)}},
        in_dst={"input_ids": {TP: Replicate(), CP: Shard(1)}},
        out_src={"output": {TP: Replicate(), CP: Shard(1)}},   # D-07 本地 chunk
        out_dst={"output": {TP: Replicate(), CP: Shard(1)}},
    )

    for mode in ("production", "validate"):
        model = _build(causal=True)
        # 根 spec 插入 + CP wrapper 显式注入（glob merge，契约继承推导）
        planner = ShardingPlanner(plan_overrides={
            "": root_spec,
            **cp_sdpa_hf_injection(),
        })
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


def _worker_nested_moe(rank, world_size):
    ref_model = _build_moe()
    torch.manual_seed(7)
    x = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        ref = ref_model(x)

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))

    for mode in ("production", "validate"):
        model = _build_moe()
        # 外层 local_map + 内层 HF 原生 MoE：EP compute 显式注入（改造后
        # 无自动注入，region_dispatch 已被 planner 清除）
        planner = ShardingPlanner(plan_overrides={
            "model.layers.0": _block_spec(region_dispatch=False),
            **ep_archetype_injection(),
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
