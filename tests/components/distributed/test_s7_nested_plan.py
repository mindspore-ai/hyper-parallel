# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S7.1: D-14 嵌套 spec 的 plan 期行为（05 §13.2/§13.3，单进程）。

覆盖点：
- _is_terminal 新语义：链式传播废除后，仅 forward 顺序最后一个边界为
  terminal（含嵌套外层 spec 的场景）；
- 参数唯一归属：外层声明直属/中间层参数合法；与内层边界冲突 fail-fast
  （test_s1_plan_overrides.py 的 TestNestedOverrideD14 为负例主体）；
- 嵌套根 spec（fqn ""）合法：整个 LM 外层契约的旗舰形态。
"""

import torch.nn as nn

from hyper_models.components.distributed.sharding_config import (
    TP,
    ModuleShardingSpec,
)
from hyper_models.components.distributed.sharding_planner import ShardingPlanner
from hyper_parallel.core.dtensor.placement_types import Shard


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
