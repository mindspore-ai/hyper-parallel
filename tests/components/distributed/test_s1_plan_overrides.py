# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S1.13: ShardingPlanner(plan_overrides=...) —— 用户手写 spec 合并（05 §3.6.7）。

覆盖点：
- 替换：fqn 命中 planner 已生成 spec → 整体替换，结构标记从模板补齐；
- 插入：fqn 未命中（容器/无参数模块）→ 插入并参与链式传播与 terminal 标记；
- 校验：覆盖 spec 与上游契约冲突 → PlacementMismatchError（合并发生在
  Phase 5 之前，链式校验仍然生效）；
- 容错：fqn 拼写错误 → ValueError；非 ModuleShardingSpec → TypeError；
- 隔离：用户传入的 spec 对象不被 plan() 改写（深拷贝），plan() 可重复调用。
"""

import copy

import pytest

from hyper_parallel.components.distributed.sharding_config import (
    CP,
    TP,
    ModuleShardingSpec,
    PlacementMismatchError,
    resolve_placements,
)
from hyper_parallel.components.distributed.sharding_planner import ShardingPlanner
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard


def _attn_override_spec(key="x"):
    """自研多输入 attention 的手写 spec：契约 key 为真实签名参数名。"""
    return ModuleShardingSpec(
        params={
            "q_proj.weight": {TP: Shard(0), CP: Replicate()},
            "k_proj.weight": {TP: Shard(0), CP: Replicate()},
            "v_proj.weight": {TP: Shard(0), CP: Replicate()},
            "o_proj.weight": {TP: Shard(1), CP: Replicate()},
        },
        in_src={key: {TP: Shard(1)}},
        in_dst={key: {TP: Replicate()}},
        out_src={TP: Partial()},   # 标量简写，合并时应归一化为 {"output": ...}
        out_dst={TP: Shard(1)},
    )


def test_override_replaces_spec_and_fills_flags(tiny_llama, make_mesh):
    """替换语义：用户 spec 为权威；_needs_cp_attn 从模板补齐；标量简写归一化。"""
    mesh = make_mesh((1,), ("tp",))
    planner = ShardingPlanner(plan_overrides={
        "model.layers.0.self_attn": _attn_override_spec(key="x"),
    })
    plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)

    spec = plan.modules["model.layers.0.self_attn"]
    # 用户 spec 生效：契约 key 为 "x"
    assert set(spec.in_src) == {"x"}
    assert set(spec.in_dst) == {"x"}
    assert tuple(resolve_placements(spec.in_src["x"], ("tp",))) == (Shard(1),)
    # 结构标记从 attention 模板补齐（用户未设置）
    assert spec._needs_cp_attn is True
    assert spec._use_local_map is False
    # 标量简写已归一化
    assert set(spec.out_src) == {"output"}
    assert tuple(resolve_placements(spec.out_src["output"], ("tp",))) == (Partial(),)
    # _is_terminal 由 Phase 5 统一标记（非末端）
    assert spec._is_terminal is False
    # 参数分片声明原样保留
    assert spec.params["q_proj.weight"][TP] == Shard(0)
    assert spec.params["o_proj.weight"][TP] == Shard(1)

    # 未被覆盖的模块保持模板推导结果
    other = plan.modules["model.layers.1.self_attn"]
    assert set(other.in_src) == {"hidden_states"}


def test_override_insert_for_missed_module(tiny_llama, make_mesh):
    """插入语义：planner 不生成 spec 的容器模块也可插入，并参与链式传播。"""
    mesh = make_mesh((1,), ("tp",))
    identity = ModuleShardingSpec(
        params={},
        in_src={"hidden_states": {TP: Shard(1)}},
        in_dst={"hidden_states": {TP: Shard(1)}},   # identity
        out_src={TP: Shard(1)},
        out_dst={TP: Shard(1)},
    )
    planner = ShardingPlanner(plan_overrides={"model.layers.0": identity})
    plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)

    assert "model.layers.0" in plan.modules
    # 插入后与上下游契约一致（embed → layers.0 → input_layernorm 全 Shard(1)），
    # 均被下游引用 → 非末端；lm_head 仍为末端
    assert plan.modules["model.layers.0"]._is_terminal is False
    assert plan.modules["model.embed_tokens"]._is_terminal is False
    assert plan.modules["lm_head"]._is_terminal is True


def test_override_chain_conflict_raises(tiny_llama, make_mesh):
    """覆盖 spec 声明与上游 out_dst 冲突 → Phase 5 链式校验报错。"""
    mesh = make_mesh((1,), ("tp",))
    bad = _attn_override_spec(key="x")
    # mlp 声明 in_src=Replicate，但上游 post_attention_layernorm out_dst=Shard(1)
    bad_mlp = ModuleShardingSpec(
        params={
            "gate_proj.weight": {TP: Shard(0)},
            "up_proj.weight": {TP: Shard(0)},
            "down_proj.weight": {TP: Shard(1)},
        },
        in_src={"hidden_states": {TP: Replicate()}},
        in_dst={"hidden_states": {TP: Replicate()}},
        out_src={TP: Partial()},
        out_dst={TP: Shard(1)},
    )
    planner = ShardingPlanner(plan_overrides={"model.layers.0.mlp": bad_mlp})
    with pytest.raises(PlacementMismatchError):
        planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)
    del bad


def test_override_invalid_fqn_raises(tiny_llama, make_mesh):
    """fqn 未命中 named_modules（拼写错误）→ fail-fast ValueError。"""
    mesh = make_mesh((1,), ("tp",))
    planner = ShardingPlanner(plan_overrides={
        "model.layers.0.self_atn": _attn_override_spec(),   # 拼写错误
    })
    with pytest.raises(ValueError, match="named_modules"):
        planner.plan(tiny_llama, mesh, tp_size=2)


def test_override_wrong_type_raises(tiny_llama, make_mesh):
    """override 值必须是 ModuleShardingSpec。"""
    mesh = make_mesh((1,), ("tp",))
    planner = ShardingPlanner(plan_overrides={
        "model.layers.0.self_attn": {"params": {}},
    })
    with pytest.raises(TypeError, match="ModuleShardingSpec"):
        planner.plan(tiny_llama, mesh, tp_size=2)


def test_override_input_spec_not_mutated(tiny_llama, make_mesh):
    """plan() 深拷贝用户 spec：归一化/标记/链式填充不污染调用方对象，可重复调用。"""
    mesh = make_mesh((1,), ("tp",))
    user_spec = _attn_override_spec(key="x")
    snapshot = copy.deepcopy(user_spec)
    planner = ShardingPlanner(plan_overrides={"model.layers.0.self_attn": user_spec})

    planner.plan(tiny_llama, mesh, tp_size=2)
    assert user_spec.out_src == snapshot.out_src            # 未被归一化改写
    assert user_spec._needs_cp_attn is False                # 未被模板补齐改写
    assert user_spec._is_terminal is False                  # 未被 Phase 5 改写
    assert user_spec.in_src == snapshot.in_src              # 未被链式填充改写

    # 重复调用结果一致（不累积污染）
    plan2 = planner.plan(tiny_llama, mesh, tp_size=2)
    spec2 = plan2.modules["model.layers.0.self_attn"]
    assert set(spec2.in_src) == {"x"}
    assert spec2._needs_cp_attn is True
