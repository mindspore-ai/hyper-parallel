# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S1.13: ShardingPlanner(plan_overrides=...) —— 用户手写 spec 合并（05 §3.6.7）。

覆盖点：
- 替换：fqn 命中 planner 已生成 spec → 整体替换，结构标记从模板补齐；
- 插入：fqn 未命中派生边界 → 插入并参与 terminal 标记；
- 嵌套（D-14，05 §13）：祖孙边界放行；外层 params 声明内层子树参数 →
  参数唯一归属 fail-fast；spec 缺 in_src → 全声明强制化 fail-fast；
- 契约不一致（D-14）：覆盖 spec 与上游契约不一致不再检查（无 warning
  无报错），声明原样保留，正确性由双模式数值对拍兜底；
- 容错：fqn 拼写错误 → ValueError；非 ModuleShardingSpec → TypeError；
- 隔离：用户传入的 spec 对象不被 plan() 改写（深拷贝），plan() 可重复调用。
"""

import copy
import logging

import pytest
import torch.nn as nn

from hyper_models.components.distributed.sharding_config import (
    CP,
    TP,
    ModuleShardingSpec,
    resolve_placements,
)
from hyper_models.components.distributed.sharding_planner import ShardingPlanner
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
    assert spec.use_local_map is False
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
    """插入语义：planner 不生成 spec 且与所有派生边界无祖孙关系的模块
    （此处为顶层旁路 dropout）可插入，并参与链式传播。"""
    mesh = make_mesh((1,), ("tp",))
    # 顶层旁路模块：planner 不覆盖，且与所有派生边界是兄弟（非祖孙）关系
    tiny_llama.model.dropout = nn.Dropout(0.0)
    identity = ModuleShardingSpec(
        params={},
        in_src={"hidden_states": {TP: Shard(1)}},
        in_dst={"hidden_states": {TP: Shard(1)}},   # identity
        out_src={TP: Shard(1)},
        out_dst={TP: Shard(1)},
    )
    planner = ShardingPlanner(plan_overrides={"model.dropout": identity})
    plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)

    assert "model.dropout" in plan.modules
    # 插入位置（named_modules 顺序）：model.norm → model.dropout → lm_head，
    # 均被下游引用 → 非末端；lm_head 仍为末端
    assert plan.modules["model.norm"]._is_terminal is False
    assert plan.modules["model.dropout"]._is_terminal is False
    assert plan.modules["lm_head"]._is_terminal is True


class TestNestedOverrideD14:
    """D-14（05 §13）嵌套 spec：祖孙边界放行，仅守两条 plan 期不变式——
    参数唯一归属（双切 fail-fast）与全声明强制化（缺 in_src fail-fast）。"""

    def test_nested_outer_allowed(self, tiny_llama, make_mesh):
        """override model.layers.0 作为外层边界（params={} 仅 I/O 契约），
        内层派生边界保留 → plan 成功，外层 spec 插入且内层不被改写。"""
        mesh = make_mesh((1,), ("tp",))
        block = ModuleShardingSpec(
            params={},
            in_src={"hidden_states": {TP: Shard(1)}},
            in_dst={"hidden_states": {TP: Shard(1)}},
            out_src={TP: Shard(1)},
            out_dst={TP: Shard(1)},
        )
        planner = ShardingPlanner(plan_overrides={"model.layers.0": block})
        plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)
        assert "model.layers.0" in plan.modules
        # 内层派生边界照常存在（不被外层接管）
        assert "model.layers.0.self_attn" in plan.modules
        assert plan.modules["model.layers.0.self_attn"].params["q_proj.weight"]

    def test_param_double_declaration_raises(self, tiny_llama, make_mesh):
        """外层 spec.params 声明了内层边界子树的参数 → ValueError（不变式 1：
        参数唯一归属，双切在 production 静默错误）。"""
        mesh = make_mesh((1,), ("tp",))
        block = ModuleShardingSpec(
            # self_attn.q_proj.weight 已被派生 attention 边界声明
            params={"self_attn.q_proj.weight": {TP: Shard(0)}},
            in_src={"hidden_states": {TP: Shard(1)}},
            in_dst={"hidden_states": {TP: Shard(1)}},
            out_src={TP: Shard(1)},
            out_dst={TP: Shard(1)},
        )
        planner = ShardingPlanner(plan_overrides={"model.layers.0": block})
        with pytest.raises(ValueError, match="exactly one boundary") as exc:
            planner.plan(tiny_llama, mesh, tp_size=2)
        assert "self_attn.q_proj.weight" in str(exc.value)
        assert "model.layers.0" in str(exc.value)

    def test_leaf_override_double_declaration_raises(self, tiny_llama, make_mesh):
        """override 派生边界的叶模块（q_proj）——参数与祖先边界冲突 →
        同样的唯一归属报错（嵌套本身合法，双切非法）。"""
        mesh = make_mesh((1,), ("tp",))
        leaf = ModuleShardingSpec(
            params={"weight": {TP: Shard(0)}},
            in_src={"hidden_states": {TP: Shard(1)}},
            in_dst={"hidden_states": {TP: Replicate()}},
            out_src={TP: Partial()},
            out_dst={TP: Shard(1)},
        )
        planner = ShardingPlanner(plan_overrides={
            "model.layers.0.self_attn.q_proj": leaf,
        })
        with pytest.raises(ValueError, match="exactly one boundary"):
            planner.plan(tiny_llama, mesh, tp_size=2)

    def test_missing_in_src_raises(self, tiny_llama, make_mesh):
        """全声明强制化（Scenario 1 填充已废除）：in_dst 非空而 in_src 为空
        → plan 期 ValueError。"""
        mesh = make_mesh((1,), ("tp",))
        block = ModuleShardingSpec(
            params={},
            in_src={},                                    # ← 空：D-14 后不再填充
            in_dst={"hidden_states": {TP: Shard(1)}},
            out_src={TP: Shard(1)},
            out_dst={TP: Shard(1)},
        )
        planner = ShardingPlanner(plan_overrides={"model.layers.0": block})
        with pytest.raises(ValueError, match="in_src"):
            planner.plan(tiny_llama, mesh, tp_size=2)


def test_override_chain_conflict_no_check(tiny_llama, make_mesh, caplog):
    """D-14：相邻契约一致性不再做任何静态/运行期检查——覆盖 spec 与上游
    out_dst 不一致时无 warning 无报错，声明原样保留（各模块只断言自身
    策略传播，端到端正确性由双模式数值对拍兜底）。

    """
    mesh = make_mesh((1,), ("tp",))
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
    with caplog.at_level(logging.WARNING):
        plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)
    assert "chain contract mismatch" not in caplog.text
    # 声明不被改写，plan 照常生成
    declared = plan.modules["model.layers.0.mlp"].in_src["hidden_states"]
    assert tuple(resolve_placements(declared, ("tp",))) == (Replicate(),)


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


def test_override_user_set_use_local_map(tiny_llama, make_mesh):
    """use_local_map 公开可配置（05 §3.6.7）：用户对自研模块显式置 True →
    合并后保留（模板未推断该标记时不受覆盖）；模板推断 True 的模块即使用户
    未设置也强制置位（防数值错误）。"""
    mesh = make_mesh((1,), ("tp",))
    # 用户对自研数据相关模块（此处借 mlp 演示）显式置 True
    custom = ModuleShardingSpec(
        params={
            "gate_proj.weight": {TP: Shard(0)},
            "up_proj.weight": {TP: Shard(0)},
            "down_proj.weight": {TP: Shard(1)},
        },
        in_src={"hidden_states": {TP: Shard(1)}},
        in_dst={"hidden_states": {TP: Shard(1)}},   # identity
        out_src={TP: Shard(1)},
        out_dst={TP: Shard(1)},
        use_local_map=True,
    )
    planner = ShardingPlanner(plan_overrides={"model.layers.0.mlp": custom})
    plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)
    # 用户显式 True 保留（mlp 模板 use_local_map=False，不会被清掉）
    assert plan.modules["model.layers.0.mlp"].use_local_map is True
    # 未覆盖的 mlp 保持模板 False
    assert plan.modules["model.layers.1.mlp"].use_local_map is False


def test_override_inner_wrap_fields_not_mutating_flags(tiny_llama, make_mesh):
    """inner-wrap 自定义入口（05 §4.4.2）：声明 inner_target/inner_wrapper
    后 inner-wrap 门控由 applier 解析链派生——**不改写 _needs_cp_attn**
    （声明互不嵌套），字段随深拷贝保留。"""
    mesh = make_mesh((1,), ("tp",))
    base = dict(
        params={
            "gate_proj.weight": {TP: Shard(0)},
            "up_proj.weight": {TP: Shard(0)},
            "down_proj.weight": {TP: Shard(1)},
        },
        in_src={"hidden_states": {TP: Shard(1)}},
        in_dst={"hidden_states": {TP: Shard(1)}},   # identity
        out_src={TP: Shard(1)},
        out_dst={TP: Shard(1)},
    )
    # inner_target 声明 → _needs_cp_attn 保持原值（False），不被隐式置位
    planner = ShardingPlanner(plan_overrides={
        "model.layers.0.mlp": ModuleShardingSpec(inner_target="self", **base),
    })
    plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec._needs_cp_attn is False
    assert spec.inner_target == "self"

    # inner_wrapper（str 注册表名/callable）→ 同样不改写，随深拷贝保留
    my_wrapper = lambda target, cp_mesh: None  # noqa: E731
    planner = ShardingPlanner(plan_overrides={
        "model.layers.0.mlp": ModuleShardingSpec(inner_wrapper=my_wrapper,
                                                 **base),
    })
    plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec._needs_cp_attn is False
    assert spec.inner_wrapper is my_wrapper


def test_override_local_compute_fn_gate_derived(tiny_llama, make_mesh):
    """local-region 自定义计算（05 §4.4.3）：声明 local_compute_fn 后骨架
    门控由 applier 解析链派生——**不改写 use_local_map**（声明互不嵌套），
    callable 随深拷贝保留。"""
    mesh = make_mesh((1,), ("tp",))

    def my_compute(module, hidden_states):
        return module.gate_proj(hidden_states)

    custom = ModuleShardingSpec(
        params={
            "gate_proj.weight": {TP: Shard(0)},
            "up_proj.weight": {TP: Shard(0)},
            "down_proj.weight": {TP: Shard(1)},
        },
        in_src={"hidden_states": {TP: Shard(1)}},
        in_dst={"hidden_states": {TP: Shard(1)}},   # identity
        out_src={TP: Shard(1)},
        out_dst={TP: Shard(1)},
        local_compute_fn=my_compute,
    )
    planner = ShardingPlanner(plan_overrides={"model.layers.0.mlp": custom})
    plan = planner.plan(tiny_llama, mesh, tp_size=2, sequence_parallel=True)
    spec = plan.modules["model.layers.0.mlp"]
    # 门控派生：use_local_map 保持声明原值（False），不被隐式改写
    assert spec.use_local_map is False
    assert spec.local_compute_fn is my_compute


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
