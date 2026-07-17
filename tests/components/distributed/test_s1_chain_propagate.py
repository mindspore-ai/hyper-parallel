# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S1.9: Phase 5 链式传播（4 场景）+ _is_terminal + 拓扑排序。"""

import logging

import torch.nn as nn

from hyper_models.components.distributed.sharding_config import (
    CP,
    TP,
    ModuleShardingSpec,
    ShardingPlan,
)
from hyper_models.components.distributed.sharding_planner import ShardingPlanner
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard

P = ShardingPlanner()
SP1 = {TP: Shard(1), CP: Shard(1), "ep": Replicate()}


class _Chain(nn.Module):
    """a → b → c 三节模块链。"""

    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)
        self.c = nn.Linear(4, 4)


def _plan_with(specs):
    plan = ShardingPlan(mesh_dim_names=("tp", "cp"))
    plan.modules.update(specs)
    return plan


def _spec(**kw):
    defaults = dict(
        in_src={"hidden_states": dict(SP1)},
        in_dst={"hidden_states": dict(SP1)},
        out_src={"output": dict(SP1)},
        out_dst={"output": dict(SP1)},
    )
    defaults.update(kw)
    return ModuleShardingSpec(**defaults)


class TestChainPropagate:
    def test_scenario1_fill_missing_in_src(self):
        """in_src 为空 → 用上游 out_dst 填充。"""
        plan = _plan_with({
            "a": _spec(),
            "b": _spec(in_src={}),
            "c": _spec(),
        })
        P._chain_propagate_and_validate(plan, _Chain())
        assert plan.modules["b"].in_src == {"hidden_states": SP1}

    def test_scenario1_fill_empty_dict_value(self):
        """key 存在但值为空 dict → 同样填充。"""
        plan = _plan_with({
            "a": _spec(),
            "b": _spec(in_src={"hidden_states": {}}),
            "c": _spec(),
        })
        P._chain_propagate_and_validate(plan, _Chain())
        assert plan.modules["b"].in_src["hidden_states"] == SP1

    def test_scenario2_first_and_last_module(self):
        """首模块 in_src 已由模板声明（不被填充）；末模块 out_dst 无下游校验。"""
        first = _spec(in_src={"input": {TP: Replicate(), CP: Replicate(),
                                        "ep": Replicate()}})
        plan = _plan_with({"a": first, "b": _spec(), "c": _spec()})
        P._chain_propagate_and_validate(plan, _Chain())
        # 首模块声明不被覆盖
        assert plan.modules["a"].in_src["input"][TP] == Replicate()
        # 末模块被标 terminal
        assert plan.modules["c"]._is_terminal is True
        assert plan.modules["a"]._is_terminal is False

    def test_scenario3_mismatch_warns(self, caplog):
        """a.out_dst=Replicate ≠ b.in_src=Shard(1) → 仅 warning，plan 保留。

        值相等比较无 shape 感知，边上的 reshape/transpose（合法场景）必然
        不等，故不报错；声明正确性由 validate 模式兜 correctness。
        """
        bad = _spec(out_dst={"output": {TP: Replicate(), CP: Replicate(),
                                        "ep": Replicate()}})
        plan = _plan_with({"a": bad, "b": _spec(), "c": _spec()})
        with caplog.at_level(logging.WARNING):
            P._chain_propagate_and_validate(plan, _Chain())
        assert "chain contract mismatch" in caplog.text
        assert "a" in caplog.text and "b" in caplog.text
        # 声明不被改写，plan 照常生成，terminal 标记不受影响
        declared = plan.modules["b"].in_src["hidden_states"]
        assert declared[TP] == Shard(1)
        assert plan.modules["b"]._is_terminal is False

    def test_scenario3_reshape_edge_legitimate(self):
        """边上有 reshape：a.out_dst Shard(1)（3D S 维）≠ b.in_src Shard(0)
        （2D 折叠维）→ 合法场景，不抛错即通过。"""
        up = _spec(out_dst={"output": {TP: Shard(1), CP: Shard(1),
                                       "ep": Replicate()}})
        down = _spec(in_src={"hidden_states": {TP: Shard(0), CP: Shard(0),
                                               "ep": Replicate()}},
                     in_dst={"hidden_states": {TP: Shard(0), CP: Shard(0),
                                               "ep": Replicate()}})
        plan = _plan_with({"a": up, "b": down, "c": _spec()})
        P._chain_propagate_and_validate(plan, _Chain())  # 不抛错即通过
        assert plan.modules["b"].in_src["hidden_states"][TP] == Shard(0)

    def test_scenario4_custom_module_inserted(self):
        """自定义模块（标量简写 + 空 in_src）插入后契约连接。"""
        custom = ModuleShardingSpec(
            params={},
            in_src={},
            in_dst={"hidden_states": dict(SP1)},
            out_src=dict(SP1),           # 标量简写
            out_dst={"output": dict(SP1)},
        )
        plan = _plan_with({"a": _spec(), "b": custom, "c": _spec()})
        P._chain_propagate_and_validate(plan, _Chain())
        assert plan.modules["b"].in_src["hidden_states"] == SP1

    def test_name_agnostic_single_entry_pairing(self):
        """上游 out key 'output' 与下游 in key 'x_BLD' 单 entry 名字无关配对。"""
        moe_in = _spec(in_src={"x_BLD": dict(SP1)}, in_dst={"x_BLD": dict(SP1)})
        plan = _plan_with({"a": _spec(), "b": moe_in, "c": _spec()})
        P._chain_propagate_and_validate(plan, _Chain())  # 不抛错即配对成功

    def test_out_dst_none_skips(self):
        plan = _plan_with({
            "a": _spec(out_dst=None),
            "b": _spec(),
            "c": _spec(),
        })
        P._chain_propagate_and_validate(plan, _Chain())
        assert plan.modules["a"]._is_terminal is True


class TestTopologicalSort:
    def test_named_modules_order(self):
        model = _Chain()
        out = P._topological_sort_by_forward_order(["c", "a", "b"], model)
        assert out == ["a", "b", "c"]

    def test_missing_fqn_appended_with_warning(self, caplog):
        model = _Chain()
        with caplog.at_level(logging.WARNING):
            out = P._topological_sort_by_forward_order(["a", "ghost"], model)
        assert out == ["a", "ghost"]
        assert "not found in named_modules" in caplog.text
