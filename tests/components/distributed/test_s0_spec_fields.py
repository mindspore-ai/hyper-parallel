# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S0.2: ShardingPlan / ModuleShardingSpec 字段与 05 §3.1-3.2 对齐。"""

from hyper_models.components.distributed.sharding_config import (
    ModuleShardingSpec,
    ShardingPlan,
)


def test_spec_defaults():
    spec = ModuleShardingSpec()
    assert spec.params == {}
    assert spec.in_src == {}
    assert spec.in_dst == {}
    assert spec.out_src is None
    assert spec.out_dst is None
    assert spec.out_names is None
    assert spec.is_boundary is True
    # 内部标记存在且默认 False
    assert spec._is_terminal is False
    assert spec.use_local_map is False
    assert spec._needs_cp_attn is False


def test_plan_defaults():
    plan = ShardingPlan()
    assert plan.modules == {}
    assert plan.sequence_parallel is True
    assert plan.loss_parallel is False
    assert plan.special_handlers == {}
    assert plan.mesh_dim_names == ()
    assert plan.tied_pairs == []


def test_spec_mutable_fields_independent():
    """default_factory 不共享可变默认值。"""
    a, b = ModuleShardingSpec(), ModuleShardingSpec()
    a.params["w"] = {}
    assert b.params == {}
