# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S1.10: Phase 6 _collect_special_handlers + SPECIAL_HANDLERS 注册表。"""

from hyper_models.components.distributed.param_role import ParamRole
from hyper_models.components.distributed.sharding_planner import (
    SPECIAL_HANDLERS,
    ShardingPlanner,
)

P = ShardingPlanner()


def test_special_role_mapped_to_handler():
    roles = {
        "model.layers.0.gated_delta.a_log": ParamRole.SPECIAL,
        "model.layers.0.self_attn.q_proj.weight": ParamRole.COLWISE,
    }
    out = P._collect_special_handlers(roles)
    assert out == {"model.layers.0.gated_delta.a_log": "gated_delta_tp_shard"}


def test_unregistered_pattern_defaults():
    class _P(ShardingPlanner):
        def __init__(self):
            super().__init__()
            self._special_handler_patterns = {}

    p = _P()
    out = p._collect_special_handlers({"m.x.special_w": ParamRole.SPECIAL})
    assert out == {"m.x.special_w": "default"}


def test_non_special_roles_ignored():
    out = P._collect_special_handlers({
        "a.b.weight": ParamRole.COLWISE,
        "a.c.weight": ParamRole.SKIP,
    })
    assert out == {}


def test_special_handlers_registry():
    assert "gated_delta_tp_shard" in SPECIAL_HANDLERS
    assert callable(SPECIAL_HANDLERS["gated_delta_tp_shard"])
