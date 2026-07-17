# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S2.9: build_tp_grad_info + tied 归一化（单进程，mock mesh）。"""

from hyper_models.components.distributed.sharding_config import (
    TP,
    ModuleShardingSpec,
    ShardingPlan,
)
from hyper_models.components.distributed.tp_grad import build_tp_grad_info
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard


class _FakeTpMesh:
    pass


def _plan():
    plan = ShardingPlan(mesh_dim_names=("tp",))
    plan.modules["model.embed_tokens"] = ModuleShardingSpec(
        params={"weight": {TP: Shard(0)}})
    plan.modules["model.layers.0.input_layernorm"] = ModuleShardingSpec(
        params={"weight": {TP: Replicate()}})
    plan.modules["lm_head"] = ModuleShardingSpec(
        params={"weight": {TP: Shard(0)}})
    return plan


def test_reads_from_plan_not_dtensor():
    mesh = _FakeTpMesh()
    info = build_tp_grad_info(_plan(), mesh)
    assert info["model.embed_tokens.weight"] == (Shard(0), mesh)
    assert info["model.layers.0.input_layernorm.weight"] == (Replicate(), mesh)
    assert info["lm_head.weight"] == (Shard(0), mesh)


def test_tied_consistent_placements_unchanged():
    plan = _plan()
    plan.tied_pairs = [("model.embed_tokens.weight", "lm_head.weight")]
    info = build_tp_grad_info(plan, _FakeTpMesh())
    assert info["model.embed_tokens.weight"][0] == Shard(0)
    assert info["lm_head.weight"][0] == Shard(0)


def test_tied_inconsistent_shard_wins():
    """tied 对 placement 不一致 → 取 Shard 优先。"""
    plan = _plan()
    plan.modules["lm_head"] = ModuleShardingSpec(
        params={"weight": {TP: Replicate()}})
    plan.tied_pairs = [("model.embed_tokens.weight", "lm_head.weight")]
    info = build_tp_grad_info(plan, _FakeTpMesh())
    assert info["model.embed_tokens.weight"][0] == Shard(0)
    assert info["lm_head.weight"][0] == Shard(0)  # 归一化为 Shard


def test_tied_pair_not_in_plan_ignored():
    plan = _plan()
    plan.tied_pairs = [("ghost.a", "ghost.b")]
    info = build_tp_grad_info(plan, _FakeTpMesh())
    assert "ghost.a" not in info


def test_explicit_tied_pairs_override():
    plan = _plan()
    info = build_tp_grad_info(
        plan, _FakeTpMesh(),
        tied_pairs=[("model.embed_tokens.weight", "lm_head.weight")])
    assert info["lm_head.weight"][0] == Shard(0)


def test_no_tp_axis_mesh_none():
    plan = ShardingPlan(mesh_dim_names=("cp",))
    plan.modules["m"] = ModuleShardingSpec(params={"w": {"cp": Replicate()}})
    info = build_tp_grad_info(plan, None)
    # 无 tp 键 → 默认 Replicate
    assert info["m.w"][0] == Replicate()
