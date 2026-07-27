# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S1.7: SP on/off × loss_parallel on/off 四组合 I/O 契约。"""

import pytest

from hyper_models.components.distributed.sharding_config import (
    CP,
    TP,
)
from hyper_models.components.distributed.sharding_planner import ShardingPlanner
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard


def _plan(tiny_llama, make_mesh, sp, lp):
    mesh = make_mesh((1,), ("tp",))
    return ShardingPlanner().plan(
        tiny_llama, mesh, tp_size=2, sequence_parallel=sp, loss_parallel=lp)


@pytest.mark.parametrize("sp,lp", [
    (True, False), (True, True), (False, False), (False, True),
])
def test_embed_contract(tiny_llama, make_mesh, sp, lp):
    spec = _plan(tiny_llama, make_mesh, sp, lp).modules["model.embed_tokens"]
    assert spec.in_src["input"][TP] == Replicate()
    assert spec.out_src["output"][TP] == Partial()
    want_out = Shard(1) if sp else Replicate()
    assert spec.out_dst["output"][TP] == want_out


@pytest.mark.parametrize("sp,lp", [
    (True, False), (True, True), (False, False), (False, True),
])
def test_attention_contract(tiny_llama, make_mesh, sp, lp):
    spec = _plan(tiny_llama, make_mesh, sp, lp).modules["model.layers.0.self_attn"]
    want_in = Shard(1) if sp else Replicate()
    assert spec.in_src["hidden_states"][TP] == want_in
    assert spec.in_dst["hidden_states"][TP] == Replicate()
    assert spec.out_src["output"][TP] == Partial()
    assert spec.out_dst["output"][TP] == want_in


@pytest.mark.parametrize("sp,lp,want_out_dst", [
    (True, False, Replicate()), (True, True, Shard(-1)),
    (False, False, Replicate()), (False, True, Shard(-1)),
])
def test_lm_head_out_dst_loss_parallel(tiny_llama, make_mesh, sp, lp, want_out_dst):
    spec = _plan(tiny_llama, make_mesh, sp, lp).modules["lm_head"]
    assert spec.out_src["output"][TP] == Shard(-1)
    assert spec.out_dst["output"][TP] == want_out_dst


def test_sp_cp_dim(tiny_llama, make_mesh):
    """SP 开启时 embed out_dst / norm in_src 的 CP 维为 Shard(1)。"""
    spec = _plan(tiny_llama, make_mesh, True, False).modules["model.norm"]
    assert spec.in_src["hidden_states"][CP] == Shard(1)
