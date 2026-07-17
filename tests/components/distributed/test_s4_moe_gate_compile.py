# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S4.3: moe_gate 模板 EP redistribute（out_dst {EP: Shard(0)}）编译。"""

from hyper_models.components.distributed.precompiled_boundary import (
    PrecompiledBoundary,
)
from hyper_models.components.distributed.sharding_config import (
    EP,
    TEMPLATES,
    ModuleShardingSpec,
    _normalize_out_fields,
)
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard


class _FakeMesh:
    mesh_dim_names = ("tp", "ep")


def _moe_gate_spec():
    t = TEMPLATES["moe_gate"]
    spec = ModuleShardingSpec(
        in_src=t.sp_in_src,
        in_dst=t.sp_in_dst,
        out_src=t.sp_out_src,
        out_dst=t.sp_out_dst,
    )
    return _normalize_out_fields(spec)


def test_moe_gate_out_plan_has_ep_redistribute():
    spec = _moe_gate_spec()
    b = PrecompiledBoundary(spec, _FakeMesh(), ("tp", "ep"))
    assert len(b.out_plan) == 1
    op = b.out_plan[0]
    ep_idx = ("tp", "ep").index("ep")
    # out_dst EP 维 Replicate → Shard(0)
    assert op.src_placements[ep_idx] == Replicate()
    assert op.dst_placements[ep_idx] == Shard(0)
    assert op.collective_type == "redistribute"


def test_moe_gate_in_plan_tp_allgather():
    spec = _moe_gate_spec()
    b = PrecompiledBoundary(spec, _FakeMesh(), ("tp", "ep"))
    assert len(b.in_plan) == 1
    assert b.in_plan[0].collective_type == "all_gather"
