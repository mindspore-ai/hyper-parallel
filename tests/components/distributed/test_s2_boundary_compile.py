# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S2.4: PrecompiledBoundary 编译期（identity 跳过 / 多输出映射 / None 分支）。"""

from hyper_models.components.distributed.precompiled_boundary import (
    PrecompiledBoundary,
)
from hyper_models.components.distributed.sharding_config import (
    CP,
    TP,
    ModuleShardingSpec,
)
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard


class _FakeMesh:
    mesh_dim_names = ("tp", "cp")


def _attention_spec():
    """attention TP×CP 契约：TP 维通信、CP 维 identity。"""
    return ModuleShardingSpec(
        in_src={"hidden_states": {TP: Shard(1), CP: Shard(1)}},
        in_dst={"hidden_states": {TP: Replicate(), CP: Shard(1)}},
        out_src={"output": {TP: Partial(), CP: Shard(1)}},
        out_dst={"output": {TP: Shard(1), CP: Shard(1)}},
    )


class TestCompileInputPlan:
    def test_attention_cp_dim_identity(self):
        """attention CP 维 in_dst=Shard(1) identity 断言：in_plan 只有 TP all-gather。"""
        b = PrecompiledBoundary(_attention_spec(), _FakeMesh(), ("tp", "cp"))
        assert len(b.in_plan) == 1
        op = b.in_plan[0]
        assert op.collective_type == "all_gather"
        # CP 维 src==dst==Shard(1)（identity 体现在 placement 对相等，而非 CP 维通信）
        assert op.src_placements[1] == Shard(1)
        assert op.dst_placements[1] == Shard(1)

    def test_identity_input_still_compiled_as_passthrough(self):
        """in_src==in_dst → identity op（直通）。"""
        spec = ModuleShardingSpec(
            in_src={"x": {TP: Shard(1)}}, in_dst={"x": {TP: Shard(1)}})
        b = PrecompiledBoundary(spec, _FakeMesh(), ("tp",))
        assert len(b.in_plan) == 1
        assert b.in_plan[0].collective_type == "identity"


class TestCompileOutputPlan:
    def test_attention_out_cp_identity_skipped(self):
        """out_plan：CP 维 identity 不产生额外 op，仅 TP reduce-scatter。"""
        b = PrecompiledBoundary(_attention_spec(), _FakeMesh(), ("tp", "cp"))
        assert len(b.out_plan) == 1
        assert b.out_plan[0].collective_type == "reduce_scatter"

    def test_identity_output_plan_empty(self):
        spec = ModuleShardingSpec(
            out_src={"output": {TP: Shard(1)}},
            out_dst={"output": {TP: Shard(1)}},
        )
        b = PrecompiledBoundary(spec, _FakeMesh(), ("tp",))
        assert b.out_plan == []

    def test_multi_output_arg_index_from_out_names(self):
        spec = ModuleShardingSpec(
            out_src={"hidden_states": {TP: Partial()}, "present_kv": {TP: Shard(1)}},
            out_dst={"hidden_states": {TP: Shard(1)}, "present_kv": {TP: Replicate()}},
            out_names=["hidden_states", "present_kv"],
        )
        b = PrecompiledBoundary(spec, _FakeMesh(), ("tp",))
        idx = {op.arg_name: op.arg_index for op in b.out_plan}
        assert idx == {"hidden_states": 0, "present_kv": 1}

    def test_multi_output_arg_index_default_key_order(self):
        spec = ModuleShardingSpec(
            out_src={"a": {TP: Partial()}, "b": {TP: Shard(1)}},
            out_dst={"a": {TP: Shard(1)}, "b": {TP: Replicate()}},
        )
        b = PrecompiledBoundary(spec, _FakeMesh(), ("tp",))
        idx = {op.arg_name: op.arg_index for op in b.out_plan}
        assert idx == {"a": 0, "b": 1}

    def test_out_src_none_no_out_plan(self):
        spec = ModuleShardingSpec(out_src=None, out_dst={"output": {TP: Shard(1)}})
        b = PrecompiledBoundary(spec, _FakeMesh(), ("tp",))
        assert b.out_plan == []

    def test_out_dst_none_no_out_plan(self):
        spec = ModuleShardingSpec(out_src={"output": {TP: Partial()}}, out_dst=None)
        b = PrecompiledBoundary(spec, _FakeMesh(), ("tp",))
        assert b.out_plan == []
