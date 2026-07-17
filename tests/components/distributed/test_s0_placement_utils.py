# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S0.3: resolve_placements / _multi_dim / _normalize_out_fields。"""

from hyper_models.components.distributed.sharding_config import (
    CP,
    EP,
    TP,
    ModuleShardingSpec,
    _multi_dim,
    _normalize_out_fields,
    resolve_placements,
)
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard


class TestResolvePlacements:
    def test_axis_order_follows_mesh_dim_names(self):
        named = {TP: Shard(0), CP: Replicate(), EP: Shard(0)}
        # mesh 轴序为 (ep, cp, tp) → 输出按该顺序重排
        out = resolve_placements(named, ("ep", "cp", "tp"))
        assert out == [Shard(0), Replicate(), Shard(0)]

    def test_missing_axis_fills_replicate(self):
        named = {TP: Shard(1)}
        out = resolve_placements(named, ("tp", "cp", "ep"))
        assert out == [Shard(1), Replicate(), Replicate()]

    def test_extra_keys_dropped(self):
        named = {TP: Shard(1), CP: Shard(1), EP: Replicate()}
        out = resolve_placements(named, ("tp",))
        assert out == [Shard(1)]

    def test_str_enum_key_interop(self):
        """plain string key 与 MeshAxisName key 互通。"""
        named = {"tp": Shard(0)}
        assert resolve_placements(named, ("tp",)) == [Shard(0)]


class TestMultiDim:
    def test_none_dims_filtered(self):
        out = _multi_dim(tp=Shard(0), cp=Replicate(), ep=None)
        assert EP not in out and out[TP] == Shard(0) and out[CP] == Replicate()

    def test_all_none_empty(self):
        assert _multi_dim() == {}


class TestNormalizeOutFields:
    def test_scalar_shorthand_wrapped(self):
        spec = ModuleShardingSpec(out_src={TP: Partial(), CP: Replicate()})
        _normalize_out_fields(spec)
        assert spec.out_src == {"output": {TP: Partial(), CP: Replicate()}}

    def test_dict_contract_untouched(self):
        spec = ModuleShardingSpec(
            out_src={"hidden_states": {TP: Shard(1)}},
            out_dst={"output": {TP: Replicate()}},
        )
        _normalize_out_fields(spec)
        assert spec.out_src == {"hidden_states": {TP: Shard(1)}}
        assert spec.out_dst == {"output": {TP: Replicate()}}

    def test_none_untouched(self):
        spec = ModuleShardingSpec(out_src=None, out_dst=None)
        _normalize_out_fields(spec)
        assert spec.out_src is None and spec.out_dst is None

    def test_idempotent(self):
        spec = ModuleShardingSpec(out_src={TP: Partial()})
        _normalize_out_fields(spec)
        _normalize_out_fields(spec)
        assert spec.out_src == {"output": {TP: Partial()}}
