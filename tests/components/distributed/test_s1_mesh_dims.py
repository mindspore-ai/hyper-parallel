# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S1.8: _build_mesh_dim_names。"""

from hyper_models.components.distributed.sharding_planner import ShardingPlanner

P = ShardingPlanner()


class _FakeMesh:
    def __init__(self, names):
        self.mesh_dim_names = names


class TestBuildMeshDimNames:
    def test_authority_order_from_mesh(self):
        """以 mesh.mesh_dim_names 为权威顺序过滤 tp/cp/ep。"""
        mesh = _FakeMesh(("dp", "ep", "cp", "tp"))
        out = P._build_mesh_dim_names(mesh, tp_size=2, cp_size=2, ep_size=2)
        assert out == ("ep", "cp", "tp")

    def test_fallback_order(self):
        """未声明 mesh_dim_names 时按 (tp, cp, ep) 回退。"""
        mesh = _FakeMesh(None)
        out = P._build_mesh_dim_names(mesh, tp_size=2, cp_size=1, ep_size=4)
        assert out == ("tp", "ep")

    def test_size_one_axis_dropped(self):
        mesh = _FakeMesh(("tp", "cp", "ep"))
        out = P._build_mesh_dim_names(mesh, tp_size=2, cp_size=1, ep_size=1)
        assert out == ("tp",)

    def test_dp_axis_never_included(self):
        mesh = _FakeMesh(("dp_shard", "tp"))
        out = P._build_mesh_dim_names(mesh, tp_size=2, cp_size=1, ep_size=1)
        assert out == ("tp",)
