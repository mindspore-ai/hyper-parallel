# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S6.1: D-10 TP-extend-EP（05 §6.4.8）单进程用例。

TP-extend-EP 语义：ep_size 即扩展 EP 组大小（a2a 通信域，由 TP 组向相邻
dp/cp rank 扩展）；expert 权重仅在 expert 维 Shard(0)——扩展 EP 组每个
rank 持 num_experts/ep_size 个完整 expert，无 TP/ETP 第二轴切分
（MindSpeed TP-extend-EP / Megatron etp=1 + ep 跨 TP 同构）。
覆盖：planner 契约（identity 边界、{EP: S0} 无 TP 键、_ep_size == ep_size）、
校验（ep_size 不超过且整除 dense 区域、num_experts 整除 ep_size）、
派生 expert mesh (edp, ep) 的 rank 映射（EP 组先跨完 TP 组再跨 dp）。
"""

import pytest

from hyper_models.components.distributed.sharding_applier import _expert_mesh_layout
from hyper_models.components.distributed.sharding_config import CP, EP, TP
from hyper_models.components.distributed.sharding_planner import ShardingPlanner
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard


def _meta_mesh(shape, names):
    """仅元数据的 mesh（planner 测试不需要真实进程组，但 DeviceMesh
    构造需要默认 PG 存在——与 make_mesh 的 _ensure_pg 同理）。"""
    from tests.components.distributed.conftest import _ensure_pg
    _ensure_pg()
    n = 1
    for s in shape:
        n *= s
    return init_device_mesh("cpu", tuple(shape), mesh_dim_names=tuple(names),
                            rank_list=tuple(range(n)), init_backend=False)


def test_planner_ep_extend_contract(tiny_hf_native_moe):
    """mesh (dp=4, tp=2)，ep=4 → 扩展 EP 组 {0,1,2,3}/{4,5,6,7}（用户示例）。"""
    mesh = _meta_mesh((4, 2), ("dp", "tp"))
    plan = ShardingPlanner().plan(tiny_hf_native_moe, mesh, tp_size=2, ep_size=4)
    spec = plan.modules["model.layers.0.mlp"]

    assert spec._ep_size == 4           # ep_size 即扩展 EP 组大小
    assert spec._ep_stack               # 堆叠元数据不变

    # expert 参数：仅 {EP: Shard(0)}（expert 维切分），无 TP 键、无第二轴
    for proj in ("gate_proj", "up_proj", "down_proj"):
        p = spec.params[f"experts.{proj}"]
        assert p[EP] == Shard(0)
        assert TP not in p and p[CP] == Replicate()
        assert len(p) == 2              # 只有 CP(Replicate) + EP 两个键

    # router 全复制（本地 chunk 计算）
    assert spec.params["gate.weight"][TP] == Replicate()

    # 边界契约 identity（SP-in）：in_dst/out_src/out_dst 均 TP Shard(1)
    assert spec.in_dst["x_BLD"][TP] == Shard(1)
    assert spec.out_src["output"][TP] == Shard(1)
    assert spec.out_dst["output"][TP] == Shard(1)
    # in_src 与上游契约不变（链式校验通过）
    assert spec.in_src["x_BLD"][TP] == Shard(1)


def test_planner_ep1_no_extend(tiny_hf_native_moe, make_mesh):
    """ep=1 → 无 TP-extend-EP，per-expert 条目保留（TP-only 语义正确）。"""
    mesh = make_mesh((1,), ("tp",))
    plan = ShardingPlanner().plan(tiny_hf_native_moe, mesh, tp_size=2)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec._ep_size == 0
    assert spec._ep_stack == {}
    assert "experts.0.gate_proj.weight" in spec.params
    assert spec.params["experts.0.gate_proj.weight"][TP] == Shard(0)


def test_planner_ep_extend_invalid(tiny_hf_native_moe):
    """ep_size 超过 dense 区域 / 不整除 dense / num_experts 不整除 → ValueError。"""
    # mesh (1,2) D=2：ep=4 > D → 报错
    mesh = _meta_mesh((1, 2), ("dp", "tp"))
    with pytest.raises(ValueError, match="dense"):
        ShardingPlanner().plan(tiny_hf_native_moe, mesh, tp_size=2, ep_size=4)
    # mesh (4,2) D=8：ep=3 不整除 D → 报错
    mesh8 = _meta_mesh((4, 2), ("dp", "tp"))
    with pytest.raises(ValueError, match="dense"):
        ShardingPlanner().plan(tiny_hf_native_moe, mesh8, tp_size=2, ep_size=3)
    # mesh (4,2) D=8：ep=8 合法但 num_experts=4 不整除 ep=8 → 报错
    with pytest.raises(ValueError, match="num_experts"):
        ShardingPlanner().plan(tiny_hf_native_moe, mesh8, tp_size=2, ep_size=8)


def test_planner_batched_contract(tiny_hf_batched_moe):
    """D-11 batched 布局（experts.gate_up_proj [E,2I,H]）：无需堆叠，直接标
    {EP: Shard(0)}；arch=qwen3moe → TopKRouter 模块 adapter。"""
    mesh = _meta_mesh((4, 2), ("dp", "tp"))
    plan = ShardingPlanner().plan(tiny_hf_batched_moe, mesh, tp_size=2, ep_size=4)
    spec = plan.modules["model.layers.0.mlp"]

    assert spec._ep_size == 4
    assert spec._ep_stack == {}          # batched 天生 stacked，无需堆叠
    assert spec._moe_router == "qwen3moe"

    # expert 参数：仅 {EP: Shard(0)}（expert 维切分），无 TP 键、无第二轴
    for proj in ("gate_up_proj", "down_proj"):
        p = spec.params[f"experts.{proj}"]
        assert p[EP] == Shard(0)
        assert TP not in p and p[CP] == Replicate()
        assert len(p) == 2

    # router（TopKRouter.weight）全复制；边界 identity
    assert spec.params["gate.weight"][TP] == Replicate()
    assert spec.in_dst["x_BLD"][TP] == Shard(1)
    assert spec.out_src["output"][TP] == Shard(1)
    assert spec.out_dst["output"][TP] == Shard(1)


def test_planner_batched_ep1_no_mark(tiny_hf_batched_moe, make_mesh):
    """batched 布局 ep=1 → 不标记，TP-only 语义（ndim=3 → TP Shard(1)，D-08）。"""
    mesh = make_mesh((1,), ("tp",))
    plan = ShardingPlanner().plan(tiny_hf_batched_moe, mesh, tp_size=2)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec._ep_size == 0
    assert spec.params["experts.gate_up_proj"][TP] == Shard(1)


def test_expert_mesh_layout_mapping():
    """派生 expert mesh：dense 区域 flatten → (edp, ep)，EP 组 = flatten 连续
    ep_size 个 rank（先跨完 TP 组再向相邻 dp rank 扩展）。"""
    mesh = _meta_mesh((4, 2), ("dp", "tp"))   # rank = d*2 + t

    # ep=4（用户示例）：EP 组 {0,1,2,3} / {4,5,6,7}——跨 2 个 TP 组 × 2 个 dp
    shape, names, rank_list = _expert_mesh_layout(mesh, ("dp", "tp"), 4)
    assert shape == (2, 4)
    assert names == ("edp", "ep")
    assert rank_list == (0, 1, 2, 3, 4, 5, 6, 7)

    # ep=2：EP 组 {0,1}/{2,3}/{4,5}/{6,7}——即 TP 组
    shape, names, _ = _expert_mesh_layout(mesh, ("dp", "tp"), 2)
    assert shape == (4, 2)
    assert names == ("edp", "ep")

    # 不整除报错
    with pytest.raises(ValueError, match="must divide"):
        _expert_mesh_layout(mesh, ("dp", "tp"), 3)
