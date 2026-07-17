# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S5.8: D-09/D-10 HF 原生 MoE EP 直通（05 §6.4.7/§6.4.8）单进程用例。

覆盖：planner 标记（堆叠元数据 + TP-extend-EP 契约）、ep=1 与 pre-stacked
不标记、_stack_moe_experts 堆叠 handler、router adapter 数值、
_swiglu_weights 两套命名。
"""

import pytest
import torch

from hyper_models.components.distributed.ep_utils import (
    MOE_ROUTER_ADAPTERS,
    _softmax_topk_router,
    _swiglu_weights,
)
from hyper_models.components.distributed.sharding.apply import (
    _StackedExperts,
    _stack_moe_experts,
)
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


def test_planner_marks_hf_native_moe(tiny_hf_native_moe):
    """per-expert 参数 + ep>1 → stacked 元数据 + TP-extend-EP 契约（D-09a/D-10）。"""
    mesh = _meta_mesh((4, 2), ("dp", "tp"))
    plan = ShardingPlanner().plan(tiny_hf_native_moe, mesh, tp_size=2, ep_size=2)

    spec = plan.modules["model.layers.0.mlp"]
    # 数字段守卫生效：边界聚合在 mlp，无 per-expert 边界
    assert not any("experts.0" in fqn for fqn in plan.modules)

    # stacked 条目（D-10 TP-extend-EP：仅 {EP: S0} expert 维切分，无 TP 键、
    # 无第二轴）
    for proj in ("gate_proj", "up_proj", "down_proj"):
        p = spec.params[f"experts.{proj}"]
        assert p[EP] == Shard(0)
        assert TP not in p and p[CP] == Replicate()

    # per-expert 条目已移除；router 全复制
    assert not any("experts.0" in k for k in spec.params)
    assert spec.params["gate.weight"][TP] == Replicate()

    # _ep_stack 元数据：stacked 名 → 按 expert idx 排序的源路径
    assert set(spec._ep_stack) == {
        "experts.gate_proj", "experts.up_proj", "experts.down_proj"}
    assert spec._ep_stack["experts.gate_proj"] == [
        f"experts.{i}.gate_proj.weight" for i in range(4)]
    assert spec._moe_router == "default"
    # TP-extend-EP：_ep_size = ep_size，边界 identity
    assert spec._ep_size == 2
    assert spec.in_dst["x_BLD"][TP] == Shard(1)


def test_planner_no_mark_without_ep(tiny_hf_native_moe, make_mesh):
    """ep=1 → 不堆叠，per-expert 条目保留（TP-only 语义正确）。"""
    mesh = make_mesh((1,), ("tp",))
    plan = ShardingPlanner().plan(tiny_hf_native_moe, mesh, tp_size=2)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec._ep_stack == {}
    assert "experts.0.gate_proj.weight" in spec.params
    assert spec.params["experts.0.gate_proj.weight"][TP] == Shard(0)


def test_planner_no_mark_for_pre_stacked(tiny_moe):
    """pre-stacked 布局（experts.w1 3D）→ 不命中 per-expert 模式，原路径。"""
    mesh = _meta_mesh((4, 2), ("dp", "tp"))
    plan = ShardingPlanner().plan(tiny_moe, mesh, tp_size=2, ep_size=2)
    spec = plan.modules["model.layers.0.mlp"]
    assert spec._ep_stack == {}


def test_stack_moe_experts(tiny_hf_native_moe):
    """堆叠 handler：stacked 值 == 原 per-expert 值，原参数移除。"""
    mesh = _meta_mesh((4, 2), ("dp", "tp"))
    mlp = tiny_hf_native_moe.model.layers[0].mlp
    orig = {
        proj: torch.stack([getattr(mlp.experts[i], proj).weight.data
                           for i in range(4)])
        for proj in ("gate_proj", "up_proj", "down_proj")
    }
    plan = ShardingPlanner().plan(tiny_hf_native_moe, mesh, tp_size=2, ep_size=2)
    ep_stack = plan.modules["model.layers.0.mlp"]._ep_stack

    _stack_moe_experts(mlp, ep_stack)

    assert isinstance(mlp.experts, _StackedExperts)
    for proj in ("gate_proj", "up_proj", "down_proj"):
        stacked = getattr(mlp.experts, proj)
        assert stacked.shape == orig[proj].shape
        torch.testing.assert_close(stacked, orig[proj])
    # 原 per-expert 参数已移除
    assert not any("experts.0" in n
                   for n, _ in mlp.named_parameters())


def test_stack_moe_experts_rejects_bias(tiny_hf_native_moe):
    """带 bias 的 expert → NotImplementedError（v1 限制）。"""
    import torch.nn as nn
    mlp = tiny_hf_native_moe.model.layers[0].mlp
    mlp.experts[0].gate_proj.bias = nn.Parameter(torch.zeros(32))
    ep_stack = {"experts.gate_proj": [f"experts.{i}.gate_proj.weight" for i in range(4)]}
    with pytest.raises(NotImplementedError, match="bias"):
        _stack_moe_experts(mlp, ep_stack)


def test_softmax_topk_router(tiny_hf_native_moe):
    """default adapter 与玩具模型 forward 的路由语义一致。"""
    mlp = tiny_hf_native_moe.model.layers[0].mlp
    torch.manual_seed(5)
    hidden = torch.randn(2, 3, 16)
    topk_idx, topk_w = _softmax_topk_router(mlp, hidden)
    logits = mlp.gate(hidden).view(-1, 4)
    w = logits.softmax(-1)
    ref_w, ref_idx = w.topk(2, dim=-1)
    ref_w = ref_w / ref_w.sum(-1, keepdim=True)
    assert torch.equal(topk_idx, ref_idx)
    torch.testing.assert_close(topk_w, ref_w)
    assert MOE_ROUTER_ADAPTERS["default"] is _softmax_topk_router


def test_swiglu_weights_two_naming_families():
    """gate/up/down_proj 与 w1/w2/w3 两套命名均可解析；缺矩阵报错。"""
    import torch.nn as nn

    class Holder(nn.Module):
        pass

    h1 = Holder()
    h1.gate_proj = nn.Parameter(torch.randn(4, 8, 16))
    h1.up_proj = nn.Parameter(torch.randn(4, 8, 16))
    h1.down_proj = nn.Parameter(torch.randn(4, 16, 8))
    g, u, d = _swiglu_weights(h1)
    assert g is h1.gate_proj and u is h1.up_proj and d is h1.down_proj

    h2 = Holder()
    h2.w1 = nn.Parameter(torch.randn(4, 8, 16))
    h2.w3 = nn.Parameter(torch.randn(4, 8, 16))
    h2.w2 = nn.Parameter(torch.randn(4, 16, 8))
    g, u, d = _swiglu_weights(h2)
    assert g is h2.w1 and u is h2.w3 and d is h2.w2

    with pytest.raises(NotImplementedError, match="SwiGLU"):
        _swiglu_weights(Holder())


def test_swiglu_weights_fused_layout():
    """D-11 fused 布局：gate_up_proj + down_proj → (fused, None, down)。"""
    import torch.nn as nn

    class Holder(nn.Module):
        pass

    h = Holder()
    h.gate_up_proj = nn.Parameter(torch.randn(4, 16, 8))
    h.down_proj = nn.Parameter(torch.randn(4, 8, 8))
    g, u, d = _swiglu_weights(h)
    assert g is h.gate_up_proj and u is None and d is h.down_proj

    # automodel 命名（gate_and_up_projs/down_projs）同构
    h2 = Holder()
    h2.gate_and_up_projs = nn.Parameter(torch.randn(4, 16, 8))
    h2.down_projs = nn.Parameter(torch.randn(4, 8, 8))
    g, u, d = _swiglu_weights(h2)
    assert g is h2.gate_and_up_projs and u is None and d is h2.down_projs


def test_topk_router_module_adapter(tiny_hf_batched_moe):
    """qwen3moe adapter：直接取 TopKRouter 模块返回的 (indices, scores)。"""
    from hyper_models.components.distributed.ep_utils import _topk_router_module
    mlp = tiny_hf_batched_moe.model.layers[0].mlp
    torch.manual_seed(5)
    hidden = torch.randn(2, 3, 16)
    idx, w = _topk_router_module(mlp, hidden)
    _, ref_w, ref_idx = mlp.gate(hidden)
    assert torch.equal(idx, ref_idx)
    torch.testing.assert_close(w, ref_w)
    from hyper_models.components.distributed.ep_utils import MOE_ROUTER_ADAPTERS
    assert MOE_ROUTER_ADAPTERS["qwen3moe"] is _topk_router_module


def test_sigmoid_group_router_adapter():
    """deepseekv3/glm4moe adapter：sigmoid + correction bias + norm + scaling
    （n_group=1 跳过 group 过滤），与手算参考一致。"""
    import torch.nn as nn
    import torch.nn.functional as F
    from hyper_models.components.distributed.ep_utils import _sigmoid_group_router

    class Gate(nn.Module):
        def __init__(self, e, h):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(e, h) * 0.02)
            self.register_buffer("e_score_correction_bias", torch.randn(e) * 0.01)

        def forward(self, x):
            return F.linear(x.view(-1, x.shape[-1]).float(), self.weight.float())

    class MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = Gate(4, 16)
            self.top_k = 2
            self.n_group = 1
            self.norm_topk_prob = True
            self.routed_scaling_factor = 2.5

    torch.manual_seed(5)
    moe = MoE()
    hidden = torch.randn(2, 3, 16)
    idx, w = _sigmoid_group_router(moe, hidden)

    logits = moe.gate(hidden)
    scores = logits.sigmoid()
    choice = scores + moe.gate.e_score_correction_bias
    ref_idx = choice.topk(2, dim=-1, sorted=False)[1]
    ref_w = scores.gather(1, ref_idx)
    ref_w = ref_w / (ref_w.sum(-1, keepdim=True) + 1e-20) * 2.5
    assert torch.equal(idx, ref_idx)
    torch.testing.assert_close(w, ref_w.to(w.dtype))
