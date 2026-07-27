# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""S2.8（2 进程）: D-17 头数改写端到端 —— 非 TP 容错 attention（reshape 显式
使用全局 num_heads）在三种执行位置下的数值与属性语义。

- production：全量改写（属性 = heads/tp），local tensor 前向数值对齐单卡；
- validate（boundary）：**不改写**——DTensor dispatch 在全局逻辑形状上
  自动推导 reshape，无需任何手动改写即数值对齐（显式 num_heads 写法在
  validate 下天然正确）；
- validate（local-region，plan_overrides 声明 use_local_map）：区域内
  两模式都是 local tensor → validate 同样改写，数值对齐。
"""

import torch
import torch.nn.functional as F

from hyper_models.components.distributed import ShardingPlanner, apply_sharding_plan
from hyper_models.components.distributed.sharding_config import (
    TP,
    ModuleShardingSpec,
)
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from tests.components.distributed.conftest import (
    TinyConfig,
    TinyLlamaAttention,
    TinyLlamaForCausalLM,
    run_dist,
)

_HEADS = TinyConfig().num_attention_heads   # 4


class ExplicitHeadsAttention(TinyLlamaAttention):
    """非 TP 容错写法：reshape 显式使用 self.num_heads（HF 生态常见写法）。"""

    def forward(self, hidden_states, position_ids=None):
        b, s, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(b, s, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(b, s, self.num_heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        o = o.transpose(1, 2).reshape(b, s, -1)
        return self.o_proj(o)


def _build():
    torch.manual_seed(1234)
    model = TinyLlamaForCausalLM(TinyConfig()).eval()
    for layer in model.model.layers:
        attn = ExplicitHeadsAttention(model.config)
        attn.load_state_dict(layer.self_attn.state_dict())
        layer.self_attn = attn
    return model


def _reference():
    ref_model = _build()
    torch.manual_seed(7)
    x = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        ref = ref_model(x)
    return x, ref


def _local_region_attn_spec():
    """契约与 planner 派生的 attention spec 一致（SP 默认开启），仅多声明
    use_local_map 使模块走 local-region。"""
    return ModuleShardingSpec(
        params={
            "q_proj.weight": {TP: Shard(0)},
            "k_proj.weight": {TP: Shard(0)},
            "v_proj.weight": {TP: Shard(0)},
            "o_proj.weight": {TP: Shard(1)},
        },
        in_src={"hidden_states": {TP: Shard(1)}},
        in_dst={"hidden_states": {TP: Replicate()}},
        out_src={TP: Partial()},
        out_dst={TP: Shard(1)},
        use_local_map=True,
    )


def _worker_production(rank, world_size):
    """production：头数改写为本地值，显式 num_heads reshape 数值对齐。"""
    x, ref = _reference()

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    model = _build()
    plan = ShardingPlanner().plan(model, mesh, tp_size=world_size)
    model, _ = apply_sharding_plan(model, plan, mesh, validate_mode=False)
    for layer in model.model.layers:
        assert layer.self_attn.num_heads == _HEADS // world_size
        assert layer.self_attn.config.num_attention_heads == _HEADS  # config 不改写
    with torch.no_grad():
        out = model(x)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def _worker_validate_boundary(rank, world_size):
    """validate（boundary）：不改写属性，DTensor 全局逻辑形状自动推导。"""
    x, ref = _reference()

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    model = _build()
    plan = ShardingPlanner().plan(model, mesh, tp_size=world_size)
    model, _ = apply_sharding_plan(model, plan, mesh, validate_mode=True)
    for layer in model.model.layers:
        # 关键断言：属性保持全局值，未做任何手动/自动改写
        assert layer.self_attn.num_heads == _HEADS
        assert not hasattr(layer.self_attn, "_hp_full_head_counts")
    with torch.no_grad():
        out = model(x)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def _worker_validate_local_region(rank, world_size):
    """validate（local-region）：区域内 local tensor → 改写且数值对齐。"""
    x, ref = _reference()

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
    model = _build()
    planner = ShardingPlanner(plan_overrides={
        "model.layers.0.self_attn": _local_region_attn_spec(),
        "model.layers.1.self_attn": _local_region_attn_spec(),
    })
    plan = planner.plan(model, mesh, tp_size=world_size)
    model, _ = apply_sharding_plan(model, plan, mesh, validate_mode=True)
    for layer in model.model.layers:
        assert layer.self_attn.num_heads == _HEADS // world_size
    with torch.no_grad():
        out = model(x)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_head_count_production_tp2():
    run_dist(2, _worker_production)


def test_head_count_validate_boundary_tp2():
    run_dist(2, _worker_validate_boundary)


def test_head_count_validate_local_region_tp2():
    run_dist(2, _worker_validate_local_region)
