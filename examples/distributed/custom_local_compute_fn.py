# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""自定义模块示例 1：local_compute_fn —— 替换 local-region 骨架内的计算。

用法:
    PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/custom_local_compute_fn.py

场景：自研 MoE（top-1 路由 + 自定义 batched expert 布局 w1/w3/w2），
不想用内置 `_hf_native_ep_compute`，注入自己的 compute fn。

要点：
- 经 plan_overrides 手写 spec（contracts + params + local_compute_fn）；
- **声明即生效**：local_compute_fn 是解析链环 1，无需也不应再设
  use_local_map（门控派生，声明互不嵌套）；
- fn 契约：fn(module, *args, **kwargs) -> Tensor，输入输出均为 local
  tensor，布局与 spec 的 in_dst/out_src 声明一致；
- 骨架四步只换 compute：in_src→in_dst 边界通信（本例 TP all-gather）、
  out_src 重包装、out_src→out_dst 边界通信（reduce-scatter）全部保留；
- 区域内可自由使用显式 process group 通信（本例无需——纯本地 top-1）。
"""

import torch
import torch.distributed as dist
import torch.nn as nn

from hyper_models.components.distributed import (
    ModuleShardingSpec,
    ShardingPlanner,
    apply_sharding_plan,
)
from hyper_models.components.distributed.sharding_config import TP
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import (
    Partial,
    Replicate,
    Shard,
)


class TinyRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states):
        var = hidden_states.pow(2).mean(-1, keepdim=True)
        return hidden_states * torch.rsqrt(var + self.eps) * self.weight


class TinyAttention(nn.Module):
    def __init__(self, h, n_heads):
        super().__init__()
        self.head_dim = h // n_heads
        self.q_proj = nn.Linear(h, h, bias=False)
        self.k_proj = nn.Linear(h, h, bias=False)
        self.v_proj = nn.Linear(h, h, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)

    def forward(self, hidden_states):
        b, s, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, s, -1, self.head_dim)
        k = self.k_proj(hidden_states).view(b, s, -1, self.head_dim)
        v = self.v_proj(hidden_states).view(b, s, -1, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return self.o_proj(o.transpose(1, 2).reshape(b, s, -1))


class MyCustomExperts(nn.Module):
    """自定义 batched expert 布局：w1/w3 [E, I, H]（gate/up 分离）、w2 [E, H, I]。

    注：gate/up 必须分离存放——融合 w13 [E, 2I, H] 沿 2I 维 Shard 时每个
    rank 拿到的是连续行块（rank0 全 gate、rank1 全 up），无法直接 chunk
    出 gate/up（这正是模板 FUSED_GATE_UP 需要 SpecialHandler 调整的原因）。
    """

    def __init__(self, h, inter, num_experts):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(num_experts, inter, h) * 0.02)
        self.w3 = nn.Parameter(torch.randn(num_experts, inter, h) * 0.02)
        self.w2 = nn.Parameter(torch.randn(num_experts, h, inter) * 0.02)


class MyCustomMoe(nn.Module):
    """自研 top-1 MoE：自定义 expert 布局 + 自定义路由（argmax）。

    forward 即单卡参考实现；sharded 版本由下面的 my_top1_moe_compute
    注入（同一份数学，作用在 local 分片上）。
    """

    def __init__(self, h, inter, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(h, num_experts, bias=False)
        self.experts = MyCustomExperts(h, inter, num_experts)

    def forward(self, hidden_states):
        return my_top1_moe_compute(self, hidden_states)


def my_top1_moe_compute(module, hidden_states):
    """注入骨架的自定义 compute（local tensor 世界）。

    in_dst 声明 Replicate → 输入是全量 hidden；experts.w1/w3/w2 是 TP 本地
    分片（w1/w3 [E, I/tp, H]、w2 [E, H, I/tp]）→ 输出是 contraction 维的
    部分和，与 out_src 声明 {TP: Partial()} 一致，骨架出口负责求和。
    """
    b, s, h = hidden_states.shape
    x = hidden_states.view(-1, h)
    idx = module.gate(x).argmax(-1)                # 自定义 top-1 路由
    out = torch.zeros_like(x)
    for e in range(module.num_experts):
        mask = idx == e
        if not mask.any():
            continue
        xe = x[mask]
        h_e = torch.nn.functional.silu(xe @ module.experts.w1[e].T) \
            * (xe @ module.experts.w3[e].T)
        out[mask] = h_e @ module.experts.w2[e].T        # 部分和 [n, H]
    return out.view(b, s, h)


class TinyBlock(nn.Module):
    def __init__(self, h, n_heads, inter, num_experts):
        super().__init__()
        self.input_layernorm = TinyRMSNorm(h)
        self.self_attn = TinyAttention(h, n_heads)
        self.post_attention_layernorm = TinyRMSNorm(h)
        self.moe = MyCustomMoe(h, inter, num_experts)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.moe(self.post_attention_layernorm(x))


class TinyModel(nn.Module):
    def __init__(self, vocab=64, h=32, n_heads=4, n_layers=2,
                 inter=16, num_experts=4):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, h)
        self.model.layers = nn.ModuleList(
            TinyBlock(h, n_heads, inter, num_experts) for _ in range(n_layers))
        self.model.norm = TinyRMSNorm(h)
        self.lm_head = nn.Linear(h, vocab, bias=False)

    def forward(self, input_ids):
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        return self.lm_head(self.model.norm(h))


def build_overrides(num_layers):
    """对每个自研 MoE 层声明 spec：contracts + params + local_compute_fn。

    契约与 moe_mlp 模板同构（SP 模式）：in all-gather → region → out
    reduce-scatter；expert 3D 权重的 TP 切分按 ndim 平移（D-08）。
    """
    overrides = {}
    for i in range(num_layers):
        overrides[f"model.layers.{i}.moe"] = ModuleShardingSpec(
            params={
                "gate.weight": {TP: Replicate()},      # router 全复制
                "experts.w1": {TP: Shard(1)},          # [E, I, H] → 切 I 维
                "experts.w3": {TP: Shard(1)},          # [E, I, H] → 切 I 维
                "experts.w2": {TP: Shard(2)},          # [E, H, I] → 切 contraction 维
            },
            in_src={"hidden_states": {TP: Shard(1)}},
            in_dst={"hidden_states": {TP: Replicate()}},   # 进 region 前 all-gather
            out_src={TP: Partial()},                        # 本地部分和
            out_dst={TP: Shard(1)},                         # 出口 reduce-scatter
            local_compute_fn=my_top1_moe_compute,           # 链环 1，声明即生效
        )
    return overrides


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("tp",))

    # 单卡参考
    torch.manual_seed(0)
    ref = TinyModel().eval()
    torch.manual_seed(7)
    input_ids = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        expected = ref(input_ids)

    for mode in ("production", "validate"):
        torch.manual_seed(0)
        model = TinyModel().eval()
        planner = ShardingPlanner(plan_overrides=build_overrides(num_layers=2))
        plan = planner.plan(model, mesh, tp_size=world)
        model, _ = apply_sharding_plan(
            model, plan, mesh, validate_mode=(mode == "validate"))

        with torch.no_grad():
            out = model(input_ids)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
        print(f"[rank{rank}] {mode}: custom local_compute_fn output "
              f"matches single-card reference")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
