# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""自定义模块示例 1：local_compute_fn —— 替换 local-region 骨架内的计算。

region_dispatch 判断口诀：注入物含通信原语/自定义 kernel/数据依赖分支 → False（黑盒托管）；纯 aten 标准算子 → True（validate 穿透真校验）。拿不准时用 check_dispatchable(fn, example_inputs, mesh) 在开发期探明。

用法:
    PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/custom_local_compute_fn.py

场景：自研 MoE（top-1 路由 + 自定义 batched expert 布局 w1/w3/w2），
不用仓内默认实现（ep_compute.hf_native_ep_compute_fn），注入自己的
compute fn。

要点：
- 经 plan_overrides 声明 spec——**统一改造后走 merge 语义**：命中推导
  边界时只需写注入字段，params/I-O 契约从模板推导继承（本例推导结果与
  手工声明等价，见 build_overrides 注释）；
- **声明即生效**：local_compute_fn 是解析链环 1；region_dispatch 必须
  显式声明（无默认——本例注入物含数据依赖逻辑，传 False）；
- fn 契约：fn(module, *args, **kwargs) -> Tensor，输入输出均为 local
  tensor，布局与 spec 的 in_dst/out_src 声明一致；也可写成工厂 Target
  （apply 时按签名过滤注入 module/ep_group/tp_group 上下文）；
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
    local_compute,
)
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh


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

    **入参必须与原 forward 匹配**（apply 时校验——本例
    MyCustomMoe.forward(hidden_states) 与本 fn 的 (module, hidden_states)
    对齐）。
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


@local_compute
def my_top1_moe_factory(mesh, tp_mesh, cp_mesh, ep_mesh):
    """local_compute_fn 的唯一形态：@local_compute 区域计算工厂
    （injection.py 强制纪律）。mesh 家族四参数必选、框架按名填充，
    用不用随你（本例不使用——统一接口规范）；apply 期被调用一次，
    返回的 compute_fn 无需再装饰。
    """
    return my_top1_moe_compute


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
    """只需声明注入字段：merge 语义下 params/I-O 契约全部从推导结果继承。

    planner 会把 `*.moe` 推导为 moe_mlp 边界（experts.w1/w3 colwise
    Shard(1)、w2 rowwise Shard(2)——D-08 ndim=3 平移；gate 全复制；
    SP 契约 in all-gather → out reduce-scatter），与手工逐项声明完全等价
    （统一改造前的本示例曾手写这些契约，留作对照）。
    glob key "*.moe" 一条覆盖所有层。
    """
    return {
        "*.moe": ModuleShardingSpec(
            local_compute_fn=my_top1_moe_factory,   # 链环 1，声明即生效
            # 自定义 top-1 路由含 argmax/掩码索引等数据依赖逻辑 → 显式 False
            region_dispatch=False,
        ),
    }


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
