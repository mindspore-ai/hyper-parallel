# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""EP（Expert Parallel）独立示例：HF 原生 MoE + TP-extend-EP（D-09/D-10）。

region_dispatch 判断口诀：注入物含通信原语/自定义 kernel/数据依赖分支 → False（黑盒托管）；纯 aten 标准算子 → True（validate 穿透真校验）。拿不准时用 check_dispatchable(fn, example_inputs, mesh) 在开发期探明。

用法:
    PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/ep.py

要点（显式注入）：
- 模型是 HF 原生风格 MoE：mlp.gate（router）+ mlp.experts（per-expert
  ModuleList，forward 无 all_to_all）。planner 自动识别布局；
- ep_size>1 即激活 TP-extend-EP：Phase A 先把 per-expert 参数堆叠为
  [E, ...]（D-09），再在派生 expert mesh 上按 {EP: Shard(0)} 分片
  （每 rank 持 num_experts/ep_size 个完整 expert）——参数分片是 planner
  的声明式推导，无需配置；
- **compute 必须显式注入**（改造后无自动注入）：local_compute_fn 指向
  仓内默认实现的工厂 Target ——
  hyper_models.components.distributed.ep_compute.hf_native_ep_compute_fn
  （router → dispatch all-to-all → 本地 expert → combine all-to-all，
  与 Megatron MoEAlltoAllTokenDispatcher 同构）；不注入会在 apply 时
  fail-fast（_preflight_compute_injection）；
- 约束（plan 时 fail-fast）：ep_size ≤ 且整除 dense 区域（dp×cp×tp）；
  num_experts % ep_size == 0。
"""

import torch
import torch.distributed as dist
import torch.nn as nn

from hyper_models.components.distributed import (
    ModuleShardingSpec,
    ShardingPlanner,
    apply_sharding_plan,
    hf_native_ep_compute_fn,
)
from hyper_models.trainer.config import Target
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


class TinyExpertMLP(nn.Module):
    def __init__(self, h, inter):
        super().__init__()
        self.gate_proj = nn.Linear(h, inter, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)

    def forward(self, x):
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyNativeMoE(nn.Module):
    """HF 原生风格 MoE（D-09 直通目标）：gate + per-expert ModuleList，
    forward 逐 expert 循环——无 all_to_all、无 dispatcher 钩子。

    路由语义与内置 default adapter（_softmax_topk_router）一致：
    softmax → top-2 → 归一化。本 forward 即单卡参考实现。
    """

    def __init__(self, h, inter, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(h, num_experts, bias=False)
        self.experts = nn.ModuleList(
            TinyExpertMLP(h, inter) for _ in range(num_experts))

    def forward(self, hidden_states):
        b, s, h = hidden_states.shape
        x = hidden_states.view(-1, h)
        logits = self.gate(hidden_states).view(-1, self.num_experts)
        weights = logits.softmax(-1)
        topk_w, topk_idx = weights.topk(self.top_k, dim=-1)
        topk_w = topk_w / topk_w.sum(-1, keepdim=True)
        out = torch.zeros_like(x)
        for e_idx, expert in enumerate(self.experts):
            tok, slot = (topk_idx == e_idx).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            out.index_add_(0, tok, expert(x[tok]) * topk_w[tok, slot].unsqueeze(-1))
        return out.view(b, s, h)


class TinyBlock(nn.Module):
    def __init__(self, h, n_heads, inter, num_experts):
        super().__init__()
        self.input_layernorm = TinyRMSNorm(h)
        self.self_attn = TinyAttention(h, n_heads)
        self.post_attention_layernorm = TinyRMSNorm(h)
        self.mlp = TinyNativeMoE(h, inter, num_experts)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class TinyMoeModel(nn.Module):
    def __init__(self, vocab=64, h=32, n_heads=4, n_layers=2,
                 inter=16, num_experts=4):
        super().__init__()
        self.config = type("Cfg", (), {
            "architectures": ["TinyMoeForCausalLM"],
            "num_experts": num_experts,
            "num_experts_per_tok": 2,
        })()
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


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()   # 2：mesh (dp=1, tp=2)，ep=2（EP 组=TP 组）

    mesh = init_device_mesh("cpu", (1, world), mesh_dim_names=("dp", "tp"))

    # 单卡参考
    torch.manual_seed(0)
    ref = TinyMoeModel().eval()
    torch.manual_seed(7)
    input_ids = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        expected = ref(input_ids)

    for mode in ("production", "validate"):
        torch.manual_seed(0)
        model = TinyMoeModel().eval()
        # ep_size=world：扩展 EP 组大小（D-10）。
        # 显式注入仓内默认 EP compute（工厂 Target；路由内嵌 default
        # softmax top-k——路由是注入函数的一部分，框架不参与选择；其他
        # 路由语义写自己的工厂）。expert mesh 由框架在 apply 时统一派生
        # （与专家参数分片共享同一对象，派生有 INFO 日志），经 ep_mesh
        # 上下文传入工厂——用户只管使用
        planner = ShardingPlanner(plan_overrides={
            "*.mlp": ModuleShardingSpec(
                # EP compute 内含 all-to-all 通信 → 不可 dispatch，显式 False
                region_dispatch=False,
                local_compute_fn=Target(
                    hf_native_ep_compute_fn,
                    target_path="hyper_models.components.distributed."
                                "ep_compute.hf_native_ep_compute_fn"),
            ),
        })
        plan = planner.plan(model, mesh, tp_size=world, ep_size=world)
        spec = plan.modules["model.layers.0.mlp"]
        assert spec._ep_size == world        # TP-extend-EP 元数据已生成
        assert spec._ep_stack                # per-expert 布局 → Phase A 堆叠（D-09）
        model, _ = apply_sharding_plan(
            model, plan, mesh, validate_mode=(mode == "validate"))

        with torch.no_grad():
            out = model(input_ids)           # 无 CP：全量 logits 逐 rank 对拍
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
        print(f"[rank{rank}] {mode}: TP={world}×EP={world} MoE output "
              f"matches single-card reference")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
