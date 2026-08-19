# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""TP×CP×EP 三维组合独立示例：causal attention + HF 原生 MoE。

用法:
    PYTHONPATH=. torchrun --nproc_per_node=4 examples/distributed/tp_cp_ep.py

拓扑：mesh (cp=2, tp=2)，ep_size=2（D-10 TP-extend-EP：mesh 无 "ep" 轴，
扩展 EP 组从 dense 区域 flatten 派生——本例 EP 组 == 各 CP 组内的 TP 组）。

要点（三维各自职责在一个模型内叠加；compute 注入全部显式）：
- TP=2：参数列/行切 + boundary 层 all-gather/reduce-scatter；SP 开启，
  序列维在 TP 轴上 Shard(1)；
- CP=2：数据管道 shard_batch_for_cp 先按序列维外层粗切（D-05）；attention
  内部由显式声明的 "sdpa_hf" wrapper 完成 K/V all-gather + D-04
  offset-aware causal mask（改造后无启发式自动分派）；boundary 层 CP 维
  恒 identity（R8）；
- EP=2：HF 原生 MoE（gate + per-expert ModuleList）→ D-09 堆叠 +
  {EP: Shard(0)} 分片由 planner 推导；EP compute（a2a dispatch）显式注入
  —— local_compute_fn 指向仓内默认工厂 routed_only_ep_compute_fn；
- 序列布局（05 §6.3.1/§6.3.2，教程 §6.6）：{TP: Shard(1), CP: Shard(1)}
  为 cp-major 嵌套切分——序列先按 cp 粗切、块内再按 tp 细切，
  rank (cp_i, tp_j) 持 chunk[cp_i*tp + tp_j]，长度 S/(cp*tp)。
  本例 S=16 → 每 rank 4 个连续 token：(cp0,tp0)→[0,4)、(cp0,tp1)→[4,8)、
  (cp1,tp0)→[8,12)、(cp1,tp1)→[12,16)；
- 输出契约（D-07）：lm_head 在本地 CP chunk 上出全 vocab logits
  （loss_parallel=False → TP 维 Replicate），对拍时沿 CP 组 all-gather
  拼回全序列即可。
"""

import torch
import torch.distributed as dist
import torch.nn as nn

from hyper_models.components.distributed import (
    ModuleShardingSpec,
    ShardingPlanner,
    apply_sharding_plan,
    routed_only_ep_compute_fn,
)
from hyper_models.components.distributed.cp_utils import shard_batch_for_cp
from hyper_models.trainer.config import Target
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh

S = 16          # 全局序列长：需整除 2*cp（数据管道 pad 约束）与 cp*tp（布局约束）
VOCAB = 64


class TinyRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states):
        var = hidden_states.pow(2).mean(-1, keepdim=True)
        return hidden_states * torch.rsqrt(var + self.eps) * self.weight


class TinyCausalAttention(nn.Module):
    """HF 风格 causal attention（同 cp.py）：内部调 F.sdpa + is_causal
    → 启发式分派到内置 "sdpa_hf" CP wrapper（D-04 offset mask）。"""

    def __init__(self, h, n_heads):
        super().__init__()
        self.config = type("Cfg", (), {"_attn_implementation": "sdpa"})()
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
        o = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True)     # ← CP wrapper 在此拦截（D-04）
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
    """HF 原生风格 MoE（同 ep.py）：gate + per-expert ModuleList，forward
    逐 expert 循环——无 all_to_all；D-09 直通 + D-10 TP-extend-EP 注入。"""

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
        self.self_attn = TinyCausalAttention(h, n_heads)
        self.post_attention_layernorm = TinyRMSNorm(h)
        self.mlp = TinyNativeMoE(h, inter, num_experts)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class TinyMoeModel(nn.Module):
    def __init__(self, vocab=VOCAB, h=32, n_heads=4, n_layers=2,
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
    world = dist.get_world_size()   # 4：mesh (cp=2, tp=2)，ep=2
    cp_size, tp_size, ep_size = 2, 2, 2
    assert world == cp_size * tp_size

    # cp 在 tp 之前 → 序列嵌套切分为 cp-major（05 §6.3.1）
    mesh = init_device_mesh("cpu", (cp_size, tp_size),
                            mesh_dim_names=("cp", "tp"))
    cp_mesh = mesh["cp"]
    cp_rank, tp_rank = cp_mesh.get_local_rank(), mesh["tp"].get_local_rank()

    # 单卡参考（全序列）
    torch.manual_seed(0)
    ref = TinyMoeModel().eval()
    torch.manual_seed(7)
    input_ids = torch.randint(0, VOCAB, (2, S))
    with torch.no_grad():
        expected = ref(input_ids)

    for mode in ("production", "validate"):
        torch.manual_seed(0)
        model = TinyMoeModel().eval()
        # 显式注入：CP wrapper（"sdpa_hf" + inner_target="self"）+
        # 仓内默认 EP compute（工厂 Target；expert mesh 由框架在 apply 时
        # 统一派生并与参数分片共享，经 ep_mesh 上下文传入工厂）
        planner = ShardingPlanner(plan_overrides={
            "*.self_attn": ModuleShardingSpec(
                inner_target="self", inner_wrapper="sdpa_hf",
                # CP wrapper 内含 K/V all-gather 通信 → 不可 dispatch
                region_dispatch=False),
            "*.mlp": ModuleShardingSpec(
                # EP compute 内含 all-to-all 通信 → 不可 dispatch
                region_dispatch=False,
                local_compute_fn=Target(
                    routed_only_ep_compute_fn,
                    target_path="hyper_models.components.distributed."
                                "ep_compute.routed_only_ep_compute_fn"),
            ),
        })
        plan = planner.plan(
            model, mesh, tp_size=tp_size, cp_size=cp_size, ep_size=ep_size)

        # plan 内省：三维各自的结构元数据
        moe_spec = plan.modules["model.layers.0.mlp"]
        assert moe_spec._ep_size == ep_size   # TP-extend-EP 元数据（D-10）
        assert moe_spec._ep_stack             # per-expert → Phase A 堆叠（D-09）
        attn_spec = plan.modules["model.layers.0.self_attn"]
        assert attn_spec._needs_cp_attn       # CP 元数据标记（模板识别）
        assert attn_spec.inner_wrapper == "sdpa_hf"   # 显式注入的 wrapper
        norm_spec = plan.modules["model.layers.0.input_layernorm"]
        io = norm_spec.in_dst["hidden_states"]   # norm：全 identity，零通信

        model, _ = apply_sharding_plan(
            model, plan, mesh, validate_mode=(mode == "validate"))

        # 数据管道：CP 外层粗切（D-05）；TP 内层细分由 embed 出口
        # reduce-scatter 完成 → 本 rank 持 cp-major 布局的连续 token 段
        local_ids = shard_batch_for_cp({"input_ids": input_ids}, cp_mesh)["input_ids"]
        with torch.no_grad():
            out_local = model(local_ids)       # [B, S/cp, V]（D-07 本地 chunk）

        # cp-major 布局自检：本 rank 的全局 token 偏移 = cp_rank*S/cp
        #（TP 维在 lm_head 入口已 all-gather 复原地层 chunk，输出 TP 维 Replicate）
        chunk = S // (cp_size * tp_size)
        lo = (cp_rank * tp_size + tp_rank) * chunk
        assert local_ids.shape[1] == S // cp_size
        assert out_local.shape == (2, S // cp_size, VOCAB)
        assert io and out_local.shape[1] == chunk * tp_size

        # 对拍：沿 CP 组收集各 rank 的本地 chunk 拼回全序列
        gathered = [torch.empty_like(out_local) for _ in range(cp_size)]
        dist.all_gather(gathered, out_local, group=cp_mesh.get_group())
        out = torch.cat(gathered, dim=1)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
        print(f"[rank{rank}] {mode}: TP={tp_size}×CP={cp_size}×EP={ep_size} "
              f"MoE output matches single-card reference "
              f"(hidden chunk: token [{lo}, {lo + chunk}) of {S})")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
