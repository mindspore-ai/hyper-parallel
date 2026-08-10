# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""CP（Context Parallel）独立示例：CP=2，HF 风格 causal attention。

region_dispatch 判断口诀：注入物含通信原语/自定义 kernel/数据依赖分支 → False（黑盒托管）；纯 aten 标准算子 → True（validate 穿透真校验）。拿不准时用 check_dispatchable(fn, example_inputs, mesh) 在开发期探明。

用法:
    PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/cp.py

要点：
- 数据管道先用 shard_batch_for_cp 把 batch 按序列维切到各 CP rank（D-05）；
- **显式注入**（改造后无启发式自动分派）：CP wrapper 通过
  plan_overrides glob merge 声明——本例模块是
  HF 风格（forward(hidden_states) 内调 F.scaled_dot_product_attention），
  选择注册表方案 "sdpa_hf"（拦截 F.sdpa 调用点）；也可写成 Target 形式
  指向 hyper_models.components.distributed.cp_wrappers.sdpa_hf_cp_wrapper；
- is_causal=True 触发 D-04：cp_size>1 时 is_causal 被替换为 offset-aware
  显式 mask（torch is_causal 在 q_len≠kv_len 时左上角对齐，对 rank>0 的
  CP chunk 是错的）；
- lm_head 的 CP 契约（D-07/R8）：每个 rank 输出本地 chunk 的 logits，
  不做 CP gather；对拍时把各 rank 输出沿序列维 all-gather 拼接即可。
"""

import torch
import torch.distributed as dist
import torch.nn as nn

from hyper_models.components.distributed import (
    ModuleShardingSpec,
    ShardingPlanner,
    apply_sharding_plan,
)
from hyper_models.components.distributed.cp_utils import shard_batch_for_cp
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh


class TinyRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states):
        var = hidden_states.pow(2).mean(-1, keepdim=True)
        return hidden_states * torch.rsqrt(var + self.eps) * self.weight


class TinyCausalAttention(nn.Module):
    """HF 风格 causal attention：投影 + SDPA 全在 forward 内。

    config._attn_implementation="sdpa" + 内部调用 F.sdpa
    → 启发式分派到内置 "sdpa_hf" wrapper（拦截原语调用点）。
    """

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


class TinyMLP(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.gate_proj = nn.Linear(h, 4 * h, bias=False)
        self.up_proj = nn.Linear(h, 4 * h, bias=False)
        self.down_proj = nn.Linear(4 * h, h, bias=False)

    def forward(self, hidden_states):
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(hidden_states))
            * self.up_proj(hidden_states))


class TinyBlock(nn.Module):
    def __init__(self, h, n_heads):
        super().__init__()
        self.input_layernorm = TinyRMSNorm(h)
        self.self_attn = TinyCausalAttention(h, n_heads)
        self.post_attention_layernorm = TinyRMSNorm(h)
        self.mlp = TinyMLP(h)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class TinyModel(nn.Module):
    def __init__(self, vocab=64, h=32, n_heads=4, n_layers=2):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, h)
        self.model.layers = nn.ModuleList(
            TinyBlock(h, n_heads) for _ in range(n_layers))
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
    cp_size = dist.get_world_size()
    mesh = init_device_mesh("cpu", (cp_size,), mesh_dim_names=("cp",))
    cp_mesh = mesh["cp"]

    # 单卡参考（全序列）
    torch.manual_seed(0)
    ref = TinyModel().eval()
    torch.manual_seed(7)
    input_ids = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        expected = ref(input_ids)

    for mode in ("production", "validate"):
        torch.manual_seed(0)
        model = TinyModel().eval()
        # 显式 CP 注入：HF 风格 attention（内部调 F.sdpa）→ "sdpa_hf" wrapper
        # （glob merge：契约继承推导，只需声明注入字段；inner_target="self"
        #  显式声明包装目标=attention 模块自身——不依赖自动定位）
        planner = ShardingPlanner(plan_overrides={
            "*.self_attn": ModuleShardingSpec(
                inner_target="self", inner_wrapper="sdpa_hf",
                # wrapper 内含 K/V all-gather 通信 → 不可 dispatch，显式 False
                region_dispatch=False),
        })
        plan = planner.plan(model, mesh, cp_size=cp_size)
        model, _ = apply_sharding_plan(
            model, plan, mesh, validate_mode=(mode == "validate"))

        # 数据管道：按 CP 切分 batch（D-05，embed 不会二次切分）
        local_ids = shard_batch_for_cp({"input_ids": input_ids}, cp_mesh)["input_ids"]
        with torch.no_grad():
            out_local = model(local_ids)          # [B, S/cp, V]（D-07）

        # 对拍：沿序列维收集各 rank 的本地 chunk 拼回全序列
        gathered = [torch.empty_like(out_local) for _ in range(cp_size)]
        dist.all_gather(gathered, out_local, group=cp_mesh.get_group())
        out = torch.cat(gathered, dim=1)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
        print(f"[rank{rank}] {mode}: CP={cp_size} causal attention output "
              f"matches single-card reference")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
