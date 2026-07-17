# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""自定义模块示例 2：inner_target + inner_wrapper —— 自定义 CP wrapper。

用法:
    PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/custom_inner_wrapper.py

场景：NeMo 风格自研 attention，inner 子模块名是非标准的 `core_attention`
（自动定位失败），且想用自己的 CP wrapper（注册为命名方案）。

要点：
- `inner_target="core_attention"`：纯位置——显式指定 inner 子模块
  （自动定位失败时 fail-fast，此字段即指定入口）；
- `inner_wrapper="demo_allgather"`：纯行为——str 引用 CP_WRAPPER_REGISTRY
  里的命名方案（内置四路 "sdpa_qkv"/"sdpa_hf"/"flex_qkv"/"flex_hf"
  之外的第 5 种出口）；也可以直接给 callable；
- wrapper 契约：fn(target, cp_mesh, *, spec=None, mesh=None,
  mesh_dim_names=()) -> None，原地替换 target.forward；
- 本例 wrapper 只做 K/V all-gather（复用 flex_cp_allgather），语义与内置
  "sdpa_qkv" 相同；非 causal 简化。若需支持 validate 模式（输入为
  DTensor），wrapper 内要做 unwrap/rewrap 容错——参考内置
  `_wrap_sdpa_for_cp` 的实现。
"""

import torch
import torch.distributed as dist
import torch.nn as nn

from hyper_models.components.distributed import (
    ModuleShardingSpec,
    ShardingPlanner,
    apply_sharding_plan,
)
from hyper_models.components.distributed.cp_utils import (
    flex_cp_allgather,
    shard_batch_for_cp,
)
from hyper_models.components.distributed.sharding_applier import (
    CP_WRAPPER_REGISTRY,
)
from hyper_models.components.distributed.sharding_config import CP
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Shard


# ── 自定义 CP wrapper（注册为命名方案） ─────────────────────────────────────

def demo_cp_wrapper(target, cp_mesh, *, spec=None, mesh=None, mesh_dim_names=()):
    """在 target.forward(q, k, v) 前对 K/V 做 CP all-gather。

    通信组取 cp_mesh.get_group()（DeviceMesh 缓存组，不得 new_group）。
    """
    orig_forward = target.forward

    def cp_forward(q, k, v, **kwargs):
        k, v = flex_cp_allgather(k, v, 2, cp_mesh)   # [B, N, S_local, H] → S_full
        return orig_forward(q, k, v, **kwargs)

    target.forward = cp_forward


CP_WRAPPER_REGISTRY["demo_allgather"] = demo_cp_wrapper


# ── 模型 ────────────────────────────────────────────────────────────────────

class TinyRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states):
        var = hidden_states.pow(2).mean(-1, keepdim=True)
        return hidden_states * torch.rsqrt(var + self.eps) * self.weight


class CoreAttention(nn.Module):
    """NeMo 风格 inner attention：forward(q, k, v) 签名。"""

    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


class MyNeMoAttention(nn.Module):
    """外层 attention：q/k/v/o 投影 + 非标准命名的 inner 子模块。

    `core_attention` 不在自动定位的属性名单（inner_attention/attn/
    attention）里——必须靠 inner_target 显式指定。
    """

    def __init__(self, h, n_heads):
        super().__init__()
        self.head_dim = h // n_heads
        self.q_proj = nn.Linear(h, h, bias=False)
        self.k_proj = nn.Linear(h, h, bias=False)
        self.v_proj = nn.Linear(h, h, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)
        self.core_attention = CoreAttention()

    def forward(self, hidden_states):
        b, s, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, s, -1, self.head_dim)
        k = self.k_proj(hidden_states).view(b, s, -1, self.head_dim)
        v = self.v_proj(hidden_states).view(b, s, -1, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        o = self.core_attention(q, k, v)      # ← CP wrapper 替换此调用目标
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
        self.self_attn = MyNeMoAttention(h, n_heads)
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


def build_overrides(num_layers):
    """attention 边界 spec：CP 契约 + inner_target/inner_wrapper 声明。

    CP 不切参数（params 空声明即全复制）；in/out 沿序列维 Shard(1)。
    """
    overrides = {}
    for i in range(num_layers):
        overrides[f"model.layers.{i}.self_attn"] = ModuleShardingSpec(
            params={},
            in_src={"hidden_states": {CP: Shard(1)}},
            in_dst={"hidden_states": {CP: Shard(1)}},
            out_src={CP: Shard(1)},
            out_dst={CP: Shard(1)},
            inner_target="core_attention",        # 纯位置：非标准属性名
            inner_wrapper="demo_allgather",       # 纯行为：注册表命名方案
        )
    return overrides


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

    torch.manual_seed(0)
    model = TinyModel().eval()
    planner = ShardingPlanner(plan_overrides=build_overrides(num_layers=2))
    plan = planner.plan(model, mesh, cp_size=cp_size)
    # 解析结果回写：可观察性（INFO 日志同样可见）
    model, _ = apply_sharding_plan(model, plan, mesh)
    spec = plan.modules["model.layers.0.self_attn"]
    assert spec._resolved_inner_wrapper == "demo_allgather"

    local_ids = shard_batch_for_cp({"input_ids": input_ids}, cp_mesh)["input_ids"]
    with torch.no_grad():
        out_local = model(local_ids)

    gathered = [torch.empty_like(out_local) for _ in range(cp_size)]
    dist.all_gather(gathered, out_local, group=cp_mesh.get_group())
    out = torch.cat(gathered, dim=1)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
    print(f"[rank{rank}] custom inner_wrapper='demo_allgather' "
          f"(inner_target='core_attention') output matches single-card reference")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
