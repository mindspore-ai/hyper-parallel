# Copyright 2025-2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""独立使用示例：components/distributed 零依赖 TP=2 分片（不依赖训练流程）。

用法:
    PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/tp.py

任意 HF 风格模型均可按此方式使用 ShardingPlanner + apply_sharding_plan，
之后接入任意训练框架（HF Trainer / PyTorch Lightning / 手写循环）。
"""

import torch
import torch.distributed as dist
import torch.nn as nn

from hyper_models.components.distributed import (
    ShardingPlanner,
    apply_sharding_plan,
)
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh


class TinyRMSNorm(nn.Module):
    """RMSNorm（weight 全复制，逐元素计算，TP/SP 下与单卡恒等）。"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states):
        var = hidden_states.pow(2).mean(-1, keepdim=True)
        return hidden_states * torch.rsqrt(var + self.eps) * self.weight


class TinyAttention(nn.Module):
    """HF 风格 attention：q/k/v 投影 + SDPA 全在 forward 内（TP 安全）。"""

    def __init__(self, h, n_heads):
        super().__init__()
        self.head_dim = h // n_heads
        self.q_proj = nn.Linear(h, h, bias=False)
        self.k_proj = nn.Linear(h, h, bias=False)
        self.v_proj = nn.Linear(h, h, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)

    def forward(self, hidden_states):
        b, s, _ = hidden_states.shape
        # view 用 -1 推断本地 head 数（TP 切分后本地为 heads/tp 个头）
        q = self.q_proj(hidden_states).view(b, s, -1, self.head_dim)
        k = self.k_proj(hidden_states).view(b, s, -1, self.head_dim)
        v = self.v_proj(hidden_states).view(b, s, -1, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
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
        self.self_attn = TinyAttention(h, n_heads)
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
    mesh = init_device_mesh("cpu", (dist.get_world_size(),), mesh_dim_names=("tp",))

    torch.manual_seed(0)
    ref = TinyModel().eval()
    x = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        expected = ref(x)

    for mode in ("production", "validate"):
        # 1. 自动推导分片策略（零模型代码改动）
        torch.manual_seed(0)
        model = TinyModel().eval()
        planner = ShardingPlanner()
        plan = planner.plan(model, mesh, tp_size=dist.get_world_size())

        # 2. 应用分片（production：零 DTensor dispatch；validate：DTensor 对拍）
        model, source_shard_info = apply_sharding_plan(
            model, plan, mesh, validate_mode=(mode == "validate"))

        # 3. 前向——输出与单卡一致
        with torch.no_grad():
            out = model(x)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
        n_grad = len(source_shard_info) if source_shard_info is not None else 0
        print(f"[rank{rank}] {mode}: TP={dist.get_world_size()} output matches "
              f"single-card reference; source_shard_info entries: {n_grad}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
