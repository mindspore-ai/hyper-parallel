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
"""tests/components/distributed 公共 fixtures 与分布式测试 harness。

- tiny_llama / tiny_moe / tiny_hf_llama：手写小模型，FQN 仿 HF Llama；
  （本环境无 transformers 包，tiny_hf_llama 用手写 mock + config 对象替代
  dev_plan 中的 LlamaConfig meta 实例——FQN 布局与 HF Llama 一致。）
- make_mesh：init_device_mesh 封装；
- run_dist：fork 多进程 gloo 测试执行器（macOS 可跑）。
"""

import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh


# ────────────────────────────────────────────────────────────────────────────
# toy 模型
# ────────────────────────────────────────────────────────────────────────────

class TinyConfig:
    """模拟 HF config 的最小属性集。"""

    def __init__(self, hidden_size=16, num_attention_heads=4, num_key_value_heads=4,
                 num_hidden_layers=2, vocab_size=32, intermediate_size=32,
                 num_experts=0, moe_intermediate_size=8,
                 tie_word_embeddings=False, architectures=None,
                 model_type="tiny_llama", attn_implementation="sdpa"):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.tie_word_embeddings = tie_word_embeddings
        self.architectures = architectures or ["TinyLlamaForCausalLM"]
        self.model_type = model_type
        self._attn_implementation = attn_implementation


class TinyRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states):
        var = hidden_states.pow(2).mean(-1, keepdim=True)
        return hidden_states * torch.rsqrt(var + self.eps) * self.weight


class TinyLlamaAttention(nn.Module):
    """HF 风格：forward(hidden_states,...)，q/k/v 投影 + SDPA 全在 forward 内。"""

    def __init__(self, config, causal=False):
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = h // self.num_heads
        self.causal = causal
        self.q_proj = nn.Linear(h, h, bias=False)
        self.k_proj = nn.Linear(h, h, bias=False)
        self.v_proj = nn.Linear(h, h, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)

    def forward(self, hidden_states, position_ids=None):
        b, s, _ = hidden_states.shape
        # HF 惯例：view 用 -1 推断本地 head 数（TP 切分后本地为 heads/tp 个头）
        q = self.q_proj(hidden_states).view(b, s, -1, self.head_dim)
        k = self.k_proj(hidden_states).view(b, s, -1, self.head_dim)
        v = self.v_proj(hidden_states).view(b, s, -1, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        o = o.transpose(1, 2).reshape(b, s, -1)
        return self.o_proj(o)


class TinyLlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        h, inter = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(h, inter, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)

    def forward(self, hidden_states):
        return self.down_proj(F.silu(self.gate_proj(hidden_states))
                              * self.up_proj(hidden_states))


class TinyMoEMLP(nn.Module):
    """toy MoE：确定性 router（argmax top-1）+ 逐 expert 计算。

    forward(x_BLD)：x [B, S, H]。expert 权重为 batched 参数 [E, ...]。
    EP 语义：分片后本地只持 [E/ep] 个 expert（expert_offset 由测试在
    apply_sharding_plan 后按 ep_rank 设置，等价于 §6.4.3 的
    init_token_dispatcher 外部初始化）；ep_group 存在时 forward 末尾做
    combine all-reduce（all-to-all combine 的 toy 等价——未被路由到本地
    expert 的 token 贡献为 0，求和即合并）。
    """

    def __init__(self, config):
        super().__init__()
        h, inter, e = (config.hidden_size, config.moe_intermediate_size,
                       config.num_experts)
        self.num_experts = e
        self.gate = nn.Linear(h, e, bias=False)
        self.experts = TinyExperts(e, h, inter)
        self.ep_group = None

    def forward(self, x_BLD):
        logits = self.gate(x_BLD)                       # [B, S, E]
        idx = logits.argmax(dim=-1)                     # 确定性 top-1
        out = torch.zeros_like(x_BLD)
        w1, w2, w3 = self.experts.w1, self.experts.w2, self.experts.w3
        n_local = w1.shape[0]
        expert_offset = getattr(self.experts, "expert_offset", 0)
        for e_local in range(n_local):
            e_global = expert_offset + e_local
            mask = idx == e_global
            if not mask.any():
                continue
            x_e = x_BLD[mask]                           # [n, H]
            h_e = F.silu(x_e @ w1[e_local].t()) * (x_e @ w3[e_local].t())
            out[mask] = h_e @ w2[e_local].t()
        if self.ep_group is not None:
            # EP combine：各 rank 只算了本地 expert 的贡献，求和合并
            import torch.distributed as dist
            dist.all_reduce(out, group=self.ep_group)
        return out


class TinyExperts(nn.Module):
    def __init__(self, num_experts, hidden, inter):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(num_experts, inter, hidden) * 0.02)
        self.w2 = nn.Parameter(torch.randn(num_experts, hidden, inter) * 0.02)
        self.w3 = nn.Parameter(torch.randn(num_experts, inter, hidden) * 0.02)


class TinyDecoderLayer(nn.Module):
    def __init__(self, config, causal=False):
        super().__init__()
        self.input_layernorm = TinyRMSNorm(config.hidden_size)
        self.self_attn = TinyLlamaAttention(config, causal=causal)
        self.post_attention_layernorm = TinyRMSNorm(config.hidden_size)
        if config.num_experts > 0:
            self.mlp = TinyMoEMLP(config)
        else:
            self.mlp = TinyLlamaMLP(config)

    def forward(self, hidden_states, position_ids=None):
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), position_ids)
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states))
        return hidden_states


class TinyHFNativeMoEMLP(nn.Module):
    """HF 原生风格 MoE（D-09 直通目标）：gate + per-expert MLP ModuleList，
    forward 逐 expert 循环——无 all_to_all、无 dispatcher 钩子、无 EP 感知。

    路由语义与 ep_utils._softmax_topk_router 一致（softmax → top-2 → 归一化），
    作为 EP 计算路径的单卡参考实现。
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = 2
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            TinyLlamaMLP(config) for _ in range(config.num_experts))

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


class TinyHFNativeMoEDecoderLayer(nn.Module):
    def __init__(self, config, causal=False):
        super().__init__()
        self.input_layernorm = TinyRMSNorm(config.hidden_size)
        self.self_attn = TinyLlamaAttention(config, causal=causal)
        self.post_attention_layernorm = TinyRMSNorm(config.hidden_size)
        self.mlp = TinyHFNativeMoEMLP(config)

    def forward(self, hidden_states, position_ids=None):
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), position_ids)
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states))
        return hidden_states


class TinyLlamaModel(nn.Module):
    def __init__(self, config, causal=False):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            TinyDecoderLayer(config, causal=causal)
            for _ in range(config.num_hidden_layers))
        self.norm = TinyRMSNorm(config.hidden_size)

    def forward(self, input_ids, position_ids=None):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids)
        return self.norm(hidden_states)


class TinyLlamaForCausalLM(nn.Module):
    def __init__(self, config=None, causal=False):
        super().__init__()
        self.config = config or TinyConfig()
        self.model = TinyLlamaModel(self.config, causal=causal)
        self.lm_head = nn.Linear(self.config.hidden_size,
                                 self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, position_ids=None):
        hidden = self.model(input_ids, position_ids)
        return self.lm_head(hidden)


class TinyHFNativeMoEForCausalLM(TinyLlamaForCausalLM):
    """HF 原生 MoE 版小模型（FQN: model.layers.N.mlp.experts.{i}.gate_proj.weight）。"""

    def __init__(self, config=None, causal=False):
        super().__init__(config, causal)
        self.model.layers = nn.ModuleList(
            TinyHFNativeMoEDecoderLayer(self.config, causal=causal)
            for _ in range(self.config.num_hidden_layers))


class TinyBatchedTopKRouter(nn.Module):
    """Qwen3MoeTopKRouter 风格（HF 2025 重构后）：forward 返回
    (logits, scores [T,K], indices [T,K])。"""

    def __init__(self, config):
        super().__init__()
        self.top_k = 2
        self.weight = nn.Parameter(
            torch.randn(config.num_experts, config.hidden_size) * 0.02)

    def forward(self, hidden_states):
        h = hidden_states.shape[-1]
        logits = F.linear(hidden_states.view(-1, h), self.weight)
        probs = logits.softmax(-1)
        top_value, indices = probs.topk(self.top_k, dim=-1)
        top_value = top_value / top_value.sum(dim=-1, keepdim=True)
        return logits, top_value.to(logits.dtype), indices


class TinyBatchedExperts(nn.Module):
    """HF 2025 batched 布局（D-11）：gate_up_proj [E, 2I, H] + down_proj
    [E, H, I]——天生 stacked 3D 参数，gate/up 融合。"""

    def __init__(self, config):
        super().__init__()
        h, inter, e = (config.hidden_size, config.moe_intermediate_size,
                       config.num_experts)
        self.num_experts = e
        self.gate_up_proj = nn.Parameter(torch.randn(e, 2 * inter, h) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(e, h, inter) * 0.02)

    def forward(self, x, topk_idx, topk_w):
        out = torch.zeros_like(x)
        for e in range(self.num_experts):
            tok, slot = (topk_idx == e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            gate, up = F.linear(x[tok], self.gate_up_proj[e]).chunk(2, dim=-1)
            y = F.linear(F.silu(gate) * up, self.down_proj[e])
            out.index_add_(0, tok, y * topk_w[tok, slot].unsqueeze(-1))
        return out


class TinyBatchedMoEMLP(nn.Module):
    """HF 2025 SparseMoeBlock 风格：gate（TopKRouter 模块）+ batched experts，
    forward 无 all_to_all（D-11 直通目标）。"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = TinyBatchedTopKRouter(config)
        self.experts = TinyBatchedExperts(config)

    def forward(self, hidden_states):
        b, s, h = hidden_states.shape
        x = hidden_states.view(-1, h)
        _, scores, indices = self.gate(x)
        return self.experts(x, indices, scores).view(b, s, h)


class TinyBatchedMoEDecoderLayer(nn.Module):
    def __init__(self, config, causal=False):
        super().__init__()
        self.input_layernorm = TinyRMSNorm(config.hidden_size)
        self.self_attn = TinyLlamaAttention(config, causal=causal)
        self.post_attention_layernorm = TinyRMSNorm(config.hidden_size)
        self.mlp = TinyBatchedMoEMLP(config)

    def forward(self, hidden_states, position_ids=None):
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), position_ids)
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states))
        return hidden_states


class TinyBatchedMoEForCausalLM(TinyLlamaForCausalLM):
    """HF 2025 batched MoE 版小模型（FQN: model.layers.N.mlp.experts.gate_up_proj）。"""

    def __init__(self, config=None, causal=False):
        super().__init__(config, causal)
        self.model.layers = nn.ModuleList(
            TinyBatchedMoEDecoderLayer(self.config, causal=causal)
            for _ in range(self.config.num_hidden_layers))


# ────────────────────────────────────────────────────────────────────────────
# fixtures
# ────────────────────────────────────────────────────────────────────────────

def _meta_mesh(shape, names):
    """仅元数据的 mesh（planner/preflight 测试不需要真实进程组，但 DeviceMesh
    构造需要默认 PG 存在——与 make_mesh 的 _ensure_pg 同理）。"""
    _ensure_pg()
    n = 1
    for s in shape:
        n *= s
    return init_device_mesh("cpu", tuple(shape), mesh_dim_names=tuple(names),
                            rank_list=tuple(range(n)), init_backend=False)


def cp_sdpa_hf_injection(match="*.self_attn"):
    """显式 CP 注入（改造后无启发式自动分派）：HF 风格 attention → "sdpa_hf"。

    返回 plan_overrides 片段（{match: spec}），直接并入 ShardingPlanner 的
    plan_overrides dict。"""
    from hyper_models.components.distributed.sharding_config import (
        ModuleShardingSpec,
    )
    return {match: ModuleShardingSpec(inner_target="self",
                                        inner_wrapper="sdpa_hf",
                                        region_dispatch=False)}


def ep_hf_native_injection(match="*.mlp"):
    """显式 EP 注入：仓内默认 TP-extend-EP compute 的工厂 Target（内嵌
    default softmax top-k 路由；其他路由语义写自己的工厂）。

    返回 plan_overrides 片段（{match: spec}）。"""
    from hyper_models.components.distributed.ep_compute import (
        hf_native_ep_compute_fn,
    )
    from hyper_models.components.distributed.sharding_config import (
        ModuleShardingSpec,
    )
    from hyper_models.trainer.config import Target
    return {match: ModuleShardingSpec(
        local_compute_fn=Target(
            hf_native_ep_compute_fn,
            target_path="hyper_models.components.distributed."
                        "ep_compute.hf_native_ep_compute_fn"), region_dispatch=False)}

@pytest.fixture
def tiny_llama():
    """2 层手写 Llama 风格模型（FQN: model.layers.N.self_attn.q_proj ...）。"""
    torch.manual_seed(1234)
    return TinyLlamaForCausalLM(TinyConfig())


@pytest.fixture
def tiny_moe():
    """MoE 版小模型（gate + experts + 无 shared_experts）。"""
    torch.manual_seed(1234)
    return TinyLlamaForCausalLM(TinyConfig(num_experts=4))


@pytest.fixture
def tiny_hf_native_moe():
    """HF 原生 MoE 小模型（per-expert Linear 列表，D-09 直通目标）。"""
    torch.manual_seed(1234)
    return TinyHFNativeMoEForCausalLM(TinyConfig(num_experts=4))


@pytest.fixture
def tiny_hf_batched_moe():
    """HF 2025 batched MoE 小模型（experts.gate_up_proj [E,2I,H]，D-11 直通目标）。

    architectures=["Qwen3MoeForCausalLM"] → arch="qwen3moe" → TopKRouter
    模块 adapter。
    """
    torch.manual_seed(1234)
    return TinyBatchedMoEForCausalLM(TinyConfig(
        num_experts=4, architectures=["Qwen3MoeForCausalLM"]))


@pytest.fixture
def tiny_hf_llama():
    """HF 风格 mock：FQN 与 HF Llama 一致 + config.architectures。

    （环境无 transformers 包，用手写 mock 替代 dev_plan 的 LlamaConfig meta 实例。）
    """
    torch.manual_seed(1234)
    cfg = TinyConfig(architectures=["LlamaForCausalLM"], model_type="llama")
    return TinyLlamaForCausalLM(cfg)


@pytest.fixture
def make_mesh():
    """init_device_mesh 封装：make_mesh((2,), ("tp",))。"""
    def _make(mesh_shape, dim_names, device_type="cpu"):
        _ensure_pg()
        return init_device_mesh(device_type, tuple(mesh_shape),
                                mesh_dim_names=tuple(dim_names))
    return _make


# ────────────────────────────────────────────────────────────────────────────
# 分布式测试 harness（fork + gloo，macOS 可跑）
# ────────────────────────────────────────────────────────────────────────────

def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ensure_pg():
    """确保父进程有 world_size=1 的 gloo 进程组（单进程测试的 mesh 构造依赖）。

    早退各测试模块原本隐式依赖其他 fixture 先初始化 PG（顺序耦合），
    统一在此显式保证。
    """
    if not dist.is_initialized():
        dist.init_process_group(
            "gloo", init_method=f"tcp://127.0.0.1:{_free_port()}",
            rank=0, world_size=1,
        )


def _dist_worker(rank, world_size, port, target, args, err_queue):
    import os
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    try:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        target(rank, world_size, *args)
    except Exception:  # pylint: disable=W0718
        err_queue.put(f"[rank{rank}]\n{traceback.format_exc()}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _attach_ep(model, mesh, ep_size, num_experts=4):
    """模拟 §6.4.3 init_token_dispatcher：设置 expert_offset + ep_group。"""
    ep_mesh = mesh["ep"]
    ep_rank = ep_mesh.get_local_rank()
    n_local = num_experts // ep_size
    for layer in model.model.layers:
        layer.mlp.experts.expert_offset = ep_rank * n_local
        layer.mlp.ep_group = ep_mesh.get_group()


def run_dist(world_size, target, args=()):
    """在 world_size 个 spawn 子进程中以 gloo 运行 target(rank, world_size, *args)。

    用 spawn 而非 fork——父进程可能已被单进程测试初始化了 process group，
    fork 继承的 gloo 状态在子进程不可用（SIGABRT）。spawn 起全新解释器，
    target 必须是可 pickle 的模块级函数。
    任一 rank 抛异常则汇总重新抛出（非 rank0 断言失败不会静默丢失）。
    """
    ctx = mp.get_context("spawn")
    # 注意用 Queue 而非 SimpleQueue——SimpleQueue 在子进程退出前不保证
    # flush，错误信息可能丢失（表现为 "worker exited with code 1" 但无详情）。
    err_queue = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_dist_worker,
                    args=(rank, world_size, port, target, args, err_queue))
        for rank in range(world_size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(180)
    errors = []
    while not err_queue.empty():
        errors.append(err_queue.get())
    for p in procs:
        if p.is_alive():
            p.terminate()
        if p.exitcode not in (0, None) and not errors:
            errors.append(f"worker exited with code {p.exitcode}")
    if errors:
        raise AssertionError("distributed test failed:\n" + "\n".join(errors))
