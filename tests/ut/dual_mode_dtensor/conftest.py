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
"""Shared fixtures and distributed test harness for tests/ut/dual_mode_dtensor.

- tiny_llama / tiny_moe / tiny_hf_llama: hand-written tiny models with
  HF-Llama-style FQNs. (This environment has no transformers package, so
  tiny_hf_llama uses a hand-written mock + config object in place of the
  LlamaConfig meta instance from dev_plan — the FQN layout matches HF Llama.)
- make_mesh: wrapper around init_device_mesh;
- run_dist: fork-based multi-process gloo test runner (runs on macOS).
"""

import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh


# ────────────────────────────────────────────────────────────────────────────
# toy models
# ────────────────────────────────────────────────────────────────────────────

class TinyConfig:
    """Minimal attribute set mimicking an HF config."""

    def __init__(self, hidden_size=16, num_attention_heads=4, num_key_value_heads=4,
                 num_hidden_layers=2, vocab_size=32, intermediate_size=32,
                 num_experts=0, moe_intermediate_size=8,
                 tie_word_embeddings=False, architectures=None,
                 model_type="tiny_llama", attn_implementation="sdpa",
                 bias=False):
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
        # bias=True: q/k/v/o_proj and gate/up/down_proj all carry bias
        # (OPT/GPT-NeoX style; used by the D-22 rowwise bias tail test)
        self.bias = bias


class TinyRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states):
        var = hidden_states.pow(2).mean(-1, keepdim=True)
        return hidden_states * torch.rsqrt(var + self.eps) * self.weight


class TinyLlamaAttention(nn.Module):
    """HF style: forward(hidden_states, ...); q/k/v projections + SDPA all inside forward."""

    def __init__(self, config, causal=False):
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = h // self.num_heads
        self.causal = causal
        self.q_proj = nn.Linear(h, h, bias=config.bias)
        self.k_proj = nn.Linear(h, h, bias=config.bias)
        self.v_proj = nn.Linear(h, h, bias=config.bias)
        self.o_proj = nn.Linear(h, h, bias=config.bias)

    def forward(self, hidden_states, position_ids=None):
        """Project to q/k/v, run SDPA, and project back with o_proj."""
        b, s, _ = hidden_states.shape
        # HF convention: view uses -1 to infer the local head count (after TP
        # sharding, each rank holds heads/tp heads)
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
        self.gate_proj = nn.Linear(h, inter, bias=config.bias)
        self.up_proj = nn.Linear(h, inter, bias=config.bias)
        self.down_proj = nn.Linear(inter, h, bias=config.bias)

    def forward(self, hidden_states):
        return self.down_proj(F.silu(self.gate_proj(hidden_states))
                              * self.up_proj(hidden_states))


class TinyMoEMLP(nn.Module):
    """Toy MoE: deterministic router (argmax top-1) + per-expert computation.

    forward(x_BLD): x [B, S, H]. Expert weights are batched parameters [E, ...].
    EP semantics: after sharding, each rank holds only [E/ep] experts
    (expert_offset is set by the test per ep_rank after apply_sharding_plan,
    equivalent to the external init_token_dispatcher setup in §6.4.3); when
    ep_group exists, forward ends with a combine all-reduce (a toy equivalent
    of all-to-all combine — tokens not routed to local experts contribute 0,
    so summing merges them).
    """

    def __init__(self, config):
        super().__init__()
        h, inter, e = (config.hidden_size, config.moe_intermediate_size,
                       config.num_experts)
        self.num_experts = e
        self.gate = nn.Linear(h, e, bias=False)
        self.experts = TinyExperts(e, h, inter)
        self.ep_group = None

    # x_BLD is a contract key of the framework MoE boundary spec
    # (sharding_config moe sp_in_src/sp_in_dst), so it must not be renamed.
    def forward(self, x_BLD):  # pylint: disable=invalid-name
        """Route tokens to local experts (argmax top-1) and EP-combine."""
        logits = self.gate(x_BLD)                       # [B, S, E]
        idx = logits.argmax(dim=-1)                     # deterministic top-1
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
            # EP combine: each rank computed only its local experts'
            # contributions; sum to merge them
            dist.all_reduce(out, group=self.ep_group)
        return out


class TinyExperts(nn.Module):
    """Batched expert weights: w1/w3 [E, I, H] gate/up, w2 [E, H, I] down."""

    def __init__(self, num_experts, hidden, inter):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(num_experts, inter, hidden) * 0.02)
        self.w2 = nn.Parameter(torch.randn(num_experts, hidden, inter) * 0.02)
        self.w3 = nn.Parameter(torch.randn(num_experts, inter, hidden) * 0.02)


class TinyDecoderLayer(nn.Module):
    """Pre-norm decoder layer: self-attention block + MLP block with residuals."""

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
    """HF-native-style MoE (D-09 passthrough target): gate + per-expert MLP
    ModuleList; forward loops over experts — no all_to_all, no dispatcher
    hooks, no EP awareness.

    Routing semantics match ep_utils._softmax_topk_router (softmax → top-2 →
    normalize), serving as the single-card reference implementation of the EP
    compute path.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = 2
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            TinyLlamaMLP(config) for _ in range(config.num_experts))

    def forward(self, hidden_states):
        """Softmax top-k routing over a per-expert MLP ModuleList."""
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
    """Decoder layer whose mlp is an HF-native per-expert MoE block."""

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
    """Embedding + stacked decoder layers + final RMSNorm (HF FQN layout)."""

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
    """Causal LM wrapper: TinyLlamaModel backbone + lm_head (tieable)."""

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
    """HF-native MoE tiny model (FQN: model.layers.N.mlp.experts.{i}.gate_proj.weight)."""

    def __init__(self, config=None, causal=False):
        super().__init__(config, causal)
        self.model.layers = nn.ModuleList(
            TinyHFNativeMoEDecoderLayer(self.config, causal=causal)
            for _ in range(self.config.num_hidden_layers))


class TinyBatchedTopKRouter(nn.Module):
    """Qwen3MoeTopKRouter style (post-2025 HF refactor): forward returns
    (logits, scores [T,K], indices [T,K])."""

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
    """HF 2025 batched layout (D-11): gate_up_proj [E, 2I, H] + down_proj
    [E, H, I] — natively stacked 3D parameters with fused gate/up."""

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
    """HF 2025 SparseMoeBlock style: gate (TopKRouter module) + batched experts;
    forward has no all_to_all (D-11 passthrough target)."""

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
    """Decoder layer whose mlp is an HF 2025 batched MoE block."""

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
    """HF 2025 batched MoE tiny model (FQN: model.layers.N.mlp.experts.gate_up_proj)."""

    def __init__(self, config=None, causal=False):
        super().__init__(config, causal)
        self.model.layers = nn.ModuleList(
            TinyBatchedMoEDecoderLayer(self.config, causal=causal)
            for _ in range(self.config.num_hidden_layers))


# ────────────────────────────────────────────────────────────────────────────
# fixtures
# ────────────────────────────────────────────────────────────────────────────

def _meta_mesh(shape, names):
    """Metadata-only mesh (planner/preflight tests need no real process group,
    but DeviceMesh construction requires a default PG to exist — same as
    _ensure_pg in make_mesh)."""
    _ensure_pg()
    n = 1
    for s in shape:
        n *= s
    return init_device_mesh("cpu", tuple(shape), mesh_dim_names=tuple(names),
                            rank_list=tuple(range(n)), init_backend=False)


def cp_sdpa_hf_injection(match="*.self_attn"):
    """Explicit CP injection (no heuristic auto-dispatch after the refactor):
    HF-style attention → "sdpa_hf".

    Returns a plan_overrides fragment ({match: spec}) to merge directly into
    the ShardingPlanner plan_overrides dict."""
    from hyper_parallel.auto_models.components.distributed.sharding_config import (
        ModuleShardingSpec,
    )
    return {match: ModuleShardingSpec(inner_target="self",
                                        inner_wrapper="sdpa_hf",
                                        region_dispatch=False)}


def ep_archetype_injection(match="*.mlp"):
    """Explicit EP injection: factory Target for the repo-default TP-extend-EP
    compute (embeds the default softmax top-k routing; other routing semantics
    require writing your own factory).

    Returns a plan_overrides fragment ({match: spec})."""
    from hyper_parallel.auto_models.components.distributed.ep_compute import (
        routed_only_ep_compute_fn,
    )
    from hyper_parallel.auto_models.components.distributed.sharding_config import (
        ModuleShardingSpec,
    )
    from hyper_parallel.auto_models.trainer.config import Target
    return {match: ModuleShardingSpec(
        local_compute_fn=Target(
            routed_only_ep_compute_fn,
            target_path="hyper_parallel.auto_models.components.distributed."
                        "ep_compute.routed_only_ep_compute_fn"), region_dispatch=False)}

@pytest.fixture
def tiny_llama():
    """2-layer hand-written Llama-style model (FQN: model.layers.N.self_attn.q_proj ...)."""
    torch.manual_seed(1234)
    return TinyLlamaForCausalLM(TinyConfig())


@pytest.fixture
def tiny_moe():
    """MoE tiny model (gate + experts, no shared_experts)."""
    torch.manual_seed(1234)
    return TinyLlamaForCausalLM(TinyConfig(num_experts=4))


@pytest.fixture
def tiny_hf_native_moe():
    """HF-native MoE tiny model (per-expert Linear list, D-09 passthrough target)."""
    torch.manual_seed(1234)
    return TinyHFNativeMoEForCausalLM(TinyConfig(num_experts=4))


@pytest.fixture
def tiny_hf_batched_moe():
    """HF 2025 batched MoE tiny model (experts.gate_up_proj [E,2I,H], D-11 passthrough target).

    architectures=["Qwen3MoeForCausalLM"] → arch="qwen3moe" → TopKRouter
    module adapter.
    """
    torch.manual_seed(1234)
    return TinyBatchedMoEForCausalLM(TinyConfig(
        num_experts=4, architectures=["Qwen3MoeForCausalLM"]))


@pytest.fixture
def tiny_hf_llama():
    """HF-style mock: FQN matches HF Llama + config.architectures.

    (No transformers package in this environment; the hand-written mock
    replaces the LlamaConfig meta instance from dev_plan.)
    """
    torch.manual_seed(1234)
    cfg = TinyConfig(architectures=["LlamaForCausalLM"], model_type="llama")
    return TinyLlamaForCausalLM(cfg)


@pytest.fixture
def make_mesh():
    """init_device_mesh wrapper: make_mesh((2,), ("tp",))."""
    def _make(mesh_shape, dim_names, device_type="cpu"):
        _ensure_pg()
        return init_device_mesh(device_type, tuple(mesh_shape),
                                mesh_dim_names=tuple(dim_names))
    return _make


# ────────────────────────────────────────────────────────────────────────────
# Distributed test harness (fork + gloo, runs on macOS)
# ────────────────────────────────────────────────────────────────────────────

def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ensure_pg():
    """Ensure the parent process has a world_size=1 gloo process group
    (required for mesh construction in single-process tests).

    Early test modules implicitly relied on other fixtures initializing the PG
    first (ordering coupling); this guarantees it explicitly in one place.
    """
    if not dist.is_initialized():
        dist.init_process_group(
            "gloo", init_method=f"tcp://127.0.0.1:{_free_port()}",
            rank=0, world_size=1,
        )


def _dist_worker(rank, world_size, port, target, args, err_queue):
    """Spawned worker: init a gloo group, run target, report failures to err_queue."""
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
    """Simulate §6.4.3 init_token_dispatcher: set expert_offset + ep_group."""
    ep_mesh = mesh["ep"]
    ep_rank = ep_mesh.get_local_rank()
    n_local = num_experts // ep_size
    for layer in model.model.layers:
        layer.mlp.experts.expert_offset = ep_rank * n_local
        layer.mlp.ep_group = ep_mesh.get_group()


def run_dist(world_size, target, args=()):
    """Run target(rank, world_size, *args) with gloo in world_size spawned child
    processes.

    Uses spawn rather than fork — the parent may already have a process group
    initialized by single-process tests, and fork-inherited gloo state is
    unusable in children (SIGABRT). spawn starts fresh interpreters, so target
    must be a pickleable module-level function.
    If any rank raises, aggregate and re-raise (non-rank0 assertion failures
    are not silently lost).
    """
    ctx = mp.get_context("spawn")
    # Use Queue rather than SimpleQueue — SimpleQueue does not guarantee flush
    # before the child exits, so error details can be lost (symptom: "worker
    # exited with code 1" with no details).
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
