# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""DeepSeek-V4Pro text model.

This is a trainer-friendly PyTorch port of the DeepSeek-V4Pro architecture
shape used by the HF snapshot in ``hfhub/models/deepseek-ai/DeepSeek-V4Pro``.
The implementation keeps the core ingredients that matter for training
integration:

- q-lora / o-lora attention projections
- grouped-query attention with a single KV head
- partial RoPE on the trailing rotary channels
- token-choice MoE with shared experts
- trainer-compatible ``loss`` / ``logits`` outputs

The first HyperParallel integration targets FSDP2 only. TP / CP / EP are
intentionally left out of the default path so the model can be validated in a
small smoke configuration first.
"""
# pylint: disable=forbidden-backend-import,missing-public-type-hints,missing-public-docstring,not-callable
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.models.modules.rmsnorm import RMSNorm
from hyper_parallel.models.modules.rope import RotaryEmbedding, apply_rotary_pos_emb


def _masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return a finite mean when a sequence shard has no valid labels."""
    flat_logits = logits.view(-1, logits.size(-1))
    flat_labels = labels.view(-1)
    loss_sum = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=-100,
        reduction="sum",
    )
    valid_tokens = (flat_labels != -100).sum().clamp_min(1)
    return loss_sum / valid_tokens


def _build_sqrtsoftplus(scores: torch.Tensor) -> torch.Tensor:
    """Stable ``sqrt(softplus(x))`` router score."""
    return torch.log1p(torch.exp(-scores.abs())) + torch.relu(scores)


def _build_hash_table(
    vocab_size: int,
    num_experts: int,
    top_k: int,
    *,
    seed: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build a deterministic token-id → expert-id routing table."""
    if top_k > num_experts:
        raise ValueError(f"top_k ({top_k}) must be <= num_experts ({num_experts})")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    tid2eid = torch.empty((vocab_size, top_k), dtype=torch.long, device=device)
    for start in range(0, vocab_size, 8192):
        end = min(start + 8192, vocab_size)
        tid2eid[start:end] = torch.rand(
            (end - start, num_experts),
            generator=generator,
            device=device,
        ).topk(top_k, dim=-1).indices
    return tid2eid


@dataclass
class DeepSeekV4ProConfig:
    """DeepSeek-V4Pro text configuration.

    The field list intentionally keeps the HF snapshot names and the smaller
    smoke-test overrides in one place. Most torchtitan-only / HF-only fields are
    accepted and ignored by the simplified training path so that ``config.json``
    can be loaded directly and then shrunk through ``model.config_overrides``.
    """

    vocab_size: int = 129280
    hidden_size: int = 7168
    intermediate_size: Optional[int] = None
    num_hidden_layers: int = 61
    num_attention_heads: int = 128
    num_key_value_heads: int = 1
    head_dim: int = 512
    q_lora_rank: int = 1536
    o_lora_rank: int = 1024
    o_groups: int = 16
    qk_rope_head_dim: int = 64
    rope_head_dim: Optional[int] = None
    sliding_window: int = 128
    window_size: Optional[int] = None
    n_routed_experts: int = 384
    num_experts: Optional[int] = None
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    moe_intermediate_size: int = 3072
    scoring_func: str = "sqrtsoftplus"
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    score_before_experts: bool = False
    num_hash_layers: int = 3
    load_balance_coeff: float = 1e-3
    debug_force_load_balance: bool = False
    swiglu_limit: float = 10.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 1_048_576
    rope_theta: float = 10_000.0
    compress_rope_theta: float = 160_000.0
    rope_scaling_factor: float = 16.0
    original_seq_len: int = 65_536
    beta_fast: int = 32
    beta_slow: int = 1
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 1_024
    compress_ratios: tuple[int, ...] = field(default_factory=tuple)
    num_nextn_predict_layers: int = 1
    num_mtp_modules: int = 0
    mtp_layer_compress_ratio: int = 1
    use_smla: bool = False
    expert_dtype: str = "fp4"
    dtype: str = "bfloat16"
    torch_dtype: str = "bfloat16"
    use_cache: bool = False
    model_type: str = "deepseek_v4"
    architectures: Optional[list[str]] = None

    def __post_init__(self):
        if self.rope_head_dim is None:
            self.rope_head_dim = self.qk_rope_head_dim
        if self.window_size is None:
            self.window_size = self.sliding_window
        if self.num_experts is None:
            self.num_experts = self.n_routed_experts
        self.n_routed_experts = int(self.num_experts)
        self.num_key_value_heads = int(self.num_key_value_heads)
        self.compress_ratios = tuple(int(v) for v in self.compress_ratios)
        self._validate()

    @property
    def layers(self):
        """Return a layer-range view for framework helpers."""
        return range(self.num_hidden_layers + self.num_mtp_modules)

    def _validate(self) -> None:
        """Fail fast on obviously invalid geometry."""
        for field_name in (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "q_lora_rank",
            "o_lora_rank",
            "o_groups",
            "qk_rope_head_dim",
            "rope_head_dim",
            "sliding_window",
            "window_size",
            "n_routed_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "max_position_embeddings",
        ):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.num_attention_heads % self.o_groups != 0:
            raise ValueError("num_attention_heads must be divisible by o_groups")
        if self.rope_head_dim > self.head_dim:
            raise ValueError("rope_head_dim must be <= head_dim")
        if self.num_experts_per_tok > self.n_routed_experts:
            raise ValueError("num_experts_per_tok must be <= n_routed_experts")
        if self.num_mtp_modules < 0:
            raise ValueError("num_mtp_modules must be non-negative")


class DeepSeekSwiGLUMLP(nn.Module):
    """Single SwiGLU block with DeepSeek-style parameter names."""

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DeepSeekPackedExperts(nn.Module):
    """Packed expert bank with one parameter slice per expert."""

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.w1 = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.w3 = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for param in (self.w1, self.w2, self.w3):
            nn.init.trunc_normal_(param, mean=0.0, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass over packed experts."""
        num_tokens, hidden_dim = hidden_states.shape
        top_k = top_k_index.size(-1)
        expert_outputs = torch.zeros(
            num_tokens,
            top_k,
            hidden_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = int(expert_idx[0])
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate = F.linear(current_state, self.w1[expert_idx])
            up = F.linear(current_state, self.w3[expert_idx])
            current_hidden_states = F.silu(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.w2[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            expert_outputs[token_idx, top_k_pos] = current_hidden_states.to(expert_outputs.dtype)
        return expert_outputs.sum(dim=1)


class DeepSeekTokenChoiceTopKRouter(nn.Module):
    """DeepSeek-V4Pro router with sqrtsoftplus and optional hash routing."""

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        layer_id: int,
        args: DeepSeekV4ProConfig,
        vocab_size: int,
    ):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.route_norm = args.norm_topk_prob
        self.route_scale = args.routed_scaling_factor
        self.score_func = args.scoring_func
        self._debug_force_load_balance = args.debug_force_load_balance
        self.hash = layer_id < args.num_hash_layers
        self.vocab_size = vocab_size
        if self.hash:
            self.register_buffer(
                "tid2eid",
                _build_hash_table(
                    vocab_size,
                    num_experts,
                    top_k,
                    seed=layer_id + 17,
                ),
                persistent=True,
            )

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        expert_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = self.gate(x)
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores.to(torch.float32))
        elif self.score_func == "softmax":
            scores = F.softmax(scores.to(torch.float32), dim=1)
        elif self.score_func == "sqrtsoftplus":
            scores = _build_sqrtsoftplus(scores.to(torch.float32)).sqrt()
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        if self.hash:
            selected_experts_indices = self.tid2eid[input_ids.flatten()]
        else:
            scores_for_choice = scores if expert_bias is None else scores + expert_bias
            selected_experts_indices = scores_for_choice.topk(self.top_k, dim=-1)[1]

        top_scores = scores.gather(dim=1, index=selected_experts_indices)
        if self._debug_force_load_balance:
            # Balance forcing is intentionally omitted in the first port.
            pass
        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale
        num_tokens_per_expert = F.one_hot(
            selected_experts_indices.view(-1),
            num_classes=self.num_experts,
        ).sum(dim=0).to(scores.dtype)
        return top_scores, selected_experts_indices, num_tokens_per_expert


class DeepSeekV4ProMoE(nn.Module):
    """DeepSeek-V4Pro token-choice MoE."""

    def __init__(self, config: DeepSeekV4ProConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.experts = DeepSeekPackedExperts(
            config.n_routed_experts,
            config.hidden_size,
            config.moe_intermediate_size,
        )
        self.router = DeepSeekTokenChoiceTopKRouter(
            dim=config.hidden_size,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            layer_id=layer_id,
            args=config,
            vocab_size=config.vocab_size,
        )
        n_shared = config.n_shared_experts
        if n_shared > 0:
            shared_intermediate = config.moe_intermediate_size * n_shared
            self.shared_experts = DeepSeekSwiGLUMLP(
                config.hidden_size,
                shared_intermediate,
                bias=False,
            )
        else:
            self.shared_experts = None
        self.score_before_experts = config.score_before_experts
        self.load_balance_coeff = config.load_balance_coeff
        if self.load_balance_coeff is not None:
            if self.load_balance_coeff <= 0.0:
                raise ValueError("load_balance_coeff must be greater than 0.0")
            self.register_buffer(
                "expert_bias",
                torch.zeros(config.n_routed_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(config.n_routed_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        bs, slen, dim = x.shape
        x_flat = x.view(-1, dim)
        input_ids_flat = input_ids.flatten()
        bias = getattr(self, "expert_bias", None)
        top_scores, selected_experts_indices, num_tokens_per_expert = self.router(
            x_flat,
            input_ids_flat,
            bias,
        )

        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)

        if self.score_before_experts:
            raise NotImplementedError(
                "score_before_experts=True is not wired in the first DeepSeek-V4Pro port"
            )

        routed_output = self.experts(x_flat, selected_experts_indices, top_scores)
        shared_output = self.shared_experts(x_flat) if self.shared_experts is not None else None
        if shared_output is None:
            output = routed_output
        else:
            output = routed_output + shared_output
        return output.view(bs, slen, dim)


class DeepSeekV4ProAttention(nn.Module):
    """DeepSeek-V4Pro attention with q-lora/o-lora projections."""

    def __init__(self, config: DeepSeekV4ProConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.attn_output_gate = False
        self.o_groups = config.o_groups
        self.window_size = config.window_size
        self.rope_head_dim = config.rope_head_dim

        self.q_lora = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
        self.q_norm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=config.attention_bias)
        self.kv_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = nn.Linear(
            self.num_heads * self.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.attn_sink = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        self.rotary_emb = RotaryEmbedding(
            dim=self.rope_head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )
        nn.init.trunc_normal_(self.attn_sink, mean=0.0, std=0.02)

    def _apply_attention_mask(
        self,
        attn_weights: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask is None:
            return attn_weights
        if attention_mask.ndim == 4:
            if attention_mask.shape[1] == 1 and attn_weights.shape[1] > 1:
                attention_mask = attention_mask.expand(
                    attention_mask.shape[0],
                    attn_weights.shape[1],
                    -1,
                    -1,
                )
            return attn_weights + attention_mask
        if attention_mask.ndim != 2:
            return attn_weights
        if torch.all(attention_mask == 1):
            return attn_weights
        padding_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
        return attn_weights.masked_fill(~padding_mask, torch.finfo(attn_weights.dtype).min)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del input_ids
        bsz, seq_len, _ = hidden_states.shape
        if position_ids is None:
            position_ids = torch.arange(
                seq_len,
                device=hidden_states.device,
                dtype=torch.long,
            ).view(1, -1).expand(bsz, -1)
        rope_position_ids = position_ids[0] if position_ids.ndim > 1 else position_ids

        q = self.q_norm(self.q_lora(hidden_states))
        q = self.q_proj(q).view(bsz, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)

        kv = self.kv_norm(self.wkv(hidden_states)).view(bsz, seq_len, 1, self.head_dim).transpose(1, 2)
        k = kv
        v = kv

        q_nope, q_rope = torch.split(q, [self.head_dim - self.rope_head_dim, self.rope_head_dim], dim=-1)
        k_nope, k_rope = torch.split(k, [self.head_dim - self.rope_head_dim, self.rope_head_dim], dim=-1)
        cos, sin = self.rotary_emb(hidden_states, rope_position_ids)
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) * (self.head_dim**-0.5)
        causal = torch.triu(
            torch.full(
                (seq_len, seq_len),
                float("-inf"),
                device=hidden_states.device,
                dtype=attn_weights.dtype,
            ),
            diagonal=1,
        )
        if self.window_size is not None and self.window_size > 0:
            positions = torch.arange(seq_len, device=hidden_states.device)
            local_mask = positions[None, :] < (positions[:, None] - self.window_size + 1)
            causal = causal.masked_fill(local_mask, float("-inf"))
        attn_weights = attn_weights + causal
        attn_weights = self._apply_attention_mask(attn_weights, attention_mask)
        sink = self.attn_sink.view(1, -1, 1, 1).expand(bsz, -1, seq_len, -1)
        combined_logits = torch.cat([attn_weights, sink], dim=-1)
        combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        probs = F.softmax(combined_logits.float(), dim=-1, dtype=combined_logits.dtype)
        attn_probs = probs[..., :-1]
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads, self.head_dim)

        o_nope, o_rope = torch.split(
            attn_output,
            [self.head_dim - self.rope_head_dim, self.rope_head_dim],
            dim=-1,
        )

        def _rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        cos, sin = self.rotary_emb(hidden_states, rope_position_ids)
        if cos.ndim == 2:
            cos = cos.unsqueeze(0).unsqueeze(2).to(o_rope.dtype)
            sin = sin.unsqueeze(0).unsqueeze(2).to(o_rope.dtype)
        elif cos.ndim == 3:
            cos = cos.unsqueeze(2).to(o_rope.dtype)
            sin = sin.unsqueeze(2).to(o_rope.dtype)
        # The output projection reverses the rotary basis before grouping heads.
        o_rope = (o_rope * cos) - (_rotate_half(o_rope) * sin)
        attn_output = torch.cat([o_nope, o_rope], dim=-1)
        attn_output = attn_output.view(bsz, seq_len, self.o_groups, -1)
        wo_a = self.wo_a.weight.view(self.o_groups, self.wo_a.out_features // self.o_groups, -1)
        attn_output = torch.einsum("bsgd,grd->bsgr", attn_output, wo_a)
        return self.wo_b(attn_output.reshape(bsz, seq_len, -1))


class DeepSeekHcSplitSinkhorn(nn.Module):
    """MHC split + Sinkhorn normalization, ported from torchtitan-npu."""

    def forward(
        self,
        mixes: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
        comb = comb.unflatten(-1, (hc_mult, hc_mult))

        pre = torch.sigmoid(pre * hc_scale[0] + hc_base[:hc_mult].view(1, 1, -1)) + eps
        post = 2 * torch.sigmoid(post * hc_scale[1] + hc_base[hc_mult: 2 * hc_mult].view(1, 1, -1))
        comb = comb * hc_scale[2] + hc_base[2 * hc_mult:].view(hc_mult, hc_mult).view(1, 1, hc_mult, hc_mult)

        comb = comb.softmax(-1) + eps
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
        for _ in range(max(sinkhorn_iters - 1, 0)):
            comb = comb / (comb.sum(-1, keepdim=True) + eps)
            comb = comb / (comb.sum(-2, keepdim=True) + eps)
        return pre, post, comb


class DeepSeekHcPre(nn.Module):
    """Pre attention/FFN MHC reduction: ``[B, S, hc, D] -> [B, S, D]``."""

    def __init__(self, config: DeepSeekV4ProConfig):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        self.split_sinkhorn = DeepSeekHcSplitSinkhorn()

    def forward(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape, input_dtype = x.size(), x.dtype
        x_flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, hc_fn) * rsqrt
        pre, post, comb = self.split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        y = torch.sum(pre.unsqueeze(-1) * x.float().view(shape), dim=2)
        return y.to(input_dtype), post, comb


class DeepSeekHcPost(nn.Module):
    """Post attention/FFN MHC expansion: ``[B, S, D] -> [B, S, hc, D]``."""

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        y = post.unsqueeze(-1) * x.unsqueeze(-2)
        y = y + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.to(x.dtype)


class DeepSeekHcHead(nn.Module):
    """Final MHC aggregation head."""

    def __init__(self, config: DeepSeekV4ProConfig):
        super().__init__()
        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        hc_dim = config.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(torch.empty(config.hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(config.hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.hc_head_fn, mean=0.0, std=0.02)
        nn.init.zeros_(self.hc_head_base)
        nn.init.ones_(self.hc_head_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape, input_dtype = x.size(), x.dtype
        x_flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, self.hc_head_fn) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.float().view(shape), dim=2)
        return y.to(input_dtype)


class DeepSeekV4ProDecoder(nn.Module):
    """One DeepSeek-V4Pro decoder block."""

    def __init__(self, config: DeepSeekV4ProConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hc_mult = config.hc_mult
        mix_hc = (2 + config.hc_mult) * config.hc_mult
        hc_dim = config.hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_pre = DeepSeekHcPre(config)
        self.hc_post = DeepSeekHcPost()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DeepSeekV4ProAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.moe = DeepSeekV4ProMoE(config, layer_id=layer_idx)
        self.is_last_layer = layer_idx == config.num_hidden_layers - 1
        self.hc_head = DeepSeekHcHead(config) if self.is_last_layer else None
        self.reset_mhc_parameters()

    def reset_mhc_parameters(self) -> None:
        for param in (self.hc_attn_fn, self.hc_ffn_fn):
            nn.init.trunc_normal_(param, mean=0.0, std=0.02)
        for param in (self.hc_attn_base, self.hc_ffn_base):
            nn.init.zeros_(param)
        for param in (self.hc_attn_scale, self.hc_ffn_scale):
            nn.init.ones_(param)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hidden_states.ndim == 3:
            hidden_states = hidden_states.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        residual = hidden_states
        hidden_states, post, comb = self.hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
        )
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = self.hc_post(hidden_states, residual, post, comb)

        residual = hidden_states
        hidden_states, post, comb = self.hc_pre(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.moe(hidden_states, input_ids)
        hidden_states = self.hc_post(hidden_states, residual, post, comb)
        if self.hc_head is not None:
            hidden_states = self.hc_head(hidden_states)
        return hidden_states


class DeepSeekV4ProTextModel(nn.Module):
    """Inner decoder stack."""

    def __init__(self, config: DeepSeekV4ProConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DeepSeekV4ProDecoder(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states.unsqueeze(2).repeat(1, 1, self.config.hc_mult, 1)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class DeepSeekV4ProForCausalLM(nn.Module):
    """DeepSeek-V4Pro causal LM wrapper."""

    _tp_plan = {
        "*.self_attn.q_lora": "colwise",
        "*.self_attn.q_proj": "colwise",
        "*.self_attn.wkv": "colwise",
        "*.self_attn.wo_a": "colwise",
        "*.self_attn.wo_b": "rowwise",
        "*.moe.router.gate": "colwise",
        "*.moe.experts.w1": "colwise",
        "*.moe.experts.w3": "colwise",
        "*.moe.experts.w2": "rowwise",
        "*.moe.shared_experts.w1": "colwise",
        "*.moe.shared_experts.w3": "colwise",
        "*.moe.shared_experts.w2": "rowwise",
    }
    _ep_modules = ["*.moe.experts"]

    def __init__(self, config: DeepSeekV4ProConfig):
        super().__init__()
        self.config = config
        self.model = DeepSeekV4ProTextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_weights()

    @property
    def layers(self):
        return self.model.layers

    @property
    def embed_tokens(self):
        return self.model.embed_tokens

    @property
    def norm(self):
        return self.model.norm

    def tie_weights(self) -> None:
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def _loss(
        self,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if labels is None:
            return None
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous()
        return _masked_cross_entropy(shift_logits, shift_labels)

    def _build_output(
        self,
        loss: Optional[torch.Tensor],
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        return_hidden_states: bool,
    ) -> dict[str, Any]:
        output: dict[str, Any] = {"loss": loss, "logits": logits}
        if return_hidden_states:
            output["hidden_states"] = hidden_states
        return output

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        """Run the causal LM forward pass."""
        del use_cache
        return_hidden_states = kwargs.pop("return_hidden_states", False)
        del kwargs
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.size(1),
                device=input_ids.device,
                dtype=torch.long,
            ).view(1, -1).expand(input_ids.size(0), -1)

        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logits = self.lm_head(hidden_states)
        loss = self._loss(logits, labels)
        return self._build_output(loss, logits, hidden_states, return_hidden_states)


__all__ = [
    "DeepSeekV4ProAttention",
    "DeepSeekV4ProConfig",
    "DeepSeekV4ProDecoder",
    "DeepSeekV4ProForCausalLM",
    "DeepSeekV4ProMoE",
    "DeepSeekV4ProTextModel",
]
