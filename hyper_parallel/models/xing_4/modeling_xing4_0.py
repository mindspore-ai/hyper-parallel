# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Training-capable Xing4 model built on the bundled Hugging Face sources.

The unmodified official sources stay in ``transformer/``. This module subclasses
them and assembles, at construction time, every shared module that carries
training semantics: fused MLA with per-head logit maxima, the fused mHC pre
operation, grouped-GEMM experts and the training router.

The recipe still selects, through ``plan_overrides``:

- ``SwiGLUMLP`` for the dense decoder MLPs and ``mlp.shared_experts``; the
  official MLP computes the same result, so this is a pure performance choice.
- the context- and expert-parallel rules, which need meshes that do not exist
  while the model is built.

Weights come from a converted training checkpoint the Trainer restores by the
model's own tensor names, so no checkpoint conversion is declared here. Loading
the published Hugging Face weights is not a supported entry point: it would
need the fused layouts declared, the released configuration resolved to the
bundled ``Xing4_0Config``, and the MTP decoder mapped out of its backbone
layer index.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from torch import nn
from transformers import PreTrainedModel
from transformers.masking_utils import create_causal_mask
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.utils import ModelOutput

from hyper_parallel.components.functional.aux_loss import aux_loss_auto_scale
from hyper_parallel.components.functional.mhc_post import mhc_post
from hyper_parallel.components.functional.mhc_pre import mhc_pre
from hyper_parallel.components.functional.npu_fusion_attention import (
    _prepare_fusion_attention_context, PACKED_SEQUENCE_ARGUMENTS,
)
from hyper_parallel.components.functional.rotary_embedding import (
    apply_rotary_pos_emb_interleave as fused_rotary,
)
from hyper_parallel.components.losses.chunked_cross_entropy import chunked_cross_entropy
from hyper_parallel.components.modules.grouped_experts import GroupedExperts
from hyper_parallel.components.modules.mla_attention import MLAAttention
from hyper_parallel.components.modules.mtp import MTPLayer, build_mtp
from hyper_parallel.components.modules.rms_norm import RMSNorm
from hyper_parallel.core.activation_memory.recompute_state import is_recomputing
from hyper_parallel.core.multicore.modules._precision import preserve_fp32_parameters
from hyper_parallel.core.utils.communication import differentiable_all_reduce
from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.distributed.expert_parallel.experts import SHARED_MERGE_MODES
from hyper_parallel.distributed.expert_parallel.recipes import build_ep_compute
from hyper_parallel.distributed.recipe_spec import local_compute
from hyper_parallel.models import HyperAutoModelForCausalLM
from hyper_parallel.models.materialization import register_rebuildable_buffer
from hyper_parallel.models.xing_4.transformer.configuration_xing4_0 import Xing4_0Config
from hyper_parallel.models.xing_4.transformer.modeling_xing4_0 import (
    Xing4_0DecoderLayer, Xing4_0ForCausalLM, Xing4_0HyperConnection,
    Xing4_0Model, Xing4_0MoE, Xing4_0PreTrainedModel, Xing4_0RMSNorm, Xing4_0TopkRouter,
    apply_rotary_pos_emb_interleave,
)

__all__ = [
    "Xing4ForCausalLM",
    "build_model",
    "create_config",
    "xing4_ep_compute_fn",
]

# MindFormers recipe options this model does not implement. Nothing here reads
# them, so accepting a different value would train something other than what the
# recipe asks for, without any error: ``mhc_pre`` has no CMHC, HC-head or clamp
# argument at all. ``create_config`` supplies the one supported value when the
# recipe leaves the option out, and names the option when it does not.
_REQUIRED_OPTIONS = {
    "use_fused_mhc": True,
    "enable_cmhc": False,
    "enable_hc_head": False,
    "mtp_enable_hyper_connections": True,
    "hc_sinkhorn_clamp": None,
    "mhc_h_res_clamp_min": None,
    "mhc_h_res_clamp_max": None,
}


def fusion_attention_with_logit_maxima(module: nn.Module, query: torch.Tensor, key: torch.Tensor,
                                       value: torch.Tensor, attention_mask: torch.Tensor | None,
                                       **kwargs: Any) -> tuple[torch.Tensor, None]:
    """Run causal TND attention and accumulate maxima for QK clipping.

    Query/key are BNSD, value is BNSV, and output is BSNV. CP supplies
    packed sequence boundaries and the head range within the TP shard."""
    del attention_mask
    if kwargs.get("indices") is not None or kwargs.get("dropout", 0.0):
        raise ValueError("Xing4 MLA requires dense attention with dropout=0")
    batch, heads, sequence = query.shape[:3]
    options = dict(kwargs)
    head_offset = options.pop("mla_head_offset", 0)
    local_heads = options.pop("mla_local_heads", heads)
    if not any(options.get(name) is not None for name in PACKED_SEQUENCE_ARGUMENTS):
        options["actual_seq_len"] = tuple(range(sequence, batch * sequence + 1, sequence))
    context = _prepare_fusion_attention_context(module, query, key, value, None, options)
    output, softmax_max = torch_npu.npu_fusion_attention(
        context.query, context.key, context.value, heads, context.input_layout,
        pse=None, padding_mask=None, atten_mask=context.attention_mask, scale=kwargs["scaling"],
        pre_tockens=context.pre_tokens, next_tockens=context.next_tokens, keep_prob=1.0,
        inner_precise=0, sparse_mode=context.sparse_mode, actual_seq_qlen=context.query_lengths,
        actual_seq_kvlen=context.key_lengths, softmax_layout="TND")[:2]
    if module.training and module.record_logit_maxima:
        # QK clipping consumes detached maxima over the token and redundant-lane axes.
        maximum = softmax_max.detach().amax(dim=(0, 2))
        if heads != local_heads:
            # Ulysses owns a subset of TP-local heads; QK clipping's MAX reduction joins them.
            placed = maximum.new_full((local_heads,), float("-inf"))
            placed[head_offset:head_offset + heads] = maximum
            maximum = placed
        module.max_logits = maximum if module.max_logits is None else torch.maximum(module.max_logits, maximum)
    return output.view(batch, sequence, heads, -1), None


class Xing4MLAAttention(MLAAttention):
    """Fuse MLA projections with FP32 rotary channels and per-head maxima.

    The decoder applies the recipe's latent-norm epsilon and attention scaling
    to the official module before this wrapper snapshots them.
    """

    def __init__(self, module: nn.Module) -> None:
        """Adopt the official attention weights and select the rotary implementation."""
        super().__init__(module=module, attention_interface=fusion_attention_with_logit_maxima)
        if not self.rotary_interleaved:
            raise ValueError("Xing4 MLA requires interleaved rotary weights")
        self.q_a_layernorm = RMSNorm(module=self.q_a_layernorm)
        self.kv_a_layernorm = RMSNorm(module=self.kv_a_layernorm)
        self.apply_rotary_pos_emb_interleave = (
            fused_rotary if getattr(self.config, "apply_rope_fusion", False) else apply_rotary_pos_emb_interleave
        )
        self.record_logit_maxima = False
        self.max_logits: torch.Tensor | None = None

    def _project_attention_inputs(self, hidden_states: torch.Tensor,
                                  position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
                                  past_key_values: Any | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reuse latent and KV projections, rounding RoPE at each concatenation."""
        del past_key_values  # The training entry always calls the backbone with use_cache=False.
        latents = self._project_latents(hidden_states)
        query_rope, key_rope = self.apply_rotary_pos_emb_interleave(
            latents.query_rope.float(), latents.key_rope.float(), *position_embeddings,
        )
        kv = self.kv_b_proj(latents.kv_nope).view(latents.batch_size, latents.sequence_length, self.num_heads, -1)
        key_nope, value = kv.transpose(1, 2).split((self.qk_nope_head_dim, self.v_head_dim), dim=-1)
        query_pass = latents.query_pass.transpose(1, 2)
        query = torch.cat((query_pass, query_rope.to(query_pass.dtype)), dim=-1)
        key = torch.cat((key_nope, key_rope.to(key_nope.dtype).expand(-1, self.num_heads, -1, -1)), dim=-1)
        return query, key, value


class Xing4Router(Xing4_0TopkRouter):
    """HF learned selection with the recipe's training state and objectives."""

    fp32_buffer_names = ("e_score_correction_bias",)

    def __init__(self, config: Any, source: nn.Module, layer_idx: int) -> None:
        """Retain the HF weights and register all resumable routing state."""
        super().__init__(config)
        self.weight = source.weight
        self.e_score_correction_bias = source.e_score_correction_bias
        self.padding = config.n_routed_experts if config.use_pad_tokens else 0
        self.aux_loss_coeff = config.moe_aux_loss_coeff
        self.aux_loss_cp_group = None
        self.register_buffer("_token_counts", torch.zeros(self.num_experts, dtype=torch.int64))
        self.input_jitter_eps = config.moe_input_jitter_eps
        self.last_aux_loss = None
        self.group_score_topk = config.moe_router_group_score_topk
        # Benchmarks only: an exactly balanced round robin replacing the decision.
        self.force_expert_balance = config.moe_router_force_expert_balance
        # MindFormers hashes the leading decoder layers, never an MTP depth, and
        # drops the load bias wherever the hash table decides the selection.
        self.is_hash_layer = layer_idx < config.moe_n_hash_layers
        self.token_ids = None
        if self.is_hash_layer:
            self.register_buffer("tid2eid", torch.empty(config.vocab_size, self.top_k, dtype=torch.int32))
        self.enable_expert_bias = config.moe_router_enable_expert_bias and not self.is_hash_layer
        self.expert_bias_update_rate = config.moe_router_bias_update_rate

    def _apply(self, fn: Callable, recurse: bool = True) -> "Xing4Router":
        """Keep load-balancing bias updates in FP32 during model dtype conversion."""
        return super()._apply(preserve_fp32_parameters(self, fn, self.fp32_buffer_names), recurse=recurse)

    def _select_experts(self, scores: torch.Tensor) -> torch.Tensor:
        """Select experts the way MindFormers does: round robin, hash table, groups or top-k.

        Forced expert balance replaces the decision with an exactly balanced
        round robin, for benchmarks. A hash layer reads fixed expert ids from
        ``tid2eid``. Otherwise the load bias steers a top-k that node-limited
        routing may restrict to the best ``topk_group`` groups, each group
        scoring as the sum of its ``moe_router_group_score_topk`` best biased
        affinities.
        """
        if self.force_expert_balance:
            tokens = scores.shape[0]
            return (torch.arange(tokens * self.top_k, device=scores.device)
                    .view(tokens, self.top_k) % self.num_experts)
        if self.is_hash_layer:
            return self.tid2eid.index_select(0, self.token_ids.reshape(-1)).long()
        biased = scores + self.e_score_correction_bias
        if self.num_group:
            grouped = biased.unflatten(-1, (self.num_group, -1))
            group_scores = grouped.topk(self.group_score_topk, dim=-1).values.sum(dim=-1)
            keep = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(
                1, group_scores.topk(self.topk_group, dim=-1, sorted=False).indices, True)
            biased = grouped.masked_fill(~keep.unsqueeze(-1), float("-inf")).flatten(-2)
        return biased.topk(self.top_k, dim=-1, sorted=False).indices

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route real tokens once; safe-token padding consumes neither jitter nor counts."""
        real_hidden = hidden_states[:, self.padding:]
        flat = real_hidden.reshape(-1, hidden_states.shape[-1]).float()
        if self.training and self.input_jitter_eps:
            flat = flat * (1.0 + (2.0 * torch.rand_like(flat) - 1.0) * self.input_jitter_eps)
        logits = F.linear(flat, self.weight.float())
        scores = logits.sigmoid()
        indices = self._select_experts(scores)
        # The load bias steers selection only; the gate value stays the raw affinity.
        weights = scores.gather(1, indices)
        if self.norm_topk_prob or self.is_hash_layer:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.routed_scaling_factor
        weights = self._attach_balance_loss(weights, scores, indices, batch_size=hidden_states.shape[0])
        if self.training and self.enable_expert_bias and not is_recomputing():
            counts = torch.zeros_like(self._token_counts).scatter_add_(
                0, indices.reshape(-1), torch.ones_like(indices.reshape(-1), dtype=torch.int64))
            self._token_counts.add_(counts)
        if self.padding:
            batch = hidden_states.shape[0]
            padding = torch.arange(self.padding * self.top_k, device=indices.device)
            padding = (padding.view(1, self.padding, self.top_k) % self.num_experts).expand(batch, -1, -1)
            indices = torch.cat((padding, indices.reshape(batch, -1, self.top_k)), dim=1).reshape(-1, self.top_k)
            weights = torch.cat((weights.new_zeros(batch, self.padding, self.top_k),
                                 weights.reshape(batch, -1, self.top_k)), dim=1).reshape(-1, self.top_k)
        return logits, weights.to(hidden_states.dtype), indices

    def _attach_balance_loss(self, weights: torch.Tensor, scores: torch.Tensor, indices: torch.Tensor,
                             batch_size: int) -> torch.Tensor:
        """Inject ``coeff * E * mean_b(sum_i(f_bi * P_bi))`` through the routing weights.

        ``f_bi`` counts unbiased top-k affinities per sequence, independently of the expert-load bias. ``P_bi`` averages
        the normalized affinities within each sequence. The sequence objectives are averaged over
        the batch.

        Args:
            weights: Routing weights of the real tokens.
            scores: Router affinities of every expert.
            batch_size: Number of independent sequences before flattening the tokens.
        """
        # Hash routing, the load bias and group limiting all leave the auxiliary
        # statistics on the raw affinities; only forced balance replaces them.
        selected = (indices if self.force_expert_balance
                    else scores.topk(self.top_k, dim=-1, sorted=False).indices).reshape(batch_size, -1)
        counts = scores.new_zeros(batch_size, self.num_experts).scatter_add_(
            1, selected, torch.ones_like(selected, dtype=scores.dtype))
        if self.aux_loss_cp_group is None:
            load = counts / selected.shape[1]
            affinity = (scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)).reshape(
                batch_size, -1, self.num_experts).mean(dim=1)
        else:
            normalized = (scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)).reshape(
                batch_size, -1, self.num_experts)
            group = self.aux_loss_cp_group
            with torch.no_grad():
                counts = counts.detach()
                dist.all_reduce(counts, group=group)
                tokens = counts.new_tensor(normalized.shape[1])
                dist.all_reduce(tokens, group=group)
            load = counts / (tokens * self.top_k)
            affinity = differentiable_all_reduce(normalized.sum(dim=1), "sum", group) / tokens
        loss = (affinity * load).sum(dim=-1).mean() * self.num_experts
        self.last_aux_loss = loss.detach()
        return aux_loss_auto_scale(weights, loss * self.aux_loss_coeff)

    @property
    def tokens_per_expert(self) -> torch.Tensor:
        """Expose the persistent counts to the shared MoE monitor."""
        return self._token_counts

    @torch.no_grad()
    def update_expert_bias(self, lr: float, num_recomputations: int = 1) -> None:
        """Apply one load-based update after the optimizer step."""
        # Forward already excludes recomputation from the accumulated token counts.
        del num_recomputations
        counts = self._token_counts
        self.e_score_correction_bias.add_((counts.sum() - counts * self.num_experts).sign(), alpha=lr)
        counts.zero_()


def build_grouped_experts(config: Xing4_0Config, experts: nn.ModuleList) -> GroupedExperts:
    """Build the shared grouped-GEMM experts for the official per-expert projections.

    The packed parameters stay empty: the Trainer's checkpoint restore or
    ``GroupedExperts.reset_parameters`` fills them.

    Args:
        config: Resolved Xing4 configuration.
        experts: Official per-expert ``Xing4_0MLP`` modules.

    Returns:
        The packed grouped-GEMM experts replacing the official module list.
    """
    gate, down = experts[0].gate_proj.weight, experts[0].down_proj.weight
    intermediate_size, hidden_size = gate.shape
    count = len(experts)
    source = nn.Module()
    source.config = config
    source.num_local_experts = count
    source.hidden_size = hidden_size
    source.intermediate_size = intermediate_size
    source.has_gate = True
    source.is_concatenated = True
    source.is_transposed = True
    source.gate_up_proj = nn.Parameter(
        gate.new_empty((count, hidden_size, 2 * intermediate_size)), requires_grad=gate.requires_grad)
    source.down_proj = nn.Parameter(
        down.new_empty((count, intermediate_size, hidden_size)), requires_grad=down.requires_grad)
    source.train(experts.training)
    return GroupedExperts(module=source)


class Xing4MoE(Xing4_0MoE):
    """Reuse the HF MoE facade with grouped-GEMM experts and training routing."""

    def __init__(self, config: Any, source: nn.Module, layer_idx: int) -> None:
        """Pack the official experts and add the training router and EP boundary."""
        nn.Module.__init__(self)  # pylint: disable=non-parent-init-called
        self.config = config
        self.experts = build_grouped_experts(config, source.experts)
        self.gate = Xing4Router(config, source.gate, layer_idx)
        # The recipe replaces this dense MLP with the shared SwiGLUMLP.
        self.shared_experts = source.shared_experts
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.padding = config.n_routed_experts if config.use_pad_tokens else 0
        self.ep_executor = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Route real tokens through the experts, with optional padding and EP dispatch."""
        padded = hidden_states
        if self.padding:
            padded = torch.cat((padded.new_zeros(padded.shape[0], self.padding, padded.shape[-1]), padded), dim=1)
        if self.ep_executor is not None:
            routed = self.ep_executor(self, padded)
        else:
            shape = padded.shape
            routed = self.moe(padded.view(-1, shape[-1]), *self.route(padded)).view(*shape)
        # Safe-token padding never reaches the shared experts: they are token-wise.
        return routed[:, self.padding:] + self.shared_experts(hidden_states)

    def moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor,
            topk_weights: torch.Tensor) -> torch.Tensor:
        """Run the packed grouped-GEMM experts on flattened tokens."""
        return self.experts(hidden_states, topk_indices, topk_weights)

    def route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose the same once-only router contract to the expert-parallel executor."""
        _, weights, indices = self.gate(hidden)
        return indices, weights


class Xing4FusedHyperConnection(Xing4_0HyperConnection):
    """Adapt official mHC parameters to the shared NPU pre operation."""

    def __init__(self, module: Xing4_0HyperConnection) -> None:
        """Keep official parameter names, values and ownership."""
        nn.Module.__init__(self)
        self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps = module.hc_mult, module.hc_sinkhorn_iters, module.hc_eps
        self.input_norm = module.input_norm
        self.hc_fn, self.hc_base, self.hc_scale = module.hc_fn, module.hc_base, module.hc_scale
        self.train(module.training)

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run the fused pre operator and preserve its residual-gradient output."""
        hidden, post, mix, residual = mhc_pre(
            hidden_streams.flatten(2), self.hc_fn, self.hc_scale, self.hc_base, self.hc_mult,
            sinkhorn_iters=self.hc_sinkhorn_iters, norm_eps=self.input_norm.eps, hc_eps=self.hc_eps)
        # Keep the pre operator's residual output in the graph: its backward
        # consumes the post gradient through grad_x_post.
        return post, mix, hidden, residual.unflatten(-1, (self.hc_mult, hidden.shape[-1]))


class Xing4DecoderLayer(Xing4_0DecoderLayer):
    """Install the high-performance blocks around the official decoder weights."""

    def __init__(self, config: Any, layer_idx: int, source: nn.Module | None = None) -> None:
        """Apply the recipe's norm and scaling values, then adopt the shared modules."""
        nn.Module.__init__(self)
        source = Xing4_0DecoderLayer(config, layer_idx) if source is None else source
        self.hidden_size = source.hidden_size
        # The official attention builds its latent norms with the module default epsilon and
        # the wrapper snapshots them, so the recipe values must be applied before wrapping.
        source.self_attn.q_a_layernorm.variance_epsilon = config.rms_norm_eps
        source.self_attn.kv_a_layernorm.variance_epsilon = config.rms_norm_eps
        if config.attention_softmax_scale is not None:
            source.self_attn.scaling = config.attention_softmax_scale
        self.self_attn = Xing4MLAAttention(source.self_attn)
        # Dense layers keep the official MLP; the recipe replaces it with the shared SwiGLUMLP.
        self.mlp = Xing4MoE(config, source.mlp, layer_idx) if isinstance(source.mlp, Xing4_0MoE) else source.mlp
        self.input_layernorm = source.input_layernorm
        self.post_attention_layernorm = source.post_attention_layernorm
        source.attn_hc.input_norm.eps = source.ffn_hc.input_norm.eps = config.mhc_layernorm_epsilon
        self.attn_hc = Xing4FusedHyperConnection(source.attn_hc)
        self.ffn_hc = Xing4FusedHyperConnection(source.ffn_hc)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None,
                position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, **kwargs: Any) -> torch.Tensor:
        """Call the official blocks through the fused mHC pre/post operations."""
        post, mix, collapsed, residual = self.attn_hc(hidden_states)
        output = self.self_attn(hidden_states=self.input_layernorm(collapsed),
                                position_embeddings=position_embeddings, attention_mask=attention_mask, **kwargs)[0]
        hidden_states = self._post(output, residual, post, mix)
        post, mix, collapsed, residual = self.ffn_hc(hidden_states)
        output = self.mlp(self.post_attention_layernorm(collapsed))
        return self._post(output, residual, post, mix)

    @staticmethod
    def _post(output: torch.Tensor, residual: torch.Tensor,
              post: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
        """Convert destination-by-source mixing to the shared post layout."""
        streams = residual.shape[2]
        result = mhc_post(output, residual.flatten(2), post, mix.transpose(-1, -2).contiguous(), streams)
        return result.unflatten(-1, (streams, output.shape[-1]))


@torch.no_grad()
def _init_training_weights(owner: Xing4_0PreTrainedModel, module: nn.Module) -> None:
    """Initialize the modules the official ``_init_weights`` no longer sees.

    The official implementation initializes ``Xing4_0HyperConnection`` through
    ``fn``/``base``/``scale``, which that class does not define, so its branch
    is replaced here. The shared RMSNorm, grouped experts and the additional
    router count are not official module types. Every call goes through HF's
    partial-load guards so already loaded values are preserved.
    """
    if isinstance(module, Xing4_0HyperConnection):
        nn.init.normal_(module.hc_fn, mean=0.0, std=1e-4)
        nn.init.constant_(module.hc_scale, owner.config.mhc_init_gating_factor)
        nn.init.zeros_(module.hc_base)
    elif isinstance(module, RMSNorm):
        nn.init.ones_(module.weight)
    elif isinstance(module, GroupedExperts):
        module.reset_parameters()
    else:
        Xing4_0PreTrainedModel._init_weights(owner, module)
        if isinstance(module, Xing4Router):
            nn.init.zeros_(module._token_counts)


class Xing4Model(Xing4_0Model):
    """Use the unchanged HF backbone forward with training-aware decoder blocks."""

    config_class = Xing4_0Config
    _init_weights = _init_training_weights
    _supports_attention_backend = False

    def __init__(self, config: Any) -> None:
        """Reuse the HF backbone and register RoPE tables for post-sharding materialization."""
        super().__init__(config)
        self.layers = nn.ModuleList(
            Xing4DecoderLayer(config, index, source=layer) for index, layer in enumerate(self.layers))
        rotary = self.rotary_emb
        rope_init_fn = (rotary.compute_default_rope_parameters if rotary.rope_type == "default"
                        else ROPE_INIT_FUNCTIONS[rotary.rope_type])
        with torch.device("cpu"):
            frequencies, rotary.attention_scaling = rope_init_fn(config, device="cpu")
        for name in ("inv_freq", "original_inv_freq"):
            register_rebuildable_buffer(rotary, name, value=frequencies)
        self.post_init()

    @staticmethod
    def shift_tokens(tokens: torch.Tensor, pad_value: int) -> torch.Tensor:
        """Shift a full training sequence; CP recipes replace its boundary exchange."""
        return F.pad(tokens[:, 1:], (0, 1), value=pad_value)

    @staticmethod
    def mean_loss(loss_sum: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        """Average one sequence objective; CP recipes reduce its numerator and count."""
        return loss_sum / count.clamp_min(1)


class Xing4MTPBlock(Xing4DecoderLayer):
    """Adapt the shared MTP block's 3-D input to official Xing4 residual streams."""

    def __init__(self, config: Any, layer_idx: int) -> None:
        """Use the HF decoder selected for this MTP depth."""
        super().__init__(config, layer_idx)
        self.hc_mult = config.hc_mult

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Expand and collapse streams, leaving fusion and norms to MTPLayer."""
        streams = hidden_states.unsqueeze(2).expand(-1, -1, self.hc_mult, -1).contiguous()
        return super().forward(streams, **kwargs).mean(dim=2)


class Xing4ForCausalLM(Xing4_0ForCausalLM):
    """Add teacher-forced LM/MTP losses to the official causal language model."""

    config_class = Xing4_0Config
    _init_weights = _init_training_weights
    _no_split_modules = ["Xing4DecoderLayer", "MTPLayer"]
    _keys_to_ignore_on_load_unexpected = None
    _can_record_outputs = None
    _supports_flash_attn = False
    _supports_flex_attn = False
    _supports_attention_backend = False

    def __init__(self, config: Any) -> None:
        """Build the training backbone and the configured MTP depths."""
        Xing4_0PreTrainedModel.__init__(self, config)
        compute_dtype = getattr(config, "compute_dtype", None)
        self.embedding_compute_dtype = (getattr(torch, compute_dtype)
                                        if isinstance(compute_dtype, str) else compute_dtype)
        self.model = Xing4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp = build_mtp(config.num_nextn_predict_layers, layer_factory=lambda depth: MTPLayer(
            config.hidden_size, Xing4MTPBlock(config, config.num_hidden_layers + depth),
            norm_factory=partial(Xing4_0RMSNorm, eps=config.rms_norm_eps)))
        if self.mtp is not None:
            self.mtp.gradient_checkpointing = False
        self.post_init()
        self._hash_routers = [module for module in self.modules()
                              if isinstance(module, Xing4Router) and module.is_hash_layer]

    @classmethod
    def from_config(cls, config: Any, **kwargs: Any) -> "Xing4ForCausalLM":
        """Construct through the shared custom-model builder contract."""
        del kwargs
        return cls(config)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None,
                shift_labels: torch.Tensor | None = None, position_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None, use_cache: bool = False) -> ModelOutput:
        """Return LM and weighted MTP losses for aligned causal training sequences."""
        del labels
        if attention_mask is not None or use_cache:
            raise ValueError("Xing4 trains causal sequences without attention masks or KV cache")
        if shift_labels is None or shift_labels.shape != input_ids.shape:
            raise ValueError("Xing4 training requires shift_labels aligned to input_ids")
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        for router in self._hash_routers:
            router.token_ids = input_ids
        embeddings = self.model.embed_tokens(input_ids).to(dtype=self.embedding_compute_dtype)
        hidden = self.model(inputs_embeds=embeddings, position_ids=position_ids, use_cache=False).last_hidden_state
        targets = shift_labels.masked_fill(shift_labels < 0, IGNORE_INDEX)
        losses = {"foundation_loss": self._cross_entropy(hidden, targets)}
        if self.mtp is not None:
            position_embeddings = self.model.rotary_emb(hidden, position_ids)
            losses["foundation_loss/mtp"] = self._mtp_loss(
                hidden, input_ids, targets, position_embeddings, position_ids)
        return ModelOutput(loss=losses)

    def _mtp_loss(self, hidden: torch.Tensor, input_ids: torch.Tensor, targets: torch.Tensor,
                  position_embeddings: tuple[torch.Tensor, torch.Tensor],
                  position_ids: torch.Tensor) -> torch.Tensor:
        """Return ``mtp_loss_factor / D * sum_k CE_k``, where depth ``k`` predicts ``k`` tokens further.

        Inputs and targets shift left by one token per depth without wrapping.
        """
        total = torch.zeros((), device=hidden.device, dtype=torch.float32)
        for layer in self.mtp.layers:
            input_ids = self.model.shift_tokens(input_ids, 0)
            targets = self.model.shift_tokens(targets, IGNORE_INDEX)
            embeddings = self.model.embed_tokens(input_ids).to(dtype=self.embedding_compute_dtype)
            attention_mask = create_causal_mask(
                config=self.config, inputs_embeds=embeddings, attention_mask=None,
                past_key_values=None, position_ids=position_ids)
            hidden = layer(hidden, embeddings, position_embeddings=position_embeddings,
                           attention_mask=attention_mask)
            total = total + self._cross_entropy(hidden, targets)
        return total * (self.config.mtp_loss_factor / len(self.mtp.layers))

    def _cross_entropy(self, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return the LM-head cross entropy averaged over the non-ignored targets.

        A head holding the whole vocabulary uses the chunked cross entropy, which never
        materializes the sequence's logits. A tensor-parallel head runs as a module, so its
        sharding plan gathers the logits or, under loss parallelism, keeps them
        vocabulary-sharded for the ``loss_function`` the Trainer installs.
        """
        count = targets.ne(IGNORE_INDEX).sum()
        weight = self.lm_head.weight
        if getattr(weight, "placements", None) is None and weight.shape[0] == self.config.vocab_size:
            loss_sum = chunked_cross_entropy(hidden, targets, weight,
                                             chunk_size=self.config.chunk_loss_size, ignore_index=IGNORE_INDEX)
        else:
            loss_sum = self.loss_function(self.lm_head(hidden), None, self.config.vocab_size, num_items_in_batch=1,
                                          ignore_index=IGNORE_INDEX, shift_labels=targets)
        return self.model.mean_loss(loss_sum, count)


@local_compute
def xing4_ep_compute_fn(*, module: Any, mesh: Any, tp_mesh: Any, cp_mesh: Any,
                        ep_mesh: Any, fix_router: bool = False) -> Callable:
    """Dispatch routed experts while retaining local shared experts and router state.

    ``fix_router`` balances dispatch indices for benchmarks; safe-token padding
    and statistics from the learned router keep their separate semantics.

    Args:
        module: One :class:`Xing4MoE` layer.
        mesh: Full framework mesh.
        tp_mesh: Tensor parallel mesh, or None.
        cp_mesh: Context parallel mesh, or None.
        ep_mesh: Expert parallel mesh owning the routed experts.
        fix_router: Benchmark-only balanced destinations.

    Returns:
        The MoE forward that the rewriter binds to the module.
    """
    del mesh, tp_mesh, cp_mesh
    module.ep_executor = build_ep_compute(
        module, ep_mesh, router_fn=type(module).route, archetype_key="xing4",
        expected_attrs=["gate", "experts", "shared_experts"], combine=SHARED_MERGE_MODES["none"],
        use_grouped_gemm=True, fix_router=fix_router)
    return type(module).forward


def create_config(config: dict[str, Any]) -> Xing4_0Config:
    """Map training aliases to HF fields, preferring an explicit HF value.

    Args:
        config: Model fields supplied by the training recipe.

    Returns:
        The bundled HF configuration carrying HP training options.

    Raises:
        ValueError: If the recipe selects semantics the shared components do not implement.
    """
    values = dict(config)
    for alias, field in (("num_residual_streams", "hc_mult"), ("mhc_sinkhorn_iterations", "hc_sinkhorn_iters"),
                         ("router_jitter_noise", "moe_input_jitter_eps")):
        if alias in values:
            values.setdefault(field, values.pop(alias))
    values.setdefault("moe_input_jitter_eps", 0.0)
    values.setdefault("attention_softmax_scale", None)
    values.setdefault("use_cache", False)
    values.setdefault("moe_n_hash_layers", 0)
    values.setdefault("moe_router_group_score_topk", 2)
    values.setdefault("moe_router_force_expert_balance", False)
    unsupported = sorted(name for name, required in _REQUIRED_OPTIONS.items()
                         if values.setdefault(name, required) != required)
    if unsupported:
        raise ValueError(f"Xing4 training implements one fixed recipe; unsupported options: {unsupported}")
    values.update(architectures=["Xing4ForCausalLM"])
    return Xing4_0Config(**values)


def build_model(*, config: dict[str, Any], **options: Any) -> PreTrainedModel:
    """Build and parallelize the model without loading weights.

    The Trainer restores the converted training checkpoint afterwards, by the
    model's own tensor names.

    Args:
        config: Model configuration with HF fields and HP training options.
        **options: Construction options the Trainer passes to the shared builder.

    Returns:
        The constructed, parallelized model.
    """
    return HyperAutoModelForCausalLM.from_config(create_config(config), **options)
