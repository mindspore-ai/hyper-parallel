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
"""Xing4.0: a DeepSeek-V3 causal LM with manifold-constrained hyper-connections (mHC) and MTP.

The attention, router, routed and shared experts, dense MLP, RMSNorm, rotary embedding
and RoPE are the Transformers DeepSeek-V3 components. This module adds what DeepSeek-V3
lacks, with MindFormers numerics: the mHC residual streams; MoE pad tokens, the
sequence-wise balance loss and the expert-load bias; multi-token prediction; and the
training losses. The attention and the experts are the sources of the framework's
``MLAAttention`` (as ``adapter.conversion.module_replacement.Xing40MLAAttention``) and
``GroupedExperts``, which the recipe swaps in on NPU; the mHC pre/post modules call the
framework's ``mhc_pre``/``mhc_post``, the fused AscendC operators on NPU. The recipe
supplies every configuration value.
"""

# This model uses the Torch/HF runtime, like the existing model families.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from transformers import AutoConfig, DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3MLP,
    DeepseekV3MoE,
    DeepseekV3PreTrainedModel,
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
)
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperHead
from transformers.utils import ModelOutput

from hyper_parallel.components.functional.aux_loss import aux_loss_auto_scale
from hyper_parallel.components.functional.mhc_post import mhc_post
from hyper_parallel.components.functional.mhc_pre import mhc_pre
from hyper_parallel.components.losses.chunked_cross_entropy import chunked_cross_entropy
from hyper_parallel.components.losses.projected_cross_entropy_sum import projected_cross_entropy_sum
from hyper_parallel.core.activation_memory.recompute_state import is_recomputing
from hyper_parallel.core.utils.communication import differentiable_all_reduce
from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.models.materialization import register_rebuildable_buffer

__all__ = [
    "Xing40Config",
    "Xing40ForCausalLM",
    "Xing40MhcPost",
    "Xing40MhcPre",
    "Xing40MoE",
    "Xing40Output",
]

# Stabilizer of the mHC sigmoid gates and Sinkhorn normalization.
_HC_EPS = 1e-6


class Xing40Config(DeepseekV3Config):
    """Xing4.0 configuration: DeepSeek-V3 fields whose model type selects this model and its adapter.

    Besides the DeepSeek-V3 fields, the recipe sets ``num_residual_streams``,
    ``mhc_sinkhorn_iterations``, ``mhc_init_gating_factor``, ``mhc_layernorm_epsilon``,
    ``mtp_loss_factor``, ``moe_aux_loss_coeff``, ``moe_router_enable_expert_bias``,
    ``moe_router_bias_update_rate``, ``use_pad_tokens`` and ``chunk_loss_size``.
    ``moe_router_force_expert_balance`` is optional and defaults to False.
    ``use_fused_mhc`` must be True; the shared mHC pre/post require NPU fused operators.
    ``attention_softmax_scale`` optionally sets the attention score scale,
    independently of the rotary embedding frequencies. None derives the scale from
    ``scaling_factor`` using the YaRN mscale formula; a factor of 1 keeps DeepSeek-V3's scale.
    """

    model_type = "xing40"
    attention_softmax_scale: float | None = None
    use_fused_mhc: bool = True

    def __post_init__(self, **kwargs: Any) -> None:
        """Default ``architectures`` to the model class this configuration builds.

        Args:
            **kwargs: Remaining Transformers configuration arguments.
        """
        if getattr(self, "architectures", None) is None:
            self.architectures = ["Xing40ForCausalLM"]
        super().__post_init__(**kwargs)
        if self.use_fused_mhc is not True:
            raise ValueError("Xing40 requires use_fused_mhc=True and the fused NPU mHC operators")
        scale = self.attention_softmax_scale
        if scale is not None and (isinstance(scale, bool) or not isinstance(scale, (int, float))
                                  or not math.isfinite(scale) or scale <= 0):
            raise ValueError("attention_softmax_scale must be positive and finite or None")


AutoConfig.register(Xing40Config.model_type, Xing40Config, exist_ok=True)


class Xing40RotaryEmbedding(DeepseekV3RotaryEmbedding):
    """DeepSeek-V3 RoPE tables with the MindFormers inverse frequencies."""

    def __init__(self, config: Xing40Config) -> None:
        """Compute the inverse frequencies and rebuild them after meta-device materialization.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        for name in ("inv_freq", "original_inv_freq"):
            register_rebuildable_buffer(self, name, value=self.compute_default_rope_parameters(config)[0])

    @staticmethod
    def compute_default_rope_parameters(config: Xing40Config, device: torch.device | str | None = None,
                                        seq_len: int | None = None, **kwargs: Any) -> tuple[torch.Tensor, float]:
        """Return the inverse frequencies and the unit attention scaling.

        NumPy FP32 powers match MindFormers bitwise; ``torch.pow`` rounds some entries differently.
        Transformers passes ``device`` positionally in some releases.

        Args:
            config: Model configuration.
            device: Device of the inverse frequencies; CPU when None.
            seq_len: Unused; the default RoPE does not depend on the sequence length.
            **kwargs: Unused Transformers RoPE options.
        """
        del seq_len, kwargs
        dim = config.qk_rope_head_dim
        inv_freq = 1.0 / (config.rope_parameters["rope_theta"] ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        return torch.from_numpy(inv_freq.astype(np.float32)).to(device=device), 1.0


class Xing40MoE(DeepseekV3MoE):
    """DeepSeek-V3 MoE with the MindFormers training extensions.

    The router, the routed experts and the shared experts are the Transformers
    modules. Each call prefixes each sequence with one zero pad token per expert,
    injects the sequence-wise balance loss through the routing weights and counts
    the routed tokens. The Trainer's ``MoEMonitorCallback`` reads
    ``enable_expert_bias`` and
    ``expert_bias_update_rate``, sums ``tokens_per_expert`` over the data-parallel
    ranks and calls :meth:`update_expert_bias`.
    """

    def __init__(self, config: Xing40Config) -> None:
        """Build the Transformers router and experts and the training state.

        Args:
            config: Model configuration.
        """
        super().__init__(config)
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.padding = config.n_routed_experts if config.use_pad_tokens else 0
        self.aux_loss_coeff = config.moe_aux_loss_coeff
        self.force_expert_balance = getattr(config, "moe_router_force_expert_balance", False)
        self.enable_expert_bias = config.moe_router_enable_expert_bias
        self.expert_bias_update_rate = config.moe_router_bias_update_rate
        self.tokens_per_expert: torch.Tensor | None = None
        self.last_aux_loss: torch.Tensor | None = None
        self.aux_loss_cp_group = None
        # Expert-parallel routed computation installed by the EP adapter; None computes locally.
        self.ep_executor = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Add the routed and shared experts' outputs for ``[batch, tokens, hidden]`` inputs.

        Args:
            hidden_states: Normalized MLP input.
        """
        hidden = hidden_states
        if self.padding:
            hidden = torch.cat((hidden.new_zeros(hidden.shape[0], self.padding, hidden.shape[-1]), hidden), dim=1)
        if self.ep_executor is None:
            indices, weights = self.route(hidden)
            # As MindFormers, the experts combine with the routing weights in the activation dtype.
            routed = self.experts(hidden.reshape(-1, hidden.shape[-1]), indices,
                                  weights.to(hidden.dtype)).view_as(hidden)
        else:
            routed = self.ep_executor(self, hidden)  # pylint: disable=not-callable
        return routed[:, self.padding:] + self.shared_experts(hidden_states)

    def route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route real tokens of a pad-prefixed ``[batch, padding + tokens, hidden]`` input.

        Attaches the balance loss to the routing weights and counts the routed tokens.
        Pad tokens are routed to every expert with zero weight. Expert parallelism uses
        this method as its router.

        Args:
            hidden: Pad-prefixed router inputs.

        Returns:
            Expert indices and routing weights, ``[batch * (padding + tokens), top_k]``.
        """
        real_hidden = hidden[:, self.padding:]
        if self.force_expert_balance:
            logits = F.linear(  # pylint: disable=not-callable
                real_hidden.reshape(-1, hidden.shape[-1]).float(), self.gate.weight.float())
            scores = logits.sigmoid()
            # Cycle the T*K slots over experts in order, as MindFormers' forced routing.
            slots = torch.arange(scores.shape[0] * self.top_k, device=scores.device)
            indices = (slots % self.num_experts).view(scores.shape[0], self.top_k)
            # The reference router numbers [sequence, batch] tokens before flattening.
            # Restore our [batch, sequence] layout without changing the expert assignment.
            indices = indices.reshape(real_hidden.shape[1], real_hidden.shape[0], self.top_k).transpose(0, 1)
            indices = indices.reshape(-1, self.top_k)
            weights = scores.gather(1, indices)
            if self.gate.norm_topk_prob:
                weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
            weights = weights * self.gate.routed_scaling_factor
        else:
            # Removing each sequence's pad prefix leaves a strided batch; HF's gate uses view().
            logits, weights, indices = self.gate(real_hidden.reshape(-1, hidden.shape[-1]))
            scores = logits.sigmoid()
        weights = self._attach_balance_loss(weights, scores, batch_size=hidden.shape[0],
                                            selected_indices=indices if self.force_expert_balance else None)
        # A recomputed forward replays the same routing; count each token once.
        if not is_recomputing():
            counts = torch.zeros_like(scores[0]).scatter_add_(
                0, indices.reshape(-1), torch.ones_like(indices.reshape(-1), dtype=scores.dtype))
            self.tokens_per_expert = counts if self.tokens_per_expert is None else self.tokens_per_expert.add_(counts)
        if self.padding:
            batch = hidden.shape[0]
            pad_indices = torch.arange(self.padding * self.top_k, device=indices.device)
            pad_indices = (pad_indices.view(1, self.padding, self.top_k) % self.num_experts).expand(batch, -1, -1)
            indices = torch.cat((pad_indices, indices.reshape(batch, -1, self.top_k)), dim=1).reshape(-1, self.top_k)
            weights = torch.cat((weights.new_zeros(batch, self.padding, self.top_k),
                                 weights.reshape(batch, -1, self.top_k)), dim=1).reshape(-1, self.top_k)
        return indices, weights

    def _attach_balance_loss(self, weights: torch.Tensor, scores: torch.Tensor, batch_size: int,
                             selected_indices: torch.Tensor | None = None) -> torch.Tensor:
        """Inject ``coeff * E * mean_b(sum_i(f_bi * P_bi))`` through the routing weights.

        ``f_bi`` counts unbiased top-k affinities per sequence, or the prescribed cyclic routes
        when forced balancing is enabled, independently of the expert-load bias. ``P_bi`` averages
        the normalized affinities within each sequence. The sequence objectives are averaged over
        the batch.

        Args:
            weights: Routing weights of the real tokens.
            scores: Router affinities of every expert.
            batch_size: Number of independent sequences before flattening the tokens.
            selected_indices: Forced routes, or None to use unbiased top-k affinities.
        """
        if selected_indices is None:
            selected_indices = scores.topk(self.top_k, dim=-1, sorted=False).indices
        selected = selected_indices.reshape(batch_size, -1)
        counts = scores.new_zeros(batch_size, self.num_experts).scatter_add_(
            1, selected, torch.ones_like(selected, dtype=scores.dtype))
        if getattr(self, 'aux_loss_cp_group', None) is None:
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

    @torch.no_grad()
    def update_expert_bias(self, lr: float, num_recomputations: int = 1) -> None:
        """Step the expert-load bias toward balance with the step's summed token counts.

        As in MindFormers the sign step is not re-centered.

        Args:
            lr: Bias update rate.
            num_recomputations: Unused; recomputed forwards do not count tokens.
        """
        del num_recomputations
        counts = self.tokens_per_expert
        self.gate.e_score_correction_bias.add_((counts.mean() - counts).sign(), alpha=lr)
        counts.zero_()


class Xing40MhcPre(nn.Module):
    """mHC mixing coefficients and sublayer input of the flattened residual streams.

    ``phi`` projects the RMS-normalized streams to the pre, post and residual logits;
    ``branch_alpha`` scales and ``branch_beta`` shifts the three groups. The residual
    mixing is made doubly stochastic by Sinkhorn normalization. The computation is the
    framework's ``mhc_pre`` fused AscendC operator on NPU.
    """

    def __init__(self, config: Xing40Config) -> None:
        """Create the mixing projection and the per-group scales and shifts.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.num_stream = num_stream = config.num_residual_streams
        self.sinkhorn_iterations = config.mhc_sinkhorn_iterations
        self.norm_eps = config.mhc_layernorm_epsilon
        self.phi = nn.Linear(num_stream * config.hidden_size, (num_stream + 2) * num_stream, bias=False)
        self.branch_alpha = nn.Parameter(torch.empty(3))
        self.branch_beta = nn.Parameter(torch.empty((num_stream + 2) * num_stream))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the sublayer input, post scale, residual mixing and the streams themselves.

        Args:
            x: Flattened residual streams ``[batch, sequence, n * hidden]``.

        Returns:
            ``[b, s, hidden]`` sublayer input, ``[b, s, n]`` post scale, ``[b, s, n, n]``
            residual mixing and ``x``.
        """
        return mhc_pre(x, self.phi.weight, self.branch_alpha, self.branch_beta, self.num_stream,
                       sinkhorn_iters=self.sinkhorn_iterations, norm_eps=self.norm_eps, hc_eps=_HC_EPS)


class Xing40MhcPost(nn.Module):
    """Mix the residual streams and add the scaled sublayer output to each stream.

    The computation is the framework's ``mhc_post`` fused AscendC operator on NPU.
    """

    def __init__(self, config: Xing40Config) -> None:
        """Record the stream count.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.num_stream = config.num_residual_streams

    def forward(self, x: torch.Tensor, residual: torch.Tensor, h_post: torch.Tensor,
                h_res: torch.Tensor) -> torch.Tensor:
        """Return the updated flattened streams ``[batch, sequence, n * hidden]``.

        Args:
            x: Sublayer output ``[batch, sequence, hidden]``.
            residual: Flattened streams before the sublayer.
            h_post: Post scale from :class:`Xing40MhcPre`.
            h_res: Residual mixing from :class:`Xing40MhcPre`.
        """
        return mhc_post(x, residual, h_post, h_res, self.num_stream)


def _hyper_head(config: Xing40Config) -> DeepseekV4HyperHead:
    """Build the learned collapse of the residual streams (MindFormers ``hc_head``).

    Transformers' DeepSeek-V4 head computes the MindFormers ``HyperConnectionHead``; it reads
    the stream count as ``hc_mult`` and normalizes with the mHC epsilon. It takes the streams
    as ``[batch, sequence, n, hidden]``.

    Args:
        config: Model configuration.
    """
    return DeepseekV4HyperHead(SimpleNamespace(
        hc_mult=config.num_residual_streams, hidden_size=config.hidden_size,
        rms_norm_eps=config.mhc_layernorm_epsilon, hc_eps=_HC_EPS))


class Xing40DecoderLayer(nn.Module):
    """Decoder layer whose attention and MLP sublayers are wrapped by mHC."""

    def __init__(self, config: Xing40Config, layer_idx: int, *, sparse: bool) -> None:
        """Build the attention, MLP, norms and the two hyper-connections.

        Args:
            config: Model configuration.
            layer_idx: Index of the layer among the decoder layers and the MTP depths.
            sparse: Use the routed MoE instead of the dense MLP.
        """
        super().__init__()
        self.self_attn = DeepseekV3Attention(config, layer_idx)
        mscale = 0.1 * math.log(getattr(config, "scaling_factor", 1.0)) + 1.0
        self.self_attn.scaling = config.attention_softmax_scale or mscale * mscale / (
            config.qk_nope_head_dim + config.qk_rope_head_dim) ** 0.5
        # Transformers builds the latent norms with the default epsilon; MindFormers uses rms_norm_eps.
        self.self_attn.q_a_layernorm.variance_epsilon = config.rms_norm_eps
        self.self_attn.kv_a_layernorm.variance_epsilon = config.rms_norm_eps
        self.mlp = Xing40MoE(config) if sparse else DeepseekV3MLP(config)
        self.input_layernorm = DeepseekV3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn_hc_pre = Xing40MhcPre(config)
        self.attn_hc_post = Xing40MhcPost(config)
        self.ffn_hc_pre = Xing40MhcPre(config)
        self.ffn_hc_post = Xing40MhcPost(config)

    def forward(self, streams: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Update the flattened residual streams ``[batch, sequence, n * hidden]`` with both sublayers.

        Args:
            streams: Residual streams.
            position_embeddings: FP32 RoPE ``(cos, sin)`` tables.
        """
        layer_input, post, mix, residual = self.attn_hc_pre(streams)
        output = self.self_attn(hidden_states=self.input_layernorm(layer_input),
                                position_embeddings=position_embeddings, attention_mask=None)[0]
        streams = self.attn_hc_post(output, residual, post, mix)
        layer_input, post, mix, residual = self.ffn_hc_pre(streams)
        output = self.mlp(self.post_attention_layernorm(layer_input))
        return self.ffn_hc_post(output, residual, post, mix)


class Xing40Model(nn.Module):
    """Token embedding, decoder layers and the mHC collapse head."""

    def __init__(self, config: Xing40Config) -> None:
        """Build the backbone; the first ``first_k_dense_replace`` layers are dense.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.num_residual_streams = config.num_residual_streams
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            Xing40DecoderLayer(config, index, sparse=index >= config.first_k_dense_replace)
            for index in range(config.num_hidden_layers))
        self.hc_head = _hyper_head(config)
        self.norm = DeepseekV3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Xing40RotaryEmbedding(config)
        # Marks ``layers`` as the repeated block container for FSDP units and activation checkpointing.
        self.gradient_checkpointing = False

    def embed_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up tokens and cast the activation to BF16 at the embedding boundary.

        MindFormers keeps the embedding table in FP32 (``embedding_params_dtype``) but computes
        the rest of the model in BF16 (``compute_dtype``); this is the one cast MindFormers makes
        between the two, unconditionally, with no other supported width. Parameter precision and
        the native embedding backward are unaffected.

        Args:
            input_ids: Token IDs with the batch and sequence dimensions.

        Returns:
            Lookup activations cast to ``torch.bfloat16``.
        """
        return self.embed_tokens(input_ids).to(torch.bfloat16)

    def forward(self, input_ids: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Return the final-normed hidden states the LM head and the first MTP depth read.

        Args:
            input_ids: Token IDs ``[batch, sequence]``.
            position_embeddings: FP32 RoPE ``(cos, sin)`` tables.
        """
        streams = self.embed_inputs(input_ids).repeat(1, 1, self.num_residual_streams)
        for layer in self.layers:
            streams = layer(streams, position_embeddings)
        return self.norm(self.hc_head(streams.unflatten(-1, (self.num_residual_streams, -1))))


class Xing40MTPLayer(nn.Module):
    """One sequential multi-token-prediction depth (MindFormers/Megatron layout).

    It fuses the embeddings of the next tokens with the previous depth's state, runs an
    MoE decoder layer on copied residual streams, collapses them with its own mHC head
    and normalizes the result for the shared LM head and the next depth.
    """

    def __init__(self, config: Xing40Config, layer_idx: int) -> None:
        """Build the fusion norms and projection, the decoder layer, the head and the output norm.

        Args:
            config: Model configuration.
            layer_idx: Index of the depth's decoder layer, after the backbone's layers.
        """
        super().__init__()
        self.num_residual_streams = config.num_residual_streams
        self.enorm = DeepseekV3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hnorm = DeepseekV3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.transformer_layer = Xing40DecoderLayer(config, layer_idx, sparse=True)
        self.hc_head = _hyper_head(config)
        self.final_layernorm = DeepseekV3RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden: torch.Tensor, embeddings: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Return this depth's normalized state.

        Args:
            hidden: Previous normalized state: the backbone output or the previous depth's state.
            embeddings: Embeddings of the tokens shifted to this depth.
            position_embeddings: FP32 RoPE ``(cos, sin)`` tables.
        """
        fused = self.eh_proj(torch.cat((self.enorm(embeddings), self.hnorm(hidden)), dim=-1))
        streams = self.transformer_layer(fused.repeat(1, 1, self.num_residual_streams), position_embeddings)
        return self.final_layernorm(self.hc_head(streams.unflatten(-1, (self.num_residual_streams, -1))))


class Xing40MTP(nn.Module):
    """Container of the multi-token-prediction depths."""

    def __init__(self, config: Xing40Config) -> None:
        """Build ``num_nextn_predict_layers`` depths.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            Xing40MTPLayer(config, config.num_hidden_layers + depth)
            for depth in range(config.num_nextn_predict_layers))
        # Marks ``layers`` as the repeated block container for FSDP units and activation checkpointing.
        self.gradient_checkpointing = False


@dataclass
class Xing40Output(ModelOutput):
    """Named training losses computed by the forward."""

    # ``logits`` stays first and empty so that ModelOutput keeps ``loss`` as one mapping.
    logits: torch.Tensor | None = None
    loss: dict[str, torch.Tensor] | None = None


class Xing40PreTrainedModel(DeepseekV3PreTrainedModel):
    """DeepSeek-V3 weight initialization extended with the MindFormers mHC initialization."""

    config: Xing40Config
    config_class = Xing40Config
    _no_split_modules = ["Xing40DecoderLayer", "Xing40MTPLayer"]
    _keys_to_ignore_on_load_unexpected = None
    _can_record_outputs = None
    # The attention calls the NPU fused kernel on NPU and SDPA elsewhere, whatever implementation is requested.
    _supports_flash_attn = False
    _supports_flex_attn = False
    _supports_attention_backend = False

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize one module; also runs on FSDP shards after meta-device materialization.

        Args:
            module: Module to initialize.
        """
        super()._init_weights(module)
        if isinstance(module, Xing40MhcPre):
            nn.init.normal_(module.phi.weight, mean=0.0, std=1e-4)
            module.branch_alpha.fill_(self.config.mhc_init_gating_factor)
            # Zeroed, not the mHC reference identity-residual shift: every real run loads a
            # converted checkpoint (mHC has no from-scratch reference path).
            module.branch_beta.zero_()
        elif isinstance(module, DeepseekV4HyperHead):
            nn.init.normal_(module.hc_fn, mean=0.0, std=self.config.initializer_range)
            module.hc_base.zero_()
            module.hc_scale.zero_()


class Xing40ForCausalLM(Xing40PreTrainedModel):
    """Causal LM whose forward returns the LM and MTP losses of one ``[batch, sequence]`` training batch."""

    def __init__(self, config: Xing40Config) -> None:
        """Build the backbone, LM head and MTP depths.

        Args:
            config: Model configuration.

        Raises:
            ValueError: If the configuration selects an unsupported variant.
        """
        super().__init__(config)
        if config.hidden_act != "silu":
            raise ValueError("Xing4.0 experts require hidden_act='silu'")
        if config.rope_parameters["rope_type"] != "default" or not config.rope_interleave:
            raise ValueError("Xing4.0 requires default, interleaved RoPE")
        if config.n_group != 1 or config.topk_group != 1:
            raise ValueError("Xing4.0 routes without expert groups; set n_group and topk_group to 1")
        if config.q_lora_rank is None:
            raise ValueError("Xing4.0 requires the low-rank query projection (q_lora_rank)")
        if config.attention_dropout:
            raise ValueError("Xing4.0 requires attention_dropout=0")
        self.model = Xing40Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp = Xing40MTP(config)
        self.post_init()

    @classmethod
    def from_config(cls, config: Xing40Config, **kwargs: object) -> Xing40ForCausalLM:
        """Build from a resolved configuration, as the shared model builder requires.

        Args:
            config: Model configuration.
            **kwargs: Construction options of the shared builder; unused.

        Returns:
            The constructed model; the shared builder initializes or loads its weights.
        """
        del kwargs
        return cls(config)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None,
                shift_labels: torch.Tensor | None = None, position_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None, use_cache: bool = False) -> Xing40Output:
        """Compute the training losses of one batch.

        Args:
            input_ids: Token IDs ``[batch, sequence]``.
            labels: Unused; ``shift_labels`` carries the same targets.
            shift_labels: Next-token targets aligned with ``input_ids``; negative targets are ignored.
            position_ids: RoPE positions; defaults to ``0..sequence-1``.
            attention_mask: Must be None: attention is causal over the whole sequence.
            use_cache: Must be False.

        Returns:
            ``loss`` maps ``foundation_loss`` (LM) and ``foundation_loss/mtp`` to 0-d losses;
            the router balance loss reaches the gradients through the routing weights.

        Raises:
            ValueError: If the inputs do not describe a nonempty batch of causal training sequences.
        """
        del labels
        if attention_mask is not None or use_cache:
            raise ValueError("Xing4.0 trains causal sequences without attention masks or KV cache")
        if shift_labels is None or input_ids.ndim != 2 or min(input_ids.shape) < 1 \
                or shift_labels.shape != input_ids.shape:
            raise ValueError("Xing4.0 requires nonempty [batch, sequence] input_ids with matching shift_labels")
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        # The empty FP32 tensor selects the device and the FP32 dtype of the RoPE tables.
        position_embeddings = self.model.rotary_emb(position_ids.new_empty(0, dtype=torch.float32), position_ids)
        hidden = self.model(input_ids, position_embeddings)
        targets = shift_labels.masked_fill(shift_labels < 0, IGNORE_INDEX)
        losses = {"foundation_loss": self._cross_entropy(hidden, targets)}
        if self.mtp.layers:
            losses["foundation_loss/mtp"] = self._mtp_loss(hidden, input_ids, targets, position_embeddings)
        return Xing40Output(loss=losses)

    def _mtp_loss(self, hidden: torch.Tensor, input_ids: torch.Tensor, targets: torch.Tensor,
                  position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Return ``mtp_loss_factor / D * sum_k CE_k``, where depth ``k`` predicts ``k`` tokens further.

        Inputs and targets shift left by one token per depth without wrapping.
        """
        total = torch.zeros((), device=hidden.device, dtype=torch.float32)
        for layer in self.mtp.layers:
            input_ids = F.pad(input_ids[:, 1:], (0, 1), value=0)
            targets = F.pad(targets[:, 1:], (0, 1), value=IGNORE_INDEX)
            hidden = layer(hidden, self.model.embed_inputs(input_ids), position_embeddings)
            total = total + self._cross_entropy(hidden, targets)
        return total * (self.config.mtp_loss_factor / len(self.mtp.layers))

    def _cross_entropy(self, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return the mean LM-head cross entropy without materializing the sequence's logits."""
        weight = self.lm_head.weight
        if getattr(weight, "placements", None) is None and weight.shape[0] == self.config.vocab_size:
            loss_sum = chunked_cross_entropy(hidden, targets, weight,
                                             chunk_size=self.config.chunk_loss_size, ignore_index=IGNORE_INDEX)
        else:
            loss_sum = projected_cross_entropy_sum(
                hidden, targets, self.lm_head, vocab_size=self.config.vocab_size,
                chunk_size=self.config.chunk_loss_size, ignore_index=IGNORE_INDEX,
            )
        return loss_sum / targets.ne(IGNORE_INDEX).sum().clamp_min(1)
