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
"""Teacher-forced LM/MTP training over the official Xing4 backbone."""
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers.masking_utils import create_causal_mask
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.utils import ModelOutput

from hyper_parallel.components.functional.aux_loss import aux_loss_auto_scale
from hyper_parallel.components.functional.mhc_post import mhc_post
from hyper_parallel.components.losses.chunked_cross_entropy import chunked_cross_entropy
from hyper_parallel.components.modules.mtp import MTPLayer, build_mtp
from hyper_parallel.core.activation_memory.recompute_state import is_recomputing
from hyper_parallel.core.multicore.modules._precision import preserve_fp32_parameters
from hyper_parallel.core.utils.communication import differentiable_all_reduce
from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.models.materialization import register_rebuildable_buffer
from hyper_parallel.models.xing_4.transformer.configuration_xing4_0 import Xing4_0Config
from hyper_parallel.models.xing_4.transformer.modeling_xing4_0 import (
    Xing4_0DecoderLayer, Xing4_0ForCausalLM, Xing4_0HyperConnection,
    Xing4_0MLP, Xing4_0Model, Xing4_0MoE, Xing4_0PreTrainedModel, Xing4_0RMSNorm, Xing4_0TopkRouter,
)


class Xing4Router(Xing4_0TopkRouter):
    """HF learned selection with the recipe's training state and objectives."""

    fp32_buffer_names = ("e_score_correction_bias",)

    def __init__(self, config: Any, source: nn.Module) -> None:
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
        self.enable_expert_bias = config.moe_router_enable_expert_bias
        self.expert_bias_update_rate = config.moe_router_bias_update_rate

    def _apply(self, fn: Callable, recurse: bool = True) -> "Xing4Router":
        """Keep load-balancing bias updates in FP32 during model dtype conversion."""
        return super()._apply(preserve_fp32_parameters(self, fn, self.fp32_buffer_names), recurse=recurse)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route real tokens once; safe-token padding consumes neither jitter nor counts."""
        real_hidden = hidden_states[:, self.padding:]
        flat = real_hidden.reshape(-1, hidden_states.shape[-1]).float()
        if self.training and self.input_jitter_eps:
            flat = flat * (1.0 + (2.0 * torch.rand_like(flat) - 1.0) * self.input_jitter_eps)
        logits, weights, indices = super().forward(flat)
        scores = logits.sigmoid()
        weights = self._attach_balance_loss(weights, scores, batch_size=hidden_states.shape[0])
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

    def _attach_balance_loss(self, weights: torch.Tensor, scores: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Inject ``coeff * E * mean_b(sum_i(f_bi * P_bi))`` through the routing weights.

        ``f_bi`` counts unbiased top-k affinities per sequence, independently of the expert-load bias. ``P_bi`` averages
        the normalized affinities within each sequence. The sequence objectives are averaged over
        the batch.

        Args:
            weights: Routing weights of the real tokens.
            scores: Router affinities of every expert.
            batch_size: Number of independent sequences before flattening the tokens.
        """
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


class Xing4MLP(Xing4_0MLP):
    """Retain HF projections with an optional training activation policy."""

    def __init__(self, source: nn.Module) -> None:
        """Adopt the HF parameters without creating a second weight owner."""
        nn.Module.__init__(self)  # pylint: disable=non-parent-init-called
        self.config = source.config
        self.hidden_size = source.hidden_size
        self.intermediate_size = source.intermediate_size
        self.gate_proj, self.up_proj, self.down_proj = source.gate_proj, source.up_proj, source.down_proj
        # Only the optional clipped/compute-dtype policy needs the extended SwiGLU API.
        from hyper_parallel.components.functional.activation_swiglu import (  # pylint: disable=import-outside-toplevel
            swiglu_from_projections,
        )
        self.act_fn = source.act_fn
        self._swiglu = swiglu_from_projections
        self.limit = getattr(self.config, "swiglu_limit", None)
        dtype_name = getattr(self.config, "swiglu_compute_dtype", None)
        self.activation_compute_dtype = getattr(torch, dtype_name) if dtype_name is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the configured shared SwiGLU policy to the HF projections."""
        activation = self._swiglu(
            self.gate_proj(x), self.up_proj(x), clamp_limit=self.limit,
            compute_dtype=self.activation_compute_dtype,
            use_fused=bool(getattr(self.config, "use_fused_swiglu", True)))
        return self.down_proj(activation)


def adapt_mlp(mlp: nn.Module) -> nn.Module:
    """Apply the training activation policy when requested."""
    config = mlp.config
    if getattr(config, "swiglu_limit", None) is None and getattr(config, "swiglu_compute_dtype", None) is None:
        return mlp
    return Xing4MLP(mlp)


class Xing4MoE(Xing4_0MoE):
    """Reuse the HF MoE facade and experts with explicit training extensions."""

    def __init__(self, config: Any, source: nn.Module) -> None:
        """Retain HF experts and add the training router and EP execution boundary."""
        nn.Module.__init__(self)  # pylint: disable=non-parent-init-called
        self.config = config
        self.experts = nn.ModuleList(adapt_mlp(expert) for expert in source.experts)
        self.gate = Xing4Router(config, source.gate)
        self.shared_experts = adapt_mlp(source.shared_experts)
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.padding = config.n_routed_experts if config.use_pad_tokens else 0
        self.ep_executor = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Reuse HF expert computation, with optional padding and EP dispatch."""
        padded = hidden_states
        if self.padding:
            padded = torch.cat((padded.new_zeros(padded.shape[0], self.padding, padded.shape[-1]), padded), dim=1)
        if self.ep_executor is not None:
            routed = self.ep_executor(self, padded)
            return routed[:, self.padding:] + self.shared_experts(hidden_states)
        return super().forward(padded)[:, self.padding:]

    def moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor,
            topk_weights: torch.Tensor) -> torch.Tensor:
        """Select the HF per-expert or declared grouped-expert implementation."""
        if isinstance(self.experts, nn.ModuleList):
            return super().moe(hidden_states, topk_indices, topk_weights)
        return self.experts(hidden_states, topk_indices, topk_weights)

    def route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose the same once-only router contract to the expert-parallel executor."""
        _, weights, indices = self.gate(hidden)
        return indices, weights


class Xing4DecoderLayer(Xing4_0DecoderLayer):
    """Keep HF submodules and expose its inline residual mixing for replacement."""

    def __init__(self, config: Any, layer_idx: int, source: nn.Module | None = None) -> None:
        """Retain HF attention, norms and feed-forward weights."""
        nn.Module.__init__(self)
        source = Xing4_0DecoderLayer(config, layer_idx) if source is None else source
        self.hidden_size = source.hidden_size
        self.self_attn = source.self_attn
        self.mlp = (Xing4MoE(config, source.mlp) if isinstance(source.mlp, Xing4_0MoE)
                    else adapt_mlp(source.mlp))
        self.input_layernorm = source.input_layernorm
        self.post_attention_layernorm = source.post_attention_layernorm
        self.attn_hc, self.ffn_hc = source.attn_hc, source.ffn_hc
        self.attn_hc.input_norm.eps = self.ffn_hc.input_norm.eps = config.mhc_layernorm_epsilon
        # The training recipe applies its norm epsilon to both MLA latent projections.
        self.self_attn.q_a_layernorm.variance_epsilon = config.rms_norm_eps
        self.self_attn.kv_a_layernorm.variance_epsilon = config.rms_norm_eps
        if config.attention_softmax_scale is not None:
            self.self_attn.scaling = config.attention_softmax_scale

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None,
                position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, **kwargs: Any) -> torch.Tensor:
        """Call the official blocks through replaceable pre/post mHC operations."""
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
def initialize(owner: Any, module: nn.Module) -> None:
    """Use guarded initializers to preserve tensors already loaded from a checkpoint."""
    if isinstance(module, Xing4_0HyperConnection):
        nn.init.normal_(module.hc_fn, mean=0.0, std=1e-4)
        nn.init.constant_(module.hc_scale, owner.config.mhc_init_gating_factor)
        nn.init.zeros_(module.hc_base)
    elif isinstance(module, Xing4Router):
        nn.init.normal_(module.weight, mean=0.0, std=owner.config.initializer_range)
        nn.init.zeros_(module._token_counts)
        nn.init.zeros_(module.e_score_correction_bias)
    else:
        Xing4_0PreTrainedModel._init_weights(owner, module)


class Xing4Model(Xing4_0Model):
    """Use the unchanged HF backbone forward with training-aware decoder blocks."""

    config_class = Xing4_0Config
    _supports_attention_backend = False

    def __init__(self, config: Any) -> None:
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

    def _init_weights(self, module: nn.Module) -> None:
        initialize(self, module)

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
        super().__init__(config, layer_idx)
        if not isinstance(self.mlp, Xing4MoE):
            raise ValueError("Xing4 MTP requires an MoE decoder at its native layer index")
        self.hc_mult = config.hc_mult

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Expand and collapse streams, leaving fusion and norms to MTPLayer."""
        streams = hidden_states.unsqueeze(2).expand(-1, -1, self.hc_mult, -1).contiguous()
        return super().forward(streams, **kwargs).mean(dim=2)


class Xing4ForCausalLM(Xing4_0ForCausalLM):
    """Add teacher-forced LM/MTP losses to the official causal language model."""

    config_class = Xing4_0Config
    _no_split_modules = ["Xing4DecoderLayer", "MTPLayer"]
    _keys_to_ignore_on_load_unexpected = None
    _can_record_outputs = None
    _supports_flash_attn = False
    _supports_flex_attn = False
    _supports_attention_backend = False

    def __init__(self, config: Any) -> None:
        Xing4_0PreTrainedModel.__init__(self, config)
        if config.hidden_act != "silu" or config.q_lora_rank is None:
            raise ValueError("Xing4 training requires SiLU and low-rank query projection")
        if config.rope_parameters["rope_type"] not in ("default", "yarn") or not config.rope_interleave:
            raise ValueError("Xing4 training requires default or YaRN interleaved RoPE")
        if getattr(config, "enable_cmhc", False) or getattr(config, "enable_hc_head", False):
            raise ValueError("This Xing4 backbone uses the official mHC streams and mean collapse")
        if not getattr(config, "mtp_enable_hyper_connections", True):
            raise ValueError("Xing4 MTP uses the same mHC decoder as the backbone")
        if config.attention_dropout:
            raise ValueError("Xing4 training requires attention_dropout=0")
        compute_dtype = getattr(config, "compute_dtype", None)
        self.embedding_compute_dtype = (getattr(torch, compute_dtype, compute_dtype)
                                        if isinstance(compute_dtype, str) else compute_dtype)
        if self.embedding_compute_dtype not in (None, torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"Unsupported Xing4 compute_dtype: {compute_dtype!r}")
        self.model = Xing4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp = build_mtp(config.num_nextn_predict_layers, layer_factory=lambda depth: MTPLayer(
            config.hidden_size, Xing4MTPBlock(config, config.num_hidden_layers + depth),
            norm_factory=partial(Xing4_0RMSNorm, eps=config.rms_norm_eps)))
        if self.mtp is not None:
            self.mtp.gradient_checkpointing = False
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        initialize(self, module)

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
        if shift_labels is None or input_ids.ndim != 2 or min(input_ids.shape) < 1 \
                or shift_labels.shape != input_ids.shape:
            raise ValueError("Xing4 requires nonempty [batch, sequence] input_ids with matching shift_labels")
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
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
            chunk_size = getattr(self.config, "chunk_loss_size", None)
            if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
                raise ValueError("Full-vocabulary Xing4 cross entropy requires a positive integer chunk_loss_size")
            loss_sum = chunked_cross_entropy(hidden, targets, weight,
                                             chunk_size=chunk_size, ignore_index=IGNORE_INDEX)
        else:
            loss_sum = self.loss_function(self.lm_head(hidden), None, self.config.vocab_size, num_items_in_batch=1,
                                          ignore_index=IGNORE_INDEX, shift_labels=targets)
        return self.model.mean_loss(loss_sum, count)
