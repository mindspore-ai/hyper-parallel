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
"""Complete HF-derived causal model, prediction depths and reference training semantics."""

# This model uses the Torch/HF runtime, like the existing Trainer model families.
# pylint: disable=forbidden-backend-import
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch_npu
from torch import nn
from torch.nn import functional as F
from transformers import DeepseekV32ForCausalLM
from transformers.utils import ModelOutput
from transformers.models.deepseek_v32.modeling_deepseek_v32 import (
    DeepseekV32Attention, DeepseekV32DecoderLayer, DeepseekV32Experts, DeepseekV32MLP,
    DeepseekV32MoE, DeepseekV32Model, DeepseekV32PreTrainedModel, DeepseekV32RMSNorm,
    DeepseekV32RotaryEmbedding, DeepseekV32TopkRouter,
)

from hyper_parallel.components.modules.mtp import DeepseekV3MTP
from hyper_parallel.components.modules.mla_attention import MLAAttention
from hyper_parallel.components.functional.npu_fusion_attention import (
    _attention_options, _prepare_attention_inputs, resolve_packed_sequence_lengths,
)
from hyper_parallel.components.functional.npu_grouped_swiglu import npu_grouped_swiglu
from hyper_parallel.models.replacement import module_replacement
from hyper_parallel.distributed.expert_parallel.routing import MOE_ROUTER_ADAPTERS
from hyper_parallel.models.jt_deepseek_v3.configuration_jt_deepseek_v3 import JTDeepseekV3Config



@dataclass
class JTDeepseekV3Output(ModelOutput):
    """Model output carrying named JT losses without HF first-field remapping."""

    # Keep logits first and optional so ModelOutput preserves the named loss
    # mapping instead of interpreting it as a field iterator.
    logits: torch.Tensor | None = None
    loss: dict[str, torch.Tensor] | None = None



class JTDeepseekV3Experts(DeepseekV32Experts):
    """Keep HF packed expert tensors and expose Hyper's grouped-compute hook."""

    def forward_expert_major(self, inputs: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Run grouped SwiGLU without an adapter-imposed dtype conversion."""
        return npu_grouped_swiglu(inputs, self.gate_up_proj, self.down_proj, counts)


class JTDeepseekV3RotaryEmbedding(nn.Module):
    """Apply rotary products without an explicit FP32 conversion."""

    def forward(self, values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotate values using the tensors' native arithmetic."""
        ordered = torch.cat((values[..., ::2], values[..., 1::2]), dim=-1)
        first, second = ordered.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return ordered * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)


def observed_fusion_attention(module: nn.Module, query: torch.Tensor, key: torch.Tensor,
                              value: torch.Tensor, attention_mask: Any, dropout: float = 0.0,
                              scaling: float | None = None, **kwargs: Any) -> tuple[torch.Tensor, None]:
    """Retain Hyper's NPU attention preparation and collect per-head QK maxima.

    Reuse the upstream packed-length, option and mask preparation helpers.
    The observed kernel outputs also supply the QK clipping statistics.

    Args:
        module: Attention module owning kernel settings.
        query: Projected query states.
        key: Projected key states.
        value: Projected value states.
        attention_mask: Optional attention mask.
        dropout: Attention dropout probability.
        scaling: Attention score scaling factor.
    """
    batch_size, _, query_length, head_dim = query.shape
    query_lengths, key_lengths = resolve_packed_sequence_lengths(
        kwargs, batch_size * query_length, key.shape[0] * key.shape[2])
    pre_tokens, next_tokens, sparse_mode, window, causal = _attention_options(module, kwargs)
    query, key, value, layout, mask, sparse_mode = _prepare_attention_inputs(
        query, key, value, attention_mask, is_packed=query_lengths is not None,
        is_causal=causal, sliding_window=window, sparse_mode=sparse_mode)
    result = torch_npu.npu_fusion_attention(
        query, key, value, query.shape[1], layout,
        pse=None, padding_mask=None, atten_mask=mask,
        scale=head_dim**-0.5 if scaling is None else scaling,
        pre_tockens=pre_tokens, next_tockens=next_tokens,
        keep_prob=1.0 - dropout, inner_precise=0, sparse_mode=sparse_mode,
        actual_seq_qlen=query_lengths, actual_seq_kvlen=key_lengths,
        softmax_layout="TND" if layout == "TND" else "")
    with torch.no_grad():
        # The default TND kernel statistics use NTD storage; request TND explicitly above.
        maximum = result[1].amax(dim=(0, 2) if layout == "TND" else (0, 2, 3))
        if module.max_logits_val is None:
            module.max_logits_val = torch.zeros_like(maximum)
        module.max_logits_val.copy_(torch.maximum(module.max_logits_val, maximum))
    if query_lengths is not None:
        return result[0].reshape(batch_size, query_length, *result[0].shape[1:]), None
    return result[0].transpose(1, 2), None


@module_replacement
class JTDeepseekV3MLAAttention(MLAAttention):
    """Reuse Hyper MLA parameters and projections with reference SP and RoPE boundaries."""

    def __init__(self, *, module: nn.Module, module_fqn: str = "", context: Any = None) -> None:
        """Initialize the configured components and retained parameter state.

        Args:
            module: Module.
            module_fqn: Module fqn.
            context: Context.
        """
        super().__init__(module=module, module_fqn=module_fqn, context=context)
        if not self.linear_qkv.weight.is_meta:
            with torch.no_grad():
                self.linear_qkv.weight.copy_(torch.cat((module.q_a_proj.weight, module.kv_a_proj_with_mqa.weight)))
        self.explicit_rotary = JTDeepseekV3RotaryEmbedding()
        self.key_rope_gather = nn.Identity()
        self.register_buffer("max_logits_val", None, persistent=False)
        self.attention_interface = observed_fusion_attention

    def project_latent_inputs(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return normalized Q/KV latents and the shared key at their existing SP boundaries.

        Args:
            hidden_states: Input activations in the layout selected by the model recipe.
        """
        latent_states = self.linear_qkv(hidden_states)
        query_local, kv_local = latent_states.split(
            (self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim), dim=-1
        )
        kv_local, rope_local = kv_local.split((self.kv_lora_rank, self.qk_rope_head_dim), dim=-1)
        query_latent = self.q_a_layernorm(query_local)
        kv_latent = self.kv_a_layernorm(kv_local)
        key_rope = self.key_rope_gather(rope_local)
        return query_latent, kv_latent, key_rope

    def _project_attention_inputs(self, hidden_states: torch.Tensor, position_embeddings: Any,
                                  past_key_values: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project shared MLA children with the JT SP and rotary boundaries.

        Separate down projections retain the two input-gradient GEMMs. The
        current upstream MLA has no split latent-projection extension point.
        """
        if past_key_values is not None or position_embeddings is None:
            raise ValueError("Reference MLA requires explicit positions and no KV cache")
        query_latent, kv_latent, key_rope = self.project_latent_inputs(hidden_states)
        batch, sequence = query_latent.shape[:2]
        query = self.q_b_proj(query_latent).reshape(batch, sequence, self.num_heads, self.qk_head_dim)
        query_pass, query_rope = query.split((self.qk_nope_head_dim, self.qk_rope_head_dim), dim=-1)
        kv_latent = kv_latent.reshape(batch, 1, sequence, self.kv_lora_rank)
        kv_states = self.kv_b_proj(kv_latent).view(
            batch, sequence, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        key_pass, value = kv_states.split((self.qk_nope_head_dim, self.v_head_dim), dim=-1)
        cos, sin = position_embeddings
        query = torch.cat((query_pass.transpose(1, 2),
                           self.explicit_rotary(query_rope.transpose(1, 2), cos, sin)), dim=-1)
        key_rope = self.explicit_rotary(key_rope.unsqueeze(1), cos, sin)
        key = torch.cat((key_pass, key_rope.expand(-1, self.num_heads, -1, -1)), dim=-1)
        return query, key, value

    def forward(self, hidden_states: torch.Tensor, position_embeddings: Any = None,
                attention_mask: Any = None, past_key_values: Any = None,
                actual_seq_len: Any = None, **kwargs: Any) -> tuple[torch.Tensor, Any]:
        """Use global projected sequence length with Hyper's MLA children and TP output.

        Args:
            hidden_states: Input hidden states.
            position_embeddings: Explicit rotary frequencies.
            attention_mask: Optional attention mask.
            past_key_values: Unsupported cached decoding state.
            actual_seq_len: Packed sequence lengths.
        """
        query, key, value = self._project_attention_inputs(hidden_states, position_embeddings, past_key_values)
        output, weights = self.attention_interface(
            self, query, key, value, attention_mask,
            dropout=self.attention_dropout if self.training else 0.0, scaling=self.scaling,
            sliding_window=self.sliding_window, actual_seq_len=actual_seq_len, **kwargs)
        output = output.reshape(query.shape[0], query.shape[2], -1).contiguous()
        return self.o_proj(output), weights


class JTDeepseekV3Attention(DeepseekV32Attention):
    """Full causal MLA semantics with HF projection names and no DSA indexer."""

    def __init__(self, config: Any, layer_idx: int) -> None:
        """Construct only the projections used by the configured causal model."""
        nn.Module.__init__(self)
        self.config, self.layer_idx = config, layer_idx
        self.num_heads = config.num_attention_heads
        self.q_lora_rank, self.kv_lora_rank = config.q_lora_rank, config.kv_lora_rank
        self.qk_rope_head_dim, self.qk_nope_head_dim = config.qk_rope_head_dim, config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        self.scaling = self.qk_head_dim ** -0.5
        self.attention_dropout, self.is_causal, self.sliding_window = config.attention_dropout, True, None
        self.q_a_proj = nn.Linear(config.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = DeepseekV32RMSNorm(self.q_lora_rank, config.rms_norm_eps)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(config.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = DeepseekV32RMSNorm(self.kv_lora_rank, config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False)
        self.explicit_rotary = JTDeepseekV3RotaryEmbedding()
        self.register_buffer("max_logits_val", None, persistent=False)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: Any = None,
                past_key_values: Any = None, attention_mask: Any = None, **kwargs: Any) -> tuple:
        """Run the complete causal attention before optional high-performance replacement.

        Args:
            hidden_states: Input hidden states.
            position_embeddings: Explicit rotary frequencies.
            past_key_values: Unsupported cached decoding state.
            attention_mask: Optional attention mask.
        """
        if past_key_values is not None or position_embeddings is None:
            raise ValueError("Reference attention requires explicit positions and no KV cache")
        batch, sequence = hidden_states.shape[:2]
        query = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        query = query.reshape(batch, sequence, self.num_heads, self.qk_head_dim).transpose(1, 2)
        kv, rope = self.kv_a_proj_with_mqa(hidden_states).split((self.kv_lora_rank, self.qk_rope_head_dim), dim=-1)
        kv = self.kv_b_proj(self.kv_a_layernorm(kv)).reshape(
            batch, sequence, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        key, value = kv.split((self.qk_nope_head_dim, self.v_head_dim), dim=-1)
        cos, sin = position_embeddings
        query = torch.cat((query[..., :self.qk_nope_head_dim],
                           self.explicit_rotary(query[..., self.qk_nope_head_dim:], cos, sin)), dim=-1)
        rope = self.explicit_rotary(rope.unsqueeze(1), cos, sin).expand(-1, self.num_heads, -1, -1)
        key = torch.cat((key, rope), dim=-1)
        if hidden_states.device.type == "npu":
            output, _ = observed_fusion_attention(self, query, key, value, attention_mask,
                                                  scaling=self.scaling, **kwargs)
        else:
            lengths, _ = resolve_packed_sequence_lengths(kwargs, batch * sequence, batch * sequence)
            if lengths is not None:
                if batch != 1 or attention_mask is not None:
                    raise ValueError("JT packed attention requires batch one and no explicit mask")
                starts = (0, *lengths[:-1])
                output = torch.cat([
                    # Pylint cannot infer this Torch C-extension callable.
                    F.scaled_dot_product_attention(  # pylint: disable=not-callable
                        query[:, :, begin:end], key[:, :, begin:end], value[:, :, begin:end],
                        is_causal=True, scale=self.scaling)
                    for begin, end in zip(starts, lengths)
                ], dim=2)
            else:
                output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask,
                                                        is_causal=attention_mask is None, scale=self.scaling)
            output = output.transpose(1, 2)
        return self.o_proj(output.reshape(batch, sequence, -1)), None




class _ModelParallelMean(torch.autograd.Function):
    """Replicate one global mean while differentiating each local contribution once."""

    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, group: Any) -> torch.Tensor:
        """Average equally weighted contributions across the model-parallel group."""
        ctx.world_size = dist.get_world_size(group)
        result = value.clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group)
        return result / ctx.world_size

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Scale the local derivative without summing identical output replicas."""
        return gradient / ctx.world_size, None


def _model_parallel_mean(value: torch.Tensor, group: Any = None) -> torch.Tensor:
    """Reduce a partitioned objective; absent an explicit group, keep it local.

    Contributions must be equally weighted, with the same upstream derivative
    on all ranks. This does not implement DDP averaging or independently
    consumed all-reduce outputs; uneven token partitions need explicit weights.
    """
    return value if group is None else _ModelParallelMean.apply(value, group)


class JTDeepseekV3Gate(DeepseekV32TopkRouter):
    """Expose HF gate logits to the public router and the model's auxiliary loss."""

    def __init__(self, config: Any) -> None:
        """Keep the HF parameters and bias without duplicating its top-k computation."""
        super().__init__(config)
        self.router_logits = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project router inputs using their native dtype."""
        self.router_logits = F.linear(hidden_states.reshape(-1, self.hidden_dim), self.weight)
        return self.router_logits


class JTDeepseekV3MoE(DeepseekV32MoE):
    """Own routing, balancing and combine semantics independently of EP binding."""

    def __init__(self, config: Any, *, is_mtp: bool = False) -> None:
        """Construct all expert branches and the standalone routing contract."""
        nn.Module.__init__(self)
        self.config = config
        self.reference_is_mtp = is_mtp
        self.padding = config.n_routed_experts if config.use_pad_tokens else 0
        self.experts = JTDeepseekV3Experts(config)
        self.gate = JTDeepseekV3Gate(config)
        self.shared_experts = DeepseekV32MLP(
            config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts)
        self.ep_group, self.ep_world = None, 1
        self.ep_compute = self.local_routed_forward
        self.auxiliary_loss = None
        self.expert_load = None

    def route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select experts and compute the auxiliary sequence-balancing loss."""
        padding = self.padding
        config = self.config
        hidden = hidden[:, padding:]
        indices, selected = MOE_ROUTER_ADAPTERS["deepseekv3"](self, hidden)
        scores = self.gate.router_logits.sigmoid()
        self.gate.router_logits = None
        self.expert_load, self.auxiliary_loss = self.routing_statistics(indices, scores)
        if padding:
            pad_ids = torch.arange(padding * config.num_experts_per_tok, device=indices.device)
            pad_ids = pad_ids.reshape(padding, config.num_experts_per_tok) % padding
            indices = torch.cat((pad_ids, indices))
            selected = torch.cat((selected.new_zeros(padding, selected.shape[-1]), selected))
        return indices, selected

    def routing_statistics(self, indices: torch.Tensor, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing frequency and auxiliary loss independently of token dispatch.

        Args:
            indices: Selected expert indices for non-padding tokens.
            scores: Router probabilities before per-token normalization.
        """
        group, world, config = self.ep_group, self.ep_world, self.config
        frequency = torch.bincount(indices.flatten(), minlength=config.n_routed_experts)
        frequency = frequency / indices.numel()
        if group is not None:
            dist.all_reduce(frequency, group=group)
        frequency = frequency / world
        normalized = scores / (scores.sum(-1, keepdim=True) + 1e-20)
        auxiliary_loss = (normalized.mean(0) * frequency).sum() * scores.shape[-1] * config.moe_aux_loss_coeff
        auxiliary_loss = _model_parallel_mean(auxiliary_loss, group)
        return frequency.detach(), auxiliary_loss

    def local_routed_forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Execute routed experts; routing weights are cast to the activation dtype as in EP aggregation."""
        indices, probabilities = self.route(hidden)
        flat = hidden.reshape(-1, hidden.shape[-1])
        outputs = flat.new_zeros(indices.shape[0], indices.shape[1], flat.shape[-1])
        for expert in range(self.experts.num_experts):
            tokens, slots = torch.where(indices == expert)
            pair = F.linear(flat[tokens], self.experts.gate_up_proj[expert])
            gate, up = pair.chunk(2, dim=-1)
            values = F.silu(gate) * up
            values = F.linear(values, self.experts.down_proj[expert])
            outputs = outputs.index_put((tokens, slots), values)
        return (outputs * probabilities.to(outputs.dtype).unsqueeze(-1)).sum(1).reshape(hidden.shape)

    @staticmethod
    def combine_routed(owner: Any, hidden: torch.Tensor, routed: torch.Tensor) -> torch.Tensor:
        """Expose the routed branch before model-owned shared-expert composition.

        Args:
            owner: MoE module owning the execution.
            hidden: Hidden states.
            routed: Routed expert outputs.
        """
        del owner, hidden
        return routed

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run routed and shared experts using native tensor dtypes."""
        hidden = hidden_states
        if self.padding:
            hidden = torch.cat((hidden.new_zeros(1, self.padding, hidden.shape[-1]), hidden), dim=1)
        routed = self.ep_compute(hidden)
        return routed[:, self.padding:] + self.shared_experts(hidden_states)


class JTDeepseekV3Decoder(DeepseekV32DecoderLayer):
    """Preserve HF submodules while matching FP32 residual accumulation boundaries."""

    def __init__(self, config: Any, layer_idx: int, *, is_mtp: bool = False) -> None:
        """Construct the complete specialized decoder without a later semantic patch."""
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = JTDeepseekV3Attention(config, layer_idx)
        self.mlp = (JTDeepseekV3MoE(config, is_mtp=is_mtp)
                    if config.mlp_layer_types[layer_idx] == "sparse" else DeepseekV32MLP(config))
        self.input_layernorm = DeepseekV32RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV32RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Any = None,
                position_ids: Any = None, past_key_values: Any = None, use_cache: bool = False,
                position_embeddings: Any = None, **kwargs: Any) -> torch.Tensor:
        """Run decoder children without explicit dtype conversions."""
        if past_key_values is not None or use_cache:
            raise ValueError("DeepSeek V3.2 JT does not support cached decoding")
        residual = hidden_states
        branch, _ = self.self_attn(
            self.input_layernorm(residual),
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + branch
        residual = hidden_states
        branch = self.mlp(self.post_attention_layernorm(residual))
        return residual + branch


class JTDeepseekV3Model(DeepseekV32Model):
    """Construct specialized decoders directly while retaining the HF model contract."""

    def __init__(self, config: Any) -> None:
        """Build HF embeddings and positional state around complete specialized layers."""
        DeepseekV32PreTrainedModel.__init__(self, config)
        self.padding_idx, self.vocab_size = config.pad_token_id, config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([JTDeepseekV3Decoder(config, index) for index in range(config.num_hidden_layers)])
        self.norm = DeepseekV32RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = DeepseekV32RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()


class JTDeepseekV3ForCausalLM(DeepseekV32ForCausalLM):
    """Reuse HF construction and children; override the reference training orchestration."""

    config_class = JTDeepseekV3Config

    def __init__(self, config: Any) -> None:
        """Construct the HF skeleton with complete, unconditional JT adapters.

        Args:
            config: HF configuration carrying the explicit JT contract.
        """
        DeepseekV32PreTrainedModel.__init__(self, config)
        if config.hidden_act != "silu":
            raise ValueError("JT fused experts require hidden_act='silu'")
        if config.n_group != 1 or config.topk_group != 1:
            raise ValueError("JT routing currently requires n_group=topk_group=1")
        if config.rope_parameters["rope_type"] != "default":
            raise ValueError("JT rotary computation currently requires rope_type='default'")
        self.model = JTDeepseekV3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        mtp_config = copy.deepcopy(config)
        depth = config.num_nextn_predict_layers
        mtp_config.mlp_layer_types = ["sparse"] * depth
        self.mtp = DeepseekV3MTP(
            hidden_size=config.hidden_size, num_layers=depth,
            decoder_factory=lambda index: JTDeepseekV3Decoder(mtp_config, index, is_mtp=True),
            norm_factory=lambda size: DeepseekV32RMSNorm(size, eps=config.rms_norm_eps),
            output_norm_factory=lambda size: DeepseekV32RMSNorm(size, eps=config.rms_norm_eps),
        )
        self.loss_group = None
        # Ranks holding the same attention heads (DP+CP); the JT builder sets it for QK clipping.
        self.qk_clip_group = None
        self.post_init()

    def forward(self, input_ids: torch.Tensor, shift_labels: torch.Tensor | None = None, *,
                labels: torch.Tensor | None = None, position_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None, use_cache: bool = False,
                actual_seq_len: tuple[int, ...] | None = None, sequence_start: int = 0) -> JTDeepseekV3Output:
        """Use the same JT semantics for evaluation and Trainer backward.

        Args:
            input_ids: Unmodified token IDs.
            shift_labels: Already-shifted targets from the public text batch; masked targets are negative.
            labels: Public batch bookkeeping field; shift_labels owns supervision.
            position_ids: Public sequence positions used to construct RoPE.
            attention_mask: Must be None; cumulative lengths define independent causal documents.
            actual_seq_len: Global cumulative document ends, without a leading zero.
            sequence_start: Offset of these local tokens within the packed sequence.
            use_cache: Whether cached decoding is requested.
        """
        del labels
        if attention_mask is not None:
            raise ValueError("JT requires attention_mask=None for its full causal sequence")
        if shift_labels is None:
            raise ValueError("JT training requires explicit shift_labels")
        if use_cache:
            raise ValueError("JT does not support cached decoding")
        # Same rule as the shared text batch's loss mask; JT data folds its 0/1 mask into the labels.
        losses = self.compute_jt_losses(input_ids, shift_labels, shift_labels >= 0, position_ids=position_ids,
                                        actual_seq_len=actual_seq_len, sequence_start=sequence_start)
        # ``<token-domain>_loss[/<name>]`` keys: every JT objective is weighted by foundation tokens.
        return JTDeepseekV3Output(loss={
            "foundation_loss/lm": losses["lm_loss"],
            "foundation_loss/mtp": losses["mtp_loss"],
            "foundation_loss/aux": losses["aux_loss"],
        })

    def _token_loss(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean causal-LM loss from the framework-selected ``loss_function`` over targets with a nonzero mask."""
        valid = (mask != 0) & (labels >= 0)
        loss = self.loss_function(logits=logits, labels=None, vocab_size=self.config.vocab_size,
                                  shift_labels=labels.masked_fill(~valid, -100),
                                  num_items_in_batch=valid.sum().clamp_min(1))
        # causal_lm_loss_parallel returns shape [1]; the Trainer stacks named losses, so keep each one 0-d.
        return loss.reshape(())

    def compute_jt_losses(self, input_ids: torch.Tensor, labels: torch.Tensor,
                         loss_mask: torch.Tensor, *, position_ids: torch.Tensor | None = None,
                         actual_seq_len: tuple[int, ...] | None = None, sequence_start: int = 0
                         ) -> dict[str, torch.Tensor]:

        """Compute model-specific LM, MTP and router losses on pre-shifted labels.

        Args:
            input_ids: Unmodified token IDs.
            labels: Already-shifted target token IDs.
            loss_mask: Mask for the pre-shifted targets.
            position_ids: Optional public batch positions for RoPE.
            actual_seq_len: Cumulative ends of independent documents, shared by attention and MTP.
            sequence_start: Global offset of the local input interval.
        """
        cfg = self.config
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("JT sequence loss currently requires a two-dimensional batch-one input")
        sequence_length = input_ids.shape[1]
        if labels.shape != input_ids.shape or loss_mask.shape != input_ids.shape:
            raise ValueError("Tokens, pre-shifted labels and loss mask must have identical shapes")
        ends = (sequence_length,) if actual_seq_len is None else tuple(actual_seq_len)
        if (not ends or ends[0] <= 0 or any(left >= right for left, right in zip(ends, ends[1:]))
                or sequence_start < 0 or sequence_start + sequence_length > ends[-1]):
            raise ValueError("JT cumulative document ends must increase and cover the local token interval")
        indices = torch.arange(sequence_start, sequence_start + sequence_length, device=input_ids.device)
        sequence_end_mask = None
        if len(ends) > 1:
            boundaries = indices.new_tensor((0, *ends))
            documents = torch.bucketize(indices, boundaries[1:], right=True)
            sequence_end_mask = (indices + 1 == boundaries[documents + 1]).unsqueeze(0)
            default_positions = indices - boundaries[documents]
        else:
            default_positions = indices
        dim = cfg.qk_rope_head_dim
        inverse = 1.0 / (cfg.rope_parameters["rope_theta"] ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        inverse = torch.from_numpy(inverse.astype(np.float32)).to(input_ids.device)
        if position_ids is None:
            position_ids = default_positions.unsqueeze(0)
        if position_ids.shape != input_ids.shape:
            raise ValueError("JT position_ids must match input_ids")
        frequency = position_ids.to(device=input_ids.device, dtype=torch.float32).unsqueeze(-1) * inverse
        frequency = torch.cat((frequency, frequency), dim=-1)
        hidden = self.model.embed_tokens(input_ids)
        # Transformers rotary contract: FP32 frequencies, tables returned in the activation dtype.
        attention_kwargs = {"position_embeddings": (frequency.cos().to(hidden.dtype), frequency.sin().to(hidden.dtype)),
                            "actual_seq_len": ends}
        auxiliary = torch.zeros((), device=hidden.device, dtype=torch.float32)
        for layer in self.model.layers:
            hidden = layer(hidden, **attention_kwargs)
            if hasattr(layer.mlp, "auxiliary_loss"):
                auxiliary = auxiliary + layer.mlp.auxiliary_loss
        lm_loss = self._token_loss(self.lm_head(self.model.norm(hidden)), labels, loss_mask)
        mtp_output = self.mtp(
            hidden, input_ids, embedding=self.model.embed_tokens, head=self.lm_head,
            labels=labels, loss_mask=loss_mask, loss_factor=cfg.mtp_loss_factor,
            decoder_kwargs=attention_kwargs, loss_fn=self._token_loss,
            auxiliary_loss=auxiliary, auxiliary_fn=lambda decoder: decoder.mlp.auxiliary_loss,
            sequence_end_mask=sequence_end_mask,
        )
        mtp_loss, auxiliary = mtp_output.loss, mtp_output.auxiliary_loss
        return {"loss": (lm_loss + auxiliary) + mtp_loss, "lm_loss": lm_loss,
                "mtp_loss": mtp_loss, "aux_loss": auxiliary}
