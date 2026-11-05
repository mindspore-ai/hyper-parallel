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
"""NPU replacements for Xing4 attention, experts and hyper-connections."""

# pylint: disable=forbidden-backend-import

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

import torch
from torch import nn
from transformers.activations import ACT2FN

from hyper_parallel.components.checkpoint import weight_conversion
from hyper_parallel.components.checkpoint.conversion_ops import ConcatenateWithSections
from hyper_parallel.components.checkpoint.weight_conversion import Transpose, WeightConverter
from hyper_parallel.components.functional.mhc_pre import mhc_pre
from hyper_parallel.components.functional.npu_fusion_attention import (
    _prepare_fusion_attention_context, PACKED_SEQUENCE_ARGUMENTS,
)
from hyper_parallel.components.functional.rotary_embedding import (
    apply_rotary_pos_emb_interleave as fused_rotary,
)
from hyper_parallel.components.modules.rms_norm import RMSNorm
from hyper_parallel.components.modules.grouped_experts import GroupedExperts
from hyper_parallel.components.modules.mla_attention import MLAAttention
from hyper_parallel.models.replacement import module_replacement as register_module_replacement
from hyper_parallel.models.xing_4.adapter.moe import Xing4MLP
from hyper_parallel.models.xing_4.transformer.modeling_xing4_0 import (
    Xing4_0HyperConnection, Xing4_0MLP, apply_rotary_pos_emb_interleave,
)

__all__ = ["Xing4MLAAttention", "Xing4GroupedExperts", "Xing4FusedHyperConnection"]


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
    if not options.get("is_causal", getattr(module, "is_causal", True)):
        raise ValueError("Xing4 training requires causal attention")
    if not any(options.get(name) is not None for name in PACKED_SEQUENCE_ARGUMENTS):
        if sequence != key.shape[2]:
            raise ValueError("Unequal Xing4 CP query/key lengths require global sequence boundaries")
        options["actual_seq_len"] = tuple(range(sequence, batch * sequence + 1, sequence))
    context = _prepare_fusion_attention_context(module, query, key, value, None, options)
    # Configuration and conversion do not require the optional NPU extension.
    import torch_npu  # pylint: disable=import-outside-toplevel

    output, softmax_max = torch_npu.npu_fusion_attention(
        context.query, context.key, context.value, heads, context.input_layout,
        pse=None, padding_mask=None, atten_mask=context.attention_mask, scale=kwargs["scaling"],
        pre_tockens=context.pre_tokens, next_tockens=context.next_tokens, keep_prob=1.0,
        inner_precise=0, sparse_mode=context.sparse_mode, actual_seq_qlen=context.query_lengths,
        actual_seq_kvlen=context.key_lengths, softmax_layout="TND")[:2]
    with torch.no_grad():
        # TND softmax statistics have token, head and redundant-lane axes.
        if softmax_max.ndim != 3 or softmax_max.shape[1] != heads:
            raise ValueError(f"Expected (T, {heads}, 8) softmax_max for TND fusion attention, "
                             f"got shape {tuple(softmax_max.shape)}.")
        maximum = softmax_max.amax(dim=(0, 2))
        if heads != local_heads:
            # Ulysses owns a subset of TP-local heads; QK clipping's MAX reduction joins them.
            placed = maximum.new_full((local_heads,), float("-inf"))
            placed[head_offset:head_offset + heads] = maximum
            maximum = placed
        module.max_logits = maximum if module.max_logits is None else torch.maximum(module.max_logits, maximum)
    return output.view(batch, sequence, heads, -1), None


@register_module_replacement
class Xing4MLAAttention(MLAAttention):
    """Fuse MLA projections with FP32 rotary channels and per-head maxima."""

    def __init__(self, *, module: nn.Module, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None) -> None:
        """Retain source weights and select the configured rotary implementation."""
        super().__init__(module=module, module_fqn=module_fqn, context=context,
                         attention_interface=fusion_attention_with_logit_maxima)
        self.q_a_layernorm = RMSNorm(module=self.q_a_layernorm)
        self.kv_a_layernorm = RMSNorm(module=self.kv_a_layernorm)
        if not self.linear_qkv.weight.is_meta:
            with torch.no_grad():
                split = self._q_latent_output_size
                self.linear_qkv.weight[:split].copy_(module.q_a_proj.weight)
                self.linear_qkv.weight[split:].copy_(module.kv_a_proj_with_mqa.weight)
        self.apply_rotary_pos_emb_interleave = (
            fused_rotary if getattr(self.config, "apply_rope_fusion", False) else apply_rotary_pos_emb_interleave
        )
        # Per-head logit maxima since the last QK clipping, folded in by the attention interface.
        self.max_logits: torch.Tensor | None = None

    def _project_attention_inputs(self, hidden_states: torch.Tensor,
                                  position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
                                  past_key_values: Any | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reuse latent and KV projections, rounding RoPE at each concatenation."""
        if position_embeddings is None or past_key_values is not None:
            raise ValueError("Xing4 MLA requires RoPE tables and no KV cache")
        latents = self._project_latents(hidden_states)
        query_rope, key_rope = self.apply_rotary_pos_emb_interleave(
            latents.query_rope.float(), latents.key_rope.float(), *position_embeddings,
        )
        key_nope, value = self._expand_key_value(latents.kv_nope)
        query_pass = latents.query_pass.transpose(1, 2)
        query = torch.cat((query_pass, query_rope.to(query_pass.dtype)), dim=-1)
        key = torch.cat((key_nope, key_rope.to(key_nope.dtype).expand(-1, self.num_heads, -1, -1)), dim=-1)
        return query, key, value


def _expert_weights(module: nn.Module) -> tuple[Any, list[tuple[nn.Parameter, nn.Parameter, nn.Parameter]]]:
    """Read per-expert projections and reject incompatible source modules."""
    if not isinstance(module, nn.ModuleList) or len(module) == 0:
        raise ValueError("Xing4 grouped experts require a nonempty ModuleList")
    config = getattr(module, "config", None)
    if config is None:
        config = getattr(module[0], "config", None)
    if config is None or getattr(config, "hidden_act", None) != "silu":
        raise ValueError("Xing4 grouped experts require the source SiLU configuration")
    if any(True for _ in module.named_buffers(remove_duplicate=False)):
        raise ValueError("Xing4 grouped experts cannot discard source buffers")
    for child in module.modules():
        if "forward" in vars(child):
            raise ValueError("Xing4 grouped experts cannot migrate instance forward overrides")
        if any("hook" in name and isinstance(value, Mapping) and value for name, value in vars(child).items()):
            raise ValueError("Xing4 grouped experts cannot migrate source module hooks")
    # Packing preserves only these exact expert, projection and activation implementations.
    expert_type = type(module[0])
    if expert_type not in (Xing4_0MLP, Xing4MLP) or any(
            type(expert) is not expert_type for expert in module):  # pylint: disable=unidiomatic-typecheck
        raise ValueError("Xing4 grouped experts require a uniform supported expert class")
    policy = (
        getattr(module[0], "limit", None),
        getattr(module[0], "activation_compute_dtype", None),
        bool(getattr(config, "use_fused_swiglu", True)),
    )
    for expert in module:
        expert_config = getattr(expert, "config", None)
        expert_policy = (
            getattr(expert, "limit", None),
            getattr(expert, "activation_compute_dtype", None),
            bool(getattr(expert_config, "use_fused_swiglu", True)),
        )
        if getattr(expert_config, "hidden_act", None) != "silu" or expert_policy != policy:
            raise ValueError("Xing4 grouped experts require uniform activation policy")
    if expert_type is Xing4MLP:
        # Xing4MLP loads this optional API only when the extended policy is used.
        from hyper_parallel.components.functional.activation_swiglu import (  # pylint: disable=import-outside-toplevel
            swiglu_from_projections,
        )
        if any(expert._swiglu is not swiglu_from_projections for expert in module):  # pylint: disable=protected-access
            raise ValueError("Xing4 grouped experts require the standard SwiGLU implementation")
    weights = []
    expected_names = set()
    expected_activation = type(ACT2FN["silu"])
    for index, expert in enumerate(module):
        if type(getattr(expert, "act_fn", None)) is not expected_activation:  # pylint: disable=unidiomatic-typecheck
            raise ValueError("Xing4 grouped experts require the standard SiLU activation")
        projections = []
        for name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(expert, name, None)
            # pylint: disable-next=unidiomatic-typecheck
            if type(projection) is not nn.Linear or projection.bias is not None:
                raise ValueError(f"Xing4 expert {index} requires bias-free Linear {name}")
            projections.append(projection.weight)
            expected_names.add(f"{index}.{name}.weight")
        weights.append(tuple(projections))
    named = dict(module.named_parameters(remove_duplicate=False))
    if set(named) != expected_names or len({id(value) for value in named.values()}) != len(named):
        raise ValueError("Xing4 expert parameters contain unsupported extras or aliases")
    storages = [value.untyped_storage() for value in named.values()]
    if len({id(storage) for storage in storages}) != len(storages):
        raise ValueError("Xing4 expert parameters with shared storage cannot be packed independently")
    return config, weights


def _validate_expert_weights(
    config: Any, weights: list[tuple[nn.Parameter, nn.Parameter, nn.Parameter]],
) -> tuple[int, int]:
    """Require uniform projection shapes, dtype, device and gradient ownership."""
    first_gate, _, first_down = weights[0]
    if first_gate.ndim != 2:
        raise ValueError("Xing4 expert gate weight must be a matrix")
    intermediate_size, hidden_size = first_gate.shape
    if hidden_size <= 0 or intermediate_size <= 0:
        raise ValueError("Xing4 expert dimensions must be positive")
    count = len(weights)
    for name, expected in (("hidden_size", hidden_size), ("moe_intermediate_size", intermediate_size),
                           ("n_routed_experts", count), ("num_local_experts", count)):
        actual = getattr(config, name, None)
        if actual is not None and actual != expected:
            raise ValueError(f"Xing4 expert config {name}={actual} does not match {expected}")
    for gate, up, down in weights:
        if (tuple(gate.shape) != (intermediate_size, hidden_size)
                or tuple(up.shape) != (intermediate_size, hidden_size)
                or tuple(down.shape) != (hidden_size, intermediate_size)):
            raise ValueError("Xing4 experts must have identical gate, up and down dimensions")
        if any(value.device != first_gate.device or value.dtype != first_gate.dtype for value in (gate, up, down)):
            raise ValueError("Xing4 expert weights must share one dtype and device")
        if gate.requires_grad != first_gate.requires_grad or up.requires_grad != first_gate.requires_grad:
            raise ValueError("packing gate/up weights requires identical requires_grad flags")
        if down.requires_grad != first_down.requires_grad:
            raise ValueError("packing down weights requires identical requires_grad flags")
    return hidden_size, intermediate_size


@register_module_replacement
class Xing4GroupedExperts(GroupedExperts):
    """Pack the official ModuleList into shared grouped-GEMM parameters.

    Eager weights are copied once. Meta weights use the reversible checkpoint
    mappings. Replacement precedes optimizer construction."""

    def __init__(
        self,
        *,
        module: nn.Module,
        module_fqn: str = "",
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Validate, pack and install expert weights in the shared grouped module."""
        config, weights = _expert_weights(module)
        hidden_size, intermediate_size = _validate_expert_weights(config, weights)
        count = len(weights)
        gate, _, down = weights[0]
        packed_gate = nn.Parameter(
            gate.new_empty((count, hidden_size, 2 * intermediate_size)), requires_grad=gate.requires_grad
        )
        packed_down = nn.Parameter(
            down.new_empty((count, intermediate_size, hidden_size)), requires_grad=down.requires_grad
        )
        if not gate.is_meta:
            with torch.no_grad():
                for index, (gate_weight, up_weight, down_weight) in enumerate(weights):
                    packed_gate[index, :, :intermediate_size].copy_(gate_weight.transpose(0, 1))
                    packed_gate[index, :, intermediate_size:].copy_(up_weight.transpose(0, 1))
                    packed_down[index].copy_(down_weight.transpose(0, 1))
        carrier = nn.Module()
        carrier.config = config
        carrier.num_local_experts = count
        carrier.hidden_size = hidden_size
        carrier.intermediate_size = intermediate_size
        carrier.has_gate = True
        carrier.is_concatenated = True
        carrier.is_transposed = True
        carrier.limit = getattr(module[0], "limit", None)
        carrier.activation_compute_dtype = getattr(module[0], "activation_compute_dtype", None)
        carrier.gate_up_proj = packed_gate
        carrier.down_proj = packed_down
        carrier.train(module.training)
        super().__init__(module=carrier, module_fqn=module_fqn, context=context)

        if carrier.limit is not None or carrier.activation_compute_dtype is not None:
            expected = {
                "clamp_limit": carrier.limit,
                "compute_dtype": carrier.activation_compute_dtype,
                "use_fused": bool(getattr(config, "use_fused_swiglu", True)),
            }
            activation = self.activation_func
            if not isinstance(activation, partial) or any(
                    activation.keywords.get(name) != value for name, value in expected.items()):
                raise ValueError("GroupedExperts must support the requested SwiGLU clamp and compute dtype")

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        """Preserve the ModuleList replacement call contract for the shared builder."""
        return super().forward(*args, **kwargs)

    def make_transforms(self) -> list[WeightConverter]:
        """Return reversible mappings for every original expert projection."""
        # Prefer HP's dimension-preserving reverse operation when available.
        merge_experts = getattr(weight_conversion, "MergeModulelist", None)
        if merge_experts is None:
            from transformers.core_model_loading import MergeModulelist  # pylint: disable=import-outside-toplevel

            if 1 in (self.hidden_size, self.intermediate_size):
                raise ValueError("Singleton expert dimensions require the dimension-preserving MergeModulelist")
            merge_experts = MergeModulelist
        pack = ConcatenateWithSections((self.intermediate_size, self.intermediate_size), dim=1)
        return [
            WeightConverter(
                source_patterns=[r"*.gate_proj.weight", r"*.up_proj.weight"],
                target_patterns="gate_up_proj",
                operations=[merge_experts(dim=0), pack, Transpose(-2, -1)],
            ),
            WeightConverter(
                source_patterns=r"*.down_proj.weight",
                target_patterns="down_proj",
                operations=[merge_experts(dim=0), Transpose(-2, -1)],
            ),
        ]


@register_module_replacement
class Xing4FusedHyperConnection(Xing4_0HyperConnection):
    """Adapt official mHC parameters to the shared NPU pre operation."""

    def __init__(self, *, module: Xing4_0HyperConnection, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None) -> None:
        """Keep official parameter names, values and ownership."""
        nn.Module.__init__(self)
        del module_fqn, context
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
