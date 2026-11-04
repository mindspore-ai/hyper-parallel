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
"""replacements: Xing4 structure replacements onto the generic modules.

:class:`Xing4MLAAttention` extends the generic high-performance ``modules.MLAAttention``
instead of reimplementing kernels; the NPU kernels live in ``functional``.

The module stays importable on CPU-only checkouts: ``modules.rms_norm`` imports the
NPU-only runtime, so it is imported lazily where the latent norms are swapped.
"""

# This adapter uses the Torch runtime, like the framework attention modules.
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.core_model_loading import Concatenate

from hyper_parallel.components.checkpoint.weight_conversion import MergeModulelist, Transpose, WeightConverter
from hyper_parallel.components.functional.mhc_pre import mhc_pre
from hyper_parallel.models.xing_4.adapter.training import Xing4HyperConnection
from hyper_parallel.components.modules.grouped_experts import GroupedExperts
from hyper_parallel.components.modules.mla_attention import MLAAttention
from hyper_parallel.models.replacement import module_replacement as register_module_replacement
from hyper_parallel.models.xing_4.transformer.modeling_xing4_0 import apply_rotary_pos_emb_interleave

__all__ = ["Xing4MLAAttention", "Xing4GroupedExperts", "Xing4FusedHyperConnection"]


def fusion_attention_with_logit_maxima(module: nn.Module, query: torch.Tensor, key: torch.Tensor,
                                       value: torch.Tensor, attention_mask: torch.Tensor | None,
                                       **kwargs: Any) -> tuple[torch.Tensor, None]:
    """Run causal TND fused attention and fold the per-head logit maxima into ``module.max_logits``.

    This is the ``attention_interface`` that :class:`Xing4MLAAttention` hands to
    ``modules.MLAAttention``.

    Args:
        module: Attention module; its ``max_logits`` holds the maxima since the last QK clipping.
        query: Query states, ``[batch, heads, sequence, head_dim]``.
        key: Key states, ``[batch, heads, sequence, head_dim]``.
        value: Value states, ``[batch, heads, sequence, v_head_dim]``.
        attention_mask: Unused local HF mask; the training model accepts only implicit causality.
        **kwargs: Attention scale, optional global packed boundaries and the MLA CP head range.

    Returns:
        The attention output, ``[batch, sequence, heads, v_head_dim]``, and no attention weights.

    Raises:
        ValueError: If the kernel's ``softmax_max`` is not the expected ``(T, heads, 8)`` shape.
    """
    # torch_npu is optional outside Ascend environments.
    import torch_npu  # pylint: disable=C0415
    from hyper_parallel.components.functional.npu_fusion_attention import (  # pylint: disable=C0415
        _prepare_fusion_attention_context, PACKED_SEQUENCE_ARGUMENTS,
    )

    del attention_mask
    batch, heads, sequence = query.shape[:3]
    options = dict(kwargs)
    head_offset = options.pop("mla_head_offset", 0)
    local_heads = options.pop("mla_local_heads", heads)
    if not any(options.get(name) is not None for name in PACKED_SEQUENCE_ARGUMENTS):
        if sequence != key.shape[2]:
            raise ValueError("Unequal Xing4 CP query/key lengths require global sequence boundaries")
        options["actual_seq_len"] = tuple(range(sequence, batch * sequence + 1, sequence))
    context = _prepare_fusion_attention_context(module, query, key, value, None, options)
    output, softmax_max = torch_npu.npu_fusion_attention(
        context.query, context.key, context.value, heads, context.input_layout,
        pse=None, padding_mask=None, atten_mask=context.attention_mask, scale=kwargs["scaling"],
        pre_tockens=context.pre_tokens, next_tockens=context.next_tokens, keep_prob=1.0,
        inner_precise=0, sparse_mode=context.sparse_mode, actual_seq_qlen=context.query_lengths,
        actual_seq_kvlen=context.key_lengths, softmax_layout="TND")[:2]
    with torch.no_grad():
        # softmax_max is (T, N, 8) for TND input: core/shard/ops/parallel_npu_flash_attention_score.py's
        # sharding rule hard-codes this ("TND softmax outputs are 3-D (T, N, 8)"), and
        # test_parallel_npu_flash_attention_score.py asserts dim 0 is token, dim 1 is head. Check it
        # instead of silently maxing across the wrong axis if that assumption is ever wrong (MindFormers'
        # own max_logit_tracker.py exists specifically because this failure raises nothing on its own).
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
    """Replace HF ``Xing4_0Attention`` with the generic MLA and HF FP32 RoPE.

    ``modules.MLAAttention`` fuses ``q_a_proj``/``kv_a_proj_with_mqa`` into ``linear_qkv``
    (checkpoint conversion declared through ``make_transforms``) and keeps the latent-norm
    and projection structure. This replacement swaps the latent norms for the NPU ``RMSNorm``
    (a replacement plan cannot also select these children of the replaced attention),
    rotates in FP32 before returning to the activation dtype. On NPU it can select the
    shared fused interleaved RoPE with the same Transformers layout. It calls attention through
    :func:`fusion_attention_with_logit_maxima`, which folds in the per-head logit maxima QK
    clipping consumes.
    """

    def __init__(self, *, module: nn.Module, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None) -> None:
        """Build the replacement from the HF Xing4 attention.

        Args:
            module: Official ``Xing4_0Attention`` instance being replaced.
            module_fqn: Fully qualified name of ``module``.
            context: Replacement context of the build pipeline.
        """
        super().__init__(module=module, module_fqn=module_fqn, context=context,
                         attention_interface=fusion_attention_with_logit_maxima)
        # Lazy: modules.rms_norm imports the NPU-only runtime.
        from hyper_parallel.components.modules.rms_norm import RMSNorm  # pylint: disable=C0415

        self.q_a_layernorm = RMSNorm(module=self.q_a_layernorm)
        self.kv_a_layernorm = RMSNorm(module=self.kv_a_layernorm)
        if not self.linear_qkv.weight.is_meta:
            with torch.no_grad():
                self.linear_qkv.weight.copy_(torch.cat((module.q_a_proj.weight, module.kv_a_proj_with_mqa.weight)))
        self.apply_rotary_pos_emb_interleave = apply_rotary_pos_emb_interleave
        # Per-head logit maxima since the last QK clipping, folded in by the attention interface.
        self.max_logits: torch.Tensor | None = None

    def _project_attention_inputs(self, hidden_states: torch.Tensor,
                                  position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
                                  past_key_values: Any | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states to query, key and value, rotating in FP32 and rounding once.

        Args:
            hidden_states: Normalized layer input, ``[batch, sequence, hidden]``.
            position_embeddings: FP32 RoPE ``(cos, sin)`` tables.
            past_key_values: Must be None; training uses no KV cache.

        Returns:
            Query, key and value states, ``[batch, heads, sequence, head_dim]``.

        Raises:
            ValueError: If the RoPE tables are missing or a KV cache is given.
        """
        if position_embeddings is None or past_key_values is not None:
            raise ValueError("Xing4 MLA requires RoPE tables and no KV cache")
        latents = self._project_latents(hidden_states)
        cos, sin = position_embeddings
        apply_rotary = self.apply_rotary_pos_emb_interleave
        if getattr(self.config, "apply_rope_fusion", False) and hidden_states.device.type == "npu":
            from hyper_parallel.components.functional.rotary_embedding import (  # pylint: disable=C0415
                apply_rotary_pos_emb_interleave,
            )
            apply_rotary = apply_rotary_pos_emb_interleave
        query_rope, key_rope = apply_rotary(latents.query_rope.float(), latents.key_rope.float(), cos, sin)
        key_value = self.kv_b_proj(latents.kv_nope).view(
            latents.batch_size, latents.sequence_length, self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim)
        key_nope, value = key_value.transpose(1, 2).split((self.qk_nope_head_dim, self.v_head_dim), dim=-1)
        query_pass = latents.query_pass.transpose(1, 2)
        query = torch.cat((query_pass, query_rope.to(query_pass.dtype)), dim=-1)
        key = torch.cat((key_nope, key_rope.to(key_nope.dtype).expand(-1, self.num_heads, -1, -1)), dim=-1)
        return query, key, value


def _expert_weights(module: nn.Module) -> tuple[Any, list[tuple[nn.Parameter, nn.Parameter, nn.Parameter]]]:
    """Validate the source container without discarding parameters, buffers or hooks."""
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
        if any("hook" in name and isinstance(value, Mapping) and value for name, value in vars(child).items()):
            raise ValueError("Xing4 grouped experts cannot migrate source module hooks")
    weights = []
    expected_names = set()
    expected_activation = type(ACT2FN["silu"])
    for index, expert in enumerate(module):
        if getattr(expert, "act_fn", None).__class__ is not expected_activation:
            raise ValueError("Xing4 grouped experts require the standard SiLU activation")
        projections = []
        for name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(expert, name, None)
            if projection.__class__ is not nn.Linear or projection.bias is not None:
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
    """Ensure one packed parameter can preserve every source value and gradient flag."""
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
    """Replace the official per-expert ModuleList with the shared grouped-GEMM implementation.

    This creates new packed parameters. Eager source values are copied immediately;
    meta source values must be populated later through the declared checkpoint transforms.
    The replacement must happen before an optimizer captures the original parameters.

    Args:
        module: Official Xing4 expert ModuleList.
        module_fqn: Fully qualified name supplied by the replacement executor.
        context: Replacement context supplied by the executor.
    """

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
        carrier.router_gating_in_fp32 = getattr(module, "router_gating_in_fp32", True)
        carrier.limit = getattr(module[0], "limit", None)
        carrier.activation_compute_dtype = getattr(module[0], "activation_compute_dtype", None)
        carrier.gate_up_proj = packed_gate
        carrier.down_proj = packed_down
        carrier.train(module.training)
        super().__init__(module=carrier, module_fqn=module_fqn, context=context)

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        """Preserve the ModuleList replacement signature and delegate grouped execution."""
        return super().forward(*args, **kwargs)

    def make_transforms(self) -> list[WeightConverter]:
        """Return reversible mappings for every original expert projection."""
        if MergeModulelist is None:
            raise ImportError("Transformers MergeModulelist is required for expert conversion")
        return [
            WeightConverter(
                source_patterns=[r"*.gate_proj.weight", r"*.up_proj.weight"],
                target_patterns="gate_up_proj",
                operations=[MergeModulelist(dim=0), Concatenate(dim=1), Transpose(-2, -1)],
            ),
            WeightConverter(
                source_patterns=r"*.down_proj.weight",
                target_patterns="down_proj",
                operations=[MergeModulelist(dim=0), Transpose(-2, -1)],
            ),
        ]


@register_module_replacement
class Xing4FusedHyperConnection(Xing4HyperConnection):
    """Replace the portable mHC computation with the shared NPU operations."""

    def __init__(self, *, module: Xing4HyperConnection, module_fqn: str = "",
                 context: Mapping[str, Any] | None = None) -> None:
        """Keep official parameter names, values and ownership."""
        nn.Module.__init__(self)
        del module_fqn, context
        self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps = module.hc_mult, module.hc_sinkhorn_iters, module.hc_eps
        self.input_norm = module.input_norm
        self.hc_fn, self.hc_base, self.hc_scale = module.hc_fn, module.hc_base, module.hc_scale
        self.train(module.training)

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Use the shared fused pre operator on NPU and the portable reference otherwise."""
        if hidden_streams.device.type != "npu":
            return super().forward(hidden_streams)
        hidden, post, mix, residual = mhc_pre(
            hidden_streams.flatten(2), self.hc_fn, self.hc_scale, self.hc_base, self.hc_mult,
            sinkhorn_iters=self.hc_sinkhorn_iters, norm_eps=self.input_norm.eps, hc_eps=self.hc_eps)
        # Keep the pre operator's residual output in the graph: its backward
        # consumes the post gradient through grad_x_post.
        return post, mix, hidden, residual.unflatten(-1, (self.hc_mult, hidden.shape[-1]))
