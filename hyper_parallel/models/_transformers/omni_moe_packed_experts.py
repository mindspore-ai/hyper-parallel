# ============================================================================
"""Packed-3D-parameter sparse-MoE replacement for the Qwen3-Omni thinker.

The stock ``Qwen3OmniMoeThinkerTextSparseMoeBlock`` stores its 128 routed
experts as 128 separate ``Qwen3OmniMoeThinkerTextMLP`` modules (384 ``nn.Linear``
parameters per block — 18,432 of the 19,743 thinker parameter entries, 93%).
Every per-parameter runtime cost in the FSDP2/optimizer stack then scales with
that count: per-step flat-buffer ``set_``/``narrow``/``copy_`` traffic, 19k
``AccumulateGrad`` invocations, and ~19k per-tensor ``aclnnApplyAdamWV2``
launches. It also forces the fused forward to ``cat``/``stack`` the expert
weights into grouped-GEMM layout on every step.

This replacement stores the same weights as two packed 3D parameters per
block — ``gate_up_proj [E, 2I, H]`` and ``down_proj [E, H, I]`` (the
grouped-GEMM layout) — and runs the router +
fused grouped-GEMM kernels directly against them (no per-step packing).
Checkpoint loading is routed through ``make_transforms()`` converters whose
custom ``ConversionOps`` stack the 128 collected per-expert tensors into the
packed layout, so the original HF checkpoint loads unchanged.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from hyper_parallel.models._transformers.omni_moe_fused_npu import (
    _npu_fused_moe_forward,
)
from hyper_parallel.models.replacement import (
    module_replacement,
)
from hyper_parallel.components.checkpoint.weight_conversion import (
    ConversionOps,
    WeightConverter,
)


class PackExperts(ConversionOps):
    """Stack collected per-expert tensors into one packed 3D parameter.

    ``convert`` receives ``{source_pattern: [tensor_e0, ..., tensor_e127]}``
    from the routing layer (loaders appended in checkpoint-key order, which
    ``dot_natural_key`` sorting makes experts.0..experts.127). With two
    source patterns (gate_proj + up_proj) the two lists are concatenated in
    pattern order along dim 0 first (``cat(e, dim=0)`` per expert), then all
    experts are stacked: ``[E, 2I, H]``. With one pattern it is a plain
    stack: ``[E, H, I]`` for down_proj.
    """

    @torch.no_grad()
    def convert(
        self,
        input_dict,
        source_patterns,
        target_patterns,
        **kwargs,
    ):
        del kwargs  # model/config/... are unused by pure stacking
        expected = len(source_patterns)
        missing = [p for p in source_patterns if p not in input_dict]
        if missing:
            raise ValueError(f"PackExperts missing collected patterns: {missing}")
        stacks = []
        for pattern in source_patterns:
            values = input_dict[pattern]
            tensors = [v() if callable(v) else v for v in values]
            if len(tensors) == 0:
                raise ValueError(f"PackExperts collected no tensors for {pattern!r}")
            stacks.append(torch.stack(tensors, dim=0).contiguous())
        if expected == 1:
            packed = stacks[0]
        elif expected == 2:
            # gate_proj stack [E, I, H] + up_proj stack [E, I, H] → [E, 2I, H]
            packed = torch.cat(stacks, dim=1).contiguous()
        else:
            raise ValueError(f"PackExperts supports 1-2 source patterns, got {expected}")
        target_pattern = target_patterns[0]
        return {target_pattern: packed}

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError(
            "PackExperts reversal requires emitting 128 per-expert tensors "
            "from one packed parameter, which the transform pattern machinery "
            "cannot express; original-format saving of packed-expert blocks "
            "is unsupported"
        )


class PackedSparseMoeBlock(nn.Module):
    """Drop-in ``Qwen3OmniMoeThinkerTextSparseMoeBlock`` with packed experts.

    Keeps the router ``gate`` Linear (the same instance as the source block, so
    ``mlp.gate.weight`` loads through the untouched base mapping) and replaces
    the 128-expert ModuleList with ``gate_up_proj [E, 2I, H]`` /
    ``down_proj [E, H, I]``. Forward keeps the stock contract: router math
    verbatim (softmax fp32 / topk / norm_topk_prob) then the fused NPU
    permute → grouped GEMM → SwiGLU → grouped GEMM → unpermute, with routing
    weights folded in by the unpermute ``probs`` argument.
    """

    def __init__(self, gate: nn.Linear, num_experts: int, top_k: int, norm_topk_prob: bool,
                 hidden_size: int, moe_intermediate_size: int, dtype: torch.dtype,
                 device=None):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.gate = gate  # shared instance: base router weight loads unchanged
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * moe_intermediate_size, hidden_size,
                        dtype=dtype, device=device)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, moe_intermediate_size,
                        dtype=dtype, device=device)
        )

    def forward(self, hidden_states: torch.Tensor):
        """Router + fused packed-experts forward; returns (hidden, router_logits)."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # Router math copied verbatim from the stock 4.57.1 forward.
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = _npu_fused_moe_forward(
            hidden_states,
            routing_weights,
            selected_experts,
            self.gate_up_proj,
            self.down_proj,
            self.num_experts,
        )
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    def make_transforms(self):
        """Per-block converters routing the 128x3 checkpoint tensors into packs."""
        return [
            WeightConverter(
                source_patterns=["experts.*.gate_proj.weight", "experts.*.up_proj.weight"],
                target_patterns=["gate_up_proj"],
                operations=[PackExperts()],
            ),
            WeightConverter(
                source_patterns=["experts.*.down_proj.weight"],
                target_patterns=["down_proj"],
                operations=[PackExperts()],
            ),
        ]


@module_replacement
def replace_sparse_moe_block_with_packed_experts(
    *,
    module: nn.Module,
    module_fqn: str,
    context,
) -> nn.Module:
    """Factory building one :class:`PackedSparseMoeBlock` from the source block.

    Extracts the router, expert geometry, and dtype from the source module so
    the replacement is config-true per block. Runs under ``init_empty_weights``
    (meta parameters); materialization is handled by the shared meta →
    ``to_empty`` path with checkpoint loading routed through
    ``make_transforms()``.
    """
    del module_fqn, context  # geometry comes from the source module itself
    if not isinstance(module, nn.Module) or not isinstance(module.experts, nn.ModuleList):
        raise TypeError(
            "replace_sparse_moe_block_with_packed_experts requires the stock "
            "Qwen3OmniMoeThinkerTextSparseMoeBlock source (ModuleList experts)"
        )
    # _validate_replacement requires training-state parity with the source.
    was_training = module.training
    first_expert = module.experts[0]
    gate_proj = first_expert.gate_proj
    down_proj = first_expert.down_proj
    hidden_size = gate_proj.in_features
    moe_intermediate_size = gate_proj.out_features
    dtype = gate_proj.weight.dtype
    # Step 5.5 runs OUTSIDE the init_empty_weights context (only _init_model is
    # wrapped), so a bare torch.empty here would allocate ~57GB of host RAM for
    # the 48 blocks. When the source tree is meta (the training path), create
    # the packed parameters on the meta device too; the shared meta → to_empty
    # materialization handles them exactly like the stock parameters.
    device = gate_proj.weight.device if gate_proj.weight.device.type == "meta" else None
    replacement = PackedSparseMoeBlock(
        gate=module.gate,
        num_experts=module.num_experts,
        top_k=module.top_k,
        norm_topk_prob=module.norm_topk_prob,
        hidden_size=hidden_size,
        # Qwen3OmniMoeThinkerTextMLP: gate/up share geometry with down's in_features
        moe_intermediate_size=moe_intermediate_size,
        dtype=dtype,
        device=device,
    )
    replacement.train(was_training)
    return replacement
