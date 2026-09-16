# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""DeepSeek-V3 inline Codegen declarations."""

from __future__ import annotations

from hyper_parallel.codegen.inline.ir import ImportPatch, ModuleSnippetPatch
from hyper_parallel.codegen.inline.spec_bundle import InlineSpecBundle, StrategySpec
from hyper_parallel.codegen.inline.templates import PARALLEL_STATE_ACCESSOR


DEEPSEEK_V3_MOE_EP_FORWARD = '''"""Inline EP routed MoE forward generated from the YAML local_compute_fn target."""
ps = get_parallel_state()
if not ps.ep_enabled:
    return self._forward_impl(hidden_states)

ep_group = ps.ep_group
ep_size = ep_group.size()
ep_rank = torch.distributed.get_rank(group=ep_group)
local_expert_count = self.experts.local_expert_count
global_expert_count = local_expert_count * ep_size
expert_offset = ep_rank * local_expert_count

batch_size, sequence_length, hidden_size = hidden_states.shape
topk_indices, topk_weights = MOE_ROUTER_ADAPTERS["deepseekv3"](self, hidden_states)
(
    source_token_indices,
    flattened_expert_weights,
    dispatch_order,
    dispatched_states,
    dispatched_expert_indices,
    send_counts,
    receive_counts,
) = _prepare_ep_dispatch(
    hidden_states,
    topk_indices,
    topk_weights,
    local_expert_count=local_expert_count,
    global_expert_count=global_expert_count,
    ep_size=ep_size,
    ep_group=ep_group,
)

received_states = ep_all_to_all(dispatched_states, send_counts, receive_counts, ep_group)
received_indices = ep_all_to_all(
    dispatched_expert_indices, send_counts, receive_counts, ep_group
).squeeze(-1)
local_outputs = self.experts(received_states, received_indices - expert_offset)
combined_expert_outputs = ep_all_to_all(
    local_outputs.contiguous(), receive_counts, send_counts, ep_group
)

weighted_outputs = combined_expert_outputs * flattened_expert_weights[dispatch_order].unsqueeze(-1)
routed_outputs = torch.zeros(
    batch_size * sequence_length,
    hidden_size,
    dtype=weighted_outputs.dtype,
    device=weighted_outputs.device,
)
routed_outputs.index_add_(0, source_token_indices[dispatch_order], weighted_outputs)
routed_outputs = routed_outputs.view(batch_size, sequence_length, hidden_size)
return routed_outputs + self.shared_experts(hidden_states)
'''

_STRATEGY_SPECS = {
    "hyper_parallel.distributed.expert_parallel.recipes.deepseekv3_ep_compute_fn": StrategySpec(
        kind="deepseek_v3_ep_routed_forward",
        target_class="DeepseekV3MoE",
        body_template=DEEPSEEK_V3_MOE_EP_FORWARD,
        imports=(
            ImportPatch("hyper_parallel.codegen.runtime", ("get_inline_parallel_state",)),
            ImportPatch(
                "hyper_parallel.distributed.expert_parallel.experts",
                ("_prepare_ep_dispatch", "ep_all_to_all"),
            ),
            ImportPatch(
                "hyper_parallel.distributed.expert_parallel.routing",
                ("MOE_ROUTER_ADAPTERS",),
            ),
        ),
    ),
}

_INLINE_SPEC_BUNDLE = InlineSpecBundle(
    replacement_specs={},
    strategy_specs=_STRATEGY_SPECS,
    external_state_classes=("DeepseekV3MoE",),
)

_INLINE_SPEC_BUNDLE.strategy_specs[
    "hyper_parallel.distributed.expert_parallel.recipes.deepseekv3_ep_compute_fn"
].imports


def get_inline_spec_bundle() -> InlineSpecBundle:
    """Return DeepSeek-V3 inline Codegen declarations."""

    return InlineSpecBundle(
        replacement_specs=_INLINE_SPEC_BUNDLE.replacement_specs,
        strategy_specs=_INLINE_SPEC_BUNDLE.strategy_specs,
        module_snippets=(ModuleSnippetPatch(PARALLEL_STATE_ACCESSOR),),
        external_state_classes=_INLINE_SPEC_BUNDLE.external_state_classes,
    )


__all__ = ["get_inline_spec_bundle"]
