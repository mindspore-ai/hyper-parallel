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
"""Qwen3-MoE inline Codegen declarations."""

from __future__ import annotations

from hyper_parallel.codegen.inline.ir import ImportPatch, ModuleSnippetPatch
from hyper_parallel.codegen.inline.spec_bundle import (
    InlineSpecBundle,
    MetaNormalizer,
    ReplacementSpec,
    StrategySpec,
)
from hyper_parallel.codegen.inline.templates import QWEN3_GQA_ATTENTION_CLASS, QWEN3_MOE_EP_FORWARD, TP_OPERATORS_CLASS


QWEN3_MOE_FLASH_ATTENTION_REPLACEMENT = (
    "hyper_parallel.models.qwen3_moe.adapter.replacements."
    "replace_qwen3_moe_flash_attention"
)

_REPLACEMENT_SPECS = {
    "hyper_parallel.models.qwen3_moe.adapter.replacements.replace_qwen3_moe_rms_norm": ReplacementSpec(
        old_ctor="Qwen3MoeRMSNorm",
        new_ctor="RMSNorm",
        imports=(ImportPatch("hyper_parallel.components.modules", ("RMSNorm",)),),
        replacement_note="RMSNorm replaces Qwen3MoeRMSNorm.",
    ),
    QWEN3_MOE_FLASH_ATTENTION_REPLACEMENT: ReplacementSpec(
        old_ctor="Qwen3MoeAttention",
        new_ctor="GQAAttention",
        imports=(
            ImportPatch("hyper_parallel.codegen.runtime", ("get_inline_parallel_state",)),
            ImportPatch("hyper_parallel.components.modules", ("RMSNorm",)),
            ImportPatch(
                "hyper_parallel.models.qwen3_moe.adapter.attention",
                ("run_qwen3_moe_flash_attention",),
            ),
            ImportPatch("hyper_parallel.platform", ("get_platform",)),
        ),
        snippets=(
            ModuleSnippetPatch(TP_OPERATORS_CLASS),
            ModuleSnippetPatch(QWEN3_GQA_ATTENTION_CLASS),
        ),
        replacement_note="GQAAttention replaces Qwen3MoeAttention.",
    ),
    "hyper_parallel.models.qwen3_moe.adapter.replacements.replace_qwen3_moe_grouped_experts": ReplacementSpec(
        old_ctor="Qwen3MoeExperts",
        new_ctor="GroupedExperts",
        mode="wrap_source",
        keyword_args=("module_fqn=''", "context=None"),
        imports=(ImportPatch("hyper_parallel.components.modules", ("GroupedExperts",)),),
        remove_class=False,
        replacement_note=(
            "GroupedExperts wraps Qwen3MoeExperts; the original class is kept "
            "as the wrapper input."
        ),
    ),
}

_STRATEGY_SPECS = {
    "hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel.qwen3moe_ep_compute_fn": StrategySpec(
        kind="qwen3_moe_ep_routed_forward",
        target_class="Qwen3MoeSparseMoeBlock",
        body_template=QWEN3_MOE_EP_FORWARD,
        imports=(
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
    (
        "hyper_parallel.models.qwen3_moe.adapter.distributed.context_parallel."
        "qwen3_moe_flash_attention_cp_wrapper"
    ): StrategySpec(
        kind="qwen3_moe_cp_attention",
        imports=(
            ImportPatch(
                "hyper_parallel.distributed.context_parallel",
                ("flex_cp_allgather",),
            ),
            ImportPatch(
                "hyper_parallel.distributed.context_parallel.attention",
                ("_cp_offset_causal_mask",),
            ),
        ),
    ),
}

_META_NORMALIZERS = (
    MetaNormalizer(
        target=QWEN3_MOE_FLASH_ATTENTION_REPLACEMENT,
        param_renames={
            "linear_qkv.weight": ("q_proj.weight", "k_proj.weight", "v_proj.weight"),
            "linear_qkv.bias": ("q_proj.bias", "k_proj.bias", "v_proj.bias"),
        },
    ),
)

_INLINE_SPEC_BUNDLE = InlineSpecBundle(
    replacement_specs=_REPLACEMENT_SPECS,
    strategy_specs=_STRATEGY_SPECS,
    meta_normalizers=_META_NORMALIZERS,
    external_state_classes=("GQAAttention", "Qwen3MoeSparseMoeBlock"),
)


def get_inline_spec_bundle() -> InlineSpecBundle:
    """Return Qwen3-MoE inline Codegen declarations."""

    return _INLINE_SPEC_BUNDLE


__all__ = ["get_inline_spec_bundle"]
