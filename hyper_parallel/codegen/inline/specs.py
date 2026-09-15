# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Built-in inline patch semantics keyed by YAML target paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from hyper_parallel.codegen.inline.ir import ImportPatch, ModuleSnippetPatch
from hyper_parallel.codegen.inline.templates import QWEN3_GQA_ATTENTION_CLASS, TP_OPERATORS_CLASS


@dataclass(frozen=True)
class ReplacementSpec:
    """Source-level form of one ``replace_module`` target."""

    old_ctor: str
    new_ctor: str
    imports: tuple[ImportPatch, ...]
    mode: str = "name"
    keyword_args: tuple[str, ...] = ()
    snippets: tuple[ModuleSnippetPatch, ...] = ()
    #: When True the original HF class definition is removed from the generated
    #: file (and any surviving bare reference is rewritten to ``new_ctor``).
    #: Must be False for ``wrap_source`` replacements because the original
    #: class is still instantiated as the inner module of the fused wrapper.
    remove_class: bool = True
    #: User-facing note emitted into the generated modeling file.
    replacement_note: str | None = None


@dataclass(frozen=True)
class StrategySpec:
    """Source-level form of one parallel strategy target."""

    kind: str
    imports: tuple[ImportPatch, ...]


REPLACEMENT_SPECS = {
    "hyper_parallel.models.qwen3_moe.adapter.replacements.replace_qwen3_moe_rms_norm": ReplacementSpec(
        old_ctor="Qwen3MoeRMSNorm",
        new_ctor="RMSNorm",
        imports=(ImportPatch("hyper_parallel.components.modules", ("RMSNorm",)),),
        replacement_note="RMSNorm replaces Qwen3MoeRMSNorm.",
    ),
    "hyper_parallel.models.qwen3_moe.adapter.replacements.replace_qwen3_moe_flash_attention": ReplacementSpec(
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


STRATEGY_SPECS = {
    "hyper_parallel.models.qwen3_moe.adapter.distributed.expert_parallel.qwen3moe_ep_compute_fn": StrategySpec(
        kind="qwen3_moe_ep_routed_forward",
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
    "hyper_parallel.models.qwen3_moe.adapter.distributed.context_parallel.qwen3_moe_flash_attention_cp_wrapper": StrategySpec(
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


def replacement_spec(target: Optional[str]) -> Optional[ReplacementSpec]:
    """Return the built-in replacement spec for ``target``."""

    if target is None:
        return None
    return REPLACEMENT_SPECS.get(target)


def strategy_spec(target: Optional[str]) -> Optional[StrategySpec]:
    """Return the built-in strategy spec for ``target``."""

    if target is None:
        return None
    return STRATEGY_SPECS.get(target)
