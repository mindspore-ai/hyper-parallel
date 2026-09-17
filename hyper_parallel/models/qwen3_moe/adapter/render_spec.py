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
"""Qwen3-MoE render spec: source-level declarations for the inline pipeline.

Declares what the inline pipeline must render for this family: which source
classes are replaced by which fused modules, how their constructors are
rewritten, and which strategy bodies are inlined into the generated artifact.
Consumed through ``ModelAdapterSpec.inline_codegen``.
"""

from __future__ import annotations

from hyper_parallel.codegen.inline.ir import ImportPatch, ModuleSnippetPatch
from hyper_parallel.codegen.inline.spec_bundle import (
    InlineSpecBundle,
    ReplacementSpec,
    StrategySpec,
)
from hyper_parallel.codegen.inline.templates import QWEN3_MOE_EP_FORWARD, TP_OPERATORS_CLASS


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

_META_NORMALIZERS: tuple = ()

def _build_attention_replacement_spec() -> ReplacementSpec:
    """Render the GQA attention class from real component source.

    Build-time only: resolving the real ``GQAAttention`` component and calling
    ``render_attention_class`` imports torch / torch_npu and the generic
    component modules, so this must never run at module import. The kernel
    entry ``run_qwen3_moe_flash_attention`` (previously imported) is inlined
    into the artifact by the generator, so no import of it is needed here.
    """
    import importlib  # pylint: disable=C0415

    from hyper_parallel.codegen.inline.attention import (  # pylint: disable=C0415
        render_attention_class,
    )

    modules = importlib.import_module("hyper_parallel.components.modules")
    adapter_attention = importlib.import_module(
        "hyper_parallel.models.qwen3_moe.adapter.attention"
    )
    expanded = render_attention_class(
        modules.GQAAttention,
        interface=adapter_attention.run_qwen3_moe_flash_attention,
    )
    generated_imports = tuple(
        ImportPatch(module="", names=(), raw=line) for line in expanded.imports
    )
    return ReplacementSpec(
        old_ctor="Qwen3MoeAttention",
        new_ctor="GQAAttention",
        # The generated class copies the real component's keyword-only ctor
        # ``(*, module=..., attention_interface=...)``, so the decoder-layer
        # call must wrap the source attention module exactly like the runtime
        # ``replace_qwen3_moe_flash_attention`` does. The source class is kept
        # (``remove_class=False``) so the wrapper input can be instantiated.
        mode="wrap_source",
        keyword_args=(
            "module_fqn=''",
            "context=None",
            "attention_interface=run_qwen3_moe_flash_attention",
        ),
        remove_class=False,
        imports=(
            ImportPatch("hyper_parallel.codegen.runtime", ("get_inline_parallel_state",)),
            ImportPatch("hyper_parallel.components.modules", ("RMSNorm",)),
            ImportPatch("hyper_parallel.platform", ("get_platform",)),
            *generated_imports,
        ),
        snippets=(
            ModuleSnippetPatch(TP_OPERATORS_CLASS),
            ModuleSnippetPatch(expanded.source),
        ),
        replacement_note="GQAAttention replaces Qwen3MoeAttention.",
    )


_RENDER_SPEC_CACHE: InlineSpecBundle | None = None


def get_render_spec() -> InlineSpecBundle:
    """Return the Qwen3-MoE render spec.

    The attention replacement is assembled lazily on first call (never at
    module import) so ``render_spec`` itself does not pull torch / torch_npu or
    the real component. The bundle is cached so callers see stable ``ReplacementSpec``
    identity across lookups. All other declarations remain static.
    """
    global _RENDER_SPEC_CACHE  # pylint: disable=global-statement

    if _RENDER_SPEC_CACHE is None:
        replacement_specs = dict(_REPLACEMENT_SPECS)
        replacement_specs[QWEN3_MOE_FLASH_ATTENTION_REPLACEMENT] = _build_attention_replacement_spec()
        _RENDER_SPEC_CACHE = InlineSpecBundle(
            replacement_specs=replacement_specs,
            strategy_specs=_STRATEGY_SPECS,
            meta_normalizers=_META_NORMALIZERS,
            external_state_classes=("GQAAttention", "Qwen3MoeSparseMoeBlock"),
        )
    return _RENDER_SPEC_CACHE


__all__ = ["get_render_spec"]
