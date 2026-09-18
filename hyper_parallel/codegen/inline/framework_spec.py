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
"""Resolve inline codegen declarations from structure, not family render specs.

This module replaces per-family ``render_spec.py`` files. Every fact a
generated artifact needs is derived from the family's *existing runtime
providers* plus the frozen plan, never from a family-name dispatch table:

* a YAML ``replace_module._target_`` names the family's own
  ``@module_replacement`` factory; inspecting that factory's source reveals
  which generic component it constructs (``RMSNorm`` / ``GroupedExperts`` /
  ``GQAAttention``) and, for fused attention, the kernel entry it hands over;
* a YAML ``local_compute_fn._target_`` that names a known MoE archetype
  factory resolves to that archetype's strategy kind / target class / boundary
  sub-patterns / EP body keys;
* a CP ``inner_wrapper`` target under the ``models/<family>/adapter``
  convention resolves to the framework CP attention strategy (imports only).

The source class name an artifact must model (``Qwen3MoeRMSNorm``) is not
duplicated here either: it is read from the rule's own ``module_type``, which
the YAML already carries for its type check. No model-family codegen
declaration remains.

Heavy resolution (importing the real generic component to render a fused
attention class) stays lazy: it only runs when a fused-attention target is
actually present, so importing this module never pulls torch / torch_npu.
Resolved specs are cached so repeated lookups stay cheap and stable (callers
may rely on object identity across calls).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Mapping

from hyper_parallel.codegen.inline.components import (
    gqa_attention_replacement,
    grouped_experts_replacement,
    rms_norm_replacement,
)
from hyper_parallel.codegen.inline.ir import ImportPatch, ModuleSnippetPatch
from hyper_parallel.codegen.inline.recognition import (
    COMPONENT_GQA_ATTENTION,
    COMPONENT_GROUPED_EXPERTS,
    COMPONENT_RMS_NORM,
    adapter_path_parts,
    archetype_for_factory,
    class_name_of,
    factory_component,
    family_from_target,
)
from hyper_parallel.codegen.inline.spec_bundle import ReplacementSpec, StrategySpec
from hyper_parallel.codegen.inline.templates import PARALLEL_STATE_ACCESSOR, TP_OPERATORS_CLASS

#: The framework CP attention strategy kind (structure-keyed, not family-named).
CP_ATTENTION_KIND = "cp_attention"

#: The generic fused-attention class name this framework generates.
GENERATED_ATTENTION_CLASS = "GQAAttention"

_CP_ATTENTION_IMPORTS = (
    ImportPatch("hyper_parallel.distributed.context_parallel", ("flex_cp_allgather",)),
    ImportPatch(
        "hyper_parallel.distributed.context_parallel.attention",
        ("_cp_offset_causal_mask",),
    ),
)

_EP_IMPORTS = (
    ImportPatch("hyper_parallel.codegen.runtime", ("get_inline_parallel_state",)),
    ImportPatch("hyper_parallel.distributed.expert_parallel.experts", ("moe_ep_forward",)),
)


def _target_path(value: Any) -> str | None:
    """Return the dotted ``_target_`` of a projected override value."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        target = value.get("_target_") or value.get("target")
        return str(target) if target else None
    return None


def _family_provider(model_type: str | None, provider: str) -> Any | None:
    """Return one family adapter provider module, or ``None`` if unregistered."""
    if not model_type:
        return None
    from hyper_parallel.models.registry import get_model_adapter  # pylint: disable=C0415

    adapter = get_model_adapter(model_type)
    loader = getattr(adapter, provider, None) if adapter is not None else None
    if loader is None:
        return None
    return loader()


def _family_key(model_type: str | None, target: str | None) -> str | None:
    """Canonical family key for one lookup.

    Every spelling of an identity (``qwen3_moe`` / ``Qwen3MoeForCausalLM``)
    normalizes to the registered ``model_type``, so repeated lookups share one
    resolved spec -- callers may rely on object identity.  A target under the
    ``models/<family>/adapter`` convention supplies the family when the caller
    did not name one.  An unregistered identity is kept verbatim so resolution
    still fails closed (``None``) instead of falling back to another family.
    """
    if model_type is None:
        model_type = family_from_target(target)
    if not model_type:
        return None
    from hyper_parallel.models.registry import get_model_adapter  # pylint: disable=C0415

    adapter = get_model_adapter(model_type)
    return adapter.model_type if adapter is not None else model_type


def _factory_function(target: str, model_type: str | None):
    """Resolve a YAML replacement ``_target_`` to the family's real factory.

    Returns ``None`` when the target is not a factory exported by this family's
    replacements provider — a target the framework cannot structurally resolve.
    """
    module = _family_provider(model_type, "replacements")
    if module is None:
        return None
    prefix = module.__name__ + "."
    if not target.startswith(prefix):
        return None
    name = target[len(prefix):]
    if "." in name:
        return None
    return getattr(module, name, None)


def _build_attention_spec(
    model_type: str | None, old_ctor: str, interface_name: str
) -> ReplacementSpec:
    """Render the fused attention replacement from the real component source.

    Lazy by construction: importing the generic attention component and its
    NPU-backed kernels only happens when a fused-attention target is present.
    """
    import importlib  # pylint: disable=C0415

    from hyper_parallel.codegen.inline.attention import (  # pylint: disable=C0415
        render_attention_class,
    )

    attention_module = _family_provider(model_type, "attention")
    interface = getattr(attention_module, interface_name, None)
    if interface is None:
        raise RuntimeError(
            f"codegen: family {model_type!r} declares attention interface "
            f"{interface_name!r} but its attention provider does not export it"
        )
    modules = importlib.import_module("hyper_parallel.components.modules")
    expanded = render_attention_class(modules.GQAAttention, interface=interface)
    generated_imports = tuple(
        ImportPatch(module="", names=(), raw=line) for line in expanded.imports
    )
    return gqa_attention_replacement(
        old_ctor=old_ctor,
        attention_interface=interface_name,
        generated_imports=generated_imports,
        snippets=(
            ModuleSnippetPatch(TP_OPERATORS_CLASS),
            ModuleSnippetPatch(expanded.source),
        ),
    )


#: Resolved-spec caches. Keyed by everything the resolution reads, so a
#: repeated lookup returns the same object (callers may compare identity) and
#: the expensive fused-attention render runs once.
_REPLACEMENT_CACHE: dict[tuple[str | None, Any, str | None], ReplacementSpec | None] = {}
_STRATEGY_CACHE: dict[tuple[str | None, str | None, bool], StrategySpec | None] = {}


def replacement_spec_for(
    target: str | None,
    model_type: str | None = None,
    *,
    module_type: str | None = None,
) -> ReplacementSpec | None:
    """Resolve a ``replace_module`` target to its structural component spec.

    ``module_type`` is the matched source module's dotted type path from the
    YAML rule; the source class name (``old_ctor``) is its class name — a
    property of the matched source module, never a hand-written literal.
    """
    if target is None or not adapter_path_parts(target):
        return None
    model_type = _family_key(model_type, target)
    factory = _factory_function(target, model_type)
    if factory is None:
        return None
    # The spec is derived from the *factory's* structure, so the factory is
    # part of the key: a target re-pointed at a different factory (or a family
    # re-registered with one) must not reuse the previous resolution.
    key = (model_type, factory, module_type)
    if key in _REPLACEMENT_CACHE:
        return _REPLACEMENT_CACHE[key]
    spec = _resolve_replacement_spec(factory, model_type, module_type, target)
    _REPLACEMENT_CACHE[key] = spec
    return spec


def _resolve_replacement_spec(
    factory: Any, model_type: str | None, module_type: str | None, target: str
) -> ReplacementSpec | None:
    # Every generic component is wired *to a source class*: the constructor
    # call it replaces and (for the plain swap) the class it removes.  Without
    # the rule's ``module_type`` there is no source class to name, and guessing
    # one would silently target the wrong call site.
    old_ctor = class_name_of(module_type)
    if not old_ctor:
        return None
    component, interface_name = factory_component(factory)
    if component == COMPONENT_RMS_NORM:
        return rms_norm_replacement(old_ctor)
    if component == COMPONENT_GROUPED_EXPERTS:
        return grouped_experts_replacement(old_ctor)
    if component == COMPONENT_GQA_ATTENTION:
        if not interface_name:
            raise RuntimeError(
                f"codegen: fused attention factory {target!r} declares no "
                "attention_interface keyword"
            )
        return _build_attention_spec(model_type, old_ctor, interface_name)
    return None


def strategy_spec_for(
    target: str | None,
    model_type: str | None = None,
    *,
    inner_wrapper: bool = False,
) -> StrategySpec | None:
    """Resolve a strategy target from structure.

    A known MoE archetype factory (EP) resolves to its structural strategy; a
    CP ``inner_wrapper`` target under the adapter convention resolves to the
    framework CP attention strategy (imports only). Anything else is
    unresolvable and yields ``None`` — recognition never guesses a strategy.
    """
    if target is None:
        return None
    model_type = _family_key(model_type, target)
    key = (target, model_type, inner_wrapper)
    if key in _STRATEGY_CACHE:
        return _STRATEGY_CACHE[key]
    spec = _resolve_strategy_spec(target, model_type, inner_wrapper)
    _STRATEGY_CACHE[key] = spec
    return spec


def _resolve_strategy_spec(
    target: str, model_type: str | None, inner_wrapper: bool
) -> StrategySpec | None:
    archetype = archetype_for_factory(target)
    if archetype is not None:
        return _ep_strategy_spec(archetype, model_type)
    if inner_wrapper and adapter_path_parts(target):
        return StrategySpec(kind=CP_ATTENTION_KIND, imports=_CP_ATTENTION_IMPORTS)
    return None


def _ep_strategy_spec(archetype: Any, model_type: str | None) -> StrategySpec:
    from hyper_parallel.codegen.inline.templates import moe_ep_forward_body  # pylint: disable=C0415

    # The EP forward fully inlines the routed + shared path, which calls
    # ``get_parallel_state`` directly; a family that also generates a fused
    # attention class already provides that accessor via its attention snippet.
    snippets: tuple[ModuleSnippetPatch, ...] = ()
    if not _has_fused_attention(model_type):
        snippets = (ModuleSnippetPatch(PARALLEL_STATE_ACCESSOR),)
    return StrategySpec(
        kind=archetype.kind,
        target_class=archetype.target_class,
        body_template=moe_ep_forward_body(
            router_kind=archetype.router_kind, shared=archetype.shared
        ),
        imports=_EP_IMPORTS,
        snippets=snippets,
        strip_boundary_subpatterns=archetype.strip_boundary_subpatterns,
    )


def _has_fused_attention(model_type: str | None) -> bool:
    """Whether this family generates a fused attention class (accessor owner)."""
    module = _family_provider(model_type, "replacements")
    if module is None:
        return False
    for name in dir(module):
        factory = getattr(module, name)
        if not callable(factory) or getattr(factory, "_hp_module_replacement", False) is not True:
            continue
        component, _ = factory_component(factory)
        if component == COMPONENT_GQA_ATTENTION:
            return True
    return False


def external_state_classes_for(
    model_type: str | None,
    injections: Iterable[Mapping[str, Any]] | None = None,
) -> tuple[str, ...]:
    """Classes whose forwards the inline pipeline fully inlines.

    Structure-proven, not declared: the MoE block classes named by the EP
    compute factories this plan actually injects (``local_compute_fn._target_``
    -> archetype -> block class) plus the generated fused-attention class name
    when this family's replacements build one.
    """
    inlined: set[str] = set()
    for record in injections or ():
        archetype = archetype_for_factory(_target_path(record.get("local_compute_fn")))
        if archetype is not None:
            inlined.add(archetype.target_class)
    if _has_fused_attention(model_type):
        inlined.add(GENERATED_ATTENTION_CLASS)
    return tuple(sorted(inlined))


def declaration_summary(
    model_type: str | None,
    overrides: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Canonical, JSON-safe summary of the declarations an override set resolves to.

    Adapter declarations live outside the codegen implementation tree, so their
    resolved content must invalidate cached artifacts. Summarizing exactly the
    YAML targets this override set will inline keeps that invalidation faithful
    without a per-family declaration site.
    """
    replacement_specs: dict[str, Any] = {}
    strategy_specs: dict[str, Any] = {}
    for entry in overrides or ():
        replace_target = _target_path(entry.get("replace_module"))
        if replace_target is not None:
            spec = replacement_spec_for(
                replace_target, model_type, module_type=entry.get("module_type")
            )
            replacement_specs[replace_target] = asdict(spec) if spec is not None else None
        local_target = _target_path(entry.get("local_compute_fn"))
        if local_target is not None:
            spec = strategy_spec_for(local_target, model_type)
            strategy_specs[local_target] = asdict(spec) if spec is not None else None
        inner_target = _target_path(entry.get("inner_wrapper"))
        if inner_target is not None:
            spec = strategy_spec_for(inner_target, model_type, inner_wrapper=True)
            strategy_specs[inner_target] = asdict(spec) if spec is not None else None
    return {"replacement_specs": replacement_specs, "strategy_specs": strategy_specs}


__all__ = [
    "CP_ATTENTION_KIND",
    "GENERATED_ATTENTION_CLASS",
    "declaration_summary",
    "external_state_classes_for",
    "replacement_spec_for",
    "strategy_spec_for",
]
