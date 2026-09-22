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
import importlib
import inspect
from typing import Any, Iterable, Mapping

from hyper_parallel.codegen.inline.ir import ImportPatch, ModuleSnippetPatch
from hyper_parallel.codegen.inline.recognition import (
    ReplacementTarget,
    adapter_path_parts,
    archetype_for_factory,
    class_name_of,
    family_from_target,
    replacement_target,
)
from hyper_parallel.codegen.inline.spec_bundle import ReplacementSpec, StrategySpec

#: The framework CP attention strategy kind (structure-keyed, not family-named).
CP_ATTENTION_KIND = "cp_attention"

_EP_IMPORTS = (
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


def _component_class(target: ReplacementTarget) -> type:
    """Import the component class one factory constructs.

    Lazy by construction: the component's own module is imported only when a
    rule that constructs it is actually present (``components.modules``
    re-exports lazily, so this pulls in one submodule, never the whole set).
    """
    module = importlib.import_module(target.module)
    component = getattr(module, target.component, None)
    if not isinstance(component, type):
        raise RuntimeError(
            f"codegen: {target.module!r} does not export the class "
            f"{target.component!r} the replacement factory constructs"
        )
    return component


def _render_arguments(
    target: ReplacementTarget, component: type
) -> tuple[str, ...] | None:
    """Render the constructor keywords the generated call passes.

    Arguments the factory binds to its own parameters have no value at
    generation time, so the component's own constructor default is emitted --
    the framework never restates a component's keyword shape. ``None`` means the
    component does not accept one of the keywords the factory passes, i.e. the
    factory constructs an interface codegen cannot sink.
    """
    parameters = inspect.signature(component).parameters
    rendered: list[str] = []
    for argument in target.arguments:
        if argument.value is not None:
            rendered.append(f"{argument.name}={argument.value}")
            continue
        if argument.name not in parameters:
            return None
        rendered.append(f"{argument.name}={parameters[argument.name].default!r}")
    return tuple(rendered)


def _interface_function(factory: Any, target: ReplacementTarget) -> Any:
    """The family kernel entry the factory hands its component."""
    interface = getattr(factory, "__globals__", {}).get(target.interface)
    if not callable(interface):
        raise RuntimeError(
            f"codegen: replacement factory {getattr(factory, '__name__', factory)!r} "
            f"hands over {target.interface!r}, which its module does not define"
        )
    return interface


def _build_replacement_spec(
    factory: Any, target: ReplacementTarget, old_ctor: str
) -> ReplacementSpec | None:
    """Build the source-level replacement one factory declares.

    Every field comes from the factory's own structure: the class it constructs
    (and the module it imports it from), whether it hands the matched module over
    as a wrap, the constructor keywords it passes, and the family interface it
    hands to the component -- which is what makes the component's class rendered
    (visible in the artifact) instead of merely imported. ``None`` means the
    construction is not one codegen can sink into the generated ``__init__``.
    """
    component = _component_class(target)
    keyword_args = _render_arguments(target, component)
    if keyword_args is None:
        return None
    # A component that receives a family interface is rendered into the artifact,
    # so its class is defined there -- importing it as well would be dead weight
    # the rendered definition shadows.
    imports: list[ImportPatch] = (
        [] if target.interface else [ImportPatch(target.module, (target.component,))]
    )
    snippets: tuple[ModuleSnippetPatch, ...] = ()
    if target.interface:
        from hyper_parallel.codegen.inline.attention import (  # pylint: disable=C0415
            render_attention_class,
        )

        expanded = render_attention_class(
            component, interface=_interface_function(factory, target)
        )
        imports.extend(
            ImportPatch(module="", names=(), raw=line) for line in expanded.imports
        )
        snippets = (ModuleSnippetPatch(expanded.source),)
    mode = "wrap_source" if target.wraps_source else "name"
    return ReplacementSpec(
        old_ctor=old_ctor,
        new_ctor=target.component,
        mode=mode,
        keyword_args=keyword_args,
        imports=tuple(imports),
        remove_class=not target.wraps_source,
        snippets=snippets,
        replacement_note=(
            f"{target.component} wraps {old_ctor}; the original class is kept "
            "as the wrapper input."
            if target.wraps_source
            else f"{target.component} replaces {old_ctor}."
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
    """One construction path for every replacement: read the factory.

    Whatever component a family's factory constructs, the spec is derived from
    that factory's own source, so a new family (or a new component inside one)
    needs no codegen declaration.  A factory with no constructor call to sink --
    or a rule with no ``module_type`` naming the source class it replaces -- has
    no spec, and the rule is left to the native path.
    """
    del model_type, target
    # The source class the constructor call replaces: without the rule's
    # ``module_type`` there is no call site to rewrite, and guessing one would
    # silently target the wrong class.
    old_ctor = class_name_of(module_type)
    if not old_ctor:
        return None
    replacement = replacement_target(factory)
    if replacement is None:
        return None
    return _build_replacement_spec(factory, replacement, old_ctor)


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
        return _ep_strategy_spec(archetype)
    if inner_wrapper and adapter_path_parts(target):
        return StrategySpec(kind=CP_ATTENTION_KIND, imports=())
    return None


def _ep_strategy_spec(archetype: Any) -> StrategySpec:
    """Build the EP strategy spec: the inlined forward plus its imports.

    The inlined EP forward orchestrates through the instance-attribute channel
    (``self.ep_enable`` / ``self._hyper_ep_group``) like every other generated
    forward, so it declares no module-level snippet.
    """
    from hyper_parallel.codegen.inline.templates import moe_ep_forward_body  # pylint: disable=C0415

    return StrategySpec(
        kind=archetype.kind,
        target_class=archetype.target_class,
        body_template=moe_ep_forward_body(
            router_kind=archetype.router_kind, shared=archetype.shared
        ),
        imports=_EP_IMPORTS,
        strip_boundary_subpatterns=archetype.strip_boundary_subpatterns,
    )


def interface_component_class(model_type: str | None) -> str | None:
    """The class this family's factories render, because one hands over an interface.

    Structural: the component is named by the replacement factory itself, so
    codegen carries no fused-attention class literal.  A rendered component
    keeps the component's own ``forward`` and therefore takes a *compiled
    boundary* (``hyper_install_boundaries``) rather than external state — see
    :func:`external_state_classes_for`.
    """
    module = _family_provider(model_type, "replacements")
    if module is None:
        return None
    for name in dir(module):
        factory = getattr(module, name)
        if not callable(factory) or getattr(factory, "_hp_module_replacement", False) is not True:
            continue
        target = replacement_target(factory)
        if target is not None and target.interface:
            return target.component
    return None


def external_state_classes_for(
    model_type: str | None,
    injections: Iterable[Mapping[str, Any]] | None = None,
) -> tuple[str, ...]:
    """Classes whose ``forward`` an inline strategy body *replaces*.

    Structure-proven, not declared: every strategy target this plan injects is
    resolved to its :class:`StrategySpec`, and a spec that carries a
    ``body_template`` rewrites that class's whole ``forward`` with an inline
    orchestration body (the MoE EP shell).  Such a class owns its collectives,
    so it must keep its parameter sharding without a compiled boundary — it is
    skipped by the boundary / compute / inner-wrapper installers and reads its
    parallel channel from install-time instance attributes instead.

    A *rendered* component class (the class this family's replacement
    factories build, e.g. the fused attention) is deliberately NOT one of
    these: its forward is the component's own plus the emitted boundary form,
    so the shared compiled-boundary path
    (``hyper_install_boundaries`` / ``hyper_apply_inner_wrapper``) owns it.
    """
    inlined: set[str] = set()
    for record in injections or ():
        spec = strategy_spec_for(
            _target_path(record.get("local_compute_fn")), model_type
        )
        if spec is None or spec.body_template is None or spec.target_class is None:
            continue
        inlined.add(spec.target_class)
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
    "declaration_summary",
    "external_state_classes_for",
    "interface_component_class",
    "replacement_spec_for",
    "strategy_spec_for",
]
