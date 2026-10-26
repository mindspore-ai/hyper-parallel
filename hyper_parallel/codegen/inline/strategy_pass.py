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
"""Second inline pass: insert parallel strategies into replaced modules."""

from __future__ import annotations

import logging
from fnmatch import fnmatch
from typing import Collection, Mapping

from hyper_parallel.codegen.inline.ir import (
    ConstructorReplacePatch,
    ForwardExtractPatch,
    InlinePatchSet,
    InlineRule,
    ModuleSnippetPatch,
)
from hyper_parallel.codegen.inline.replacement_pass import _scope_attrs
from hyper_parallel.codegen.inline.spec_bundle import StrategySpec
from hyper_parallel.codegen.inline.specs import strategy_spec
from hyper_parallel.codegen.inline.templates import cp_subclass_snippet

logger = logging.getLogger(__name__)


def rule_block_classes(
    rule: InlineRule, boundary_classes: Mapping[str, str] | None
) -> set[str]:
    """The concrete classes a rule's match patterns select in the boundary map.

    Per-model fact, derived (never registered): the frozen boundary map names
    each matched FQN's class, so any custom model class picked up by the
    pattern materializes without a framework-side registration.

    Args:
        rule: The inline rule whose match patterns select host classes.
        boundary_classes: Frozen FQN -> class name map.

    Returns:
        The class names the rule's patterns select.
    """
    classes: set[str] = set()
    for fqn, class_name in (boundary_classes or {}).items():
        if any(fnmatch(fqn, pattern) for pattern in rule.match):
            classes.add(class_name)
    return classes


def build_strategy_patches(
    rules: tuple[InlineRule, ...],
    model_type: str | None = None,
    boundary_classes: Mapping[str, str] | None = None,
    external_state_classes: Collection[str] | None = None,
    source_classes: Collection[str] | None = None,
) -> InlinePatchSet:
    """Build inline forward patches from strategy targets.

    A plan carries one rule per matched FQN, so the same strategy target (and
    therefore the same imports and module snippets) is seen once per module it
    applies to.  Those contributions are deduplicated here — mirroring
    ``build_replacement_patches`` — so a module-level snippet such as the
    parallel-state accessor is emitted once per generated file instead of once
    per matched layer.

    Host derivation is deliberately *not* deduplicated per target: rules that
    share one strategy target may select different host classes (one EP
    compute target serving both dense-MLP and MoE layers), so every rule
    derives its hosts and the per-class ``emitted`` bookkeeping keeps one
    materialization per host — mirroring how the freeze side derives
    ``external_state_classes`` per record.

    A spec whose body inlines a forward but carries no literal target class
    (the EP strategy) applies to every class the rule's match selects in
    ``boundary_classes``: the block class is a per-model fact, so the patch is
    materialized per artifact rather than cached in the capability spec.  A
    spec that carries a ``subclass_shell`` (inline CP) instead materializes an
    in-artifact subclass of the attention host plus a scoped constructor swap,
    and never a forward extract — the host keeps its compiled boundary.  A
    declared strategy that selects no boundary fails fast — a silently
    unmaterialized strategy is an artifact that only looks distributed.

    ``source_classes`` names the classes the *main* modeling source defines.
    A host class missing from it is a sibling-file class (a two-file model
    whose language model lives in a copied sibling module): the forward
    extract cannot target it, so the host is skipped and its strategy
    resolves at install time through the runtime local-region skeleton
    (``hp_bind_compute``), the same channel an imported-class boundary uses.
    ``None`` (the pre-split signature) keeps the historical behavior of
    trusting every derived host.

    Args:
        rules: Inline rules recovered from the frozen meta.
        model_type: Model family key used to resolve framework declarations.
        boundary_classes: Frozen FQN -> class name map.
        external_state_classes: Classes a prior emission recorded as inline.
        source_classes: Class names the main modeling source defines, or
            ``None`` when the caller did not index the source.

    Returns:
        The inline patch set for the plan's strategy targets.
    """

    patch_set = InlinePatchSet()
    emitted: dict[tuple[str, str], str] = {}
    imported: set[tuple[str, bool]] = set()
    subclassed: set[str] = set()
    for rule in rules:
        for target in (rule.local_compute_target, rule.inner_wrapper_target):
            if target is None:
                continue
            inner_wrapper = target == rule.inner_wrapper_target
            spec = strategy_spec(target, model_type, inner_wrapper=inner_wrapper)
            if spec is None:
                continue
            if (target, inner_wrapper) not in imported:
                imported.add((target, inner_wrapper))
                patch_set.imports.extend(spec.imports)
                patch_set.module_snippets.extend(spec.snippets)
            if spec.subclass_shell is not None:
                _materialize_cp_subclass(
                    patch_set, spec, rule, target, model_type,
                    boundary_classes, subclassed,
                )
                continue
            if spec.body_template is None:
                continue
            host_classes = _host_classes_for(
                spec, rule, target, model_type,
                boundary_classes, external_state_classes,
            )
            inlineable, deferred = _split_source_hosts(
                host_classes, source_classes
            )
            for host_class in sorted(inlineable):
                key = (host_class, spec.method_name)
                if key in emitted:
                    if emitted[key] != spec.body_template:
                        raise ValueError(f"Conflicting inline strategies for {host_class}.{spec.method_name}")
                    continue
                emitted[key] = spec.body_template
                patch_set.forward_extracts.append(
                    ForwardExtractPatch(
                        class_name=host_class,
                        method_name=spec.method_name,
                        body=spec.body_template,
                    )
                )
            if deferred:
                logger.info(
                    "codegen: EP strategy %s host classes %s are defined in a "
                    "sibling module, not the main modeling source; the "
                    "forward extract is skipped and the runtime local-region "
                    "skeleton (hp_bind_compute) carries the strategy at "
                    "install time",
                    target, sorted(deferred),
                )
    return patch_set


def _host_classes_for(
    spec: StrategySpec,
    rule: InlineRule,
    target: str,
    model_type: str | None,
    boundary_classes: Mapping[str, str] | None,
    external_state_classes: Collection[str] | None,
) -> set[str]:
    """Derive the host classes a body-bearing strategy rewrites.

    A spec with a literal ``target_class`` (a framework-fixed attention host)
    rewrites that class alone.  Otherwise the rule's match patterns select
    hosts from the frozen boundary map; when they select none — re-emission
    of an already-normalized meta, whose boundary entries the first pass
    stripped — the classes a prior emission persisted in
    ``meta.external_state_classes`` are the surviving record of which classes
    an inline body replaced.  (Today the EP shell is the only body-bearing
    strategy, so the whole record is this rule's host set; a second
    body-bearing strategy would need a kind filter here.)  A declared
    strategy that selects no host at all fails fast.

    Args:
        spec: The body-bearing strategy spec being materialized.
        rule: The inline rule carrying the strategy's match patterns.
        target: Dotted strategy target path, for error reporting.
        model_type: Model family key, for error reporting.
        boundary_classes: Frozen FQN -> class name map.
        external_state_classes: Classes a prior emission recorded as inline.

    Returns:
        The host class names the strategy's patches rewrite.
    """
    if spec.target_class is not None:
        return {spec.target_class}
    host_classes = rule_block_classes(rule, boundary_classes)
    if not host_classes:
        host_classes = set(external_state_classes or ())
    if not host_classes:
        raise _unmaterialized_strategy_error(spec.kind, target, rule, model_type)
    return host_classes


def _split_source_hosts(
    host_classes: set[str], source_classes: Collection[str] | None
) -> tuple[set[str], set[str]]:
    """Split derived hosts into inlineable vs sibling-module classes.

    A host the main modeling source does not define is defined in a sibling
    module the bundle copies verbatim (a two-file model whose language model
    class tree lives in ``modeling_kimi_linear.py``).  No patch may rewrite
    the sibling: the main file is the only file codegen edits, so such hosts
    defer to the runtime local-region skeleton instead.  ``source_classes``
    ``None`` means the caller did not index the source (the pre-split call
    shape) and every host stays inlineable, preserving old behavior.
    """
    if source_classes is None:
        return host_classes, set()
    inlineable = {
        host for host in host_classes if host in source_classes
    }
    deferred = host_classes - inlineable
    return inlineable, deferred


def _materialize_cp_subclass(
    patch_set: InlinePatchSet,
    spec: StrategySpec,
    rule: InlineRule,
    target: str,
    model_type: str | None,
    boundary_classes: Mapping[str, str] | None,
    subclassed: set[str],
) -> None:
    """Materialize the in-artifact CP dispatch subclass for one rule's hosts.

    The subclass is emitted once per host class (the plan carries one rule
    per matched FQN, so repeated rules re-derive the same host): a
    ``ModuleSnippetPatch`` with the full class source — carrying the
    ``_hp_inline_cp`` marker the runtime installers read and, when the
    family declares one, the ``_hp_cp_state_attr`` companion contract its
    shared CP kernel reads — plus a scoped ``name``-mode constructor swap
    that retargets the matched assignments to
    the subclass.  No forward extract is produced: the host keeps its
    compiled boundary, and the subclass dispatches to the parent or the
    family helper inside it.
    """
    host_classes = rule_block_classes(rule, boundary_classes)
    if not host_classes:
        raise _unmaterialized_strategy_error(spec.kind, target, rule, model_type)
    helper_names = [name for patch in spec.imports for name in patch.names]
    if len(helper_names) != 1:
        raise ValueError(
            f"codegen: CP strategy {target!r} must carry exactly one helper "
            f"import for its subclass shell; got {helper_names}"
        )
    for host_class in sorted(host_classes):
        subclass_name = f"{host_class}CP"
        if subclass_name in subclassed:
            continue
        subclassed.add(subclass_name)
        patch_set.module_snippets.append(
            ModuleSnippetPatch(
                cp_subclass_snippet(
                    subclass_name, host_class, helper_names[0],
                    spec.subclass_state_attr,
                )
            )
        )
        patch_set.constructor_replaces.append(
            ConstructorReplacePatch(
                old_ctor=host_class,
                new_ctor=subclass_name,
                mode="name",
                scope_attrs=_scope_attrs(rule),
            )
        )


def _unmaterialized_strategy_error(
    kind: str, target: str, rule: InlineRule, model_type: str | None
) -> RuntimeError:
    """The error for a declared strategy that selects no boundary module."""
    return RuntimeError(
        f"codegen: {kind} strategy {target!r} matched no boundary "
        f"module (patterns {list(rule.match)}, model_type="
        f"{model_type!r}) — the declared strategy cannot be "
        "materialized; check the match pattern or the plan's "
        "boundary coverage"
    )
