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

from fnmatch import fnmatch
from typing import Collection, Mapping

from hyper_parallel.codegen.inline.ir import ForwardExtractPatch, InlinePatchSet, InlineRule
from hyper_parallel.codegen.inline.specs import strategy_spec


def rule_block_classes(
    rule: InlineRule, boundary_classes: Mapping[str, str] | None
) -> set[str]:
    """The concrete classes a rule's match patterns select in the boundary map.

    Per-model fact, derived (never registered): the frozen boundary map names
    each matched FQN's class, so any custom model class picked up by the
    pattern materializes without a framework-side registration.
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
) -> InlinePatchSet:
    """Build inline forward patches from strategy targets.

    A plan carries one rule per matched FQN, so the same strategy target (and
    therefore the same imports and module snippets) is seen once per module it
    applies to.  Those contributions are deduplicated here — mirroring
    ``build_replacement_patches`` — so a module-level snippet such as the
    parallel-state accessor is emitted once per generated file instead of once
    per matched layer.

    A spec whose body inlines a forward but carries no literal target class
    (the EP strategy) applies to every class the rule's match selects in
    ``boundary_classes``: the block class is a per-model fact, so the patch is
    materialized per artifact rather than cached in the capability spec. A
    declared EP strategy that selects no boundary fails fast — a silently
    unmaterialized strategy is an artifact that only looks distributed.
    """

    patch_set = InlinePatchSet()
    emitted: dict[tuple[str, str], str] = {}
    seen: set[tuple[str, bool]] = set()
    for rule in rules:
        for target in (rule.local_compute_target, rule.inner_wrapper_target):
            if target is None:
                continue
            inner_wrapper = target == rule.inner_wrapper_target
            if (target, inner_wrapper) in seen:
                continue
            spec = strategy_spec(target, model_type, inner_wrapper=inner_wrapper)
            if spec is None:
                continue
            seen.add((target, inner_wrapper))
            patch_set.imports.extend(spec.imports)
            patch_set.module_snippets.extend(spec.snippets)
            if spec.body_template is None:
                continue
            if spec.target_class is not None:
                host_classes = {spec.target_class}
            else:
                host_classes = rule_block_classes(rule, boundary_classes)
                if not host_classes:
                    # Re-emission of an already-normalized meta: the boundary
                    # entries the first pass stripped are gone, but the pass
                    # persisted the derived classes in
                    # ``meta.external_state_classes`` — the surviving record of
                    # which classes an inline body replaced. (Today the EP
                    # shell is the only body-bearing strategy, so the whole
                    # record is this rule's host set; a second body-bearing
                    # strategy would need a kind filter here.)
                    host_classes = set(external_state_classes or ())
                if not host_classes:
                    raise RuntimeError(
                        f"codegen: EP strategy {target!r} matched no boundary "
                        f"module (patterns {list(rule.match)}, model_type="
                        f"{model_type!r}) — the declared strategy cannot be "
                        "materialized; check the match pattern or the plan's "
                        "boundary coverage"
                    )
            for host_class in sorted(host_classes):
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
    return patch_set
