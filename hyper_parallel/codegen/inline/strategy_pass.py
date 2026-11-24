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
from typing import Mapping

from hyper_parallel.codegen.inline.ir import (
    ForwardExtractPatch,
    ImportPatch,
    InlinePatchSet,
    ModuleSnippetPatch,
    SourceObservation,
    StrategySpec,
)
from hyper_parallel.codegen.inline.framework_spec import CP_ATTENTION_KIND, strategy_spec_for as strategy_spec


def rule_block_classes(
    rule: SourceObservation, boundary_classes: Mapping[str, str] | None
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
    rules: tuple[SourceObservation, ...],
    model_type: str | None = None,
    boundary_classes: Mapping[str, str] | None = None,
) -> InlinePatchSet:
    """Materialize selected inner-wrapper forwards, deduplicating shared source bindings.

    Host classes come from each rule's observed FQNs. Local-compute methods
    are materialized separately from native records by ``emit.recorded``.
    """
    patch_set = InlinePatchSet()
    imported: set[str] = set()
    subclassed: dict[str, str] = {}
    for rule in rules:
        target = rule.inner_wrapper_target
        if target is None:
            continue
        spec = _rule_strategy(rule, target, model_type)
        if spec is None:
            continue
        patch_set.imports.extend(spec.imports)
        if target not in imported:
            imported.add(target)
            patch_set.module_snippets.extend(spec.snippets)
        if spec.subclass_shell is not None:
            _materialize_cp_forward(
                patch_set, spec, rule, target, model_type,
                boundary_classes, subclassed,
            )
    return patch_set


def _materialize_cp_forward(
    patch_set: InlinePatchSet,
    spec: StrategySpec,
    rule: SourceObservation,
    target: str,
    model_type: str | None,
    boundary_classes: Mapping[str, str] | None,
    subclassed: dict[str, str],
) -> None:
    """Materialize CP dispatch in the original host class exactly once."""
    host_classes = ({rule.observed_wrapper["host_class"]} if rule.observed_wrapper is not None
                    else rule_block_classes(rule, boundary_classes))
    if not host_classes:
        raise _unmaterialized_strategy_error(spec.kind, target, rule, model_type)
    for host_class in sorted(host_classes):
        if host_class in subclassed:
            if subclassed[host_class] != (spec.definition or spec.subclass_shell):
                raise RuntimeError(f"codegen: conflicting CP bindings for source class {host_class!r}")
            continue
        subclassed[host_class] = spec.definition or spec.subclass_shell
        patch_set.forward_extracts.append(
            ForwardExtractPatch(
                class_name=host_class,
                method_name=spec.method_name,
                body=spec.subclass_shell,
                definition=spec.definition,
            )
        )


def _unmaterialized_strategy_error(
    kind: str, target: str, rule: SourceObservation, model_type: str | None
) -> RuntimeError:
    """The error for a declared strategy that selects no boundary module."""
    return RuntimeError(
        f"codegen: {kind} strategy {target!r} matched no boundary "
        f"module (patterns {list(rule.match)}, model_type="
        f"{model_type!r}) — the declared strategy cannot be "
        "materialized; check the match pattern or the plan's "
        "boundary coverage"
    )


def _rule_strategy(
    rule: SourceObservation, target: str, model_type: str | None,
) -> StrategySpec | None:
    """Use an already-installed factory-tail method or the declared wrapper source."""
    if rule.observed_wrapper is not None:
        return StrategySpec(
            kind=CP_ATTENTION_KIND, subclass_shell=rule.observed_wrapper["body"],
            definition=rule.observed_wrapper.get("definition"),
            imports=tuple(ImportPatch(module, (name,)) for module, name in rule.observed_wrapper["imports"]),
            snippets=tuple(ModuleSnippetPatch(source) for source in rule.observed_wrapper.get("bindings", ())),
        )
    return strategy_spec(target, model_type, inner_wrapper=True, construction_kwargs=rule.construction_kwargs)
