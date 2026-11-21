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
"""First inline pass: sink YAML module replacements into source constructors."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import replace
from typing import Any

from hyper_parallel.codegen.inline.ir import (
    ClassRemovalPatch,
    ConstructorReplacePatch,
    InlinePatchSet,
    SourceObservation,
    ModuleSnippetPatch,
)
from hyper_parallel.codegen.inline.framework_spec import (
    observed_constructor_spec, replacement_spec_for as replacement_spec,
)

logger = logging.getLogger(__name__)


def build_replacement_patches(
    rules: tuple[SourceObservation, ...], model_type: str | None = None
) -> InlinePatchSet:
    """Build constructor replacement patches from module actions.

    The rule's settled FQNs (``rule.match``, recorded from the same plan the
    native path applies) define the *scope* of each rewrite: only the
    ``self.<attr>`` call sites those FQNs name are sunk.  A rule that matched no
    module is not expanded at all -- generating a replacement the plan never
    decided would make the artifact diverge from the trained model.

    The plan records one entry per matched FQN, so the entries sharing a
    module implementation must be merged: taking only the first would scope
    the rewrite to a single matched attribute (``input_layernorm``) and silently
    leave its siblings (``post_attention_layernorm``, ``model.norm``) on the
    original class.

    Args:
        rules: Inline rules recovered from the plan's frozen metadata.
        model_type: Model family key used to resolve framework declarations.

    Returns:
        The constructor replacement patches for the matched rules.
    """

    patch_set = InlinePatchSet()
    scopes_by_target: dict[tuple, set[str]] = defaultdict(set)
    fqns_by_target: dict[tuple, set[str]] = defaultdict(set)
    specs_by_target: dict[tuple, Any] = {}
    source_fqns_by_target: dict[tuple, set[str]] = defaultdict(set)
    shared_by_target: dict[tuple, list[tuple[str, ...]]] = defaultdict(list)
    for rule in rules:
        target = rule.replace_target
        if target is None:
            continue
        spec = _rule_replacement_spec(rule, target, model_type)
        if spec is None:
            continue
        key = (target, rule.module_type, spec.keyword_args)
        specs_by_target.setdefault(key, spec)
        if rule.construction is not None:
            source_fqns_by_target[key].update(rule.construction["source_fqns"])
        if len(rule.shared_fqns) > 1:
            shared_by_target[key].append(rule.shared_fqns)
        scopes_by_target[key].update(_scope_attrs(rule))
        fqns_by_target[key].update(fqn for fqn in rule.match if fqn)
    for target, spec in specs_by_target.items():
        scope_attrs = tuple(sorted(scopes_by_target[target]))
        if not scope_attrs:
            logger.warning(
                "codegen: module replacement %r matched no module and is not expanded",
                target,
            )
            continue
        patch_set.imports.extend(spec.imports)
        if spec.replacement_note:
            patch_set.module_snippets.append(
                ModuleSnippetPatch(f"# Codegen replacement: {spec.replacement_note}")
            )
        patch_set.module_snippets.extend(spec.snippets)
        deferred = _requires_deferred_scope(spec, source_fqns_by_target[target],
                                            scope_attrs, fqns_by_target[target], shared_by_target[target])
        patch_set.constructor_replaces.append(
            ConstructorReplacePatch(
                old_ctor=spec.old_ctor,
                new_ctor=spec.new_ctor,
                mode="wrap_source" if deferred else spec.mode,  # type: ignore[arg-type]
                keyword_args=spec.keyword_args,
                scope_attrs=scope_attrs,
                scope_fqns=tuple(sorted(fqns_by_target[target])),
                defer_to_fqns=deferred,
                scope_complete=bool(source_fqns_by_target[target]) and not deferred,
                shared_fqns=tuple(shared_by_target[target]),
            )
        )
        if spec.remove_class:
            patch_set.class_removals.append(
                ClassRemovalPatch(old_name=spec.old_ctor, new_name=spec.new_ctor)
            )
    return patch_set


def _scope_attrs(rule: SourceObservation) -> tuple[str, ...]:
    """Attribute names of the rule's settled FQNs, i.e. the rewrite scope.

    ``model.layers.3.post_attention_layernorm`` scopes the rewrite to
    ``self.post_attention_layernorm = <ctor>(...)`` assignments, which is the
    granularity native's FQN matching produces.  ``model.norm`` is a sole-name
    FQN, so the whole FQN is the attribute.
    """

    return tuple(sorted({fqn.rsplit(".", 1)[-1] for fqn in rule.match if fqn}))


def _rule_replacement_spec(rule: SourceObservation, target: str, model_type: str | None) -> Any:
    """Keep observed constructors separate from legacy declaration-only rules."""
    if rule.construction is not None:
        return observed_constructor_spec(rule)
    spec = replacement_spec(target, model_type, module_type=rule.module_type)
    if spec is None:
        return None
    arguments = {item.split("=", 1)[0]: item for item in spec.keyword_args}
    arguments.update({item.split("=", 1)[0]: item for item in rule.construction_kwargs})
    return replace(spec, keyword_args=tuple(arguments.values()))


def _requires_deferred_scope(spec: Any, source_fqns: set[str], attrs: tuple[str, ...],
                             matches: set[str], shared: list[tuple[str, ...]]) -> bool:
    """Do not widen a partial or shared instance replacement into a class rewrite."""
    available = {fqn for fqn in source_fqns if fqn.rsplit(".", 1)[-1] in attrs}
    return spec.defer_to_fqns or bool(shared) or bool(source_fqns) and available != matches
