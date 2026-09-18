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
from typing import Any

from hyper_parallel.codegen.inline.ir import (
    ClassRemovalPatch,
    ConstructorReplacePatch,
    InlinePatchSet,
    InlineRule,
    ModuleSnippetPatch,
)
from hyper_parallel.codegen.inline.specs import replacement_spec

logger = logging.getLogger(__name__)


def build_replacement_patches(rules: tuple[InlineRule, ...], model_type: str | None = None) -> InlinePatchSet:
    """Build constructor replacement patches from ``replace_module`` targets.

    The rule's settled FQNs (``rule.match``, recorded from the same plan the
    native path applies) define the *scope* of each rewrite: only the
    ``self.<attr>`` call sites those FQNs name are sunk.  A rule that matched no
    module is not expanded at all -- generating a replacement the plan never
    decided would make the artifact diverge from the trained model.

    The plan records one entry per matched FQN, so the entries sharing a
    ``replace_module`` target must be merged: taking only the first would scope
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
    scopes_by_target: dict[str, set[str]] = defaultdict(set)
    specs_by_target: dict[str, Any] = {}
    for rule in rules:
        target = rule.replace_target
        if target is None:
            continue
        spec = replacement_spec(target, model_type, module_type=rule.module_type)
        if spec is None:
            continue
        specs_by_target.setdefault(target, spec)
        scopes_by_target[target].update(_scope_attrs(rule))
    for target, spec in specs_by_target.items():
        scope_attrs = tuple(sorted(scopes_by_target[target]))
        if not scope_attrs:
            logger.warning(
                "codegen: replace_module %r matched no module and is not expanded",
                target,
            )
            continue
        patch_set.imports.extend(spec.imports)
        if spec.replacement_note:
            patch_set.module_snippets.append(
                ModuleSnippetPatch(f"# Codegen replacement: {spec.replacement_note}")
            )
        patch_set.module_snippets.extend(spec.snippets)
        patch_set.constructor_replaces.append(
            ConstructorReplacePatch(
                old_ctor=spec.old_ctor,
                new_ctor=spec.new_ctor,
                mode=spec.mode,  # type: ignore[arg-type]
                keyword_args=spec.keyword_args,
                scope_attrs=scope_attrs,
            )
        )
        if spec.remove_class:
            patch_set.class_removals.append(
                ClassRemovalPatch(old_name=spec.old_ctor, new_name=spec.new_ctor)
            )
    return patch_set


def _scope_attrs(rule: InlineRule) -> tuple[str, ...]:
    """Attribute names of the rule's settled FQNs, i.e. the rewrite scope.

    ``model.layers.3.post_attention_layernorm`` scopes the rewrite to
    ``self.post_attention_layernorm = <ctor>(...)`` assignments, which is the
    granularity native's FQN matching produces.  ``model.norm`` is a sole-name
    FQN, so the whole FQN is the attribute.
    """

    return tuple(sorted({fqn.rsplit(".", 1)[-1] for fqn in rule.match if fqn}))
