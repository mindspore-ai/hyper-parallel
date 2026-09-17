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

from hyper_parallel.codegen.inline.ir import (
    ClassRemovalPatch,
    ConstructorReplacePatch,
    InlinePatchSet,
    InlineRule,
    ModuleSnippetPatch,
)
from hyper_parallel.codegen.inline.specs import replacement_spec


def build_replacement_patches(rules: tuple[InlineRule, ...], model_type: str | None = None) -> InlinePatchSet:
    """Build constructor replacement patches from ``replace_module`` targets."""

    patch_set = InlinePatchSet()
    seen_targets: set[str] = set()
    for rule in rules:
        target = rule.replace_target
        if target is None or target in seen_targets:
            continue
        spec = replacement_spec(target, model_type, module_type=rule.module_type)
        if spec is None:
            continue
        seen_targets.add(target)
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
            )
        )
        if spec.remove_class:
            patch_set.class_removals.append(
                ClassRemovalPatch(old_name=spec.old_ctor, new_name=spec.new_ctor)
            )
    return patch_set
