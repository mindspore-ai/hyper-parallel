# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
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


def build_replacement_patches(rules: tuple[InlineRule, ...]) -> InlinePatchSet:
    """Build constructor replacement patches from ``replace_module`` targets."""

    patch_set = InlinePatchSet()
    seen_targets: set[str] = set()
    for rule in rules:
        target = rule.replace_target
        if target is None or target in seen_targets:
            continue
        spec = replacement_spec(target)
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
