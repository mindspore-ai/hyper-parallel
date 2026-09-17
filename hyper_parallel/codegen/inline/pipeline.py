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
"""Two-stage inline modeling source pipeline."""

from __future__ import annotations

from typing import Any

from hyper_parallel.codegen.inline.ir import InlinePatchSet
from hyper_parallel.codegen.inline.meta_plan import normalize_inline_meta
from hyper_parallel.codegen.inline.patch_engine import apply_patch_set
from hyper_parallel.codegen.inline.replacement_pass import build_replacement_patches
from hyper_parallel.codegen.inline.specs import replacement_spec, strategy_spec
from hyper_parallel.codegen.inline.strategy_pass import build_strategy_patches
from hyper_parallel.codegen.inline.yaml_rules import collect_inline_rules


def render_inline_modeling(source_text: str, meta: Any, model_type: str | None = None) -> str:
    """Render the inline-patched modeling file for a frozen plan.

    This is the only artifact route. A YAML target with no adapter declaration
    is a hard error rather than a silent fallback, so an inline coverage gap
    surfaces at generation time instead of degrading the artifact. A plan with
    no inline rules needs no patch and the source is returned unchanged.
    """

    rules = collect_inline_rules(meta)
    if not rules:
        return source_text
    _require_inline_coverage(rules, model_type)
    replacement_patches = build_replacement_patches(rules, model_type)
    strategy_patches = build_strategy_patches(rules, model_type)
    patch_set = _merge_patch_sets(replacement_patches, strategy_patches)
    rendered = apply_patch_set(source_text, patch_set)
    normalize_inline_meta(meta, rules, model_type)
    return rendered


def _require_inline_coverage(rules: tuple[Any, ...], model_type: str | None = None) -> None:
    """Fail when a YAML rule has no structural resolution behind it.

    Covers both rule kinds: a replacement target needs a ``ReplacementSpec``,
    and every strategy target needs a ``StrategySpec``.  Reporting all gaps at
    once keeps one fix cycle from hiding the next gap.
    """

    uncovered: list[str] = []
    for rule in rules:
        if (
            rule.replace_target is not None
            and replacement_spec(rule.replace_target, model_type, module_type=rule.module_type) is None
        ):
            uncovered.append(rule.replace_target)
        for target in (rule.local_compute_target, rule.inner_wrapper_target):
            if (
                target is not None
                and strategy_spec(target, model_type, inner_wrapper=target == rule.inner_wrapper_target) is None
            ):
                uncovered.append(target)
    if uncovered:
        raise RuntimeError(
            f"codegen: no inline declaration for {sorted(set(uncovered))} "
            f"(model_type={model_type!r}); the target names no known generic "
            "component or framework strategy"
        )


def _merge_patch_sets(*sets: InlinePatchSet) -> InlinePatchSet:
    merged = InlinePatchSet()
    for patch_set in sets:
        merged.imports.extend(patch_set.imports)
        merged.module_snippets.extend(patch_set.module_snippets)
        merged.constructor_replaces.extend(patch_set.constructor_replaces)
        merged.forward_extracts.extend(patch_set.forward_extracts)
        merged.forward_bodies.extend(patch_set.forward_bodies)
        merged.class_markers.extend(patch_set.class_markers)
        merged.class_removals.extend(patch_set.class_removals)
    return merged
