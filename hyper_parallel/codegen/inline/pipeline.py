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

import os
from typing import Any, Optional

from hyper_parallel.codegen.inline.ir import InlinePatchSet
from hyper_parallel.codegen.inline.meta_plan import normalize_inline_meta
from hyper_parallel.codegen.inline.patch_engine import apply_patch_set
from hyper_parallel.codegen.inline.replacement_pass import build_replacement_patches
from hyper_parallel.codegen.inline.specs import replacement_spec, strategy_spec
from hyper_parallel.codegen.inline.strategy_pass import build_strategy_patches
from hyper_parallel.codegen.inline.yaml_rules import collect_inline_rules


def try_render_inline_modeling(source_text: str, meta: Any, model_type: str | None = None) -> Optional[str]:
    """Render an inline-patched modeling file when the current rules are covered.

    This development path is intentionally gated by ``HYPER_CODEGEN_INLINE_PATCH``.
    The existing generic Codegen path remains the default until the inline
    replacement and strategy passes cover every active YAML target required by
    the generated artifact.
    """

    if os.environ.get("HYPER_CODEGEN_INLINE_PATCH") != "1":
        return None
    rules = collect_inline_rules(meta)
    if not rules or not _can_inline(rules, model_type):
        return None
    replacement_patches = build_replacement_patches(rules, model_type)
    strategy_patches = build_strategy_patches(rules, model_type)
    patch_set = _merge_patch_sets(replacement_patches, strategy_patches)
    rendered = apply_patch_set(source_text, patch_set)
    normalize_inline_meta(meta, rules, model_type)
    return rendered


def _can_inline(rules: tuple[Any, ...], model_type: str | None = None) -> bool:
    for rule in rules:
        if rule.replace_target is not None and replacement_spec(rule.replace_target, model_type) is None:
            return False
        for target in (rule.local_compute_target, rule.inner_wrapper_target):
            if target is not None and strategy_spec(target, model_type) is None:
                return False
    return True


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
