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

from hyper_parallel.codegen.inline.ir import ForwardExtractPatch, InlinePatchSet, InlineRule
from hyper_parallel.codegen.inline.specs import strategy_spec


def build_strategy_patches(rules: tuple[InlineRule, ...], model_type: str | None = None) -> InlinePatchSet:
    """Build inline forward patches from strategy targets.

    A plan carries one rule per matched FQN, so the same strategy target (and
    therefore the same imports and module snippets) is seen once per module it
    applies to.  Those contributions are deduplicated here — mirroring
    ``build_replacement_patches`` — so a module-level snippet such as the
    parallel-state accessor is emitted once per generated file instead of once
    per matched layer.
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
            if spec.target_class is None:
                continue
            key = (spec.target_class, spec.method_name)
            if key in emitted:
                if emitted[key] != spec.body_template:
                    raise ValueError(f"Conflicting inline strategies for {spec.target_class}.{spec.method_name}")
                continue
            emitted[key] = spec.body_template
            patch_set.forward_extracts.append(
                ForwardExtractPatch(
                    class_name=spec.target_class,
                    method_name=spec.method_name,
                    body=spec.body_template,
                )
            )
    return patch_set
