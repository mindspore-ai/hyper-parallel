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
    """Build inline forward patches from strategy targets."""

    patch_set = InlinePatchSet()
    emitted: dict[tuple[str, str], str] = {}
    for rule in rules:
        for target in (rule.local_compute_target, rule.inner_wrapper_target):
            spec = strategy_spec(target, model_type)
            if spec is None:
                continue
            patch_set.imports.extend(spec.imports)
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
