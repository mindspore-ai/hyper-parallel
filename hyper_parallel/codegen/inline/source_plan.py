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
"""Build one source patch plan from native replacement and strategy observations."""

from __future__ import annotations

from collections.abc import Collection, Mapping

from hyper_parallel.codegen.inline.ir import InlinePatchSet, SourceObservation
from hyper_parallel.codegen.inline.replacement_pass import build_replacement_patches
from hyper_parallel.codegen.inline.strategy_pass import build_strategy_patches


def build_source_patch_plan(
    observations: tuple[SourceObservation, ...],
    strategy_observations: tuple[SourceObservation, ...],
    model_type: str | None,
    boundary_classes: Mapping[str, str] | None,
    external_state_classes: Collection[str] | None,
) -> InlinePatchSet:
    """Build the complete ordered patch set for one modeling source."""
    replacement = build_replacement_patches(observations, model_type)
    strategy = build_strategy_patches(
        strategy_observations,
        model_type,
        boundary_classes,
        external_state_classes,
    )
    return merge_patch_sets(replacement, strategy)


def merge_patch_sets(*patch_sets: InlinePatchSet) -> InlinePatchSet:
    """Merge pass outputs without changing their established application order."""
    merged = InlinePatchSet()
    for patch_set in patch_sets:
        merged.imports.extend(patch_set.imports)
        merged.module_snippets.extend(patch_set.module_snippets)
        merged.constructor_replaces.extend(patch_set.constructor_replaces)
        merged.forward_extracts.extend(patch_set.forward_extracts)
        merged.class_removals.extend(patch_set.class_removals)
    return merged


__all__ = ["build_source_patch_plan", "merge_patch_sets"]
