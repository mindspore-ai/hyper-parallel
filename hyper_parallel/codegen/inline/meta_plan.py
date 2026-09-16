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
"""Normalize frozen meta so it matches the inline-generated source shape."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from hyper_parallel.codegen.inline.ir import InlineRule
from hyper_parallel.codegen.inline.specs import get_inline_spec_bundle


def normalize_inline_meta(meta: Any, rules: tuple[InlineRule, ...], model_type: str | None = None) -> None:
    """Align frozen parameter names with adapter-declared source replacements."""
    seen_targets: set[str] = set()
    for rule in rules:
        target = rule.replace_target
        if target is None or target in seen_targets:
            continue
        seen_targets.add(target)
        bundle = get_inline_spec_bundle(model_type, target)
        if bundle is None:
            continue
        for normalizer in bundle.meta_normalizers:
            if normalizer.target != target:
                continue
            fqns = _replacement_fqns(rules, target)
            if not fqns:
                continue
            _rewrite_param_plan(getattr(meta, "param_plan", None) or {}, fqns, normalizer.param_renames)
            meta.frozen_sharded_params = _rewrite_frozen_param_names(
                getattr(meta, "frozen_sharded_params", None) or [], fqns, normalizer.param_renames,
            )


def _replacement_fqns(rules: tuple[InlineRule, ...], target: str) -> set[str]:
    fqns: set[str] = set()
    for rule in rules:
        if rule.replace_target != target:
            continue
        fqns.update(match for match in rule.match if match)
        extra_fqns = rule.options.get("fqns") if isinstance(rule.options, dict) else ()
        fqns.update(str(fqn) for fqn in (extra_fqns or ()) if fqn)
        extra_fqn = rule.options.get("fqn") if isinstance(rule.options, dict) else None
        if extra_fqn:
            fqns.add(str(extra_fqn))
    return fqns


def _rewrite_param_plan(
    param_plan: dict[str, Any],
    boundary_fqns: set[str],
    renames: dict[str, tuple[str, ...]],
) -> None:
    for fqn in sorted(boundary_fqns):
        entry = param_plan.get(fqn)
        if not isinstance(entry, dict):
            continue
        params = entry.get("params")
        if not isinstance(params, dict):
            continue
        for old_name, new_names in renames.items():
            if old_name not in params:
                continue
            old_plan = params.pop(old_name)
            for new_name in new_names:
                params.setdefault(new_name, deepcopy(old_plan))


def _rewrite_frozen_param_names(
    frozen_sharded_params: list[str],
    boundary_fqns: set[str],
    renames: dict[str, tuple[str, ...]],
) -> list[str]:
    expanded: list[str] = []
    for param_fqn in frozen_sharded_params:
        replacements = _expanded_param_names(param_fqn, boundary_fqns, renames)
        expanded.extend(replacements or (param_fqn,))
    return sorted(dict.fromkeys(expanded))


def _expanded_param_names(
    param_fqn: str,
    boundary_fqns: set[str],
    renames: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    for boundary_fqn in boundary_fqns:
        prefix = boundary_fqn + "."
        if not param_fqn.startswith(prefix):
            continue
        local_name = param_fqn[len(prefix):]
        new_names = renames.get(local_name)
        if new_names:
            return tuple(prefix + new_name for new_name in new_names)
    return ()


__all__ = ["normalize_inline_meta"]
