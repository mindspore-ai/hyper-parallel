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
from fnmatch import fnmatch
from typing import Any

from hyper_parallel.codegen.inline.ir import InlineRule
from hyper_parallel.codegen.inline.specs import get_inline_spec_bundle, strategy_spec


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
    # EP strategy forwards inline the entire routed+shared path, so any
    # boundary entries for sub-modules of an EP target (e.g. routed experts
    # and shared_experts) must be stripped — otherwise the shared class used
    # in both EP and TP contexts fails the single-contract invariant.
    _strip_ep_sub_boundaries(meta, rules, model_type)


def _strip_ep_sub_boundaries(
    meta: Any, rules: tuple[InlineRule, ...], model_type: str | None,
) -> None:
    """Mark boundary entries owned by inline EP strategy forwards as non-boundary.

    The EP strategy forward replaces the target class's ``forward`` and calls
    sub-modules directly (routed experts via all-to-all, shared experts via
    ``self.shared_experts(...)``).  Boundary entries for the EP target itself
    and for routed-expert / shared-expert sub-modules are marked
    ``is_boundary: false`` so ``lower_forward_boundaries`` does not double-wrap
    them.  The param_plan entries are kept so FSDP / EP param sharding still
    applies and the preflight coverage check passes.
    """
    param_plan = getattr(meta, "param_plan", None) or {}
    boundary_classes = getattr(meta, "boundary_classes", None) or {}
    ep_fqns: set[str] = set()
    strip_patterns: tuple[str, ...] = ()
    for rule in rules:
        target = rule.local_compute_target
        if target is None:
            continue
        spec = strategy_spec(target, model_type)
        if spec is None or spec.target_class is None:
            continue
        if spec.strip_boundary_subpatterns:
            strip_patterns = spec.strip_boundary_subpatterns
        for pattern in rule.match:
            for fqn, cls in boundary_classes.items():
                if fnmatch(fqn, pattern) and cls == spec.target_class:
                    ep_fqns.add(fqn)
    if not ep_fqns:
        return
    mark_fqns: set[str] = set()
    for ep_fqn in ep_fqns:
        mark_fqns.add(ep_fqn)
        if not strip_patterns:
            continue
        sub_prefix = ep_fqn + "."
        for fqn in list(param_plan):
            if not fqn.startswith(sub_prefix):
                continue
            local_suffix = fqn[len(sub_prefix):]
            for pat in strip_patterns:
                if fnmatch(local_suffix, pat):
                    mark_fqns.add(fqn)
                    break
    for fqn in mark_fqns:
        entry = param_plan.get(fqn)
        if isinstance(entry, dict):
            entry["is_boundary"] = False
        boundary_classes.pop(fqn, None)


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
