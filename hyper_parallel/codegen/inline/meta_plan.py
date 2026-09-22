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

from fnmatch import fnmatch
from typing import Any

from hyper_parallel.codegen.inline.ir import InlineRule
from hyper_parallel.codegen.inline.specs import strategy_spec


def normalize_inline_meta(meta: Any, rules: tuple[InlineRule, ...], model_type: str | None = None) -> None:
    """Align the frozen plan with the inline-generated source shape.

    Every stage is structure-proven: the strategy each strategy target
    resolves to names the class whose forward is inlined and the sub-patterns
    that forward owns, and ``meta.external_state_classes`` (recorded from the
    same structure at freeze time) names the classes whose forward an inline
    strategy body *replaces* (the MoE EP shell) and which therefore read their
    parallel state from install-time instance attributes.  A rendered component
    class (the fused attention) keeps the component's own forward and is not
    stripped: its boundary is emitted and compiled like any other.
    """

    # EP strategy forwards inline the entire routed+shared path, so any
    # boundary entries for sub-modules of an EP target (e.g. routed experts
    # and shared_experts) must be stripped — otherwise the shared class used
    # in both EP and TP contexts fails the single-contract invariant.
    _strip_ep_sub_boundaries(meta, rules, model_type)
    _strip_external_state_boundaries(meta)


def _strip_external_state_boundaries(meta: Any) -> None:
    """Keep parameter sharding without duplicating inline-body-owned collectives.

    Only a class whose forward an inline strategy body replaced is stripped:
    that body carries its own orchestration, so a boundary wrapper for the same
    class would rewrite the very forward the inline patch installed.  A
    rendered component class is deliberately kept — the emitted boundary form
    wraps its ``_forward_impl`` and is compiled at install time.
    """
    external_classes = set(getattr(meta, "external_state_classes", None) or ())
    boundary_classes = getattr(meta, "boundary_classes", None) or {}
    param_plan = getattr(meta, "param_plan", None) or {}
    for fqn, class_name in list(boundary_classes.items()):
        if class_name not in external_classes:
            continue
        entry = param_plan.get(fqn)
        if isinstance(entry, dict):
            entry["is_boundary"] = False
        boundary_classes.pop(fqn)


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


__all__ = ["normalize_inline_meta"]
