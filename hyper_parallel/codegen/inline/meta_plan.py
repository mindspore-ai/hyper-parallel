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

import logging
from fnmatch import fnmatch
from typing import Any

from hyper_parallel.codegen.inline.ir import InlineRule
from hyper_parallel.codegen.inline.specs import strategy_spec

logger = logging.getLogger(__name__)


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
    # Capture the complete class map BEFORE stripping, so we can detect
    # classes shared between boundary and non-boundary instances (e.g.
    # DeepseekV3MLP used as both dense decoder layers and shared experts
    # inside MoE). Source lowering rewrites class.forward, which ALL
    # instances inherit -- a non-boundary instance would get a boundary
    # body it cannot satisfy (no _hp_boundary installed).
    full_class_map = dict(getattr(meta, "boundary_classes", None) or {})
    _strip_ep_sub_boundaries(meta, rules, model_type)
    _strip_external_state_boundaries(meta)
    _mark_shared_class_boundaries(meta, full_class_map)


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


def _mark_shared_class_boundaries(meta: Any, full_class_map: dict[str, str]) -> None:
    """Mark source lowering as skipped for classes shared with a non-boundary instance.

    Source lowering rewrites class.forward, which ALL instances of the class
    inherit -- including instances that are NOT boundaries. A non-boundary
    instance would inherit a boundary body that references
    self._hp_boundary, but hp_install_boundaries skips non-boundary FQNs
    (is_boundary=False), so the attribute is never installed and the
    forward raises AttributeError at train time.

    The classic trigger is DeepSeek-V3 hybrid stack: DeepseekV3MLP is
    used both as the dense decoder MLP (a TP boundary) and as the shared
    expert inside the MoE layer (stripped to is_boundary=False by
    _strip_ep_sub_boundaries). Source lowering the dense-layer boundary
    rewrites the class forward; the shared-expert instance inherits it;
    the runtime never installs _hp_boundary on the shared expert -> crash.

    The decision travels as an explicit ``skip_source_lowering`` marker on
    the param-plan entry (``is_boundary`` stays True; ``boundary_classes``
    is kept intact) rather than by removing the class mapping: the
    emitter's fail-fast on an unresolvable boundary class is a real safety
    contract (a silently dropped redistribution would change the model's
    numeric behavior), so "skip source lowering" must be stated, not
    implied by missing data.  ``hp_install_boundaries`` installs the
    per-instance runtime wrapper for the marked boundary, which executes
    the same contract.
    """
    param_plan = getattr(meta, "param_plan", None) or {}
    boundary_classes = getattr(meta, "boundary_classes", None) or {}

    non_boundary_classes: set[str] = set()
    for fqn, entry in param_plan.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("is_boundary") is False:
            cls = full_class_map.get(fqn)
            if cls is not None:
                non_boundary_classes.add(cls)

    if not non_boundary_classes:
        return

    for fqn, class_name in list(boundary_classes.items()):
        if class_name not in non_boundary_classes:
            continue
        entry = param_plan.get(fqn)
        if isinstance(entry, dict):
            entry["skip_source_lowering"] = True
            logger.info(
                "codegen: boundary %s (class %s) marked skip_source_lowering "
                "-- the class is shared with a non-boundary instance; the "
                "runtime per-instance wrapper is used instead",
                fqn, class_name,
            )


__all__ = ["normalize_inline_meta"]
