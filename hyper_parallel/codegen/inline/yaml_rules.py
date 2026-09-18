# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Recover inline rules from existing YAML-derived codegen metadata.

The user YAML is not extended.  Codegen reads the already frozen
``module_overrides`` and ``injections`` metadata, whose fields originate from
``plan_overrides.replace_module`` / ``inner_wrapper`` / ``local_compute_fn``.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from hyper_parallel.codegen.inline.ir import InlineRule


def collect_inline_rules(meta: Any) -> tuple[InlineRule, ...]:
    """Return normalized inline rules recovered from ``meta``."""

    rules: list[InlineRule] = []
    for record in getattr(meta, "module_overrides", None) or []:
        target = record.get("factory")
        if not target:
            continue
        rules.append(
            InlineRule(
                match=_matches(record),
                module_type=record.get("module_type"),
                replace_target=target,
                options={
                    "fqns": tuple(record.get("fqns") or ()),
                    "fqn": record.get("fqn"),
                    "exact_type": bool(record.get("exact_type", False)),
                },
            )
        )
    for record in getattr(meta, "injections", None) or []:
        local = _target_path(record.get("local_compute_fn"))
        inner = _target_path(record.get("inner_wrapper"))
        if local is None and inner is None:
            continue
        rules.append(
            InlineRule(
                match=_matches(record),
                inner_wrapper_target=inner,
                local_compute_target=local,
                region_dispatch=record.get("region_dispatch"),
                options={
                    key: value
                    for key, value in record.items()
                    if key not in {"match", "inner_wrapper", "local_compute_fn"}
                },
            )
        )
    return tuple(rules)


def _matches(record: dict[str, Any]) -> tuple[str, ...]:
    value = record.get("fqns") or record.get("match") or record.get("fqn")
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    return (str(value),)


def _target_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("_target_") or value.get("target")
    return getattr(value, "__qualname__", None)
