# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Normalize frozen meta so it matches the inline-generated source shape."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from hyper_parallel.codegen.inline.ir import InlineRule


QWEN3_MOE_FLASH_ATTENTION_REPLACEMENT = (
    "hyper_parallel.models.qwen3_moe.adapter.replacements."
    "replace_qwen3_moe_flash_attention"
)

QWEN3_MOE_FUSED_QKV_PARAM_RENAMES = {
    "linear_qkv.weight": ("q_proj.weight", "k_proj.weight", "v_proj.weight"),
    "linear_qkv.bias": ("q_proj.bias", "k_proj.bias", "v_proj.bias"),
}


def normalize_inline_meta(meta: Any, rules: tuple[InlineRule, ...]) -> None:
    """Mutate ``meta`` to match the source form emitted by inline patches.

    The generic frozen plan is derived after runtime module replacements, so
    Qwen3 MoE attention records the adapter's fused ``linear_qkv`` parameter.
    The inline product intentionally expands that adapter back into readable
    HF-style ``q_proj`` / ``k_proj`` / ``v_proj`` modules.  The generated
    ``codegen_meta.json`` must therefore use the same parameter names as the
    generated ``modeling`` file, otherwise preflight correctly reports a stale
    artifact before training starts.
    """

    _normalize_qwen3_moe_attention_plan(meta, rules)


def _normalize_qwen3_moe_attention_plan(meta: Any, rules: tuple[InlineRule, ...]) -> None:
    attention_fqns = _replacement_fqns(rules, QWEN3_MOE_FLASH_ATTENTION_REPLACEMENT)
    if not attention_fqns:
        return
    _rewrite_param_plan(
        getattr(meta, "param_plan", None) or {},
        attention_fqns,
        QWEN3_MOE_FUSED_QKV_PARAM_RENAMES,
    )
    meta.frozen_sharded_params = _rewrite_frozen_param_names(
        getattr(meta, "frozen_sharded_params", None) or [],
        attention_fqns,
        QWEN3_MOE_FUSED_QKV_PARAM_RENAMES,
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
