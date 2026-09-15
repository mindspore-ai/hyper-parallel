# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Second inline pass: insert parallel strategies into replaced modules."""

from __future__ import annotations

from hyper_parallel.codegen.inline.ir import ForwardExtractPatch, InlinePatchSet, InlineRule
from hyper_parallel.codegen.inline.specs import strategy_spec
from hyper_parallel.codegen.inline.templates import QWEN3_MOE_EP_FORWARD


def build_strategy_patches(rules: tuple[InlineRule, ...]) -> InlinePatchSet:
    """Build inline forward patches from strategy targets."""

    patch_set = InlinePatchSet()
    emitted: set[tuple[str, str]] = set()
    for rule in rules:
        target = rule.local_compute_target or rule.inner_wrapper_target
        spec = strategy_spec(target)
        if spec is None:
            continue
        patch_set.imports.extend(spec.imports)
        if spec.kind == "qwen3_moe_ep_routed_forward":
            key = ("Qwen3MoeSparseMoeBlock", spec.kind)
            if key in emitted:
                continue
            emitted.add(key)
            patch_set.forward_extracts.append(
                ForwardExtractPatch(
                    class_name="Qwen3MoeSparseMoeBlock",
                    method_name="forward",
                    body=QWEN3_MOE_EP_FORWARD,
                )
            )
    return patch_set
