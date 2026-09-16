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
"""Discover inline declarations through model adapters."""

from __future__ import annotations

from hyper_parallel.codegen.inline.spec_bundle import InlineSpecBundle, ReplacementSpec, StrategySpec
from hyper_parallel.models.registry import get_model_adapter


def get_inline_spec_bundle(model_type: str | None = None, target: str | None = None) -> InlineSpecBundle | None:
    """Resolve declarations, inferring legacy identity from the target path.

    Explicit identity is authoritative: unsupported models never fall back to
    another family. Legacy callers use the models/<family>/adapter convention.
    """
    if model_type is None:
        parts = (target or "").split(".")
        if len(parts) < 5 or parts[:2] != ["hyper_parallel", "models"] or parts[3] != "adapter":
            return None
        model_type = parts[2]
    adapter = get_model_adapter(model_type)
    if adapter is None or adapter.inline_codegen is None:
        return None
    bundle = adapter.inline_codegen().get_inline_spec_bundle()
    if not isinstance(bundle, InlineSpecBundle):
        raise TypeError(f"Inline provider for {model_type!r} must return InlineSpecBundle")
    return bundle


def replacement_spec(target: str | None, model_type: str | None = None) -> ReplacementSpec | None:
    """Return the adapter's replacement declaration for a YAML target."""
    if target is None:
        return None
    bundle = get_inline_spec_bundle(model_type, target)
    return bundle.replacement_specs.get(target) if bundle is not None else None


def strategy_spec(target: str | None, model_type: str | None = None) -> StrategySpec | None:
    """Return the adapter's parallel-strategy declaration for a YAML target."""
    if target is None:
        return None
    bundle = get_inline_spec_bundle(model_type, target)
    return bundle.strategy_specs.get(target) if bundle is not None else None
