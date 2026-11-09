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
"""Stable construction artifacts for models directly composed from module families."""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

from torch import nn

from hyper_parallel.components.modules.module_family import _plain, family_plan_snapshot
from hyper_parallel.models.module_build import construct_family_model, prepare_model_components


_SCHEMA_VERSION = 1


def _spec_record(context: Mapping[str, Any]) -> Any:
    spec = context.get("model_spec")
    if spec is None:
        return None
    return {"model_type": spec.model_type, "architecture": spec.architecture,
            "providers": {name: _plain(value) for name, value in vars(spec).items() if callable(value)}}


def direct_model_manifest(
    model: nn.Module, factory: Callable, config: Any, context: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture effective direct choices before checking a cached construction artifact.

    Args:
        model: Final or source model tree for this build.
        factory: Construction callable or protocol factory.
        config: Complete logical model configuration.
        context: Read-only job context and effective model facts.
    """
    payload = {
        "schema_version": _SCHEMA_VERSION, "factory": _plain(factory), "config": _plain(config),
        "spec": _spec_record(context), "families": family_plan_snapshot(model),
        "model_class": _plain(type(model)),
        "runtime_versions": _runtime_versions(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**payload, "signature": hashlib.sha256(encoded.encode()).hexdigest()}


def construct_cached_direct_model(
    factory: Callable, config: Any, context: Mapping[str, Any], expected_signature: str,
) -> nn.Module:
    """Construct once and reject a stale factory/config/member/backend/layout snapshot.

    Args:
        factory: Construction callable or protocol factory.
        config: Complete logical model configuration.
        context: Read-only job context and effective model facts.
        expected_signature: Effective construction fingerprint recorded by the artifact.
    """
    model = construct_family_model(factory, config, context=context)
    model, _ = prepare_model_components(model, context)
    actual = direct_model_manifest(model, factory, config, context)["signature"]
    if actual != expected_signature:
        raise ValueError("direct model codegen signature changed; regenerate the construction artifact")
    return model


def prepare_direct_model_artifact(
    model: nn.Module, factory: Callable, config: Any, context: Mapping[str, Any], *, artifact_dir: str | None = None,
) -> str:
    """Write an importable constructor and manifest without replacement or live resources.

    Args:
        model: Prepared real or meta tree, before parameter layout.
        factory: Stable importable model factory.
        config: Complete reconstruction configuration.
        context: Current logical job context and effective model spec.
        artifact_dir: Optional explicit cache directory; otherwise use a content-addressed cache.

    Returns:
        Directory containing the versioned manifest and generated entry point.
    """
    if (not getattr(model, "_hp_components_prepared", False)
            or getattr(model, "_hp_module_build_context", None) is not context):
        raise ValueError("construction artifacts require prepare_model_components with the same context")
    manifest = direct_model_manifest(model, factory, config, context)
    path = Path(artifact_dir) if artifact_dir is not None else (
        Path(".cache") / "hyper_parallel" / "direct_models" / manifest["signature"]
    )
    path.mkdir(parents=True, exist_ok=True)
    manifest_path = path / "family_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("signature") != manifest["signature"]:
            raise ValueError("direct model codegen cache differs from the effective build; choose a new artifact_dir")
    factory_path = manifest["factory"]["callable"]
    module_path, _, name = factory_path.rpartition(".")
    generated = (
        '"""Generated direct-family model constructor."""\n'
        f"from {module_path} import {name} as _model_factory\n"
        "from hyper_parallel.codegen.family_model import construct_cached_direct_model\n\n"
        "def build_model(config, *, context):\n"
        "    return construct_cached_direct_model(\n"
        f"        _model_factory, config, context, {manifest['signature']!r},\n"
        "    )\n"
    )
    _write_atomic(path / "modeling_family.py", generated)
    _write_atomic(manifest_path, json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    return str(path)


def _write_atomic(path: Path, content: str) -> None:
    """Publish complete files when multiple ranks share the construction cache."""
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_versions() -> dict[str, str]:
    result = {}
    for name in ("torch", "torch-npu", "transformers"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "unavailable"
    return result
