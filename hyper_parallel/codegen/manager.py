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
"""Generate and atomically publish single-process codegen bundles.

The manager is the only entry point the trainer / CLI talks to. It does not
perform AST patching or source emitting itself; those operations belong to
the ``emit`` layer. Every invocation emits and atomically replaces the bundle
in the current process.
"""

from __future__ import annotations

import logging
from typing import Optional

from hyper_parallel.codegen.artifact import (
    ArtifactLayout,
    artifact_generation_lock,
    clean_temp_artifacts,
    ensure_artifact_dir,
    resolve_default_artifact_layout,
    resolve_model_artifact_layout,
    write_bundle_atomic,
)
from hyper_parallel.codegen.emit import emit_bundle
from hyper_parallel.codegen.emit.replacement import replacement_records_from_snapshot
from hyper_parallel.codegen.emit.snapshot_patches import (
    emission_patches_from_snapshot_payload,
)
from hyper_parallel.codegen.emit.context import EmissionContext
from hyper_parallel.codegen.snapshot import CodegenSnapshot, CodegenStatus
from hyper_parallel.models.registry import (
    artifact_model_type,
    family_dir_for_model_type,
)

logger = logging.getLogger(__name__)


def ensure_codegen_artifact_from_snapshot(
    snapshot: CodegenSnapshot,
    artifact_dir: Optional[str] = None,
    *,
    timeout_s: float = 600.0,
) -> ArtifactLayout:
    """Generate and replace a bundle in the current single process.

    This entry must remain independent of config projection, meta-model
    construction, offline meshes, replacement matching, and planner calls.
    Distributed launchers must use a dedicated sidecar instead of invoking
    this function independently on every training rank.
    """
    if not isinstance(snapshot, CodegenSnapshot):
        raise TypeError("snapshot must be a CodegenSnapshot")
    if snapshot.status not in (CodegenStatus.DERIVED, CodegenStatus.NOT_REQUIRED):
        raise ValueError(
            "snapshot-driven generation requires status 'derived' or 'not_required', "
            f"got {snapshot.status.value!r}"
        )
    source = snapshot.source
    if not source.get("file_path"):
        raise ValueError("snapshot source must include a modeling file_path")
    model_type = source.get("model_type")
    public_model_type = artifact_model_type(str(model_type)) if model_type else None
    model_name = (
        _model_type_to_modeling_token(str(public_model_type))
        if model_type
        else source.get("architecture")
    )
    if artifact_dir is None and source.get("model_type"):
        layout = resolve_model_artifact_layout(
            family_dir_for_model_type(str(source["model_type"])),
            model_name=str(model_name) if model_name else None,
        )
    else:
        layout = resolve_default_artifact_layout(
            artifact_dir,
            model_name=str(model_name) if model_name else None,
        )
    ensure_artifact_dir(layout)

    with artifact_generation_lock(layout, timeout_s=timeout_s):
        clean_temp_artifacts(layout)
        logger.info("codegen: generating artifact %s", layout.artifact_dir)
        context = _build_emission_context(snapshot)
        files = emit_bundle(layout, context)
        write_bundle_atomic(layout, files)
    return layout


def _build_emission_context(snapshot: CodegenSnapshot) -> EmissionContext:
    """Map one immutable observation into an in-memory emission context."""
    payload = snapshot._emission_view()  # pylint: disable=protected-access
    source = dict(payload["source"])
    status = snapshot.status
    frozen_plan = dict(payload["plan"] or {})
    module_overrides = list(replacement_records_from_snapshot(payload))
    is_derived = status is CodegenStatus.DERIVED
    emission_patches = (
        emission_patches_from_snapshot_payload(payload) if is_derived else {}
    )
    return EmissionContext(
        source=source,
        parallel_dims=dict(frozen_plan.get("parallel_dims") or {}),
        status=payload["status"],
        injections=list(frozen_plan.get("injections") or ()),
        param_plan=dict(frozen_plan.get("param_plan") or {}),
        boundary_classes=dict(frozen_plan.get("boundary_classes") or {}),
        nested_components=list(frozen_plan.get("nested_components") or ()),
        mesh_dim_names=list(frozen_plan.get("mesh_dim_names") or ()) or None,
        module_overrides=(module_overrides if is_derived else []),
        emission_patches=emission_patches,
    )


def _model_type_to_modeling_token(model_type: str) -> str:
    """Convert HF ``model_type`` into the generated modeling file token."""
    parts = [part for part in model_type.replace("-", "_").split("_") if part]
    if not parts:
        return "model"
    return "_".join(part[:1].upper() + part[1:] for part in parts)


__all__ = [
    "ensure_codegen_artifact_from_snapshot",
]
