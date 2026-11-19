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
"""Project shared Trainer preparation into codegen-owned construction IR."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Optional

import torch

from hyper_parallel.codegen.ir.construction import (
    CodegenBuildIR,
    ComponentBoundaryIR,
    EmissionPlanIR,
    ModelSourceIR,
    OwnedLayoutIR,
    ParallelBindingIR,
)
from hyper_parallel.codegen.manager import ensure_codegen_artifact_from_snapshot
from hyper_parallel.codegen.plan.observation import observe_emission_plan
from hyper_parallel.codegen.session import CodegenBuildSession
from hyper_parallel.models.build_capture import ModelBuildCapture
from hyper_parallel.models.preparation_manifest import PreparationManifest, PreparedComponent


@dataclass
class CodegenProjectionResult:
    """Artifact and sealed compatibility snapshot produced from one IR."""

    artifact_layout: Optional[Any] = None
    snapshot: Optional[Any] = None


def prepare_model_source_compat(model_spec: Any) -> None:
    """Run the model-owned source import hook before normal construction."""
    loader = getattr(model_spec, "codegen", None) if model_spec is not None else None
    provider = loader() if loader is not None else None
    prepare = getattr(provider, "prepare_model_import", None)
    if prepare is not None:
        prepare()


def _contains_fqn(alias: str, fqn: str) -> bool:
    return alias == "" or alias == fqn or fqn.startswith(f"{alias}.")


def _layout_owner(
    module_fqn: str,
    components: tuple[PreparedComponent, ...],
) -> Optional[str]:
    owners = (
        (len(alias), component.boundary_id)
        for component in components
        for alias in component.aliases
        if _contains_fqn(alias, module_fqn)
    )
    return max(owners, key=lambda item: item[0], default=(0, None))[1]


def _parallel_dims(mesh: Any) -> dict[str, Any]:
    """Return stable dimensions from the logical build mesh."""
    return {
        "dp_size": getattr(mesh, "dp_size", 1),
        "tp_size": getattr(mesh, "tp_size", 1),
        "cp_size": getattr(mesh, "cp_size", 1),
        "pp_size": getattr(mesh, "pp_size", 1),
        "ep_size": getattr(mesh, "ep_size", 1),
        "sequence_parallel": getattr(mesh, "sequence_parallel", False),
        "loss_parallel": getattr(mesh, "loss_parallel", False),
    }


def _validate_manifest_links(manifest: PreparationManifest) -> None:
    """Reject incomplete shared preparation records before source emission."""
    components = {component.boundary_id: component for component in manifest.components}
    if len(components) != len(manifest.components):
        raise ValueError("preparation manifest contains duplicate component boundary IDs")
    for component in manifest.components:
        for link_name in ("parent_boundary_id", "emission_owner_boundary_id"):
            linked_id = getattr(component, link_name)
            if linked_id is not None and linked_id not in components:
                raise ValueError(
                    f"component {component.boundary_id} references unknown "
                    f"{link_name} {linked_id}"
                )
    _validate_replacement_links(manifest, components)
    for binding in manifest.native_bindings:
        if binding.layout_owner_boundary_id not in components:
            raise ValueError(
                f"binding {binding.boundary_id} references unknown layout owner "
                f"{binding.layout_owner_boundary_id}"
            )


def _validate_replacement_links(
    manifest: PreparationManifest, components: dict[str, PreparedComponent],
) -> None:
    """Verify that replacements point to known declarations and components."""
    declaration_ids = {
        item.get("rule_id")
        for item in manifest.declarations
        if item.get("rule_id") is not None
    }
    replacement_boundaries = {item.boundary_id for item in manifest.replacements}
    if len(replacement_boundaries) != len(manifest.replacements):
        raise ValueError("preparation manifest contains duplicate replacement boundary IDs")
    for replacement in manifest.replacements:
        if replacement.declaration_id not in declaration_ids:
            raise ValueError(
                f"replacement {replacement.boundary_id} references unknown declaration "
                f"{replacement.declaration_id}"
            )
        component = components.get(replacement.boundary_id)
        if component is None:
            raise ValueError(f"replacement application {replacement.boundary_id} has no component boundary")
        if (component.origin != "replacement" or component.declaration_id != replacement.declaration_id
                or component.emission_owner_boundary_id != component.boundary_id):
            raise ValueError(f"replacement application {replacement.boundary_id} has no source constructor owner")


def project_model_build(capture: ModelBuildCapture) -> CodegenBuildIR:
    """Project one completed shared build without repeating any build decision."""
    if not capture.complete:
        raise RuntimeError("native model build completed without a complete preparation capture")
    manifest = capture.manifest
    if manifest is None:
        raise RuntimeError("native model build completed without a preparation manifest")
    _validate_manifest_links(manifest)

    applications = {item.boundary_id: item for item in manifest.replacements}
    components = tuple(
        ComponentBoundaryIR(
            boundary_id=component.boundary_id,
            parent_boundary_id=component.parent_boundary_id,
            emission_owner_boundary_id=component.emission_owner_boundary_id,
            origin=component.origin,
            aliases=component.aliases,
            family_id=component.family_id,
            family_version=component.family_version,
            declaration_id=component.declaration_id,
            values=component.to_dict(),
            replacement=(
                applications[component.boundary_id].to_legacy_dict()
                if component.boundary_id in applications
                else None
            ),
        )
        for component in manifest.components
    )

    plan_ir = None
    layouts: tuple[OwnedLayoutIR, ...] = ()
    if capture.plan is not None or manifest.replacements:
        observed_plan = observe_emission_plan(capture.plan, capture.model).to_dict()
        observed_plan["parallel_dims"] = _parallel_dims(capture.mesh)
        plan_ir = EmissionPlanIR(observed_plan)
        layouts = tuple(
            OwnedLayoutIR(
                module_fqn=fqn,
                owner_boundary_id=_layout_owner(fqn, manifest.components),
                values=values,
            )
            for fqn, values in observed_plan["param_plan"].items()
        )

    bindings = tuple(
        ParallelBindingIR(
            boundary_id=binding.boundary_id,
            layout_owner_boundary_id=binding.layout_owner_boundary_id,
            aliases=binding.aliases,
            values=binding.to_dict(),
        )
        for binding in manifest.native_bindings
    )
    return CodegenBuildIR(
        schema_version=1,
        source=ModelSourceIR(capture.source or {}),
        declarations=manifest.declarations,
        components=components,
        layouts=layouts,
        bindings=bindings,
        plan=plan_ir,
        replacement_order=tuple(applications),
        reason=capture.reason,
    )


def publish_codegen_ir(
    build_ir: CodegenBuildIR,
    artifact_dir: Optional[str],
) -> CodegenProjectionResult:
    """Adapt verified IR to the existing emitter while preserving host RNG."""
    session = CodegenBuildSession()
    session.record_source(build_ir.source.to_dict())
    session.record_replacements(
        declarations=build_ir.declaration_records(),
        applications=build_ir.replacement_records(),
    )
    if build_ir.plan is None:
        session.mark_not_required(build_ir.reason or "sharding_not_required")
    else:
        session.record_plan(build_ir.plan.to_dict())

    result = CodegenProjectionResult()
    cpu_rng_state = torch.get_rng_state()
    python_rng_state = random.getstate()

    def _publish(snapshot: Any) -> None:
        try:
            result.artifact_layout = ensure_codegen_artifact_from_snapshot(snapshot, artifact_dir)
        finally:
            torch.set_rng_state(cpu_rng_state)
            random.setstate(python_rng_state)

    result.snapshot = session.finish(_publish)
    return result


__all__ = [
    "CodegenProjectionResult",
    "prepare_model_source_compat",
    "project_model_build",
    "publish_codegen_ir",
]
