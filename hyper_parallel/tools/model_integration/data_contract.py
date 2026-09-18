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
"""Data/forward-input ownership and semantic diagnostics."""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping
from typing import Any, Optional

from hyper_parallel.tools.model_integration.schemas import (
    FindingSeverity,
    ModelIntegrationFinding,
)
from hyper_parallel.models.validation_spec import ModelValidationSpec


_FRAMEWORK_FIELDS = {
    "input_ids",
    "labels",
    "shift_labels",
}


def build_field_ownership(
        runtime_adapter: Any,
) -> dict[str, str]:
    """Build declared field ownership without executing another data batch.

    Args:
        runtime_adapter: Optional model-owned runtime-input adapter.

    Returns:
        Declared field owners and any duplicate ownership claims.
    """
    ownership = {name: "framework" for name in sorted(_FRAMEWORK_FIELDS)}
    runtime_fields = getattr(runtime_adapter, "runtime_input_fields", lambda: ())()
    collisions = {}
    for field_name in runtime_fields:
        previous = ownership.get(field_name)
        if previous is not None:
            collisions.setdefault(field_name, [previous]).append("runtime_adapter")
        else:
            ownership[field_name] = "runtime_adapter"
    return {"fields": ownership, "collisions": collisions}


def validate_data_contract(
        validation_spec: Optional[ModelValidationSpec],
        runtime_adapter: Any,
        *,
        labels_are_shifted: Optional[bool],
        source_type: Optional[str],
        assets: tuple[Any, ...] = (),
) -> tuple[dict[str, Any], list[ModelIntegrationFinding]]:
    """Validate stable adapter ownership and model-declared data semantics.

    Args:
        validation_spec: Optional model-owned validation declaration.
        runtime_adapter: Optional model-owned runtime-input adapter.
        labels_are_shifted: Resolved get-batch label alignment, when declared.
        source_type: Resolved get-batch source type, when declared.
        assets: Tokenizer or processor assets used by the data pipeline.

    Returns:
        Field-ownership evidence and actionable validation findings.
    """
    ownership = build_field_ownership(runtime_adapter)
    findings = []
    if ownership["collisions"]:
        findings.append(
            ModelIntegrationFinding(
                code="HP-DATA-001",
                phase="D2",
                owner_fqn="data.forward_fields",
                severity=FindingSeverity.ERROR,
                message="multiple data stages claim the same model forward field",
                facts={"collisions": ownership["collisions"]},
                why_unsafe="a later stage can silently replace authoritative batch semantics",
                remediation="assign every forward field to exactly one framework or adapter owner",
                related_config=("dataset.data_transform", "dataloader.get_batch"),
            )
        )
    data_spec = validation_spec.data if validation_spec is not None else None
    if data_spec is not None:
        declared_runtime = set(ownership["fields"])
        missing_runtime = sorted(set(data_spec.runtime_fields) - declared_runtime)
        shift_mismatch = (
            data_spec.labels_are_shifted is not None
            and labels_are_shifted is not None
            and data_spec.labels_are_shifted != labels_are_shifted
        )
        missing_processor_revision = data_spec.processor_revision_required and not any(
            getattr(asset, "name_or_path", None)
            or getattr(asset, "_name_or_path", None)
            for asset in assets
        )
        if missing_runtime or shift_mismatch or missing_processor_revision:
            findings.append(
                ModelIntegrationFinding(
                    code="HP-DATA-002",
                    phase="D2",
                    owner_fqn="data.semantic_contract",
                    severity=FindingSeverity.ERROR,
                    message="resolved data pipeline disagrees with the model validation contract",
                    facts={
                        "missing_runtime_fields": missing_runtime,
                        "expected_labels_are_shifted": data_spec.labels_are_shifted,
                        "actual_labels_are_shifted": labels_are_shifted,
                        "processor_revision_missing": missing_processor_revision,
                    },
                    why_unsafe=(
                        "label shift, packed boundaries, modality metadata, and processor revision "
                        "change the mathematical training input"
                    ),
                    remediation=(
                        "fix the dataset/get_batch recipe or adapter declarations before using "
                        "the run for precision conclusions"
                    ),
                    related_config=("dataset", "dataloader", "model adapter validation"),
                )
            )
    ownership.update(
        {
            "source_type": source_type,
            "labels_are_shifted": labels_are_shifted,
        }
    )
    return ownership, findings


def validate_observed_forward_fields(
        validation_spec: Optional[ModelValidationSpec],
        model_inputs: Mapping[str, Any],
) -> list[ModelIntegrationFinding]:
    """Validate the actual first-step kwargs against required model semantics."""
    data_spec = validation_spec.data if validation_spec is not None else None
    if data_spec is None:
        return []
    missing = sorted(set(data_spec.required_forward_fields) - set(model_inputs))
    modality_present = sorted(set(data_spec.modality_fields).intersection(model_inputs))
    if not missing:
        return []
    return [
        ModelIntegrationFinding(
            code="HP-DATA-002",
            phase="D7",
            owner_fqn="model.forward",
            severity=FindingSeverity.ERROR,
            message="actual model forward kwargs omit required training fields",
            facts={
                "missing": missing,
                "observed": sorted(model_inputs),
                "modality_fields_present": modality_present,
            },
            why_unsafe="the executed model input is not the declared mathematical training sample",
            remediation="fix transform/collation/runtime-input ownership for the missing fields",
            related_config=("dataset", "dataloader.get_batch"),
        )
    ]


def modality_gradient_parameters(
        validation_spec: Optional[ModelValidationSpec],
        model: Any,
) -> dict[str, bool]:
    """Return whether every declared multimodal group has a finite gradient."""
    data_spec = validation_spec.data if validation_spec is not None else None
    if data_spec is None:
        return {}
    result = {}
    named_parameters = tuple(model.named_parameters(remove_duplicate=False))
    for pattern in data_spec.modality_parameter_patterns:
        matched = [
            parameter
            for parameter_name, parameter in named_parameters
            if parameter.requires_grad and fnmatch.fnmatchcase(parameter_name, pattern)
        ]
        finite_gradients = []
        for parameter in matched:
            gradient = getattr(parameter, "main_grad", None)
            if gradient is None:
                gradient = parameter.grad
            local = (
                gradient.to_local()
                if callable(getattr(gradient, "to_local", None))
                else gradient
            )
            if local is not None:
                finite_gradients.append(bool(local.isfinite().all()))
        result[pattern] = bool(finite_gradients) and all(finite_gradients)
    return result


__all__ = [
    "build_field_ownership",
    "modality_gradient_parameters",
    "validate_data_contract",
    "validate_observed_forward_fields",
]
