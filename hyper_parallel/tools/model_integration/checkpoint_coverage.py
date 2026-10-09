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
"""Complete checkpoint coverage reporting for finalized model state."""

from __future__ import annotations

import fnmatch
import re
from typing import Any, Optional

from hyper_parallel.tools.model_integration.schemas import (
    FindingSeverity,
    ModelIntegrationFinding,
)
from hyper_parallel.models.validation_spec import CheckpointValidationSpec


def _matches_target(transform: Any, target_name: str) -> bool:
    """Match processed WeightTransform regexes, not adapter-owned FQN globs."""
    scope = getattr(transform, "scope_prefix", None)
    scoped_name = target_name
    if scope and target_name.startswith(f"{scope}."):
        scoped_name = target_name.removeprefix(f"{scope}.")
    for pattern in getattr(transform, "target_patterns", ()):
        normalized = str(pattern).replace(".*.", ".0.")
        if re.search(normalized, scoped_name):
            return True
    return False


def _matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def _non_reversible_transforms(transforms: tuple[Any, ...]) -> list[str]:
    """Probe each declared transform for an executable reverse transform."""
    non_reversible = []
    for transform in transforms:
        reverse_transform = getattr(transform, "reverse_transform", None)
        if not callable(reverse_transform):
            non_reversible.append(type(transform).__name__)
            continue
        try:
            reverse_transform()
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            non_reversible.append(type(transform).__name__)
    return non_reversible


def _target_category(
        target_name: str,
        matching: list[Any],
        loaded: set[str],
        intentional_missing: set[str],
        unsupported_missing: set[str],
) -> str:
    """Classify one finalized persistent target key."""
    if target_name in loaded:
        if not matching:
            return "direct_load"
        if any(hasattr(transform, "operations") for transform in matching):
            return "reversible_transform"
        return "same_shape_rename"
    if target_name in intentional_missing:
        return "training_only_initialization"
    if target_name in unsupported_missing:
        return "unsupported_missing_target"
    return "alias_or_non_persistent"


def _coverage_entries(
        model: Any,
        transforms: tuple[Any, ...],
        loaded: set[str],
        intentional_missing: set[str],
        unsupported_missing: set[str],
        unexpected: set[str],
        inference_only: set[str],
) -> list[dict[str, Any]]:
    """Build one coverage entry per target and per unexpected source key."""
    entries = []
    for target_name in sorted(model.state_dict()):
        matching = [transform for transform in transforms if _matches_target(transform, target_name)]
        entries.append(
            {
                "target_key": target_name,
                "category": _target_category(
                    target_name,
                    matching,
                    loaded,
                    intentional_missing,
                    unsupported_missing,
                ),
                "transforms": [type(transform).__name__ for transform in matching],
            }
        )
    for source_name in sorted(unexpected):
        entries.append(
            {
                "source_key": source_name,
                "category": (
                    "inference_only_exclusion"
                    if source_name in inference_only
                    else "unsupported_architectural_mismatch"
                ),
            }
        )
    return entries


def _coverage_findings(
        unsupported_missing: set[str],
        unsupported_unexpected: set[str],
        non_reversible: list[str],
) -> list[ModelIntegrationFinding]:
    """Emit ERROR findings for unsupported keys and one-way transforms."""
    findings = []
    if unsupported_missing or unsupported_unexpected:
        findings.append(
            ModelIntegrationFinding(
                code="HP-CKPT-001",
                phase="D5",
                owner_fqn="<root>",
                severity=FindingSeverity.ERROR,
                message="checkpoint coverage is incomplete after finalization",
                facts={
                    "missing_targets": sorted(unsupported_missing),
                    "unexpected_sources": sorted(unsupported_unexpected),
                },
                why_unsafe=(
                    "strict=False logs do not prove that every authoritative tensor has an "
                    "intentional load, transform, initialization, or exclusion policy"
                ),
                remediation=(
                    "add a checkpoint mapping/transform or an explicit training-only/inference-only "
                    "declaration, then rebuild with strict finalization"
                ),
                related_config=("model.load_base_model", "model adapter checkpoint"),
            )
        )
    if non_reversible:
        findings.append(
            ModelIntegrationFinding(
                code="HP-CKPT-002",
                phase="D5",
                owner_fqn="<root>",
                severity=FindingSeverity.ERROR,
                message="checkpoint transform has no executable reverse transform",
                facts={"transforms": non_reversible},
                why_unsafe=(
                    "a one-way transform cannot preserve the authoritative namespace and values "
                    "when exporting or validating a source-target-source round trip"
                ),
                remediation="provide inverse conversion operations and add a tensor round-trip parity case",
                related_config=("model adapter checkpoint",),
            )
        )
    return findings


def build_checkpoint_coverage(
        model: Any,
        validation_spec: Optional[CheckpointValidationSpec] = None,
) -> tuple[dict[str, Any], list[ModelIntegrationFinding]]:
    """Classify every finalized persistent state key using the last load report."""
    report = getattr(model, "_hp_checkpoint_load_report", None)
    if report is None:
        return {
            "status": "NOT_LOADED",
            "entries": [],
            "reason": "model was initialized from scratch or no load report was retained",
        }, []
    loaded = set(getattr(report, "loaded_keys", ()))
    missing = set(getattr(report, "missing_keys", ()))
    unexpected = set(getattr(report, "unexpected_keys", ()))
    validation_spec = validation_spec or CheckpointValidationSpec()
    intentional_missing = {
        name
        for name in missing
        if _matches_any(name, validation_spec.training_only_target_patterns)
    }
    inference_only = {
        name
        for name in unexpected
        if _matches_any(name, validation_spec.inference_only_source_patterns)
    }
    unsupported_missing = missing - intentional_missing
    unsupported_unexpected = unexpected - inference_only
    transforms = tuple(getattr(model, "_weight_conversions", ()) or ())
    non_reversible = _non_reversible_transforms(transforms)
    entries = _coverage_entries(
        model,
        transforms,
        loaded,
        intentional_missing,
        unsupported_missing,
        unexpected,
        inference_only,
    )
    findings = _coverage_findings(
        unsupported_missing,
        unsupported_unexpected,
        non_reversible,
    )
    return {
        "status": "FAIL" if findings else "PASS",
        "loaded": len(loaded),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "intentional_training_only": len(intentional_missing),
        "intentional_inference_only": len(inference_only),
        "entries": entries,
    }, findings


__all__ = ["build_checkpoint_coverage"]
