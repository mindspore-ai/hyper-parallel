# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Codegen: bake TP/CP/EP parallel contracts into a generated HF-modeling file.

Public manager helpers:

- :func:`ensure_codegen_artifact` — generate/reuse the bundle for a YAML.
- :func:`preflight_integrity_check` — fail-fast drift detection before train.
- :func:`resolve_artifact_layout` — map a YAML path to its bundle paths.

Planning and validation APIs:

- :func:`derive_sharding_plan` / :func:`freeze_plan` — run the planner over
  an offline mesh and freeze the plan into JSON-safe structures.
- :func:`verify_signature` / :func:`verify_output_hashes` / ... — per-zone
  preflight verifiers.
"""
from hyper_parallel.codegen.artifact import resolve_artifact_layout
from hyper_parallel.codegen.check.preflight import (
    verify_generated_import,
    verify_meta_required_fields,
    verify_output_hashes,
    verify_param_plan,
    verify_signature,
    warn_skipped_overrides,
)
from hyper_parallel.codegen.manager import (
    ensure_codegen_artifact,
    preflight_integrity_check,
)
from hyper_parallel.codegen.plan.freeze import (
    FrozenPlan,
    freeze_plan,
    freeze_param_plan,
    named_placement_to_dict,
    placement_to_string,
)
from hyper_parallel.codegen.plan.offline_mesh import OfflineMesh, build_offline_mesh

__all__ = [
    "FrozenPlan",
    "OfflineMesh",
    "build_offline_mesh",
    "ensure_codegen_artifact",
    "freeze_param_plan",
    "freeze_plan",
    "named_placement_to_dict",
    "placement_to_string",
    "preflight_integrity_check",
    "resolve_artifact_layout",
    "verify_generated_import",
    "verify_meta_required_fields",
    "verify_output_hashes",
    "verify_param_plan",
    "verify_signature",
    "warn_skipped_overrides",
]
