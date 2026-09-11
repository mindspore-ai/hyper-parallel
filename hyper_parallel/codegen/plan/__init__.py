# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Plan derivation: offline mesh + ShardingPlanner -> frozen plan.

The planner runs at generation time over a static offline mesh and
freezes the result into ``CodegenMeta`` so the generated modeling file (and
preflight) can consume it without re-running the planner.
"""
from hyper_parallel.codegen.plan.boundary_forms import (
    BoundaryForm,
    FORM_GENERIC,
    FORM_IDENTITY,
    FORM_REGION,
    FORM_TP_COLLECTIVE,
    TransitionOp,
    classify_boundary_form,
)
from hyper_parallel.codegen.plan.derive import (
    build_meta_model,
    build_plan_overrides,
    derive_sharding_plan,
)
from hyper_parallel.codegen.plan.freeze import (
    FrozenPlan,
    expand_frozen_sharded_params,
    freeze_injections,
    freeze_param_plan,
    freeze_plan,
    freeze_tied_pairs,
    named_placement_to_dict,
    placement_to_string,
)
from hyper_parallel.codegen.plan.offline_mesh import OfflineMesh, build_offline_mesh

__all__ = [
    "BoundaryForm",
    "FORM_GENERIC",
    "FORM_IDENTITY",
    "FORM_REGION",
    "FORM_TP_COLLECTIVE",
    "FrozenPlan",
    "OfflineMesh",
    "TransitionOp",
    "build_meta_model",
    "build_offline_mesh",
    "build_plan_overrides",
    "classify_boundary_form",
    "derive_sharding_plan",
    "expand_frozen_sharded_params",
    "freeze_injections",
    "freeze_param_plan",
    "freeze_plan",
    "freeze_tied_pairs",
    "named_placement_to_dict",
    "placement_to_string",
]
