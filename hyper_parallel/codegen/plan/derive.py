# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""Derive the sharding plan for a model without a live process group.

The real ``ShardingPlanner`` runs at *generation* time so the
frozen plan (and the signature over it) reflects the actual model structure
— planner outputs land in ``CodegenMeta`` and, later, in the generated
modeling file.  All helpers are deliberately independent of the training
stack: they take a ``CodegenSpec`` (or a config object exposing the same
attributes) and return plain data.

``build_meta_model`` constructs the model on the meta device — no weights
are loaded, so a codegen run never needs the checkpoint or a GPU.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from hyper_parallel.codegen.plan.offline_mesh import build_offline_mesh
from hyper_parallel.codegen.spec.types import CodegenSpec

logger = logging.getLogger(__name__)


def build_meta_model(
    spec: CodegenSpec,
    *,
    config_overrides: Optional[dict[str, Any]] = None,
) -> Any:
    """Construct an empty meta-device model for the spec's source.

    ``AutoModelForCausalLM.from_config(config)`` builds the full parameter
    tree with no weights and no network; wrapping it in
    ``components.utils.model_utils.init_empty_weights`` keeps the parameters
    on the meta device so a CPU-only host never allocates the model's memory.

    Returns ``None`` only when the spec carries no model identity (the caller
    then skips plan derivation entirely).  A model that *is* configured but
    cannot be built is a hard failure — swallowing it would freeze an empty
    plan and let an incomplete artifact pass preflight.
    """
    model_id = spec.source.model_name_or_path
    if not model_id:
        return None
    try:
        from hyper_parallel import init_empty_weights

        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(model_id, **dict(config_overrides or {}))
        with init_empty_weights():
            return AutoModelForCausalLM.from_config(config)
    except Exception as exc:  # noqa: BLE001 - fail fast: an unbuildable source must not yield a plan-less artifact
        logger.error(
            "codegen: cannot build meta model for %r (%s: %s)",
            model_id, type(exc).__name__, exc,
        )
        raise


def build_plan_overrides(config: Any, spec: CodegenSpec) -> dict[str, Any]:
    """Desugar ``config.plan_overrides`` into the planner's override dict.

    Wraps ``trainer.config.entries_to_plan_overrides`` (imported lazily — the
    codegen package must stay importable without the trainer layer).
    ``when``-gated entries are skipped here against the projected dims, the
    same way the trainer's own desugar pass applies them at run time.
    """
    entries = getattr(config, "plan_overrides", None) or []
    if not entries:
        return {}
    from hyper_parallel.trainer.config import entries_to_plan_overrides

    return entries_to_plan_overrides(
        entries,
        cp_size=spec.parallel_dims.cp_size,
        ep_size=spec.parallel_dims.ep_size,
    )


def derive_sharding_plan(
    model: Any, spec: CodegenSpec, *, plan_overrides: Optional[dict[str, Any]] = None,
) -> Any:
    """Run ``ShardingPlanner.plan`` over the offline mesh for ``spec``.

    Returns the planner's ``ShardingPlan``.  A planner failure raises — the
    manager's ``_fill_plan_fields`` turns a None plan into an empty frozen
    plan, which is exactly the silent-degradation path that must not survive
    (a plan-less artifact passing preflight).  ``model`` is the meta-device
    model from :func:`build_meta_model`; the mesh is the static offline mesh,
    so no process group is needed.

    ``plan_overrides`` is the desugared override dict — the planner's native
    interface — normally produced by :func:`build_plan_overrides` from the
    live config.  ``allow_uncovered_params`` stays at its default (False):
    the F4b fail-fast guard must fire here, at generation time, not silently
    inside the trainer.
    """
    parallel_dims = spec.parallel_dims
    mesh = build_offline_mesh(parallel_dims)
    from hyper_parallel.distributed._builder.planner import ShardingPlanner

    planner = ShardingPlanner(plan_overrides=dict(plan_overrides or {}))
    return planner.plan(
        model,
        mesh,
        tp_size=parallel_dims.tp_size,
        cp_size=parallel_dims.cp_size,
        ep_size=parallel_dims.ep_size,
        sequence_parallel=parallel_dims.sequence_parallel,
        loss_parallel=parallel_dims.loss_parallel,
    )


__all__ = [
    "build_meta_model",
    "build_plan_overrides",
    "derive_sharding_plan",
]
