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
"""instantiate_infrastructure + apply_model_infrastructure.

Following design doc 01_hf_compatibility_layer.md §8.
Stub — creates ShardingPlanner, FSDP2Manager, and AutoPipeline.
"""

import logging
from typing import Any, Optional

import torch
from torch import nn

from hyper_models.components.distributed.fsdp2 import FSDP2Manager, _instantiate_fsdp2
from hyper_models.components.distributed.pipelining import _instantiate_pipeline
from hyper_models.components.distributed.sharding_applier import apply_sharding_plan
from hyper_models.components.distributed.sharding_planner import ShardingPlanner
from hyper_models.trainer.config import entries_to_plan_overrides

logger = logging.getLogger(__name__)


def instantiate_infrastructure(
    distributed_setup=None,
    device=None,
    **kwargs,
) -> tuple[Any, Any, Any]:
    """Instantiate distributed infrastructure components.

    Following design doc 01 §8.2.

    Returns:
        (sharding_planner, fsdp2_manager, autopipeline) tuple.
    """
    del kwargs, device
    # ShardingPlanner — already implemented in components/distributed.
    # plan_overrides come from the resolved TrainerConfig (YAML
    # plan_overrides → List[PlanOverride]) via DistributedSetup; they are
    # desugared HERE (placement DSL → objects, when-filtered against the
    # accelerator topology) so the planner has exactly one override
    # interface.
    entries = getattr(distributed_setup, "plan_overrides", None) or None
    if entries is not None:
        mesh_ctx = getattr(distributed_setup, "mesh_context", None)
        plan_overrides = entries_to_plan_overrides(
            entries,
            cp_size=getattr(mesh_ctx, "cp_size", 1),
            ep_size=getattr(mesh_ctx, "ep_size", 1))
    else:
        plan_overrides = None
    sharding_planner = ShardingPlanner(plan_overrides=plan_overrides)

    # FSDP2Manager: build from strategy config if available
    fsdp2_manager = None
    mesh = distributed_setup.mesh_context if distributed_setup is not None else None
    strategy_cfg = distributed_setup.strategy_config if distributed_setup is not None else None
    if strategy_cfg is not None:
        fsdp2_manager = _instantiate_fsdp2(config=strategy_cfg, mesh_context=mesh)

    if fsdp2_manager is None:
        logger.info("FSDP2Manager: no strategy_config provided; skipping FSDP2 wrap")
    else:
        logger.info("FSDP2Manager instantiated with %s", type(fsdp2_manager.config).__name__)

    # AutoPipeline: only when pp_size > 1
    autopipeline = None
    pipeline_cfg = distributed_setup.pipeline_config if distributed_setup is not None else None
    if mesh is not None and mesh.pp_size > 1:
        autopipeline = _instantiate_pipeline(pipeline_cfg, mesh)
        if autopipeline is not None:
            logger.info("AutoPipeline instantiated for pp_size=%d", mesh.pp_size)

    return sharding_planner, fsdp2_manager, autopipeline


def _plan_and_apply_sharding(
    model: nn.Module,
    mesh,
    sharding_planner,
    is_hf_model: bool,
    validate_placement: bool,
) -> tuple[nn.Module, Optional[dict]]:
    """Plan parameter layouts and apply dual-mode sharding when requested."""
    model_sharding_requested = (
        mesh is not None
        and any(size > 1 for size in (mesh.tp_size, mesh.cp_size, mesh.ep_size))
    )
    if (
        sharding_planner is None
        or mesh is None
        or (is_hf_model and not model_sharding_requested)
    ):
        return model, None
    if mesh.device_mesh is None:
        logger.warning("MeshContext has no device_mesh; skipping sharding")
        return model, None

    logger.info(
        "Running ShardingPlanner.plan(tp=%d, cp=%d, ep=%d, "
        "sequence_parallel=%s, loss_parallel=%s)",
        mesh.tp_size,
        mesh.cp_size,
        mesh.ep_size,
        mesh.sequence_parallel,
        mesh.loss_parallel,
    )
    plan = sharding_planner.plan(
        model,
        mesh.device_mesh,
        tp_size=mesh.tp_size,
        cp_size=mesh.cp_size,
        ep_size=mesh.ep_size,
        sequence_parallel=mesh.sequence_parallel,
        loss_parallel=mesh.loss_parallel,
    )
    model, tp_grad_info = apply_sharding_plan(
        model,
        plan,
        mesh,
        validate_mode=validate_placement,
    )
    logger.info("Sharding plan applied; tp_grad_info keys=%d", len(tp_grad_info or {}))
    return model, tp_grad_info


def _apply_fsdp2(model: nn.Module, fsdp2_manager, tp_grad_info) -> nn.Module:
    """Apply FSDP2 after planning and compile have finalized the model graph."""
    if fsdp2_manager is None:
        return model
    if not isinstance(fsdp2_manager, FSDP2Manager):
        logger.warning("fsdp2_manager is not an FSDP2Manager instance")
        return model
    model = fsdp2_manager.parallelize(model, tp_grad_info=tp_grad_info)
    logger.info("FSDP2 wrap applied")
    return model


def _move_model_to_device(
    model: nn.Module,
    is_meta_device: bool,
    device,
    load_base_model: bool,
    pretrained_path: Optional[str],
) -> nn.Module:
    """Materialize meta parameters or move an initialized model to its device."""
    if device is None:
        return model
    if not is_meta_device:
        model.to(device)
        logger.info("Model moved to %s", device)
        return model

    model.to_empty(device=device)
    if load_base_model and pretrained_path is not None:
        logger.warning("load_base_model not implemented in stub")
    # Stub path: without real weight loading, uninitialized meta tensors
    # would yield NaN losses. Initialize parameters/buffers so skeleton runs
    # produce finite gradients. Remove/replace this when load_base_model lands.
    for parameter in model.parameters():
        if parameter.dtype.is_floating_point:
            nn.init.normal_(parameter, mean=0.0, std=0.02)
    for buffer in model.buffers():
        if buffer.dtype.is_floating_point:
            nn.init.zeros_(buffer)
    logger.info("Model moved from meta to %s", device)
    return model


def apply_model_infrastructure(
    model: nn.Module,
    mesh=None,
    sharding_planner=None,
    fsdp2_manager=None,
    autopipeline=None,
    peft_config=None,
    qat_config=None,
    fp8_config=None,
    freeze_config=None,
    compile_config=None,
    is_meta_device: bool = False,
    is_hf_model: bool = False,
    device=None,
    load_base_model: bool = False,
    pretrained_path: Optional[str] = None,
    validate_placement: bool = False,
    **kwargs,
) -> nn.Module:
    """Apply model infrastructure (sharding, PEFT, FSDP2, weight loading).

    Following design doc 01 §8.3 canonical order:
        PP split → PEFT → QAT/FP8 → freeze → plan → apply_sharding_plan
        → torch.compile → FSDP2 wrap → to_empty + load_base_model

    D-01'': CP K/V all-gather is injected at apply_sharding_plan time;
           no extra CP hooks are registered here.

    Stub — applies sharding plan if sharding_planner is provided.
    """
    del kwargs

    # Step 3: PP split (if autopipeline)
    if autopipeline is not None:
        # build() is in-place; model stays the original nn.Module
        autopipeline.build(model)

    # Step 4: PEFT injection (before sharding)
    if peft_config is not None:
        logger.warning("PEFT injection not implemented in stub")

    # Step 5: QAT / FP8 (before sharding)
    if qat_config is not None:
        logger.warning("QAT not implemented in stub")
    if fp8_config is not None:
        logger.warning("FP8 not implemented in stub")

    # Step 6: Parameter freezing (before sharding)
    if freeze_config is not None:
        logger.warning("Parameter freezing not implemented in stub")

    # Steps 7-8: plan and apply parameter/activation layouts.
    model, tp_grad_info = _plan_and_apply_sharding(
        model,
        mesh,
        sharding_planner,
        is_hf_model,
        validate_placement,
    )

    # Step 9: torch.compile
    if compile_config is not None:
        model = torch.compile(model, **compile_config)

    # Steps 10-11: FSDP2 wrap, then materialize/move model storage.
    model = _apply_fsdp2(model, fsdp2_manager, tp_grad_info)
    return _move_model_to_device(
        model,
        is_meta_device,
        device,
        load_base_model,
        pretrained_path,
    )
