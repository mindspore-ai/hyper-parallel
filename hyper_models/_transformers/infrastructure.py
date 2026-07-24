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
import torch.nn as nn

from hyper_models.components.distributed.fsdp2 import FSDP2Manager, _instantiate_fsdp2
from hyper_models.components.distributed.pipelining import _instantiate_pipeline
from hyper_models.components.distributed.sharding_planner import ShardingPlanner
from hyper_models.components.distributed.config import _resolve_strategy_config

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
    # ShardingPlanner — already implemented in components/distributed
    sharding_planner = ShardingPlanner()

    # FSDP2Manager: build from strategy config if available
    fsdp2_manager = None
    mesh = getattr(distributed_setup, "mesh_context", None)
    strategy_cfg = getattr(distributed_setup, "strategy_config", None)
    if strategy_cfg is not None:
        fsdp2_manager = _instantiate_fsdp2(config=strategy_cfg, mesh_context=mesh)
    elif kwargs.get("strategy") is not None:
        strategy_cfg = _resolve_strategy_config(kwargs["strategy"])
        fsdp2_manager = _instantiate_fsdp2(config=strategy_cfg, mesh_context=mesh)

    if fsdp2_manager is None:
        logger.warning("FSDP2Manager: no strategy_config provided; returning None")
    else:
        logger.info("FSDP2Manager instantiated with %s", type(fsdp2_manager.config).__name__)

    # AutoPipeline: only when pp_size > 1
    autopipeline = None
    pipeline_cfg = getattr(distributed_setup, "pipeline_config", None)
    if mesh is not None and getattr(mesh, "pp_size", 1) > 1:
        autopipeline = _instantiate_pipeline(pipeline_cfg, mesh)
        if autopipeline is not None:
            logger.info("AutoPipeline instantiated for pp_size=%d", mesh.pp_size)

    return sharding_planner, fsdp2_manager, autopipeline


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
    from hyper_models.components.distributed.sharding_applier import apply_sharding_plan
    from hyper_models.components.distributed.sharding_config import ShardingPlan

    plan: Optional[ShardingPlan] = None
    tp_grad_info: Optional[dict] = None

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

    # Step 7: ShardingPlanner.plan() → ShardingPlan
    # Step 8: apply_sharding_plan (includes _local_params_context unpack + wrapping)
    if not is_hf_model and sharding_planner is not None and mesh is not None:
        device_mesh = getattr(mesh, "device_mesh", None)
        if device_mesh is not None:
            tp_size = getattr(mesh, "tp_size", 1)
            cp_size = getattr(mesh, "cp_size", 1)
            ep_size = getattr(mesh, "ep_size", 1)
            sequence_parallel = bool(getattr(mesh, "sequence_parallel", False))
            loss_parallel = bool(getattr(mesh, "loss_parallel", False))

            logger.info(
                "Running ShardingPlanner.plan(tp=%d, cp=%d, ep=%d, "
                "sequence_parallel=%s, loss_parallel=%s)",
                tp_size, cp_size, ep_size, sequence_parallel, loss_parallel,
            )
            plan = sharding_planner.plan(
                model,
                device_mesh,
                tp_size=tp_size,
                cp_size=cp_size,
                ep_size=ep_size,
                sequence_parallel=sequence_parallel,
                loss_parallel=loss_parallel,
            )

            # Step 8: apply_sharding_plan returns (model, tp_grad_info)
            model, tp_grad_info = apply_sharding_plan(
                model,
                plan,
                device_mesh,
                validate_mode=validate_placement,
            )
            logger.info("Sharding plan applied; tp_grad_info keys=%d", len(tp_grad_info or {}))
        else:
            logger.warning("MeshContext has no device_mesh; skipping sharding")

    # Step 9: torch.compile
    if compile_config is not None:
        model = torch.compile(model, **compile_config)

    # Step 10: FSDP2 wrap (on meta or real)
    if fsdp2_manager is not None:
        if isinstance(fsdp2_manager, FSDP2Manager):
            model = fsdp2_manager.parallelize(
                model,
                tp_shard_plan=plan,
                tp_grad_info=tp_grad_info,
            )
            logger.info("FSDP2 wrap applied")
        else:
            logger.warning("fsdp2_manager is not an FSDP2Manager instance")

    # Step 11: meta → device + load_base_model
    if is_meta_device and device is not None:
        model.to_empty(device=device)
        if load_base_model and pretrained_path is not None:
            logger.warning("load_base_model not implemented in stub")
        logger.info("Model moved from meta to %s", device)

    return model
