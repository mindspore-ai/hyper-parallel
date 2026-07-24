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
"""Common model utilities — build_model, OptimizerInit.

Following design doc 01_hf_compatibility_layer.md §6.7.
"""

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

from hyper_models._transformers import HyperAutoModelForCausalLM
from hyper_models.components.distributed.infrastructure import DistributedSetup
from hyper_models.components.optim.optimizer import OptimizerInit

logger = logging.getLogger(__name__)


def build_model(
    model_cfg,
    peft_config=None,
    distributed_setup: Optional[DistributedSetup] = None,
    **kwargs,
) -> tuple[nn.Module, Optional[OptimizerInit]]:
    """High-level build_model entry point.

    Following design doc 01 §6.7:
    ① Call HyperAutoModelForCausalLM.from_pretrained() (HF-compatible entry)
    ② Export OptimizerInit from distributed_setup / ShardingPlan

    Two model paths unified:
    - Path A (HyperAutoModel): from_pretrained internally handles meta→shard→load
    - Path B (HF native): _target_ is not HyperAutoModel — build separately

    Args:
        model_cfg: ModelConfig or ConfigNode (with _target_, name, weights_path etc.)
        peft_config: PEFT configuration (optional).
        distributed_setup: Distributed topology and strategy.
        **kwargs: Extra args for model construction.

    Returns:
        (model, optimizer_init) tuple.
    """
    # Determine if this is a HyperAutoModel target
    is_hyper_auto = False
    target = getattr(model_cfg, "_target_", None) or getattr(model_cfg, "target", None)
    if target is not None:
        is_hyper_auto = target in (
            HyperAutoModelForCausalLM.from_pretrained,
            HyperAutoModelForCausalLM.from_config,
        )

    if is_hyper_auto or distributed_setup is not None:
        # Path A: HyperAutoModel path — from_pretrained handles infrastructure
        pretrained_path = getattr(model_cfg, "weights_path", None) or getattr(model_cfg, "pretrained_model_name_or_path", None)
        model = HyperAutoModelForCausalLM.from_pretrained(
            pretrained_path,
            distributed_setup=distributed_setup,
            peft_config=peft_config,
            **kwargs,
        )
    else:
        # Path B: Non-HyperAutoModel path (e.g. transformers AutoModel)
        # Build model first, then apply infrastructure separately
        from transformers import AutoModelForCausalLM as HFAutoModel

        pretrained_path = getattr(model_cfg, "weights_path", None) or getattr(model_cfg, "pretrained_model_name_or_path", None)
        model = HFAutoModel.from_pretrained(pretrained_path, **kwargs)

        if distributed_setup is not None:
            from hyper_models._transformers.infrastructure import (
                apply_model_infrastructure,
                instantiate_infrastructure,
            )
            sharding_planner, fsdp2_manager, autopipeline = instantiate_infrastructure(
                distributed_setup=distributed_setup,
                device=torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu"),
            )
            model = apply_model_infrastructure(
                model,
                mesh=distributed_setup.mesh_context,
                sharding_planner=sharding_planner,
                fsdp2_manager=fsdp2_manager,
                autopipeline=autopipeline,
                peft_config=peft_config,
                is_meta_device=False,
                device=torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu"),
                load_base_model=False,
            )

    # Export OptimizerInit
    optimizer_init = OptimizerInit.from_distributed_setup(
        distributed_setup=distributed_setup,
        model=model,
        peft_config=peft_config,
    )

    return model, optimizer_init