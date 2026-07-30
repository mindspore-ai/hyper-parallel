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

import torch.nn as nn

from hyper_models._transformers import HyperAutoModelForCausalLM
from hyper_models.components.distributed.infrastructure import DistributedSetup
from hyper_models.components.optim.optimizer import OptimizerInit

logger = logging.getLogger(__name__)


def build_model(
    model_cfg: Any,
    peft_config: Any = None,
    distributed_setup: Optional[DistributedSetup] = None,
    **kwargs: Any,
) -> tuple[nn.Module, Optional[OptimizerInit]]:
    """High-level build_model entry point.

    Following design doc 01 §6.7:
    ① Call HyperAutoModelForCausalLM.from_pretrained() (HF-compatible entry)
    ② Export OptimizerInit from distributed_setup / ShardingPlan

    HyperAutoModelForCausalLM is the unified model loading path: it handles
    both single-card and distributed setups, and applies ShardingPlanner /
    FSDP2 / PEFT internally when a non-trivial DistributedSetup is provided.
    The legacy "plain HF AutoModel" fallback has been removed because
    HyperAutoModel already provides an equivalent single-card path.

    Args:
        model_cfg: ModelConfig or ConfigNode containing a model path.
        peft_config: PEFT configuration (optional).
        distributed_setup: Distributed topology and strategy.
        **kwargs: Extra args for model construction.

    Returns:
        (model, optimizer_init) tuple.
    """
    pretrained_path = (
        getattr(model_cfg, "weights_path", None)
        or getattr(model_cfg, "pretrained_model_name_or_path", None)
        or getattr(model_cfg, "model_name_or_path", None)
    )
    if not pretrained_path:
        raise ValueError(
            "model config must define model_name_or_path, weights_path, "
            "or pretrained_model_name_or_path"
        )

    model = HyperAutoModelForCausalLM.from_pretrained(
        pretrained_path,
        distributed_setup=distributed_setup,
        peft_config=peft_config,
        **kwargs,
    )

    # Export OptimizerInit
    optimizer_init = OptimizerInit.from_distributed_setup(
        distributed_setup=distributed_setup,
        model=model,
        peft_config=peft_config,
    )

    return model, optimizer_init
