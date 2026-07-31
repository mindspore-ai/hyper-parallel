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
from dataclasses import dataclass
from typing import Any, Optional

import torch.nn as nn

from hyper_parallel import HSDPModule
from hyper_parallel.trainer.config import ModelConfig
from hyper_models._transformers import HyperAutoModelForCausalLM
from hyper_models.components.distributed.infrastructure import DistributedSetup
from hyper_models.components.optim.optimizer.optimizer import OptimizerInit

logger = logging.getLogger(__name__)


@dataclass
class ModelBuildResult:
    """Runtime objects produced by the model component target."""

    model: nn.Module
    optimizer_init: Optional[OptimizerInit]
    model_config: Any
    model_parts: list[nn.Module]
    hsdp_model_parts: list[nn.Module]


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


def build_model_component(
    *,
    distributed_setup: DistributedSetup,
    peft_config: Any,
    name: str = "qwen3_5",
    weights_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    **model_kwargs: Any,
) -> ModelBuildResult:
    """Build all model-owned runtime state required by the trainer.

    Args:
        distributed_setup: Runtime distributed topology and sharding policy.
        peft_config: Optional PEFT configuration supplied by the trainer.
        name: Registered model name.
        weights_path: Local or remote pretrained checkpoint path.
        tokenizer_path: Tokenizer path retained in the model configuration.
        **model_kwargs: Additional ``ModelConfig`` fields.

    Returns:
        Built model together with optimizer initialization and model metadata.
    """
    model_config = ModelConfig(
        name=name,
        weights_path=weights_path,
        tokenizer_path=tokenizer_path,
        **model_kwargs,
    )
    model, optimizer_init = build_model(
        model_config,
        peft_config,
        distributed_setup=distributed_setup,
    )
    runtime_model_config = model.config
    model_parts = model.parts if hasattr(model, "parts") else [model]
    hsdp_model_parts = [
        model_part
        for model_part in model_parts
        if isinstance(model_part, HSDPModule)
    ]
    return ModelBuildResult(
        model=model,
        optimizer_init=optimizer_init,
        model_config=runtime_model_config,
        model_parts=model_parts,
        hsdp_model_parts=hsdp_model_parts,
    )


__all__ = ["ModelBuildResult", "build_model", "build_model_component"]
