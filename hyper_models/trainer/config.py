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
"""Typed configuration tree produced by the HyperModels YAML resolver."""

from dataclasses import dataclass, field
from typing import Literal, Optional

from hyper_models.components.loss import Loss
from hyper_models.components.optim import LRScheduler, Optimizer
from hyper_parallel.trainer import config as legacy_config


@dataclass
class TrainingConfig:
    """Training-loop parameters exposed by the initial YAML schema."""

    max_steps: int = 100
    global_batch_size: int = 8
    init_device: Literal["meta", "cpu", "cuda", "npu"] = "meta"
    loss_aggregation: Literal["token_weighted", "rank_average"] = "token_weighted"


@dataclass
class AcceleratorConfig:
    """Parallel topology exposed by the initial YAML schema."""

    dp_shard_size: int = 1
    tp_size: int = 1


@dataclass
class MixedPrecisionConfig:
    """Mixed-precision parameters exposed by the initial YAML schema."""

    enabled: bool = False


@dataclass
class GradientCheckpointingConfig:
    """Activation-checkpoint mode exposed by the initial YAML schema."""

    activation_checkpoint: Literal["off", "none", "full", "selective"] = "off"


@dataclass
class DebugConfig:
    """Debug parameters exposed by the initial YAML schema."""

    check_nan_inf: bool = False


@dataclass
class TrainerConfig:
    """Resolved component tree; runtime objects are built by the task trainer."""

    model: legacy_config.ModelConfig
    optimizer: Optional[Optimizer.Config] = None
    lr_scheduler: Optional[LRScheduler.Config] = None
    loss: Optional[Loss.Config] = None
    training: TrainingConfig = field(default_factory=TrainingConfig)
    accelerator: AcceleratorConfig = field(default_factory=AcceleratorConfig)
    mixed_precision: MixedPrecisionConfig = field(
        default_factory=MixedPrecisionConfig
    )
    gradient_checkpointing: GradientCheckpointingConfig = field(
        default_factory=GradientCheckpointingConfig
    )
    debug: DebugConfig = field(default_factory=DebugConfig)


__all__ = [
    "AcceleratorConfig",
    "DebugConfig",
    "GradientCheckpointingConfig",
    "MixedPrecisionConfig",
    "TrainerConfig",
    "TrainingConfig",
]
