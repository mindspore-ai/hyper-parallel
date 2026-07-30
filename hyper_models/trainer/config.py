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
from typing import Any, Literal, Optional

from hyper_models.components.checkpoint.config import CheckpointingConfig
from hyper_models.components.datasets import DatasetConfig
from hyper_models.components.loss import Loss
from hyper_models.components.optim import LRScheduler, Optimizer
from hyper_parallel.trainer import config as legacy_config


@dataclass
class TrainingConfig:
    """Training-loop parameters exposed by the initial YAML schema."""

    max_steps: int = 100
    num_train_epochs: int = 1
    global_batch_size: int = 8
    micro_batch_size: int = 1
    backend: Literal["nccl", "hccl", "gloo"] = "nccl"
    init_device: Literal["meta", "cpu", "cuda", "npu"] = "meta"
    loss_aggregation: Literal["token_weighted", "rank_average"] = "token_weighted"
    seed: Optional[int] = None
    enable_full_determinism: bool = False


ModelConfig = legacy_config.ModelConfig


@dataclass
class DataLoaderConfig:
    """DataLoader behavior consumed by the Trainer loop."""

    shuffle: bool = True
    drop_last: bool = True
    use_background_prefetcher: bool = False


@dataclass
class FSDPConfig:
    """FSDP runtime behavior used by the Trainer micro-batch loop."""

    fsdp_mode: Literal["fsdp2"] = "fsdp2"
    reshard_after_backward: bool = False


@dataclass
class AcceleratorConfig:
    """Parallel topology exposed by the initial YAML schema."""

    dp_shard_size: int = 1
    dp_replicate_size: int = 1
    tp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1
    pp_size: int = 1
    sequence_parallel: bool = False
    loss_parallel: bool = False
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)


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
class WandbConfig:
    """WandB remote-logging parameters (03 §4.2.5：build_callback_manager 读取）。"""

    enabled: bool = False
    project: str = ""
    entity: Optional[str] = None


@dataclass
class TrainerConfig:
    """Resolved component tree; runtime objects are built by the task trainer."""
    # model identity is the only required root component
    model: ModelConfig

    # general training configs
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # parallelism configs
    accelerator: AcceleratorConfig = field(default_factory=AcceleratorConfig)
    mixed_precision: MixedPrecisionConfig = field(
        default_factory=MixedPrecisionConfig
    )
    gradient_checkpointing: GradientCheckpointingConfig = field(
        default_factory=GradientCheckpointingConfig
    )

    # training components
    optimizer: Optional[Optimizer.Config] = None
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    lr_scheduler: Optional[LRScheduler.Config] = None
    loss: Optional[Loss.Config] = None

    # callbacks
    checkpoint: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    magi: Optional[Any] = None
    peft: Optional[Any] = None


def save_configs(config: TrainerConfig, output_dir: str) -> None:
    """Accept trainer config persistence requests without writing files.

    Args:
        config: Resolved trainer configuration.
        output_dir: Intended configuration output directory.
    """
    del config, output_dir


__all__ = [
    "AcceleratorConfig",
    "DataLoaderConfig",
    "DebugConfig",
    "DatasetConfig",
    "FSDPConfig",
    "GradientCheckpointingConfig",
    "MixedPrecisionConfig",
    "TrainerConfig",
    "TrainingConfig",
    "WandbConfig",
    "ModelConfig",
    "save_configs",
    "CheckpointingConfig",
]
