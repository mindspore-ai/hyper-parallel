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
from hyper_models.components.loss import Loss
from hyper_models.components.optim import LRScheduler, Optimizer
from hyper_models.components.training.step_scheduler import StepSchedulerConfig
from hyper_parallel.trainer import config as legacy_config


@dataclass
class TrainingConfig:
    """Training-loop parameters exposed by the initial YAML schema."""

    max_steps: int = 100
    global_batch_size: int = 8
    init_device: Literal["meta", "cpu", "cuda", "npu"] = "meta"
    loss_aggregation: Literal["token_weighted", "rank_average"] = "token_weighted"
    # 随机种子（03 §5.3 ③：StatefulRNG(seed=cfg.training.seed, ranked=True)）
    seed: int = 42


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

    # ── 训练循环扩展字段（03 §5.2/§13 规划 schema，随 Recipe 骨架落地） ──
    # Recipe 名称（03 §13：main() 经 RECIPE_REGISTRY 解析，默认 FinetuneRecipe）
    recipe: str = "FinetuneRecipe"
    # 训练节奏控制（03 §4.1：typed .build(dataloader, dp_size, local_bs)）
    step_scheduler: StepSchedulerConfig = field(default_factory=StepSchedulerConfig)
    # Checkpoint（04 §4：typed .build(dp_rank, tp_rank, ...)）
    checkpoint: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    # WandB 远程日志（03 §4.2.5）
    wandb: WandbConfig = field(default_factory=WandbConfig)
    # 以下字段由 02_data_pipeline.md 消费（build_dataloader 独立构建函数），
    # 数据管道落地前保持弱类型（Any）。
    dataset: Optional[Any] = None
    dataloader: Optional[Any] = None
    packed_sequence: Optional[Any] = None
    # MagiAttention 上下文（03 §5.3 ⑤；无配置时 setup_magi 返回 None）
    magi: Optional[Any] = None
    # PEFT 配置（03 §5.3 ⑨：传入 build_model 并用于判断 is_peft）
    peft: Optional[Any] = None


__all__ = [
    "AcceleratorConfig",
    "DebugConfig",
    "GradientCheckpointingConfig",
    "MixedPrecisionConfig",
    "TrainerConfig",
    "TrainingConfig",
    "WandbConfig",
]
