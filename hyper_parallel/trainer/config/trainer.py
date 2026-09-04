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
"""Aggregate trainer configuration tree.

Split from ``auto_models/trainer/config.py`` in stage 7 (05 §15.2.5);
class name, fields and defaults are unchanged.
"""

from dataclasses import dataclass, field, fields
from typing import Any, List, Literal, Optional

from hyper_parallel.models.build_options import CompileConfig, FSDP2Config
from hyper_parallel.components.checkpoint.config import CheckpointingConfig

from hyper_parallel.trainer.config.data import DataLoaderConfig, DatasetConfig
from hyper_parallel.trainer.config.optimization import (
    MixedPrecisionConfig,
    OptimizerConfig,
)
from hyper_parallel.trainer.config.parallelism import (
    AcceleratorConfig,
    ActivationCheckpointConfig,
    PlanOverride,
)
from hyper_parallel.trainer.config.target import Target, _serialize_config_value
from hyper_parallel.trainer.config.training import (
    DebugConfig,
    ProfilingConfig,
    TrainingConfig,
    WandbConfig,
)


@dataclass
class TrainerConfig:
    """Resolved component tree; runtime objects are built by the task trainer."""

    model: Target[Any]
    optimizer: OptimizerConfig

    lr_scheduler: Optional[Target[Any]] = None
    loss_fn: Optional[Target[Any]] = None
    # Final floating-point dtype after model weights are loaded or initialized
    # from scratch. None preserves the dtype produced by the initialization path.
    model_init_dtype: Optional[Literal["float16", "bfloat16", "float32"]] = None
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # parallelism configs
    accelerator: AcceleratorConfig = field(default_factory=AcceleratorConfig)
    fsdp_config: FSDP2Config = field(default_factory=FSDP2Config)
    plan_overrides: List[PlanOverride] = field(default_factory=list)
    mixed_precision: MixedPrecisionConfig = field(
        default_factory=MixedPrecisionConfig
    )
    activation_checkpoint: ActivationCheckpointConfig = field(
        default_factory=ActivationCheckpointConfig
    )
    activation_swap: Literal["none", "attention"] = "none"
    compile: CompileConfig = field(default_factory=CompileConfig)

    # data
    dataset: Optional[DatasetConfig] = None
    dataloader: Optional[DataLoaderConfig] = None
    packed_sequence: Optional[Any] = None

    checkpoint: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    magi: Optional[Any] = None
    peft: Optional[Any] = None

    def __post_init__(self) -> None:
        """Validate combinations that span multiple config sections."""
        if self.compile.enabled and self.accelerator.pp_size > 1:
            raise ValueError("compile is not supported together with pipeline parallelism")
        reduce_dtype = self.fsdp_config.mix_precision.reduce_dtype
        if self.optimizer.fp32_main_params and reduce_dtype != "float32":
            raise ValueError(
                "optimizer.fp32_main_params=true requires "
                "fsdp_config.mix_precision.reduce_dtype='float32'; "
                f"got {reduce_dtype!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the resolved trainer configuration for logging."""
        return {
            config_field.name: _serialize_config_value(
                getattr(self, config_field.name)
            )
            for config_field in fields(self)
        }


def save_configs(config: TrainerConfig, output_dir: str) -> None:
    """Accept trainer config persistence requests without writing files.

    Args:
        config: Resolved trainer configuration.
        output_dir: Intended configuration output directory.
    """
    del config, output_dir
