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
"""Typed trainer configuration tree (stage 7, 05 §15.2.5).

The former ``auto_models/trainer/config.py`` monolith is split into
``target`` / ``training`` / ``parallelism`` / ``optimization`` / ``data`` /
``trainer`` by configuration domain; ``manager`` and ``resolver`` moved
from ``auto_models/config`` unchanged. This package re-exports the same
public class names — including the AutoModels-owned ``CompileConfig`` /
``FSDP2Config`` / ``CheckpointingConfig`` — without copying their
definitions.
"""

from hyper_parallel.models.build_options import CompileConfig, FSDP2Config
from hyper_parallel.components.checkpoint.config import CheckpointingConfig

from hyper_parallel.trainer.config.data import (
    DataLoaderConfig,
    DatasetConfig,
    ModelAssetsConfig,
)
from hyper_parallel.trainer.config.optimization import (
    MixedPrecisionConfig,
    OptimizerConfig,
)
from hyper_parallel.trainer.config.parallelism import (
    AcceleratorConfig,
    ActivationCheckpointConfig,
    PlanOverride,
    _import_module_type,
    entries_to_module_replacements,
    entries_to_plan_overrides,
    normalize_distributed_setup_overrides,
)
from hyper_parallel.trainer.config.target import Target
from hyper_parallel.trainer.config.trainer import TrainerConfig, save_configs
from hyper_parallel.trainer.config.training import (
    DebugConfig,
    ProfilingConfig,
    TrainingConfig,
    WandbConfig,
)

__all__ = [
    "AcceleratorConfig",
    "ActivationCheckpointConfig",
    "CompileConfig",
    "DataLoaderConfig",
    "DatasetConfig",
    "DebugConfig",
    "FSDP2Config",
    "MixedPrecisionConfig",
    "OptimizerConfig",
    "ProfilingConfig",
    "Target",
    "TrainerConfig",
    "TrainingConfig",
    "WandbConfig",
    "save_configs",
    "CheckpointingConfig",
]
