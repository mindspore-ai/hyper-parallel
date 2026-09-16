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
"""Training-loop, debug, profiler, and remote-logging configuration sections."""

from dataclasses import dataclass, field
from typing import Literal, Optional

from hyper_parallel.components.quantization.config import LowPrecisionConfig


@dataclass
class TrainingConfig:
    """Training-loop parameters exposed by the initial YAML schema."""

    train_iters: Optional[int] = None
    train_samples: Optional[int] = None
    eval_iters: int = 0

    global_batch_size: int = 8
    micro_batch_size: int = 1

    backend: Literal["nccl", "hccl", "gloo"] = "nccl"
    max_grad_norm: float = 1.0
    init_device: Literal["meta", "cpu", "cuda", "npu"] = "meta"
    loss_aggregation: Literal["token_weighted", "rank_average"] = "token_weighted"
    seed: Optional[int] = None
    enable_full_determinism: bool = False
    gc_steps: int = 0
    empty_cache_steps: int = 0
    empty_cache_before_backward: bool = False
    eval_steps: int = 0
    eval_epochs: int = 0
    logging_steps: int = 1
    low_precision: LowPrecisionConfig = field(default_factory=LowPrecisionConfig)


@dataclass
class DebugConfig:
    """Debug parameters exposed by the initial YAML schema."""

    check_dataset: Optional[Literal["debug", "info", "warn"]] = None
    check_nan_inf: bool = False


@dataclass
class WandbConfig:
    """WandB remote-logging parameters (03 §4.2.5: read by build_callback_manager)."""

    enabled: bool = False
    project: str = ""
    entity: Optional[str] = None


@dataclass
class ProfilingConfig:
    """Configure bounded PyTorch profiler collection during training.

    HyperParallel groups these options under ``TrainerConfig.profiler``, so
    their names describe only the option itself rather than repeating a
    ``profile_`` prefix.

    Args:
        enabled: Whether profiler collection is enabled.
        start_step: First training step included in the profile.
        stop_step: Last training step included in the profile.
        start_on_init: Whether collection starts at ``on_train_begin`` rather than the first step.
        memory: Whether tensor memory events are collected.
        rank_ids: Explicit ranks that collect a profile. ``None`` or an empty list selects all ranks.
        pipeline_stage_leaders: Whether the first rank of every pipeline stage also collects a profile.
        output_path: Output root. Each rank writes under ``profile/rank_<rank>``.
        level: Ascend collection level, from 0 through 2. ``None`` selects level 0.
        with_stack: Whether Python stack traces are collected.
        data_simplification: Whether torch-npu removes selected auxiliary CANN data after export.
        mstx: Whether torch-npu records explicit step ranges.
    """

    enabled: bool = False
    start_step: int = 1
    stop_step: int = 10
    start_on_init: bool = False
    memory: bool = True
    rank_ids: Optional[list[int]] = None
    pipeline_stage_leaders: bool = False
    output_path: Optional[str] = None
    level: Optional[int] = 1
    with_stack: bool = False
    data_simplification: bool = False
    mstx: bool = False
