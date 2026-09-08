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
"""Training-loop, debug, wandb and profiling configuration sections.

Split from ``auto_models/trainer/config.py`` in stage 7 (05 §15.2.5);
class names, fields and defaults are unchanged.
"""

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
    """Lightweight per-step profiler settings."""

    enabled: bool = False
    start_step: int = 3
    end_step: int = 4
    trace_dir: str = "./outputs/profiling"
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False
    with_modules: bool = False
    rank: int = 0
