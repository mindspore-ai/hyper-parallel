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
"""Optional Trainer monitoring, independent of the external logging cadence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SamplingConfig:
    """Enable a recorder at optimizer-step intervals."""

    enabled: bool = False
    sampler_steps: int = 100

    def __post_init__(self) -> None:
        if (not isinstance(self.sampler_steps, int) or isinstance(self.sampler_steps, bool)) or self.sampler_steps < 1:
            raise ValueError("sampler_steps must be a positive integer")


@dataclass
class ExpertMonitorConfig(SamplingConfig):
    """Collect routing assignments without altering expert-bias updates."""

    record_distribution: bool = False
    layers: list[int] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.layers is not None and (
            not isinstance(self.layers, list)
            or not self.layers
            or any(not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in self.layers)
            or len(set(self.layers)) != len(self.layers)
        ):
            raise ValueError("monitor.moe.layers must be null or a non-empty list of unique non-negative integers")


@dataclass
class OptimizerMonitorConfig:
    """Optimizer-internal sampling."""

    ns_norm: SamplingConfig = field(default_factory=SamplingConfig)


@dataclass
class DatasetMonitorConfig:
    """CPU input statistics; never copy an accelerator batch to the host."""

    sampler_steps: int = 100
    token_distribution: bool = True
    top_n_tokens: int = 20
    packing_stats: bool = True
    packing_boundary_check: bool = True
    log_first_rendered_template: bool = False
    log_level: Literal["debug", "info", "warn"] | None = None

    def __post_init__(self) -> None:
        if (
            (not isinstance(self.sampler_steps, int) or isinstance(self.sampler_steps, bool))
            or self.sampler_steps < 1
            or (not isinstance(self.top_n_tokens, int) or isinstance(self.top_n_tokens, bool))
            or self.top_n_tokens < 0
        ):
            raise ValueError("datasets requires sampler_steps > 0 and top_n_tokens >= 0")


@dataclass
class MonitorWandbConfig:
    """Optional W&B destination; authentication stays outside YAML."""

    project: str = "hyper-parallel"
    mode: str = "online"

    def __post_init__(self) -> None:
        if self.mode not in ("online", "offline"):
            raise ValueError("monitor.wandb.mode must be online or offline")


@dataclass
class MonitorConfig:
    """Monitoring controls, with no duplicated logging or rank selectors."""

    enabled: bool = True
    sinks: list[str] = field(default_factory=lambda: ["console", "jsonl"])
    output_dir: str = "./outputs/metrics"
    timing: bool = True
    check_fsdp_runtime: bool = False
    mfu: bool = False
    wandb: MonitorWandbConfig = field(default_factory=MonitorWandbConfig)
    local_loss: bool = False
    local_grad_norm: bool = False
    moe: ExpertMonitorConfig = field(default_factory=ExpertMonitorConfig)
    optimizer: OptimizerMonitorConfig = field(default_factory=OptimizerMonitorConfig)
    datasets: DatasetMonitorConfig = field(default_factory=DatasetMonitorConfig)

    def __post_init__(self) -> None:
        allowed = {"console", "jsonl", "tensorboard", "wandb"}
        if set(self.sinks) - allowed or len(self.sinks) != len(set(self.sinks)):
            raise ValueError("monitor.sinks must contain unique supported destinations")
        if not self.output_dir:
            raise ValueError("monitor.output_dir cannot be empty")
