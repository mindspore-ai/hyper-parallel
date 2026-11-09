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

from dataclasses import dataclass, field


@dataclass
class SamplingConfig:
    """Enable a recorder at optimizer-step intervals."""

    enabled: bool = False
    sampler_steps: int = 100

    def __post_init__(self) -> None:
        if (not isinstance(self.sampler_steps, int) or isinstance(self.sampler_steps, bool)) or self.sampler_steps < 1:
            raise ValueError("sampler_steps must be a positive integer")


@dataclass
class GradientMonitorConfig(SamplingConfig):
    """Select parameter subtrees for local post-reduction L2 norms."""

    modules: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(set(self.modules)) != len(self.modules) or any(
            not isinstance(name, str) or not name or name.strip(".") != name for name in self.modules
        ):
            raise ValueError("local_grad_norm.modules requires unique nonempty module paths")


@dataclass
class ExpertMonitorConfig(SamplingConfig):
    """Collect routing assignments without altering expert-bias updates."""

    record_distribution: bool = False


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
    mfu: bool = False
    wandb: MonitorWandbConfig = field(default_factory=MonitorWandbConfig)
    local_loss: SamplingConfig = field(default_factory=SamplingConfig)
    local_grad_norm: GradientMonitorConfig = field(default_factory=GradientMonitorConfig)
    moe: ExpertMonitorConfig = field(default_factory=ExpertMonitorConfig)
    optimizer: OptimizerMonitorConfig = field(default_factory=OptimizerMonitorConfig)
    datasets: DatasetMonitorConfig = field(default_factory=DatasetMonitorConfig)

    def __post_init__(self) -> None:
        allowed = {"console", "jsonl", "tensorboard", "wandb"}
        if set(self.sinks) - allowed or len(self.sinks) != len(set(self.sinks)):
            raise ValueError("monitor.sinks must contain unique supported destinations")
        if not self.output_dir:
            raise ValueError("monitor.output_dir cannot be empty")
