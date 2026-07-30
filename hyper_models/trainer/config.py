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

import inspect
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, Generic, Literal, Optional, TypeVar

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedTokenizerBase

from hyper_models.components.checkpoint.config import CheckpointingConfig


@dataclass
class TrainingConfig:
    """Training-loop parameters exposed by the initial YAML schema."""

    max_steps: int = 100
    num_train_epochs: int = 1
    global_batch_size: int = 8
    micro_batch_size: int = 1
    backend: Literal["nccl", "hccl", "gloo"] = "nccl"
    max_grad_norm: float = 1.0
    init_device: Literal["meta", "cpu", "cuda", "npu"] = "meta"
    loss_aggregation: Literal["token_weighted", "rank_average"] = "token_weighted"
    seed: Optional[int] = None
    enable_full_determinism: bool = False


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


_T = TypeVar("_T")


def _serialize_config_value(value: Any) -> Any:
    """Convert one target argument to a plain serializable value."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {
            key: _serialize_config_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_serialize_config_value(item) for item in value]
    return value


class Target(Generic[_T]):
    """Configuration for one callable whose invocation is delayed until runtime."""

    def __init__(
        self,
        _target_: Callable[..., _T],
        *,
        target_path: str,
        **kwargs: Any,
    ) -> None:
        """Store the resolved callable, its source path, and configured arguments."""
        if not callable(_target_):
            raise TypeError("_target_ must be callable")
        if not isinstance(target_path, str) or not target_path.strip():
            raise ValueError("target_path must be a non-empty string")

        self._target_ = _target_
        self._target_path = target_path
        self._kwargs = dict(kwargs)

    def __getattr__(self, name: str) -> Any:
        kwargs = object.__getattribute__(self, "_kwargs")
        try:
            return kwargs[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def build(self, **runtime_kwargs: Any) -> _T:
        """Invoke the target with configured and applicable runtime arguments."""
        signature = inspect.signature(self._target_)
        if not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            runtime_kwargs = {
                name: value
                for name, value in runtime_kwargs.items()
                if name in signature.parameters
            }

        kwargs = {**self._kwargs, **runtime_kwargs}
        return self._target_(**kwargs)

    def replace(self, **changes: Any) -> "Target[_T]":
        """Return a new target with selected configured arguments replaced."""
        kwargs = dict(self._kwargs)
        kwargs.update(changes)
        return type(self)(
            self._target_,
            target_path=self._target_path,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this target back to its YAML-compatible form."""
        return {
            "_target_": self._target_path,
            **{
                name: _serialize_config_value(value)
                for name, value in self._kwargs.items()
            },
        }


@dataclass
class TrainerConfig:
    """Resolved component tree; runtime objects are built by the task trainer."""

    model: Target[nn.Module]
    tokenizer: Target[PreTrainedTokenizerBase]
    optimizer: Target[Optimizer]
    lr_scheduler: Target[LRScheduler]
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # parallelism configs
    accelerator: AcceleratorConfig = field(default_factory=AcceleratorConfig)
    mixed_precision: MixedPrecisionConfig = field(
        default_factory=MixedPrecisionConfig
    )
    gradient_checkpointing: GradientCheckpointingConfig = field(
        default_factory=GradientCheckpointingConfig
    )

    dataset: Optional[Target[Any]] = None
    collate_fn: Optional[Target[Any]] = None
    dataloader: Optional[Target[Any]] = None
    packed_sequence: Optional[Any] = None

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
    "DebugConfig",
    "FSDPConfig",
    "GradientCheckpointingConfig",
    "MixedPrecisionConfig",
    "Target",
    "TrainerConfig",
    "TrainingConfig",
    "WandbConfig",
    "save_configs",
    "CheckpointingConfig",
]
