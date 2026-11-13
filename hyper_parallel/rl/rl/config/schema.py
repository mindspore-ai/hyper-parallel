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
"""RL configuration schema."""
import math
from dataclasses import dataclass, field as dataclass_field, fields
from typing import Any, Mapping, Optional
from rl.workers.engine.model import ModelRegistration, resolve_model



_HCCL_MIN_PORT = 1024


_HCCL_MAX_PORT = 65520


_EXPECTED_TOP_LEVEL = frozenset(
    (
        "model",
        "data",
        "rollout",
        "agentic",
        "algorithm",
        "evaluation",
        "train",
        "logging",
        "reward_model",
        "distillation",
        "consistency",
    )
)


def required_mapping(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return one required mapping-valued configuration section."""
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration section '{name}' must be a mapping")
    return value


def optional_mapping(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return one optional mapping-valued configuration section."""
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration section '{name}' must be a mapping")
    return value


def uses_colocated_vllm(config: Mapping[str, Any]) -> bool:
    """Return whether rollout shares each trainer rank's NPU."""
    rollout = config.get("rollout", {})
    if not isinstance(rollout, Mapping) or rollout.get("engine") != "vllm":
        return False
    vllm = rollout.get("vllm", {})
    return isinstance(vllm, Mapping) and vllm.get("deployment", "disjoint") == "colocated"


def _positive_integer(section: Mapping[str, Any], field: str, prefix: str) -> int:
    """Read one strictly positive, non-Boolean integer configuration value."""
    value = section.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{prefix}.{field} must be a positive integer")
    return value


def _trainer_topology(accelerator: Mapping[str, Any]) -> dict[str, int]:
    """Return the validated Trainer topology used by Hyper-RL orchestration."""
    topology = {
        "dp_replicate": int(accelerator.get("dp_replicate", 1)),
        "dp_shard": int(accelerator.get("dp_shard", 0)),
        "tp": int(accelerator.get("tp", 1)),
        "cp": int(accelerator.get("cp", 1)),
        "pp": int(accelerator.get("pp", 1)),
    }
    non_positive = {name: size for name, size in topology.items() if size <= 0}
    if non_positive:
        raise ValueError(
            "Trainer parallel sizes must be positive integers, "
            f"got {non_positive}"
        )
    unsupported = {}
    for name, size in topology.items():
        if name == "dp_replicate" and size != 1:
            unsupported[name] = size
        elif name == "tp" and size not in (1, 2, 4):
            unsupported[name] = size
        elif name in ("cp", "pp") and size != 1:
            unsupported[name] = size
    if unsupported:
        raise ValueError(
            "Hyper-RL Trainer currently supports dp_replicate=1, TP1/TP2/TP4, "
            f"and CP=PP=1; invalid topology={unsupported}"
        )
    return topology


def _trainer_world_size(accelerator: Mapping[str, Any]) -> int:
    """Return the physical process count for one supported Trainer topology."""
    topology = _trainer_topology(accelerator)
    return math.prod(topology.values())


def _trainer_data_parallel_size(accelerator: Mapping[str, Any]) -> int:
    """Return the logical Trainer data-parallel degree excluding model axes."""
    topology = _trainer_topology(accelerator)
    return topology["dp_replicate"] * topology["dp_shard"]


def router_replay_mode(config: Mapping[str, Any]) -> str:
    """Read the single RL-owned replay switch, rejecting unsupported modes."""
    replay = optional_mapping(required_mapping(config, "train"), "router_replay")
    if set(replay) - {"mode"}:
        raise ValueError("train.router_replay only supports the mode field")
    mode = replay.get("mode", "disabled")
    if mode not in ("disabled", "R3"):
        raise ValueError("train.router_replay.mode must be disabled or R3")
    return mode


def build_model_registration(config: Mapping[str, Any]) -> ModelRegistration:
    """Resolve the configured model shared by training and rollout."""
    return resolve_model(required_mapping(config, "model"))


def _normalize_dtype_name(name: Any) -> Optional[str]:
    """Normalize a configured mixed-precision dtype for the master FSDP2 API."""
    normalized = None if name is None else str(name).strip().lower()
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp32": "float32",
        "float32": "float32",
        "fp16": "float16",
        "float16": "float16",
    }
    canonical_name = aliases.get(normalized)
    if normalized is not None and canonical_name is None:
        raise ValueError(f"Unsupported mixed-precision dtype: {name!r}")
    return canonical_name


def _activation_checkpoint_mode(accelerator_config: Mapping[str, Any]) -> str:
    """Normalize the activation-checkpoint mode."""
    configured_value = accelerator_config.get("activation_checkpoint", "off")
    if isinstance(configured_value, bool):
        mode = "full" if configured_value else "off"
    else:
        mode = str(configured_value).lower()
    if mode not in ("off", "full", "selective"):
        raise ValueError(
            "train.accelerator.activation_checkpoint must be off, full, selective, "
            f"or a Boolean, got {configured_value!r}"
        )
    return mode


def _optimizer_fp32_main_params(optimizer_config: Mapping[str, Any]) -> bool:
    """Select the existing main-parameter optimizer without accepting truthy strings."""
    value = optimizer_config.get("fp32_main_params", False)
    if not isinstance(value, bool):
        raise ValueError("train.optimizer.fp32_main_params must be a boolean")
    return value


def _require_async(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _async_integer(value: Any, minimum: int = 1) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


@dataclass(frozen=True)
class AsyncTrainingConfig:
    """Explicit fixed-topology capacities; timeouts are seconds."""
    groups_per_dp_rank: int = 2
    max_inflight_groups: int = 4
    max_buffer_groups: int = 16
    max_group_bytes: int = 64 * 1024 * 1024
    max_buffer_bytes: int = 1024 * 1024 * 1024
    publish_every_updates: int = 2
    max_policy_lag_updates: int = 4
    operation_timeout: float = 900.0
    max_candidate_retries: int = 1
    audit_dir: Optional[str] = None

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "AsyncTrainingConfig":
        """Reject unknown settings, booleans, invalid paths and unreserved capacity."""
        _require_async(isinstance(config, Mapping), "async_training must be a mapping")
        _require_async(not set(config) - {item.name for item in fields(cls)}, "Unknown async_training fields")
        result = cls(**config)
        for name, value in vars(result).items():
            if name == "audit_dir":
                valid = value is None or isinstance(value, str) and bool(value.strip())
            elif name == "operation_timeout":
                valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                         and math.isfinite(value) and value > 0)
            else:
                valid = _async_integer(value, 0 if name == "max_candidate_retries" else 1)
            _require_async(valid, f"Invalid async_training.{name}")
        _require_async(result.max_inflight_groups <= result.max_buffer_groups,
                       "In-flight capacity exceeds total capacity")
        _require_async(result.max_buffer_bytes >= result.max_buffer_groups * result.max_group_bytes,
                 "Reserve max_group_bytes for every admitted group")
        return result


@dataclass(frozen=True)
class RolloutCorrectionConfig:
    """Explicit opt-in to conditional-action correction at a frozen Actor anchor."""
    mode: str = "off"
    importance_sampling: str = "token"
    max_importance_weight: float = 2.0
    rejection: str = "disabled"
    max_abs_log_ratio: Optional[float] = None
    max_refill_rounds: int = 3

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "RolloutCorrectionConfig":
        """Validate controls independently from sampling capacity and transport."""
        _require_async(isinstance(config, Mapping), "off_policy must be a mapping")
        _require_async(not set(config) - {item.name for item in fields(cls)}, "Unknown off_policy fields")
        result = cls(**config)
        _require_async(result.mode in ("off", "metrics_only", "decoupled"), "Invalid off_policy mode")
        _require_async(result.importance_sampling == "token", "Only token importance sampling is supported")
        cap = result.max_importance_weight
        _require_async(isinstance(cap, (int, float)) and not isinstance(cap, bool)
                       and math.isfinite(cap) and 1 <= cap <= math.exp(20),
                       "max_importance_weight must be in [1, exp(20)]")
        _require_async(result.rejection in ("disabled", "group"), "Invalid off_policy rejection")
        _require_async(_async_integer(result.max_refill_rounds, 0), "max_refill_rounds must be nonnegative")
        threshold = result.max_abs_log_ratio
        if threshold is not None:
            _require_async(isinstance(threshold, (int, float)) and not isinstance(threshold, bool)
                           and math.isfinite(threshold) and threshold > 0, "max_abs_log_ratio must be positive")
        _require_async(result.rejection != "group" or result.mode == "decoupled" and threshold is not None,
                       "Group rejection requires decoupled mode and max_abs_log_ratio")
        return result


@dataclass(frozen=True)
class ColocateAsyncConfig:
    """Bounded complete-group prefetch for alternating training and generation."""
    groups_per_dp_rank: int = 2
    warmup_batches: int = 1
    max_group_bytes: int = 64 * 1024 * 1024
    max_buffer_bytes: int = 1024 * 1024 * 1024
    max_policy_lag_updates: int = 4
    operation_timeout: float = 900.0
    max_candidate_retries: int = 1
    audit_dir: Optional[str] = None
    off_policy: RolloutCorrectionConfig = dataclass_field(default_factory=RolloutCorrectionConfig)

    def sampling_config(self, dp_size: int) -> AsyncTrainingConfig:
        """Adapt to the existing producer without exposing a second capacity policy."""
        _require_async(_async_integer(dp_size), "Colocated sampling requires a positive DP size")
        capacity = (self.warmup_batches + 1) * dp_size * self.groups_per_dp_rank
        values = {key: value for key, value in vars(self).items() if key not in ("warmup_batches", "off_policy")}
        values.update(max_inflight_groups=capacity, max_buffer_groups=capacity, publish_every_updates=1)
        return AsyncTrainingConfig.from_mapping(values)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ColocateAsyncConfig":
        """Validate explicit controls before allocating models or launching a producer."""
        _require_async(isinstance(config, Mapping), "colocate_async must be a mapping")
        _require_async(not set(config) - {item.name for item in fields(cls)}, "Unknown colocate_async fields")
        values = dict(config)
        values["off_policy"] = RolloutCorrectionConfig.from_mapping(values.get("off_policy", {}))
        result = cls(**values)
        _require_async(_async_integer(result.warmup_batches, 0), "warmup_batches must be a nonnegative integer")
        _require_async(_async_integer(result.groups_per_dp_rank), "groups_per_dp_rank must be a positive integer")
        result.sampling_config(1)
        return result
