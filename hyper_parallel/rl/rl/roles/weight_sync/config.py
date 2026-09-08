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
"""Canonical weight-sync configuration and supported topology boundaries."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional


SUPPORTED_WEIGHT_SYNC_STRATEGIES = frozenset(("direct_reshard", "full_gather"))
SUPPORTED_WEIGHT_SYNC_FALLBACKS = frozenset(("full_gather", "none"))
SUPPORTED_WEIGHT_SYNC_FAMILIES = frozenset(("qwen3", "qwen3_moe", "deepseek_v3"))


@dataclass(frozen=True)
class WeightSyncConfig:
    """Normalized public weight-sync configuration."""

    strategy: str
    fallback_strategy: str
    bucket_size_bytes: int


def validate_weight_sync_support(
    *,
    deployment: str,
    model_family: str,
    rollout_tp: int,
    strategy: str,
    fallback_strategy: str,
) -> None:
    """Reject strategy/topology combinations without an implemented runtime."""
    if deployment not in ("colocated", "disjoint"):
        raise ValueError(f"Unsupported rollout deployment: {deployment!r}")
    if model_family not in SUPPORTED_WEIGHT_SYNC_FAMILIES:
        raise ValueError(
            "Weight synchronization supports Qwen3, Qwen3-MoE, and DeepSeek-V3 only; "
            f"got family={model_family!r}"
        )
    if rollout_tp <= 0:
        raise ValueError("rollout tensor_parallel_size must be positive")
    if strategy not in SUPPORTED_WEIGHT_SYNC_STRATEGIES:
        raise ValueError(f"Unsupported weight-sync strategy: {strategy!r}")
    if fallback_strategy not in SUPPORTED_WEIGHT_SYNC_FALLBACKS:
        raise ValueError(f"Unsupported weight-sync fallback: {fallback_strategy!r}")
    if strategy == "full_gather" and fallback_strategy != "none":
        raise ValueError(
            "full_gather is already a complete publication strategy and requires "
            "fallback_strategy='none'"
        )
    if model_family == "qwen3":
        return
    if rollout_tp not in (1, 2):
        raise ValueError(
            f"{model_family} weight synchronization currently supports rollout TP1/TP2, "
            f"got TP{rollout_tp}"
        )
    if deployment == "disjoint":
        raise ValueError(
            f"{model_family} disjoint weight synchronization is not implemented; "
            "use colocated rollout until disjoint expert ownership is supported"
        )


def resolve_weight_sync_config(
    value: Optional[Mapping[str, Any]],
    *,
    deployment: str,
    model_family: str,
    rollout_tp: int,
) -> WeightSyncConfig:
    """Normalize defaults once and validate the public strategy contract."""
    config = dict(value or {})
    if "full_gather_implementation" in config:
        raise ValueError(
            "rollout.vllm.weight_sync.full_gather_implementation was removed; "
            "select strategy='full_gather' for the bounded implementation"
        )
    strategy = str(config.get("strategy", "full_gather"))
    fallback_strategy = str(config.get("fallback_strategy", "none"))
    bucket_size_mb = int(config.get("bucket_size_mb", 128))
    if bucket_size_mb <= 0:
        raise ValueError("rollout.vllm.weight_sync.bucket_size_mb must be positive")
    validate_weight_sync_support(
        deployment=deployment,
        model_family=model_family,
        rollout_tp=int(rollout_tp),
        strategy=strategy,
        fallback_strategy=fallback_strategy,
    )
    return WeightSyncConfig(
        strategy=strategy,
        fallback_strategy=fallback_strategy,
        bucket_size_bytes=bucket_size_mb * 2**20,
    )


__all__ = [
    "WeightSyncConfig",
    "resolve_weight_sync_config",
    "validate_weight_sync_support",
]
