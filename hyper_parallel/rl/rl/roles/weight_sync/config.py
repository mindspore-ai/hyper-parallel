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
SUPPORTED_WEIGHT_SYNC_FAMILIES = frozenset(("qwen3",))


@dataclass(frozen=True)
class WeightSyncConfig:
    """Normalized public weight-sync configuration."""

    strategy: str
    bucket_size_bytes: int


def validate_weight_sync_support(
    *,
    deployment: str,
    model_family: str,
    rollout_tp: int,
    strategy: str,
) -> None:
    """Reject strategy/topology combinations without an implemented runtime."""
    if deployment not in ("colocated", "disjoint"):
        raise ValueError(f"Unsupported rollout deployment: {deployment!r}")
    if model_family not in SUPPORTED_WEIGHT_SYNC_FAMILIES:
        raise ValueError(
            "Weight synchronization supports Qwen3 dense only; "
            f"got family={model_family!r}"
        )
    if rollout_tp <= 0:
        raise ValueError("rollout tensor_parallel_size must be positive")
    if strategy not in SUPPORTED_WEIGHT_SYNC_STRATEGIES:
        raise ValueError(f"Unsupported weight-sync strategy: {strategy!r}")


def resolve_weight_sync_config(
    value: Optional[Mapping[str, Any]],
    *,
    deployment: str,
    model_family: str,
    rollout_tp: int,
) -> WeightSyncConfig:
    """Normalize defaults once and validate the public strategy contract."""
    config = dict(value or {})
    unexpected = sorted(set(config) - {"strategy", "bucket_size_mb"})
    if unexpected:
        raise ValueError(
            "Unsupported rollout.vllm.weight_sync field(s): "
            + ", ".join(unexpected)
        )
    strategy = str(config.get("strategy", "full_gather"))
    bucket_size_mb = int(config.get("bucket_size_mb", 128))
    if bucket_size_mb <= 0:
        raise ValueError("rollout.vllm.weight_sync.bucket_size_mb must be positive")
    validate_weight_sync_support(
        deployment=deployment,
        model_family=model_family,
        rollout_tp=int(rollout_tp),
        strategy=strategy,
    )
    return WeightSyncConfig(
        strategy=strategy,
        bucket_size_bytes=bucket_size_mb * 2**20,
    )


__all__ = [
    "WeightSyncConfig",
    "resolve_weight_sync_config",
    "validate_weight_sync_support",
]
