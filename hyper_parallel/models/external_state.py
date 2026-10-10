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
"""Model-independent lifecycle context for externally managed parameters."""
# pylint: disable=forbidden-backend-import,unsupported-binary-operation

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ExternalBuildContext:
    """Information available after source-layout planning and before FSDP."""

    mesh_context: Any
    source_shard_info: Any
    init_device: torch.device | None
    model_init_dtype: torch.dtype | None
    validate_placement: bool


@dataclass(frozen=True)
class ExternalLoadContext:
    """Information needed to initialize one rank's external parameters."""

    load_base_model: bool
    pretrained_path: str | None
    weights_mapping: Any


@dataclass(frozen=True)
class ExternalMaterializationResult:
    """Exact target and source key sets consumed by external loading."""

    target_fqns: frozenset[str]
    source_keys: frozenset[str]


@dataclass(frozen=True)
class GradientPreparation:
    """Combined gradient norm and clipping result."""

    global_norm: torch.Tensor
    clip_coefficient: torch.Tensor
    post_clip_norm: torch.Tensor


@dataclass(frozen=True)
class CheckpointRuntime:
    """Checkpoint location, topology and requested save or restore scope."""

    step_dir: str
    global_step: int | None
    mesh_context: Any
    save_optimizer: bool
    save_train_state: bool
    restore_optimizer: bool
    restore_train_state: bool
    persisted_optimizer_keys: frozenset[str] = frozenset()


@dataclass(frozen=True)
class CheckpointRequirements:
    """Dense optimizer keys actually persisted alongside an external sidecar."""

    persisted_optimizer_keys: frozenset[str]


def get_model_external_state(model: nn.Module) -> Any:
    """Return the optional model-owned external state object.

    Args:
        model: Model being built or inspected.
    """
    return getattr(model, "_hp_external_state", None)
