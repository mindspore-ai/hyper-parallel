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
"""Prepared storage and metadata retained through one native compute submission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from hyper_parallel.core.multicore.modules.routed_experts.readiness import ExpertReadiness


@dataclass(frozen=True)
class ExpertState:
    """Retain forward X/Z; the owner rebinds matrix pointers for backward."""

    x: torch.Tensor
    preact: torch.Tensor


@dataclass
class LinearWork:
    """Keep one GEMM's operands, output, metadata and fresh publication cells alive."""

    owner: object
    x: torch.Tensor
    weights: tuple[torch.Tensor, ...]
    weight_table: torch.Tensor | None
    result: torch.Tensor
    tasks: torch.Tensor | None
    config: torch.Tensor | None
    schedule_workspace: torch.Tensor
    initialized: Any
    transposed: bool
    cores: int
    ready: ExpertReadiness | None
    submitted: bool = False


@dataclass
class GradientWork:
    """Keep one complete expert-gradient reduction and its independent FP32 output."""

    owner: object
    x: torch.Tensor
    dy: torch.Tensor
    result: torch.Tensor
    tasks: torch.Tensor | None
    config: torch.Tensor | None
    initialized: Any
    cores: int
    submitted: bool = False


@dataclass
class ForwardWork:
    """Own F/replay storage; only state.x/state.preact survive as saved activations."""

    owner: object
    inputx: torch.Tensor
    z: torch.Tensor
    activation: torch.Tensor
    output: torch.Tensor
    ready: ExpertReadiness | None
    state: ExpertState
    up: LinearWork
    down: LinearWork
    submitted: bool = False


@dataclass
class BackwardWork:
    """Own B storage; activation aliases dz and da aliases the eventual dx buffer.

    The two FP32 gradients have independent storage because asynchronous owner
    return may still read the first gradient when the second is produced.
    """

    owner: object
    dv: torch.Tensor
    state: ExpertState
    da: torch.Tensor
    dz: torch.Tensor
    activation: torch.Tensor
    output: torch.Tensor
    ready: ExpertReadiness | None
    down_gradient: torch.Tensor
    gate_gradient: torch.Tensor
    da_work: LinearWork
    dx_work: LinearWork
    down_work: GradientWork
    gate_work: GradientWork
    submitted: bool = False

    @property
    def dx(self) -> torch.Tensor:
        """Return the already allocated input-gradient view."""
        return self.output
