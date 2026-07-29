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
"""YAML-configurable optimizer components."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn


class OptimizerConfig(ABC):
    """Configuration contract for constructing runtime optimizers."""

    @abstractmethod
    def build(
        self,
        model: nn.Module,
        *,
        device_mesh: Optional[Any] = None,
    ) -> list[torch.optim.Optimizer]: ...


@dataclass(kw_only=True, slots=True)
class AdamW(OptimizerConfig):
    """AdamW parameters and runtime construction."""

    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    foreach: Optional[bool] = None
    max_grad_norm: float = 1.0

    def build(
        self,
        model: nn.Module,
        *,
        device_mesh: Optional[Any] = None,
    ) -> list[torch.optim.Optimizer]:
        """Build one AdamW optimizer for each model part."""
        optimizers = []
        for part in getattr(model, "parts", [model]):
            param_groups = _build_param_groups(part, self.weight_decay)
            optimizers.append(
                torch.optim.AdamW(
                    param_groups,
                    lr=self.lr,
                    betas=self.betas,
                    eps=self.eps,
                    foreach=self.foreach if self.foreach is not None else True,
                )
            )
        return optimizers


def _build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Decay/no_decay parameter grouping.

    Following design doc §9.5.
    """
    decay_params, no_decay_params = [], []
    seen_ids = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_id = id(param)
        if param_id in seen_ids:
            continue
        seen_ids.add(param_id)

        if _is_no_decay(name):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def _is_no_decay(name: str) -> bool:
    """Check if a parameter name should be excluded from weight decay.

    Following design doc §9.5.
    """
    no_decay_patterns = ("bias", "norm", "rmsnorm", "layernorm", "ln_")
    return any(pattern in name.lower() for pattern in no_decay_patterns)



__all__ = ["AdamW", "OptimizerConfig"]