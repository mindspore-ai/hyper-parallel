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
"""YAML-configurable optimizer implementations."""

from typing import Optional

import torch
import torch.nn as nn


class AdamW(torch.optim.AdamW):
    """AdamW optimizer with Hyper model parameter grouping."""

    def __init__(
        self,
        *,
        model: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        foreach: Optional[bool] = None,
    ) -> None:
        """Initialize AdamW from a runtime model.

        Args:
            model: Runtime model injected by the trainer.
            lr: Learning rate.
            weight_decay: Weight decay for decay parameter groups.
            betas: AdamW coefficient pair.
            eps: AdamW numerical-stability term.
            foreach: Whether to use the foreach implementation.
        """
        param_groups = _build_param_groups(model, weight_decay)
        super().__init__(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            foreach=True if foreach is None else foreach,
        )


def _build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Build decay and no-decay groups across all local model parts."""
    decay_params = []
    no_decay_params = []
    seen_ids = set()

    model_parts = getattr(model, "parts", None)
    parts = model_parts if model_parts is not None else [model]

    for part in parts:
        for name, param in part.named_parameters():
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
    """Return whether a parameter should be excluded from weight decay."""
    no_decay_patterns = ("bias", "norm", "ln_")
    return any(pattern in name.lower() for pattern in no_decay_patterns)


__all__ = ["AdamW"]
