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
"""Loss component interfaces for HyperModels — following design doc §10 + §10.0."""

from dataclasses import dataclass, field
from typing import Optional

import torch.nn as nn

from .loss import CausalLMLoss, Loss
from .masked_ce import MaskedCrossEntropy
from .utils import calculate_loss, calculate_mtp_loss

# FusedLinearCrossEntropy — optional, requires cut_cross_entropy
try:
    from .linear_ce import FusedLinearCrossEntropy  # noqa: F401
except ImportError:
    pass


@dataclass(kw_only=True, slots=True)
class LossConfig(Loss.Config):
    """Loss typed config — following design doc §10.0.

    Supports two consumption paths:
    - _target_ not set: default MaskedCrossEntropy()
    - _target_ set to Loss subclass: .build() instantiates _target_(**kwargs)

    Inherits from :class:`Loss.Config` so it can be used in the typed
    ``TrainerConfig.loss`` slot.
    """
    _target_: Optional[type] = None
    loss_aggregation: str = "token_weighted"
    kwargs: dict = field(default_factory=dict)

    def build(self) -> nn.Module:
        """Build the loss module from config."""
        if self._target_ is None:
            return MaskedCrossEntropy()
        return self._target_(**self.kwargs)


def build_loss_config(factory, **kwargs) -> LossConfig:
    """Normalize _target_ factory + kwargs into LossConfig instance."""
    return LossConfig(_target_=factory, kwargs=kwargs)


__all__ = [
    "Loss", "CausalLMLoss",
    "MaskedCrossEntropy",
    "FusedLinearCrossEntropy",
    "calculate_loss", "calculate_mtp_loss",
    "LossConfig", "build_loss_config",
]