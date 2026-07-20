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
"""Loss component configuration types."""

from dataclasses import dataclass

from hyper_models.config.configurable import Configurable


class Loss(Configurable):
    """Base category for loss components."""

    @dataclass
    class Config(Configurable.Config):
        """Base configuration accepted by the loss slot."""


class CausalLMLoss(Loss):
    """Shifted-token causal language-model loss configuration owner."""

    @dataclass
    class Config(Loss.Config):
        """Causal language-model loss parameters."""

        ignore_index: int = -100

    def __init__(self, config: "CausalLMLoss.Config") -> None:
        """Store the validated causal language-model loss configuration."""
        self.config = config


__all__ = ["Loss", "CausalLMLoss"]
