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
"""Learning-rate scheduler component configuration types."""

from dataclasses import dataclass

from hyper_models.config.configurable import Configurable


class LRScheduler(Configurable):
    """Base category for learning-rate scheduler components."""

    @dataclass
    class Config(Configurable.Config):
        """Base configuration accepted by the scheduler slot."""


class CosineWithWarmup(LRScheduler):
    """Cosine scheduler with a warmup phase."""

    @dataclass
    class Config(LRScheduler.Config):
        """Cosine-with-warmup scheduler parameters."""

        warmup_ratio: float = 0.1
        min_lr: float = 1e-5
        batch_size_warmup_ratio: float = 0.0

    def __init__(self, config: "CosineWithWarmup.Config") -> None:
        """Store the validated cosine scheduler configuration."""
        self.config = config


__all__ = ["LRScheduler", "CosineWithWarmup"]
