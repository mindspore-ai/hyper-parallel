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
"""Mixed-precision and optimizer configuration sections.

Split from ``auto_models/trainer/config.py`` in stage 7 (05 §15.2.5);
class names, fields and defaults are unchanged.
"""

from dataclasses import dataclass
from typing import Any

from torch.optim import Optimizer  # pylint: disable=forbidden-backend-import

from hyper_parallel.trainer.config.target import Target, _serialize_config_value


@dataclass
class MixedPrecisionConfig:
    """Mixed-precision parameters exposed by the initial YAML schema."""

    enabled: bool = False


@dataclass
class OptimizerConfig:
    """Optimizer target plus Trainer-owned parameter precision policy."""

    target: Target[Optimizer]
    fp32_main_params: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize optimizer options in their compact target YAML shape."""
        config = self.target.to_dict()
        config["fp32_main_params"] = self.fp32_main_params
        return config
