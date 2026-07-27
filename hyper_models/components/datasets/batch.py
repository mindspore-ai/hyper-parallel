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
"""Runtime batch contract between DataLoader and model execution."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PreparedBatch:
    """Separate model arguments, loss inputs, and auxiliary batch metadata."""

    model_inputs: dict[str, Any]
    loss_inputs: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def loss_count_inputs(self) -> dict[str, Any]:
        """Return fields used by token-count and loss aggregation logic."""
        count_inputs = dict(self.model_inputs)
        count_inputs.update(self.loss_inputs)
        return count_inputs


__all__ = ["PreparedBatch"]
