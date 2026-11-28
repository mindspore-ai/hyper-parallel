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
"""Data structures shared by Dry-run's normal-training data path."""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PreparedDryRunBatch:
    """One normal-training batch and its CPU-derived token counts."""

    model_inputs: Mapping[str, Any]
    loss_inputs: Mapping[str, Any]
    token_counts: Mapping[str, int]


__all__ = ["PreparedDryRunBatch"]
