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
"""Dry-run configuration for logical LLM memory analysis."""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

@dataclass
class DryRunConfig:
    """Shape and value-profile configuration for logical LLM memory analysis."""

    enabled: bool = True
    target_device: Literal["npu", "cuda"] = "npu"
    sequence_length: int = 4096
    output_dir: str = "outputs/dry_run"
    device_memory_gib: Optional[float] = None
    value_dependencies: dict[str, Any] = field(default_factory=lambda: {"rules": []})
