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
"""Online Omni dataset interface."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def build_online_dataset(
        *,
        data_path: str,
        transform: Callable[[Any], Any] | None = None,
        **dataset_options: Any,
) -> Any:
    """Reserve the modality-aware Online Omni Dataset implementation."""
    del data_path, transform, dataset_options
    raise NotImplementedError("Online Omni Dataset is not implemented")
