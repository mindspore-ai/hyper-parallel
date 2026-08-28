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
"""Omni dataset selection and composition."""

from collections.abc import Callable
from typing import Any, Literal

from hyper_parallel.auto_models.components.datasets.omni.offline_dataset import build_offline_dataset
from hyper_parallel.auto_models.components.datasets.omni.online_dataset import build_online_dataset

OmniSourceType = Literal["online", "offline"]


def build_omni_dataset(
        source_type: OmniSourceType,
        *,
        data_path: str,
        transform: Callable[[Any], Any] | None = None,
        **dataset_options: Any,
) -> Any:
    """Build an Omni dataset from an online or offline source."""
    if source_type == "online":
        return build_online_dataset(
            data_path=data_path,
            transform=transform,
            **dataset_options,
        )
    if source_type == "offline":
        return build_offline_dataset(
            data_path=data_path,
            transform=transform,
            **dataset_options,
        )
    raise ValueError(f"Unsupported Omni source type: {source_type!r}")
