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
"""Typed contracts for selectable operator implementations."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

DeviceType = Literal["cuda", "cpu", "npu"]
Fn = TypeVar("Fn", bound=Callable[..., object])


@dataclass(frozen=True, kw_only=True)
class OpSpec(Generic[Fn]):
    """Identify one callable contract within a YAML-configurable family."""

    config_key: str
    variant: str = "standard"

    def __post_init__(self) -> None:
        if not isinstance(self.config_key, str) or not self.config_key.strip():
            raise ValueError("Operator config_key must be a nonempty string")
        if not isinstance(self.variant, str) or not self.variant.strip():
            raise ValueError("Operator variant must be a nonempty string")
