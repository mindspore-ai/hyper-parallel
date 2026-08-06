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
"""Environment registry used by the default agent rollout path."""

from typing import Callable

from rl.contracts import PromptRecord
from rl.agentic.base import Environment

EnvironmentBuilder = Callable[[PromptRecord], Environment]


class EnvironmentRegistry:
    """Map stable environment names to episode builders."""

    def __init__(self) -> None:
        """Initialize an empty environment registry."""
        self._builders: dict[str, EnvironmentBuilder] = {}

    def register(self, name: str) -> Callable[[EnvironmentBuilder], EnvironmentBuilder]:
        """Return a decorator that registers an environment builder."""
        key = name.strip().lower()

        def decorator(builder: EnvironmentBuilder) -> EnvironmentBuilder:
            """Register the decorated builder under the normalized name."""
            if key in self._builders:
                raise ValueError(f"Environment is already registered: {key}")
            self._builders[key] = builder
            return builder

        return decorator

    def build(self, name: str, prompt: PromptRecord) -> Environment:
        """Build a named environment for one prompt."""
        key = name.strip().lower()
        try:
            builder = self._builders[key]
        except KeyError as error:
            raise ValueError(
                f"Unknown environment '{name}'; available={sorted(self._builders)}"
            ) from error
        return builder(prompt)

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered environment names in deterministic order."""
        return tuple(sorted(self._builders))


ENVIRONMENTS = EnvironmentRegistry()
