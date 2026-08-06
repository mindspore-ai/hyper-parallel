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
"""Generation backend registry."""

from typing import Any, Callable, Mapping

from rl.roles.model import ModelRegistration
from rl.roles.rollout.base import GenerationEngine


class RolloutEngineRegistry:
    """Map stable backend names to generation engine builders."""

    def __init__(self) -> None:
        """Initialize an empty generation engine registry."""
        self._builders: dict[str, Callable[..., GenerationEngine]] = {}

    def register(self, name: str) -> Callable[[Callable[..., GenerationEngine]], Callable[..., GenerationEngine]]:
        """Return a decorator that registers one generation engine builder."""
        key = name.strip().lower()

        def decorator(builder: Callable[..., GenerationEngine]) -> Callable[..., GenerationEngine]:
            """Register the decorated builder under the normalized name."""
            if key in self._builders:
                raise ValueError(f"Rollout engine is already registered: {key}")
            self._builders[key] = builder
            return builder

        return decorator

    def build(
        self,
        name: str,
        config: Mapping[str, Any],
        model: ModelRegistration,
        actor: Any = None,
    ) -> GenerationEngine:
        """Build a named generation backend."""
        key = name.strip().lower()
        try:
            builder = self._builders[key]
        except KeyError as error:
            raise ValueError(
                f"Unknown rollout engine '{name}'; available={sorted(self._builders)}"
            ) from error
        return builder(config=config, model=model, actor=actor)

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered rollout engine names in deterministic order."""
        return tuple(sorted(self._builders))


ROLLOUT_ENGINES = RolloutEngineRegistry()


def build_rollout_engine(
    config: Mapping[str, Any],
    model: ModelRegistration,
    actor: Any = None,
) -> GenerationEngine:
    """Build the generation backend selected by ``rollout.engine``."""
    name = config.get("engine")
    if not isinstance(name, str) or not name:
        raise ValueError("rollout.engine must be a non-empty string")
    return ROLLOUT_ENGINES.build(name, config, model, actor)
