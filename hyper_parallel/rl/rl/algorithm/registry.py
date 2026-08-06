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
"""Small explicit registry for algorithm plugins."""

from typing import Any, Callable, Mapping

from rl.algorithm.base import RLAlgorithm

Builder = Callable[[Mapping[str, Any]], RLAlgorithm]


class AlgorithmRegistry:
    """Map stable names to complete public algorithm Recipe builders."""

    def __init__(self) -> None:
        """Initialize an empty algorithm registry."""
        self._builders: dict[str, Builder] = {}

    def register(self, name: str) -> Callable[[Builder], Builder]:
        """Return a decorator that registers one Recipe builder."""
        key = name.strip().lower()

        def decorator(builder: Builder) -> Builder:
            """Register the decorated builder under the normalized name."""
            if key in self._builders:
                raise ValueError(f"Algorithm is already registered: {key}")
            self._builders[key] = builder
            return builder

        return decorator

    def build(self, name: str, config: Mapping[str, Any]) -> RLAlgorithm:
        """Build a named algorithm Recipe from its configuration."""
        key = name.strip().lower()
        try:
            builder = self._builders[key]
        except KeyError as error:
            raise ValueError(
                f"Unknown algorithm '{name}'; available={sorted(self._builders)}"
            ) from error
        return builder(config)

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered algorithm names in deterministic order."""
        return tuple(sorted(self._builders))


ALGORITHMS = AlgorithmRegistry()


def build_algorithm(config: Mapping[str, Any]) -> RLAlgorithm:
    """Build the complete Recipe selected by ``algorithm.name``."""
    name = config.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("algorithm.name must be a non-empty string")
    return ALGORITHMS.build(name, config)
