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
from rl.registry import Registry
from rl.roles.model import ModelRegistration
from rl.roles.rollout.base import GenerationEngine
RolloutEngineBuilder = Callable[..., GenerationEngine]


class RolloutEngineRegistry(Registry[RolloutEngineBuilder]):
    """Registry specialized only by its public error label."""

    def __init__(self) -> None:
        """Create the rollout registry with its public error label."""
        super().__init__("rollout engine")
ROLLOUT_ENGINES = RolloutEngineRegistry()


def build_rollout_engine(
    config: Mapping[str, Any],
    model: ModelRegistration,
) -> GenerationEngine:
    """Build the generation backend selected by ``rollout.engine``."""
    name = config.get("engine")
    if not isinstance(name, str) or not name:
        raise ValueError("rollout.engine must be a non-empty string")
    return ROLLOUT_ENGINES.build(name, config=config, model=model)
