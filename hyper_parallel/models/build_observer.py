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
"""Internal observer contract and context for native model-build facts."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Protocol


class ModelBuildObserver(Protocol):
    """Consume native build facts without influencing build decisions."""

    def mark_not_required(self, reason: str) -> None:
        """Record why the native build produced no distributed plan."""

    def observe_plan(self, plan: Any, model: Any, mesh: Any) -> None:
        """Observe the planner result before it is applied."""

    def observe_plan_application(
        self,
        plan: Any,
        source_shard_info: Any,
        *,
        validate_mode: bool,
    ) -> None:
        """Observe how the plan was applied to the live model."""

    def record_replacements(self, **facts: Any) -> None:
        """Observe replacement declarations and applications."""


@dataclass
class ModelBuildObservation:
    """Internal build-only observation request."""

    observer: ModelBuildObserver
    artifact_dir: Optional[str] = None
    publish: bool = True
    observation_only: bool = False
    artifact_layout: Optional[Any] = None


_ACTIVE_OBSERVATION: ContextVar[Optional[ModelBuildObservation]] = ContextVar(
    "hyper_parallel_model_build_observation",
    default=None,
)


def current_model_build_observation() -> Optional[ModelBuildObservation]:
    """Return the active internal observation request, if any."""
    return _ACTIVE_OBSERVATION.get()


@contextmanager
def use_model_build_observation(observation: ModelBuildObservation) -> Iterator[None]:
    """Scope one internal build observation request to the current context."""
    token = _ACTIVE_OBSERVATION.set(observation)
    try:
        yield
    finally:
        _ACTIVE_OBSERVATION.reset(token)


__all__ = [
    "ModelBuildObservation",
    "ModelBuildObserver",
    "current_model_build_observation",
    "use_model_build_observation",
]
