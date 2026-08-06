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
"""Rank-zero fan-out across configured Hyper-RL monitoring backends."""

from typing import Any, Mapping, Optional, Sequence

from rl.utils.monitoring.backends import (
    ConsoleBackend,
    TrackingBackend,
    WandbBackend,
)


class TrainingTracker:
    """Own rank-zero tracking backends without coupling Trainer to vendors."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        backends: Sequence[str],
        project_name: str,
        experiment_name: str,
        resolved_config: Mapping[str, Any],
        wandb_mode: str = "auto",
        wandb_entity: Optional[str] = None,
        wandb_directory: str = "outputs/wandb",
        wandb_module: Optional[Any] = None,
    ) -> None:
        """Initialize configured tracking backends on rank zero only."""
        normalized = tuple(str(backend).lower() for backend in backends)
        unsupported = set(normalized) - {"console", "wandb"}
        if unsupported:
            raise ValueError(f"Unsupported logging backends: {sorted(unsupported)}")
        if wandb_mode not in {"auto", "online", "offline", "disabled"}:
            raise ValueError(f"Unsupported W&B mode: {wandb_mode}")
        self._rank = rank
        self._backends: list[TrackingBackend] = []
        if rank != 0:
            return
        if "console" in normalized:
            self._backends.append(ConsoleBackend(world_size, normalized))
        if "wandb" in normalized and wandb_mode != "disabled":
            self._backends.append(
                WandbBackend(
                    project_name=project_name,
                    experiment_name=experiment_name,
                    resolved_config=resolved_config,
                    mode=wandb_mode,
                    entity=wandb_entity,
                    directory=wandb_directory,
                    module=wandb_module,
                )
            )

    def log(
        self,
        metrics: Mapping[str, float],
        step: int,
        samples: Optional[Sequence[Mapping[str, Any]]] = None,
        sample_tables: Optional[
            Mapping[str, Sequence[Mapping[str, Any]]]
        ] = None,
    ) -> None:
        """Fan out scalar metrics and optional bounded sample tables."""
        if self._rank != 0:
            return
        tables = dict(sample_tables or {})
        if samples:
            tables.setdefault("rollout/samples", samples)
        for backend in self._backends:
            backend.log(metrics, step, tables)

    def finish(self) -> None:
        """Finish every initialized external backend exactly once."""
        for backend in self._backends:
            backend.finish()
        self._backends.clear()
