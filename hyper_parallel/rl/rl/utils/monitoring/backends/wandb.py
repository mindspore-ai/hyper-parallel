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
"""Optional Weights & Biases monitoring backend."""

import os
from pathlib import Path
from typing import Any, Mapping, Optional

from rl.utils.monitoring.backends.base import SampleTables
from rl.utils.monitoring.config import sanitize_config

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

_SAMPLE_COLUMNS = (
    "step",
    "rank",
    "prompt",
    "response",
    "ground_truth",
    "extracted_answer",
    "reward",
)


class WandbBackend:
    """Log rank-zero metrics and bounded sample tables to W&B."""

    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        resolved_config: Mapping[str, Any],
        mode: str,
        entity: Optional[str],
        directory: str,
        module: Optional[Any] = None,
    ) -> None:
        """Initialize one W&B run using a sanitized resolved configuration."""
        self._wandb = module if module is not None else _wandb
        if self._wandb is None:
            raise ValueError(
                "W&B backend is enabled but the 'wandb' package is not installed"
            )
        effective_mode = self._resolve_mode(mode)
        storage = Path(directory)
        storage.mkdir(parents=True, exist_ok=True)
        self._run = self._wandb.init(
            project=project_name,
            entity=entity,
            name=experiment_name,
            config=sanitize_config(resolved_config),
            mode=effective_mode,
            dir=str(storage),
        )

    @staticmethod
    def _resolve_mode(mode: str) -> str:
        if mode != "auto":
            return mode
        return "online" if os.environ.get("WANDB_API_KEY") else "offline"

    def log(
        self,
        metrics: Mapping[str, float],
        step: int,
        sample_tables: SampleTables,
    ) -> None:
        """Log scalars and bounded sample tables to the active W&B run."""
        payload: dict[str, Any] = dict(metrics)
        for table_name, samples in sample_tables.items():
            if not samples:
                continue
            rows = [
                [sample.get(column) for column in _SAMPLE_COLUMNS]
                for sample in samples
            ]
            payload[table_name] = self._wandb.Table(
                columns=list(_SAMPLE_COLUMNS),
                data=rows,
            )
        self._run.log(payload, step=step)

    def finish(self) -> None:
        """Finish the W&B run at most once."""
        if self._run is None:
            return
        self._run.finish()
        self._run = None
