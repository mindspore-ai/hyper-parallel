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
"""Console monitoring backend."""

import logging
from typing import Mapping

from rl.utils.monitoring.backends.base import SampleTables

logger = logging.getLogger("rl.utils.monitoring")


class ConsoleBackend:
    """Render sorted scalar metrics through the standard logger."""

    def __init__(self, world_size: int, configured_backends: tuple[str, ...]) -> None:
        """Log the initialized rank-zero tracking configuration."""
        logger.info(
            "Tracking initialized: backends=%s, world_size=%d",
            configured_backends,
            world_size,
        )

    def log(
        self,
        metrics: Mapping[str, float],
        step: int,
        sample_tables: SampleTables,
    ) -> None:
        """Render sorted scalar metrics through the standard logger."""
        del sample_tables
        rendered = ", ".join(
            f"{key}={value:.6g}" for key, value in sorted(metrics.items())
        )
        logger.info("step=%d | %s", step, rendered)

    def finish(self) -> None:
        """Console logging owns no external resources."""
