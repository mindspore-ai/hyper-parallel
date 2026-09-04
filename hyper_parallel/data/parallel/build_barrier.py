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
"""Online Dataset build synchronization barrier.

Moved verbatim from ``components/distributed/infrastructure.py`` in
stage 6 (05 §15.10 step 3). This is a Data-side build synchronization
primitive and must not import the Trainer.
"""

from datetime import timedelta
import logging
from typing import Any

import torch.distributed as dist

logger = logging.getLogger(__name__)

_DATASET_BARRIER_TIMEOUT = timedelta(hours=10)


class OnlineDatasetBarrier:
    """Synchronize long Online mapping builds through a diagnostic Gloo group."""

    def __init__(self, timeout: timedelta = _DATASET_BARRIER_TIMEOUT) -> None:
        """Store the timeout and defer auxiliary group creation."""
        if timeout.total_seconds() <= 0:
            raise ValueError("Online Dataset barrier timeout must be positive")
        self.timeout = timeout
        self._gloo_group: Any = None
        self._gloo_unavailable = False

    def __call__(self) -> None:
        """Wait up to ten hours and identify missing ranks when supported."""
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return

        if self._gloo_unavailable:
            dist.barrier()
            return

        if self._gloo_group is None:
            try:
                self._gloo_group = dist.new_group(
                    backend="gloo",
                    timeout=self.timeout,
                )
            except (RuntimeError, ValueError) as error:
                self._gloo_unavailable = True
                logger.warning(
                    "Online Dataset Gloo group is unavailable; falling back "
                    "to the default process-group barrier: %s",
                    error,
                )
                dist.barrier()
                return

        dist.monitored_barrier(
            group=self._gloo_group,
            timeout=self.timeout,
            wait_all_ranks=True,
        )
