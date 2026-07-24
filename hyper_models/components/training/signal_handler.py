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
"""DistributedSignalHandler — SIGTERM distributed coordination.

Following design doc 03_training_loop.md §11.
"""

import logging
import signal

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class DistributedSignalHandler:
    """SIGTERM distributed coordination — any rank receives → all respond."""

    def __init__(self):
        self._signal_received = False
        self._orig_handler = None

    def __enter__(self):
        self._orig_handler = signal.signal(signal.SIGTERM, self._handler)
        return self

    def __exit__(self, *args):
        signal.signal(signal.SIGTERM, self._orig_handler)

    def _handler(self, signum, frame):
        rank = dist.get_rank() if dist.is_initialized() else 0
        logger.warning("Rank %d received SIGTERM", rank)
        self._signal_received = True

    def signals_received(self) -> list[bool]:
        """all_gather: any rank received → all return True.

        NCCL doesn't support CPU tensor collective — move to CUDA device first.
        """
        if not dist.is_initialized():
            return [self._signal_received]

        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        tensor = torch.tensor([int(self._signal_received)], dtype=torch.int32, device=device)
        gathered = [torch.zeros(1, dtype=torch.int32, device=device) for _ in range(dist.get_world_size())]
        torch.distributed.all_gather(gathered, tensor)
        return [bool(t.item()) for t in gathered]