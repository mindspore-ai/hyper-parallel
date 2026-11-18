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
"""Asynchronous transfer of owned, small tensor summaries on the producer stream."""

from collections import deque
from typing import Callable

import torch


class SummaryTransfer:
    """Never read pinned memory before its completion event or retain source batches."""

    def __init__(self, capacity: int = 32, max_elements: int = 65536) -> None:
        """Bound pending device summaries and the size of each transfer."""
        self.capacity, self.max_elements = capacity, max_elements
        self.pending = deque()
        self.dropped = 0

    def submit(self, values: torch.Tensor, consume: Callable) -> bool:
        """Copy a bounded summary; producer-stream order protects against later writes."""
        if len(self.pending) >= self.capacity or values.numel() > self.max_elements:
            self.dropped += 1
            return False
        source = values.detach().clone(memory_format=torch.contiguous_format)
        if source.device.type == "cpu":
            consume(source.tolist())
            return True
        backend = getattr(torch, source.device.type)
        destination = torch.empty_like(source, device="cpu", pin_memory=True)
        destination.copy_(source, non_blocking=True)
        event = backend.Event()
        event.record(backend.current_stream(source.device))
        # Keep the owned summary alive until the copy finishes, never its input tensor.
        self.pending.append((event, source, destination, consume))
        return True

    def poll(self) -> None:
        """Consume only completed copies without synchronizing the training device."""
        while self.pending and self.pending[0][0].query():
            _, _, destination, consume = self.pending.popleft()
            consume(destination.tolist())

    def close(self) -> None:
        """Wait only at shutdown, then release every transfer-owned buffer."""
        while self.pending:
            event, _, destination, consume = self.pending.popleft()
            event.synchronize()
            consume(destination.tolist())
