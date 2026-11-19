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
"""Accumulate logging scalars while optionally deferring accelerator-to-host reads."""

from collections import defaultdict
from typing import Any, Mapping, Optional

import torch


class StepMetricAccumulator:
    """Preserve per-microbatch Python summation order with optional grouped host reads.

    Only detached scalar values are retained. Forward loss, gradients, collectives
    and optimizer calculations are outside this logging helper's ownership.
    """

    def __init__(self, *, defer: bool = False) -> None:
        """Select immediate scalar reads by default or postpone them until finalize."""
        if not isinstance(defer, bool):
            raise ValueError("defer must be a boolean")
        self.defer = defer
        self.total_loss = 0.0
        self.total_named = defaultdict(int)
        self._pending = []

    def add(self, loss: torch.Tensor, named: Mapping[str, torch.Tensor]) -> None:
        """Record one microbatch's original loss and named logging scalars.

        Args:
            loss: Scalar loss tensor from this microbatch.
            named: Named scalar tensors to include in step logging.
        """
        for key, value in ((None, loss), *named.items()):
            if self.defer:
                if not isinstance(value, torch.Tensor) or value.numel() != 1:
                    raise ValueError("Deferred step metrics require scalar tensors")
                # Metrics may alias reusable model buffers; preserve their value at this microbatch.
                self._pending.append((key, value.detach().reshape(()).clone()))
            else:
                self._accumulate(key, value.item())

    def _accumulate(self, key: Optional[str], value: Any) -> None:
        if key is None:
            self.total_loss += value
        else:
            self.total_named[key] += value

    def finalize(self) -> tuple[float, dict[str, float]]:
        """Read each device/dtype group once, then sum values in their original order.

        Grouping by dtype also preserves large integer and FP64 log values.
        Repeated calls retain totals without consuming any scalar twice.
        """
        if self._pending:
            by_storage = defaultdict(list)
            for index, (_, value) in enumerate(self._pending):
                by_storage[(value.device, value.dtype)].append((index, value))
            host_values = [None] * len(self._pending)
            for entries in by_storage.values():
                values = torch.stack([value for _, value in entries]).cpu().tolist()
                for (index, _), value in zip(entries, values):
                    host_values[index] = value
            for (key, _), value in zip(self._pending, host_values):
                self._accumulate(key, value)
            self._pending.clear()
        return self.total_loss, self.total_named
