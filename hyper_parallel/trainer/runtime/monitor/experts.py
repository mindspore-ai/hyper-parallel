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
"""Fixed-layout source assignment recorder for explicit MoE dispatch boundaries."""

from typing import Any, Callable

import torch
import torch.distributed as dist


class ExpertLoadRecorder:
    """Register all local layers once; collective participation never depends on routing."""

    def __init__(self, max_elements: int = 65536) -> None:
        """Initialize a stable layer registry and empty per-step accumulator."""
        self.specs = {}
        self.counts = {}
        self.max_elements = max_elements
        self.active = False

    def register(self, name: str, experts: int, device: Any) -> None:
        """Register a full global-expert vector before any step or collective."""
        if name in self.specs or experts <= 0:
            raise ValueError("Expert registration requires a unique name and positive size")
        if sum(size + 2 for size, _ in self.specs.values()) + experts + 2 > self.max_elements:
            raise ValueError("Registered expert summaries exceed the fixed recorder capacity")
        self.specs[name] = (experts, device)

    def begin_step(self, active: bool) -> None:
        """Allocate only bounded counter vectors on sampled steps."""
        self.active = active
        self.counts = (
            {
                name: torch.zeros(size + 2, device=device, dtype=torch.float32)
                for name, (size, device) in sorted(self.specs.items())
            }
            if active
            else {}
        )

    def record(self, name: str, counts: torch.Tensor) -> None:
        """Add real-forward assignments without retaining the borrowed router tensor."""
        if not self.active or name not in self.counts:
            return
        if counts.shape != self.counts[name][:-2].shape:
            raise ValueError(f"Expert summary shape changed for {name}")
        counts = getattr(counts, "_local_tensor", counts).detach()
        self.counts[name][:-2].add_(counts)
        self.counts[name][-2].add_(counts.sum())
        self.counts[name][-1].add_(1)

    def record_routes(self, name: str, indices: torch.Tensor, attempted: int) -> None:
        """Count accepted source-side slots when a dispatcher has no expert histogram."""
        if not self.active or name not in self.counts:
            return
        indices = indices.detach().reshape(-1)
        # Expand a scalar weight instead of allocating one value per assignment.
        ones = torch.ones((), dtype=self.counts[name].dtype, device=indices.device).expand_as(indices)
        self.counts[name][:-2].scatter_add_(0, indices, ones)
        self.counts[name][-2].add_(attempted)
        self.counts[name][-1].add_(1)

    def finalize(self, groups: list, submit: Callable, record_distribution: bool) -> None:
        """Pack all layers once, sum independent token domains, then release counters."""
        if not self.active or not self.counts:
            self.active = False
            return
        names = sorted(self.counts)
        lengths = [self.counts[name].numel() for name in names]
        packed = torch.cat([self.counts[name] for name in names])
        metadata = {"modules": names, "lengths": lengths, "physical_slots": "unavailable"}
        if record_distribution:
            submit("expert_layers_local", packed, {**metadata, "scope": "rank"})
        for group in groups:
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(packed, group=group)
        submit("expert_layers", packed, {**metadata, "scope": "dp_cp_and_sequence_tp"})
        self.counts.clear()
        self.active = False
