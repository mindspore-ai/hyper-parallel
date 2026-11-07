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
"""Invocation-owned MoonEP plans and asynchronous replica prefetch."""

from __future__ import annotations

from collections import deque
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable
import weakref

import torch
import torch.distributed as dist

from hyper_parallel.core.activation_memory.recompute_state import get_recompute_state
from hyper_parallel.distributed.expert_parallel.moonep_dedup import MoonEPTokenDedup
from hyper_parallel.distributed.expert_parallel.moonep_planner import MoonEPPlan


class MoonEPTransport:
    """Keep replica traffic independent of token traffic on the EP communicator."""

    def __init__(self, group: Any, device: torch.device) -> None:
        """Create one pair of replica communicators and a packing stream per EP group."""
        self.group = group
        ranks = dist.get_process_group_ranks(group)
        self.weight_group = dist.new_group(ranks, use_local_synchronization=True)
        self.gradient_group = dist.new_group(ranks, use_local_synchronization=True)
        self.device_module = None if device.type == "cpu" else torch.get_device_module(device)
        self.stream = None if self.device_module is None else self.device_module.Stream(device=device)
        self.gradient_stream = None if self.device_module is None else self.device_module.Stream(device=device)
        self.trace = []

    def stream_context(self, *, gradients: bool = False) -> AbstractContextManager:
        """Order weight packing after the producer without blocking subsequent compute."""
        stream = self.gradient_stream if gradients else self.stream
        if stream is None:
            return nullcontext()
        stream.wait_stream(self.device_module.current_stream())
        return self.device_module.stream(stream)

    def prefetch_previous(self, current: "MoonEPExecution") -> None:
        """Prefetch up to depth earlier live calls when weights are not FSDP-managed.

        Args:
            current: Invocation entering backward and determining the prefetch depth.
        """
        if current.fsdp or not current.prefetch_depth:
            return
        live = [reference for reference in self.trace if reference() is not None and not reference().complete]
        self.trace = live
        position = next((index for index, reference in enumerate(live) if reference() is current), None)
        if position is None:
            return
        for reference in reversed(live[max(0, position - current.prefetch_depth):position]):
            execution = reference()
            if execution is not None and not execution.fsdp:
                execution.prefetch()


_TRANSPORTS = {}


def get_transport(group: Any, device: torch.device) -> MoonEPTransport:
    """Share streams and communicators across expert layers in one EP group.

    Args:
        group: Process group owning the expert partitions.
        device: Device holding expert parameters and token payloads.
    """
    key = (group, device)
    if key not in _TRANSPORTS:
        _TRANSPORTS[key] = MoonEPTransport(group, device)
    return _TRANSPORTS[key]


@dataclass(eq=False)
class MoonEPExecution:
    """One forward's immutable route and temporary backward weight transfer."""

    plan: MoonEPPlan
    transport: MoonEPTransport
    dedup: MoonEPTokenDedup | None = None
    prefetch_depth: int = 1
    fsdp: bool = False
    prepare_weights: Callable | None = None
    incoming: Any = None
    complete: bool = False

    def prefetch(self) -> None:
        """Launch at most one outstanding weight transfer for this invocation."""
        if self.plan.weight_exchange and self.incoming is None:
            with torch.no_grad(), self.transport.stream_context():
                self.incoming = self.prepare_weights()


def checkpoint_execution(owner: Any, create: Callable[[], MoonEPExecution]) -> MoonEPExecution:
    """Reuse each invocation's plan in the same order during Hyper recomputation.

    Args:
        owner: Expert module identifying its invocation-local checkpoint tape.
        create: Build a plan for an original forward invocation.
    """
    state = get_recompute_state()
    tape = None if state is None else state.get_resource(owner, deque)
    if state is not None and state.is_recomputing:
        if not tape:
            raise RuntimeError("MoonEP recomputation has no matching forward invocation")
        return tape.popleft()
    execution = create()
    if tape is not None:
        tape.append(execution)
    if torch.is_grad_enabled():
        trace = execution.transport.trace
        trace[:] = [reference for reference in trace if reference() is not None and not reference().complete]
        trace.append(weakref.ref(execution))
    return execution
