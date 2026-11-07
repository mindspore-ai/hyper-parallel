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
"""Compile one execution's read-only transport and forward tables together."""

from __future__ import annotations

import torch

from hyper_parallel.core.multicore.modules.routed_experts.descriptor import RouteDescriptor, RouteEdge
from hyper_parallel.core.multicore.modules.routed_experts.transport import _load_native
from hyper_parallel.core.multicore.modules.routed_experts.workspace import RingWorkspace


def _edge_table(edges: tuple[RouteEdge, ...]) -> torch.Tensor:
    """Preserve strict integer validation before Python values enter the native ABI."""
    values = [(edge.source, edge.target, edge.expert, edge.source_row, edge.target_row, edge.rows) for edge in edges]
    if (any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63
            for row in values for value in row)
            or any(row[3] + row[5] >= 2**63 or row[4] + row[5] >= 2**63 for row in values)):
        raise ValueError("transport edge fields and exclusive row ends must be nonnegative int64 integers")
    return torch.tensor(values, dtype=torch.int64, device="cpu").reshape(-1, 6)


def compile_forward_metadata(route: RouteDescriptor, workspace: RingWorkspace, anchor: torch.Tensor,
                             config: torch.Tensor, tile_rows: int) -> tuple:
    """Publish all immutable first-forward tables behind one initialization event.

    Args:
        route: Validated local ranges after final expert placement.
        workspace: Owner of the two directional transport caches.
        anchor: Current input supplying the device, without reading its contents.
        config: Immutable CPU forward-kernel tiling bytes.
        tile_rows: Rows per compact compute task.

    Returns:
        Compute tasks, configuration, CPU/device expectations, CPU progress
        template and their shared recorded initialization event. Mutable frame
        counters and current weight pointers are not part of this cache.
    """
    route.validate()
    block_rows = workspace.spec.block_rows
    if (isinstance(block_rows, bool) or not isinstance(block_rows, int) or not 1 <= block_rows <= 1024
            or isinstance(tile_rows, bool) or not isinstance(tile_rows, int) or tile_rows < 1
            or route.ranks != workspace.spec.ranks or anchor.device != workspace.device):
        raise ValueError("forward metadata does not match the transport geometry and device")
    _load_native()
    host, device = torch.ops.hyper_parallel.routed_experts_forward_metadata(
        anchor, _edge_table(route.outgoing), _edge_table(route.incoming), route.active_experts,
        route.boundaries, config, route.ranks, route.source_rows, block_rows, tile_rows)
    send_host, receive_host, ranges, expected, task_expected, compute_expected, progress = host
    (send, receive, reverse_send, reverse_receive, slots, expected_device,
     task_device, tasks, device_config, compute_device) = device
    current = torch.npu.current_stream(workspace.device)
    initialized = torch.npu.Event()
    initialized.record(current)
    rows = tuple(tuple(row) for row in ranges.tolist())
    workspace.metadata[(route, "host_transport", block_rows)] = route, send_host, receive_host, ranges.numpy()
    workspace.metadata[(route, False)] = send, receive, rows, initialized, route
    workspace.metadata[(route, True)] = (reverse_send, reverse_receive,
                                         tuple((row[2], row[3], row[0], row[1]) for row in rows), initialized, route)
    workspace.metadata[(route, "readiness")] = (slots, expected, initialized, expected_device,
                                                 task_expected, task_device)
    return (tasks, device_config if route.boundaries else None, compute_expected,
            compute_device, progress if compute_expected.shape[0] else None, initialized)
