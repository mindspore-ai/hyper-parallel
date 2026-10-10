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
"""Owner-local FP32 Engram parameters and one-step sparse gradient storage."""
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class HostEngramTable(nn.Module):
    """Keep one EP owner's rows on CPU and stage only touched rows on device."""

    def __init__(self, *, source_weight: nn.Parameter, logical_rows: int,
                 physical_rows: int, width: int, max_pending_entries: int = 1000000,
                 max_sparse_rows_per_step: int = 1000000) -> None:
        """Validate the full meta weight and configure sparse entry budgets."""
        super().__init__()
        if not 0 < logical_rows <= physical_rows or width <= 0:
            raise ValueError("Invalid Engram table shape")
        if not source_weight.is_meta or source_weight.dtype != torch.float32:
            raise ValueError("Host Engram replacement requires a meta FP32 weight")
        if tuple(source_weight.shape) != (physical_rows, width):
            raise ValueError("Host Engram replacement requires the full physical table")
        if max_pending_entries <= 0 or max_sparse_rows_per_step <= 0:
            raise ValueError("Host Engram sparse budgets must be positive")
        self.weight = source_weight
        self.logical_rows = logical_rows
        self.physical_rows = physical_rows
        self.width = width
        self.max_pending_entries = max_pending_entries
        self.max_sparse_rows_per_step = max_sparse_rows_per_step
        self.rows_per_owner: int | None = None
        self.global_row_start: int | None = None
        self.ep_rank = 0
        self.ep_size = 1
        self.ep_group = None
        self.mesh_context = None
        self.lookup_compute_dtype = torch.float32
        self.pending_chunks: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.pending: torch.Tensor | None = None
        self.pending_error: str | None = None
        self.lookup_requests = 0
        self.lookup_unique_ids = 0
        self.forward_a2a_bytes = 0
        self.host_to_device_bytes = 0
        self.max_synced_nnz = 0
        self.coalesce_seconds = 0.0

    def bind_planned_shard(self, *, ep_rank: int, ep_size: int, ep_group: Any = None,
                           compute_dtype: torch.dtype = torch.float32) -> None:
        """Bind the planner's single EP row shard to its global row interval."""
        if ep_size < 1 or not 0 <= ep_rank < ep_size or self.physical_rows % ep_size:
            raise ValueError("Invalid EP row partition")
        rows = self.physical_rows // ep_size
        if not self.weight.is_meta or self.weight.dtype != torch.float32:
            raise ValueError("Planner must leave a meta FP32 Host table")
        if tuple(self.weight.shape) != (rows, self.width):
            raise ValueError("Planner did not produce one EP row shard")
        if ep_size > 1 and ep_group is None:
            raise ValueError("EP routing requires an explicit process group")
        self.rows_per_owner = rows
        self.global_row_start = ep_rank * rows
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.ep_group = ep_group
        self.lookup_compute_dtype = compute_dtype

    def _apply(self, fn, recurse=True):
        """Keep the CPU parameter outside model-wide device and dtype changes."""
        weight = self._parameters.pop("weight")
        try:
            return super()._apply(fn, recurse=recurse)
        finally:
            self.register_parameter("weight", weight)

    def append_rows(self, local_ids: torch.Tensor, grad_rows: torch.Tensor) -> None:
        """Append owner-local CPU gradients without making a dense table grad.

        Args:
            local_ids: Owner-local row IDs.
            grad_rows: Gradient values for requested rows.
        """
        if self.rows_per_owner is None:
            raise RuntimeError("Host Engram table is not bound")
        if local_ids.device.type != "cpu" or grad_rows.device.type != "cpu":
            raise ValueError("Host Engram pending gradients must reside on CPU")
        if local_ids.dtype != torch.long or grad_rows.dtype != torch.float32:
            raise ValueError("Host Engram pending gradients require int64 IDs and FP32 values")
        if tuple(grad_rows.shape) != (local_ids.numel(), self.width):
            raise ValueError("Host Engram pending gradient shape mismatch")
        if local_ids.numel() and (local_ids.min() < 0 or local_ids.max() >= self.rows_per_owner):
            raise IndexError("Host Engram pending row is outside the owner shard")
        entries = sum(ids.numel() for ids, _ in self.pending_chunks) + local_ids.numel()
        if entries > self.max_pending_entries:
            # Autograd backward may run before other EP ranks have finished their
            # collectives. Report the failure at the aligned optimizer-step gate.
            self.pending_error = "Host Engram pending entry budget exceeded"
            return
        self.pending_chunks.append((local_ids.detach(), grad_rows.detach()))

    def coalesce_pending(self) -> torch.Tensor:
        """Combine microbatches and repeated rows into one sparse CPU tensor."""
        if self.rows_per_owner is None:
            raise RuntimeError("Host Engram table is not bound")
        if self.pending_error is not None:
            raise ValueError(self.pending_error)
        ids = (torch.cat([chunk[0] for chunk in self.pending_chunks]) if self.pending_chunks
               else torch.empty(0, dtype=torch.long))
        values = (torch.cat([chunk[1] for chunk in self.pending_chunks]) if self.pending_chunks
                  else torch.empty((0, self.width), dtype=torch.float32))
        result = torch.sparse_coo_tensor(ids.unsqueeze(0), values,
                                          (self.rows_per_owner, self.width)).coalesce()
        if result._nnz() > self.max_sparse_rows_per_step:
            raise ValueError("Host Engram unique sparse row budget exceeded")
        if not torch.isfinite(result.values()).all():
            raise ValueError("Host Engram gradient contains non-finite values")
        return result

    def install_grad(self) -> None:
        """Expose only the prepared sparse gradient to SparseAdam."""
        if self.pending is None:
            raise RuntimeError("Host Engram sparse gradient has not been prepared")
        self.weight.grad = self.pending if self.pending._nnz() else None

    def clear_step(self) -> None:
        """Release sparse gradients after the optimizer step."""
        self.weight.grad = None
        self.pending = None
        self.pending_chunks.clear()
        self.pending_error = None
        self.lookup_requests = 0
        self.lookup_unique_ids = 0
        self.forward_a2a_bytes = 0
        self.host_to_device_bytes = 0
        self.max_synced_nnz = 0
        self.coalesce_seconds = 0.0

    def reset_parameters(self) -> None:
        """Preserve already materialized CPU rows during model initialization."""

    def forward(self, global_ids: torch.Tensor) -> torch.Tensor:
        """Look up global row IDs through the owner-local CPU table.

        Args:
            global_ids: Global Engram row IDs to look up.
        """
        from hyper_parallel.models.deepseek_v41.adapter.engram.host_lookup import host_lookup  # pylint: disable=C0415

        return host_lookup(self, global_ids)
