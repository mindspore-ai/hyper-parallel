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
"""Differentiable owner-row staging for CPU-resident Engram tables."""
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.distributed.expert_parallel.collectives import ep_all_to_all
from hyper_parallel.models.deepseek_v41.adapter.distributed.host_sparse_grad import stage_check


class HostRowsFn(torch.autograd.Function):
    """Fetch CPU rows and accumulate their backward values as sparse chunks."""

    @staticmethod
    def forward(ctx: Any, weight: torch.Tensor, local_ids: torch.Tensor, table: Any) -> torch.Tensor:
        """Gather touched CPU rows and retain their local IDs for backward.

        Args:
            ctx: Autograd context for the row lookup.
            weight: CPU owner-local table parameter.
            local_ids: Owner-local row IDs.
            table: Owner-local Host Engram table.
        """
        ctx.table = table
        ctx.save_for_backward(local_ids)
        return weight.index_select(0, local_ids)

    @staticmethod
    def backward(ctx: Any, grad_rows: torch.Tensor) -> tuple[None, None, None]:
        """Append touched row gradients without constructing a dense weight grad.

        Args:
            ctx: Autograd context for the row lookup.
            grad_rows: Gradient values for requested rows.
        """
        (local_ids,) = ctx.saved_tensors
        ctx.table.append_rows(local_ids, grad_rows.float().contiguous())
        return None, None, None


def host_lookup(table: Any, global_ids: torch.Tensor) -> torch.Tensor:
    """Deduplicate requests, route by EP owner, and restore input order.

    Args:
        table: Owner-local Host Engram table.
        global_ids: Global Engram row IDs to look up.
    """
    if table.rows_per_owner is None or table.global_row_start is None:
        raise RuntimeError("Host Engram table is not bound")
    if table.weight.device.type != "cpu" or table.weight.dtype != torch.float32:
        raise RuntimeError("Host Engram weight must remain CPU FP32")
    flat_ids = global_ids.reshape(-1).long()
    unique_ids, inverse = torch.unique(flat_ids, sorted=True, return_inverse=True)
    table.lookup_requests += flat_ids.numel()
    table.lookup_unique_ids += unique_ids.numel()
    invalid_ids = bool(unique_ids.numel() and
                       (unique_ids.min() < 0 or unique_ids.max() >= table.logical_rows))
    if stage_check(invalid_ids, table.mesh_context):
        raise IndexError("Engram hash requested a row outside its logical table")

    if table.ep_size == 1:
        local_ids = unique_ids.to("cpu")
        owner_values = HostRowsFn.apply(table.weight, local_ids, table)
        table.host_to_device_bytes += owner_values.numel() * owner_values.element_size()
        unique_values = owner_values.to(global_ids.device)
    else:
        owner = torch.div(unique_ids, table.rows_per_owner, rounding_mode="floor")
        send_counts_tensor = torch.bincount(owner, minlength=table.ep_size).to(torch.long)
        count_parts = [torch.empty_like(send_counts_tensor) for _ in range(table.ep_size)]
        dist.all_gather(count_parts, send_counts_tensor, group=table.ep_group)
        send_counts = send_counts_tensor.cpu().tolist()
        recv_counts = torch.stack(count_parts)[:, table.ep_rank].cpu().tolist()
        # Sorted global IDs already group requests by owner.
        owned_global_ids = ep_all_to_all(unique_ids, send_counts, recv_counts, table.ep_group)
        owned_local_ids = (owned_global_ids - table.global_row_start).to("cpu")
        invalid_owner = bool(owned_local_ids.numel() and (
            owned_local_ids.min() < 0 or owned_local_ids.max() >= table.rows_per_owner))
        if stage_check(invalid_owner, table.mesh_context):
            raise IndexError("Engram EP routing delivered a row to the wrong owner")
        owner_values = HostRowsFn.apply(table.weight, owned_local_ids, table)
        table.forward_a2a_bytes += unique_ids.numel() * unique_ids.element_size()
        table.forward_a2a_bytes += owner_values.numel() * owner_values.element_size()
        table.host_to_device_bytes += owner_values.numel() * owner_values.element_size()
        staged_values = owner_values.to(global_ids.device)
        unique_values = ep_all_to_all(staged_values, recv_counts, send_counts, table.ep_group)

    result = unique_values.index_select(0, inverse)
    return result.view(*global_ids.shape, table.width).to(table.lookup_compute_dtype)
