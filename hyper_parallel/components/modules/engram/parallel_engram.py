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
"""Existing Engram CP/EP execution, sequence alignment and sparse-gradient communication."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any
import torch
from torch import nn
import torch.distributed as dist
from hyper_parallel.distributed.expert_parallel.collectives import ep_all_to_all


def local_token_mask(token_mask: torch.Tensor | None, hidden_states: torch.Tensor,
                     *, tp_rank: int, tp_size: int) -> torch.Tensor | None:
    """Match a CP-local token mask to the current TP/SP hidden interval.

    Args:
        token_mask: CP-local mask replicated before the embedding boundary.
        hidden_states: Current TP-local hidden tensor.
        tp_rank: Local TP rank.
        tp_size: Number of TP ranks.

    Returns:
        Mask aligned to the hidden sequence, or ``None``.
    """
    if token_mask is None or token_mask.shape[1] == hidden_states.shape[1]:
        return token_mask
    local_length = hidden_states.shape[1]
    if token_mask.shape[1] != local_length * tp_size:
        raise ValueError("Engram token mask is incompatible with TP/SP hidden")
    return token_mask[:, tp_rank * local_length:(tp_rank + 1) * local_length]


def _all_gather_sequence(value: torch.Tensor, group: Any, group_size: int) -> torch.Tensor:
    """Gather equal-size sequence chunks and concatenate along dimension one."""
    gathered = [torch.empty_like(value) for _ in range(group_size)]
    dist.all_gather(gathered, value.contiguous(), group=group)
    return torch.cat(gathered, dim=1)


def aligned_hash_ids(
    self: Any,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    segment_starts: torch.Tensor | None,
    token_mask: torch.Tensor | None = None,
    *,
    cp_group: Any = None,
    cp_rank: int = 0,
    cp_size: int = 1,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> torch.Tensor:
    """Hash the local sequence plus the left halo required by n-grams.

    Args:
        self: Engram module exposing the n-gram hash mapping.
        hidden_states: Current hidden-state shard.
        input_ids: Token IDs for the local sequence.
        segment_starts: Optional segment boundary flags.
        token_mask: Optional validity mask for token IDs.
        cp_group: Context-parallel process group.
        cp_rank: Local context-parallel rank.
        cp_size: Context-parallel degree.
        tp_rank: Local tensor-parallel rank.
        tp_size: Tensor-parallel degree.

    Returns:
        Hash row IDs aligned with the local hidden-state sequence.
    """
    cp_local_ids = input_ids
    if cp_size > 1:
        input_ids = _all_gather_sequence(input_ids, cp_group, cp_size)
        if segment_starts is not None:
            segment_starts = _all_gather_sequence(segment_starts, cp_group, cp_size)
        if token_mask is not None:
            token_mask = _all_gather_sequence(token_mask, cp_group, cp_size)
    local_sequence = hidden_states.shape[1]
    cp_sequence = cp_local_ids.shape[1]
    if cp_sequence == local_sequence:
        sequence_parallel_parts = 1
        local_tp_rank = 0
    elif cp_sequence % local_sequence == 0 and cp_sequence // local_sequence == tp_size:
        sequence_parallel_parts = tp_size
        local_tp_rank = tp_rank
    else:
        raise ValueError(
            "Engram input_ids length is incompatible with the local hidden sequence: "
            f"ids={cp_sequence}, hidden={local_sequence}, tp_size={tp_size}"
        )
    del sequence_parallel_parts
    start = cp_rank * cp_sequence + local_tp_rank * local_sequence
    end = start + local_sequence
    context_start = max(0, start - self.hash_mapping.max_ngram_size + 1)
    ids_window = input_ids[:, context_start:end]
    starts_window = None if segment_starts is None else segment_starts[:, context_start:end]
    mask_window = None if token_mask is None else token_mask[:, context_start:end]
    hashes = self.hash_mapping(ids_window, starts_window, mask_window)
    target_offset = start - context_start
    return hashes[:, target_offset:target_offset + local_sequence]


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
        transfer_values = owner_values.to(table.lookup_compute_dtype)
        table.host_to_device_bytes += transfer_values.numel() * transfer_values.element_size()
        unique_values = transfer_values.to(global_ids.device)
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
        transfer_values = owner_values.to(table.lookup_compute_dtype)
        table.forward_a2a_bytes += unique_ids.numel() * unique_ids.element_size()
        table.forward_a2a_bytes += transfer_values.numel() * transfer_values.element_size()
        table.host_to_device_bytes += transfer_values.numel() * transfer_values.element_size()
        staged_values = transfer_values.to(global_ids.device)
        unique_values = ep_all_to_all(staged_values, recv_counts, send_counts, table.ep_group)

    result = unique_values.index_select(0, inverse)
    return result.view(*global_ids.shape, table.width).to(table.lookup_compute_dtype)


def _communication_device(mesh) -> torch.device:
    """Place control and row staging tensors on the collective backend device."""
    device_type = getattr(mesh, "device_type", "cpu")
    return torch.device(device_type)


def gather_sparse_along_axis(coo_cpu: torch.Tensor, axis_mesh: Any, width: int) -> torch.Tensor:
    """Gather variable coordinate-format rows and coalesce them without a dense table grad.

    Args:
        coo_cpu: Owner-local coordinate-format sparse gradient on CPU.
        axis_mesh: Mesh axis used to synchronize sparse rows.
        width: Number of values in each table row.
    """
    if axis_mesh is None or axis_mesh.size() == 1:
        return coo_cpu
    group = axis_mesh.get_group()
    device = _communication_device(axis_mesh)
    coo_cpu = coo_cpu.coalesce()
    nnz = coo_cpu._nnz()  # pylint: disable=protected-access
    local_count = torch.tensor([nnz], dtype=torch.long, device=device)
    count_parts = [torch.empty_like(local_count) for _ in range(axis_mesh.size())]
    dist.all_gather(count_parts, local_count, group=group)
    counts = [int(part.item()) for part in count_parts]
    max_count = max(counts)
    if max_count == 0:
        return coo_cpu
    ids = torch.zeros(max_count, dtype=torch.long, device=device)
    values = torch.zeros((max_count, width), dtype=torch.float32, device=device)
    if nnz:
        ids[:nnz] = coo_cpu.indices()[0].to(device)
        values[:nnz] = coo_cpu.values().to(device)
    id_parts = [torch.empty_like(ids) for _ in counts]
    value_parts = [torch.empty_like(values) for _ in counts]
    dist.all_gather(id_parts, ids, group=group)
    dist.all_gather(value_parts, values, group=group)
    gathered_ids = torch.cat([part[:count].cpu() for part, count in zip(id_parts, counts)])
    gathered_values = torch.cat([part[:count].cpu() for part, count in zip(value_parts, counts)])
    return torch.sparse_coo_tensor(
        gathered_ids.unsqueeze(0), gathered_values, coo_cpu.shape,
    ).coalesce()


def sparse_sum_same_ep(coo_cpu: torch.Tensor, mesh_context: Any, width: int) -> torch.Tensor:
    """Sum row contributions over every existing same-owner replica axis.

    Args:
        coo_cpu: Owner-local coordinate-format sparse gradient on CPU.
        mesh_context: Parallel mesh and loss-domain configuration.
        width: Number of values in each table row.
    """
    if mesh_context is None or mesh_context.device_mesh is None:
        return coo_cpu
    moe = mesh_context.fsdp_moe_mesh
    if moe is None:
        for name in ("dp", "cp", "tp"):
            if name in (mesh_context.device_mesh.mesh_dim_names or ()):
                coo_cpu = gather_sparse_along_axis(coo_cpu, mesh_context.device_mesh[name], width)
    else:
        for name in ("edp_shard", "edp_replicate"):
            if name in (moe.mesh_dim_names or ()):
                coo_cpu = gather_sparse_along_axis(coo_cpu, moe[name], width)
    return coo_cpu


def stage_reduce_scalar(value: torch.Tensor, mesh_context: Any,
                        op: Any = dist.ReduceOp.SUM) -> torch.Tensor:
    """Reduce a scalar across the current PP stage in fixed mesh-axis order.

    Args:
        value: Scalar value to reduce.
        mesh_context: Parallel mesh and loss-domain configuration.
        op: Distributed reduction operation.
    """
    if mesh_context is None or mesh_context.device_mesh is None:
        return value
    for name in mesh_context.device_mesh.mesh_dim_names or ():
        axis = mesh_context.device_mesh[name]
        if axis.size() > 1:
            dist.all_reduce(value, op=op, group=axis.get_group())
    return value


def stage_check(local_bad: bool, mesh_context: Any) -> bool:
    """Align validation failure across every rank in this PP stage.

    Args:
        local_bad: Whether this rank detected an invalid state.
        mesh_context: Parallel mesh and loss-domain configuration.
    """
    mesh = mesh_context.device_mesh if mesh_context is not None else None
    flag = torch.tensor(int(local_bad), device=_communication_device(mesh))
    stage_reduce_scalar(flag, mesh_context, dist.ReduceOp.MAX)
    return bool(flag.item())


def host_sparse_lookup(hash_ids: torch.Tensor, embedding: Any, *, ep_group: Any,
                   ep_rank: int, ep_size: int, padded_num_embeddings: int) -> torch.Tensor:
    """Use the already-bound Host row route without applying the device-table route twice.

    Args:
        hash_ids: Global row IDs generated for local tokens.
        embedding: Bound Host Engram table module.
        ep_group: Expert-parallel process group.
        ep_rank: Local expert-parallel rank.
        ep_size: Expert-parallel degree.
        padded_num_embeddings: Physical padded table-row count.

    Returns:
        Lookup values in the table compute dtype.
    """
    if (embedding.ep_rank != ep_rank or embedding.ep_size != ep_size or embedding.ep_group is not ep_group
            or embedding.physical_rows != padded_num_embeddings):
        raise ValueError("Host lookup must match the table's planned row partition")
    return embedding(hash_ids)


class ParallelEngram:
    """Keep existing parallel execution separate from ordinary Engram fusion."""

    # The concrete Engram module owns these parameters and local fusion math.
    embed: nn.Embedding
    padded_num_embeddings: int
    _fuse: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    @staticmethod
    def _sparse_lookup(
        hash_ids: torch.Tensor,
        embedding: nn.Embedding,
        *,
        ep_group: Any,
        ep_rank: int,
        ep_size: int,
        padded_num_embeddings: int,
    ) -> torch.Tensor:
        """Route requested global rows to their EP owners and restore order."""
        if ep_size == 1:
            return embedding(hash_ids)
        if padded_num_embeddings % ep_size:
            raise ValueError(
                f"Engram padded rows {padded_num_embeddings} are not divisible by ep_size={ep_size}"
            )
        num_local_rows = padded_num_embeddings // ep_size
        row_start = ep_rank * num_local_rows
        flat_ids = hash_ids.reshape(-1)
        owner = torch.div(flat_ids, num_local_rows, rounding_mode="floor")
        if owner.numel() and int(owner.max()) >= ep_size:
            raise IndexError("Engram hash requested a row outside the padded table")
        local_counts = torch.bincount(owner, minlength=ep_size).to(torch.long)
        gathered_counts = [torch.empty_like(local_counts) for _ in range(ep_size)]
        dist.all_gather(gathered_counts, local_counts, group=ep_group)
        all_counts = torch.stack(gathered_counts)
        send_counts = local_counts.cpu().tolist()
        recv_counts = all_counts[:, ep_rank].cpu().tolist()

        # Ascend ArgSort executes integer inputs on AiCPU. The owner range is
        # bounded by ep_size, so float32 preserves it exactly and stays on AiCore.
        sort_indices = torch.argsort(owner.float(), stable=True)
        sorted_ids = flat_ids.index_select(0, sort_indices)
        received_ids = ep_all_to_all(sorted_ids, send_counts, recv_counts, ep_group)
        local_ids = received_ids - row_start
        if local_ids.numel() and (int(local_ids.min()) < 0 or int(local_ids.max()) >= num_local_rows):
            raise IndexError("Engram EP routing delivered a row to the wrong owner")
        # Call the child module instead of reading ``embedding.weight``
        # directly. This preserves the FSDP pre/post-forward lifecycle when
        # the EP table is wrapped independently from the dense Engram gates.
        values = embedding(local_ids)
        returned = ep_all_to_all(values, recv_counts, send_counts, ep_group)
        restored = torch.zeros_like(returned)
        restored.scatter_add_(
            0,
            sort_indices.unsqueeze(1).expand(-1, returned.shape[1]),
            returned,
        )
        return restored.view(*hash_ids.shape, embedding.embedding_dim)

    def _aligned_hash_ids(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor, segment_starts: torch.Tensor | None,
        token_mask: torch.Tensor | None = None, *, cp_group: Any = None, cp_rank: int = 0, cp_size: int = 1,
        tp_rank: int = 0, tp_size: int = 1,
    ) -> torch.Tensor:
        """Delegate existing sequence alignment while preserving the source method."""
        return aligned_hash_ids(self, hidden_states, input_ids, segment_starts, token_mask,
                                cp_group=cp_group, cp_rank=cp_rank, cp_size=cp_size,
                                tp_rank=tp_rank, tp_size=tp_size)

    def parallel_forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        segment_starts: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        *,
        ep_group: Any,
        ep_rank: int,
        ep_size: int,
        cp_group: Any = None,
        cp_rank: int = 0,
        cp_size: int = 1,
        tp_rank: int = 0,
        tp_size: int = 1,
    ) -> torch.Tensor:
        """Run CP/TP sequence alignment and sparse EP lookup.

        Args:
            hidden_states: Current decoder hidden rows.
            input_ids: CP-local token IDs for hash construction.
            segment_starts: Optional packed sequence boundaries.
            token_mask: CP-local mask selecting Engram updates.
            ep_group: Expert-parallel process group.
            ep_rank: Local expert-parallel rank.
            ep_size: Expert-parallel degree.
            cp_group: Context-parallel process group.
            cp_rank: Local context-parallel rank.
            cp_size: Context-parallel degree.
            tp_rank: Local tensor-parallel rank.
            tp_size: Tensor-parallel degree.
        """
        hash_ids = self._aligned_hash_ids(
            hidden_states,
            input_ids,
            segment_starts,
            token_mask,
            cp_group=cp_group,
            cp_rank=cp_rank,
            cp_size=cp_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        embeddings = self._sparse_lookup(
            hash_ids,
            self.embed,
            ep_group=ep_group,
            ep_rank=ep_rank,
            ep_size=ep_size,
            padded_num_embeddings=self.padded_num_embeddings,
        )
        fused = self._fuse(hidden_states, embeddings)
        token_mask = local_token_mask(token_mask, hidden_states, tp_rank=tp_rank, tp_size=tp_size)
        if token_mask is None:
            return fused
        return torch.where(token_mask.to(torch.bool).unsqueeze(-1).unsqueeze(-1), fused, hidden_states)


class ParallelHostTable:
    """Own existing EP partition binding without placing it in local row storage."""

    # The concrete host table owns storage; this base only binds its partition.
    physical_rows: int
    width: int
    weight: nn.Parameter

    def _init_parallel_state(self) -> None:
        self.rows_per_owner: int | None = None
        self.global_row_start: int | None = None
        self.ep_rank = 0
        self.ep_size = 1
        self.ep_group = None
        self.mesh_context = None

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
