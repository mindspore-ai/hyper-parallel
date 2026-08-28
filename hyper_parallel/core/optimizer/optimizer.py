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

"""Base distributed optimizer and chain optimizer composition."""

from collections import defaultdict
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)

from hyper_parallel.core.dtensor.dtensor import SkipDTensorDispatch
from hyper_parallel.core.optimizer.dtensor_compat import to_local_if_dtensor
from hyper_parallel.core.optimizer.sharding_category import (
    HSDPGroupAssignment,
    build_owner_by_size,
    get_multi_dim_logical_info,
    select_owned_records,
    group_parameters_for_hsdp
)
from hyper_parallel.core.optimizer.utils import empty_accelerator_cache, get_current_device, get_device_count

logger = logging.getLogger(__name__)


class ChainedOptimizer:
    """Composition wrapper that dispatches step/zero_grad to sub-optimizers."""

    def __init__(
            self,
            model: torch.nn.Module,
            optimizers: Dict[str, torch.optim.Optimizer],
            flatten: bool = False
    ) -> None:
        self.optimizers_dict = optimizers
        self.chained_optimizers = list(optimizers.values())
        self.optimizers_keys = list(optimizers.keys())
        self.model = model
        self.flatten = flatten  # not flatten adamw, flatten for multi-optimizer
        self._is_multi_optimizer = flatten

        self._rebind_param_attrs()

    def __iter__(self):
        """Allow iteration over the underlying optimizers."""
        return iter(self.chained_optimizers)

    def _rebind_param_attrs(self) -> None:
        """Restore ``model_name`` and ``is_muon`` on the current parameters."""
        muon_param_ids: set = set()
        adamw_param_ids: set = set()
        self.muon_keys = []
        self.no_muon_keys = []
        for _, opt in self.optimizers_dict.items():
            target = muon_param_ids if getattr(opt, "is_muon", False) else adamw_param_ids
            for group in opt.param_groups:
                for p in group.get("params", []):
                    target.add(id(p))

        for param_name, param in self.model.named_parameters():
            setattr(param, "model_name", param_name)
            if id(param) in muon_param_ids:
                setattr(param, "is_muon", True)
                self.muon_keys.append(param_name)
            elif id(param) in adamw_param_ids:
                setattr(param, "is_muon", False)
                self.no_muon_keys.append(param_name)

    def __str__(self):
        if not self.optimizers_dict:
            return f'{self.__class__.__name__}'

        sections = []
        for name, optimizer in self.optimizers_dict.items():
            sections.append(f'{name}: {repr(optimizer)}')
        return "\n\n".join(sections)

    __repr__ = __str__

    def step(self, closure=None) -> None:
        """Call each sub-optimizer's step in order."""
        for opt in self.chained_optimizers:
            opt.step(closure=closure)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients for all sub-optimizers."""
        for opt in self.chained_optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Access underlying optimizer when only one optimizer included for backward compatibility."""
        if len(self.chained_optimizers) != 1:
            raise ValueError("ChainedOptimizer has more than one optimizer when accessing self.optimizer")
        return self.chained_optimizers[0]

    @property
    def defaults(self) -> Dict[str, Any]:
        """Return the defaults of the first sub-optimizer."""
        return self.chained_optimizers[0].defaults

    def state_dict(self) -> Dict[str, Any]:
        """Return state dicts with DTensor values localized to CPU for serialization.

        In HSDP, optimizer state is only populated on the owner rank for replicated
        parameters. This method first broadcasts state to all replicate-group peers,
        then converts DTensor values to local CPU tensors so ``torch.save`` works.
        """
        # Ensure all ranks have consistent optimizer state before snapshotting
        for opt in self.chained_optimizers:
            if hasattr(opt, "_broadcast_state_fused_for_ckpt"):
                opt._broadcast_state_fused_for_ckpt()  # pylint: disable=protected-access

        merged: Dict[str, Any] = {}
        for name, optimizer in self.optimizers_dict.items():
            sd = get_optimizer_state_dict(
                self.model,
                optimizer,
                options=StateDictOptions(flatten_optimizer_state_dict=self.flatten)
            )
            overlap = set(merged.keys()) & set(sd.keys())
            if overlap:
                raise KeyError(
                    f"Key clash detected while merging state dict for optimizer '{name}': "
                    f"{', '.join(sorted(overlap))}"
                )
            merged.update(sd)

        return merged

    def _get_param_groups(self) -> List[Dict[str, Any]]:
        """Get param_groups aggregated over underlying optimizers."""
        param_groups: List[Dict[str, Any]] = []
        for optimizer in self.chained_optimizers:
            param_groups += optimizer.param_groups
        return param_groups

    def _set_param_groups(self, new_param_groups: List[Dict[str, Any]]) -> None:
        """Set param_groups distributed across underlying optimizers."""
        if not isinstance(new_param_groups, list):
            raise TypeError("new_param_groups should be a list")
        if len(new_param_groups) != len(self.param_groups):
            raise ValueError("The size of new_param_groups must be equal to origin param_groups")

        start = 0
        for optimizer in self.chained_optimizers:
            group_len = len(optimizer.param_groups)
            optimizer.param_groups = new_param_groups[start: start + group_len]
            start += group_len

    param_groups = property(_get_param_groups, _set_param_groups)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state dicts and synchronize steps."""
        for optimizer in self.chained_optimizers:
            # PyTorch may initialize missing slots through a synthetic
            # optimizer step. Distributed parameters must stay on their local
            # storage path, matching a real Trainer optimizer step.
            with SkipDTensorDispatch():
                set_optimizer_state_dict(
                    self.model,
                    optimizer,
                    optim_state_dict=state_dict,
                    options=StateDictOptions(flatten_optimizer_state_dict=self.flatten),
                )

        self._synchronize_steps()

    def _synchronize_steps(self) -> Optional[int]:
        """Synchronize the step of all optimizers.

        TE FusedAdam will not accumulate "step" for empty param groups,
        so we need to align the step across param groups before saving and after loading.
        """
        steps = []
        for optimizer in self.chained_optimizers:
            actual_opt = getattr(optimizer, 'optimizer', optimizer)
            for param_group in actual_opt.param_groups:
                if len(param_group['params']) > 0 and 'step' in param_group:
                    steps.append(param_group['step'])

        unique_steps = list(set(steps))
        if len(unique_steps) > 1:
            raise ValueError(f"steps should <= 1, but got {unique_steps}")

        step = unique_steps[0] if len(unique_steps) == 1 else None

        for optimizer in self.chained_optimizers:
            actual_opt = getattr(optimizer, 'optimizer', optimizer)
            for param_group in actual_opt.param_groups:
                param_group['step'] = step

        return step


class BaseDistributedOptimizer(torch.optim.Optimizer):
    """Base class for distributed optimizers with HSDP-aware topology communication.

    Provides fused hierarchical broadcast for parameters and optimizer states.
    """

    def __init__(
            self,
            params: Any,
            defaults: Dict[str, Any],
            is_muon: bool,
            hsdp_replica_count: Optional[int] = None,
    ) -> None:
        super().__init__(params, defaults)
        self.is_muon = is_muon
        self.hsdp_replica_count = hsdp_replica_count
        self._param_to_broadcast_info: Dict[
            torch.nn.Parameter, Tuple[Tuple[int, ...], Tuple[dist.ProcessGroup, ...]]
        ] = {}

        # Cache: (parent_ranks_tuple, sub_size) -> {sub_idx: sub_pg}
        self._split_sub_pg_cache: Dict[Tuple[Tuple[int, ...], int], Dict[int, dist.ProcessGroup]] = {}

    def _group_dtensor_by_mesh(self):
        """Group DTensor parameters by mesh topology and shard layout."""
        self._hsdp_grouping: Dict[int, Tuple[List, List]] = {}
        for group_key, group in enumerate(self.param_groups):
            no_comm_params, hsdp_groups = group_parameters_for_hsdp(group["params"])
            self._hsdp_grouping[group_key] = (no_comm_params, hsdp_groups)

    def _auto_deduce_replica_count(self) -> Optional[int]:
        """Deduce hsdp_replica_count based on cluster topology.

        - Intra-node PGs: Full dedup (no split), high bandwidth makes broadcast cheap.
        - Inter-node PGs: Split at node boundaries to restrict communication domains
          within a single node, bypassing cross-node bottlenecks.
        """
        devices_per_node = get_device_count()

        dedup_per_dim: Dict[int, int] = {}
        needs_split = False

        # group_key, (no_comm_params, hsdp_groups)
        for _, (_, hsdp_groups) in self._hsdp_grouping.items():
            for hsdp_group in hsdp_groups:
                for dim_idx, pg in enumerate(hsdp_group.replicate_pgs):
                    if pg is None:
                        continue
                    pg_size = dist.get_world_size(pg)
                    if pg_size <= 1:
                        continue

                    if pg_size > devices_per_node:
                        # Inter-node: Find largest divisor safe for node boundary
                        dedup = min(pg_size, devices_per_node)
                        while pg_size % dedup != 0:
                            dedup -= 1
                        needs_split = True
                    else:
                        # origin Inter-node
                        dedup = pg_size

                    # Enforce conservative (smallest) dedup across shared mesh axes
                    if dim_idx not in dedup_per_dim:
                        dedup_per_dim[dim_idx] = dedup
                    else:
                        dedup_per_dim[dim_idx] = min(dedup_per_dim[dim_idx], dedup)

        if not needs_split:
            return None

        sorted_dedups = [dedup_per_dim[k] for k in sorted(dedup_per_dim.keys())]
        return min(sorted_dedups)

    def _split_replicate_groups(self) -> None:
        """Split replicate ProcessGroups into smaller sub-groups based on hsdp_replica_count."""
        if self.hsdp_replica_count is None:
            return

        # Argument validation
        if not isinstance(self.hsdp_replica_count, int):
            raise TypeError(f"Unsupported hsdp_replica_count type: {type(self.hsdp_replica_count).__name__}")
        if self.hsdp_replica_count <= 0:
            raise ValueError(f"hsdp_replica_count must be positive, got {self.hsdp_replica_count}")

        for group_key, (_, hsdp_groups) in self._hsdp_grouping.items():
            for hsdp_group in hsdp_groups:
                new_replicate_pgs: List[dist.ProcessGroup] = []
                for dim_idx, pg in enumerate(hsdp_group.replicate_pgs):
                    if pg is None:
                        new_replicate_pgs.append(pg)
                        continue

                    pg_size = dist.get_world_size(pg)
                    dedup_size = self.hsdp_replica_count

                    if pg_size <= dedup_size:
                        new_replicate_pgs.append(pg)
                        continue

                    if pg_size % dedup_size != 0:
                        raise ValueError(
                            f"hsdp_replica_count {dedup_size} must evenly divide replicate group size {pg_size}"
                        )

                    sub_pg = self._get_or_create_sub_pg(pg, dedup_size)
                    new_replicate_pgs.append(sub_pg)

                    logger.info_rank0(
                        "[HSDP Split] group_key=%s, dim_idx=%s, original_size=%s, sub_size=%s",
                        group_key, dim_idx, pg_size, dedup_size
                    )

                # replace new sub groups of hsdp groups
                hsdp_group.replicate_pgs = tuple(new_replicate_pgs)

    def _get_or_create_sub_pg(
            self,
            parent_pg: dist.ProcessGroup,
            sub_size: int,
    ) -> dist.ProcessGroup:
        """Retrieve or collectively create a synchronized sub-ProcessGroup."""
        local_parent_ranks = tuple(sorted(list(dist.get_process_group_ranks(parent_pg))))
        cache_key = (local_parent_ranks, sub_size)

        # Cache Hit Check
        if cache_key in self._split_sub_pg_cache:
            sub_pg_map = self._split_sub_pg_cache[cache_key]
            for sub_idx, sub_pg in sub_pg_map.items():
                if sub_pg is not None:
                    try:
                        dist.get_rank(group=sub_pg)
                        return sub_pg
                    except RuntimeError:
                        continue
            raise RuntimeError(f"Current rank not found in cached sub-groups for parent_pg={local_parent_ranks}")

        global_rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Global Rendezvous via GPU all_gather_into_tensor (NCCL/HCCL fast-path).
        # Far more scalable than all_gather_object for large world_size.
        device = get_current_device()
        num_parent_ranks = len(local_parent_ranks)
        local_tensor = torch.tensor(local_parent_ranks, dtype=torch.long, device=device)
        gathered_tensor = torch.empty((world_size, num_parent_ranks), dtype=torch.long, device=device)
        dist.all_gather_into_tensor(gathered_tensor.view(-1), local_tensor)

        # Deduplicate: each row is one rank's parent_ranks; convert to
        # sorted set of tuples for deterministic iteration order.
        gathered_cpu_list = gathered_tensor.cpu().tolist()
        unique_all_parent_ranks = sorted(set(
            tuple(row) for row in gathered_cpu_list
        ))

        sub_pg_map: Dict[int, dist.ProcessGroup] = {}
        my_sub_pg: Optional[dist.ProcessGroup] = None

        # Synchronized collective creation loop
        for parent_ranks in unique_all_parent_ranks:
            num_sub_groups = len(parent_ranks) // sub_size

            for sub_idx in range(num_sub_groups):
                sub_ranks = parent_ranks[sub_idx * sub_size: (sub_idx + 1) * sub_size]
                sub_pg = dist.new_group(sub_ranks)

                # Map results only if this parent ranks list matches the current rank's context
                if parent_ranks == local_parent_ranks:
                    if global_rank in sub_ranks:
                        my_sub_pg = sub_pg
                        sub_pg_map[sub_idx] = sub_pg
                    else:
                        sub_pg_map[sub_idx] = None

        self._split_sub_pg_cache[cache_key] = sub_pg_map

        if my_sub_pg is None:
            raise RuntimeError(
                f"Rank {global_rank} not found in any sub-group of parent_pg "
                f"with ranks {local_parent_ranks} and sub_size={sub_size}"
            )

        return my_sub_pg

    def _build_hsdp_batch(
            self,
            max_batch_numel: Optional[int] = None,
    ) -> None:
        """Split HSDP groups into memory-capped batches for compute-broadcast overlap."""
        if max_batch_numel is None:
            broadcast_max_bytes = getattr(
                self, "replicate_broadcast_max_bytes", 512 * 1024 * 1024,
            )
            hsdp_size = self.hsdp_replica_count if self.hsdp_replica_count is not None else 1
            max_batch_numel = broadcast_max_bytes * hsdp_size  # if hsdp_size > 1 else float('inf')

        self._hsdp_batches: Dict[int, List[Dict]] = {}

        for group_key, (_, hsdp_groups) in self._hsdp_grouping.items():
            # Sort groups by total numel descending — large groups first.
            sorted_groups = sorted(
                hsdp_groups,
                key=lambda g: -sum(r.param.numel() for r in g.records),
            )

            batch_groups: List[Dict] = []

            for hsdp_group in sorted_groups:
                # Sort records within group by numel descending.
                sorted_records = sorted(
                    hsdp_group.records,
                    key=lambda r: -r.param.numel(),
                )

                sub_batches: List[List[Dict]] = []
                current_batch: List[Dict] = []
                current_numel = 0

                for record in sorted_records:
                    p_numel = record.param.numel()

                    current_batch.append({
                        "record": record,
                        "hsdp_group": hsdp_group,
                    })
                    current_numel += p_numel

                    # Soft limit: allow the bucket to slightly exceed the cap
                    # so that symmetric structures stay together and fragmentation is reduced.
                    if current_numel >= max_batch_numel:
                        sub_batches.append(current_batch)
                        current_batch = []
                        current_numel = 0

                if current_batch:
                    sub_batches.append(current_batch)

                # Sort sub-batches by numel descending within this group.
                sub_batches.sort(key=lambda b: -sum(e["record"].param.numel() for e in b))

                batch_groups.append({
                    "hsdp_group": hsdp_group,
                    "sub_batches": sub_batches,
                })

            # log batch split info.
            total_sub_batches = sum(len(bg["sub_batches"]) for bg in batch_groups)
            logger.info_rank0(
                "[HSDP Batch] group_key=%s, num_hsdp_groups=%s, num_batch_groups=%s, "
                "total_sub_batches=%s, group_numels=%s, max_batch_numel=%s",
                group_key,
                len(hsdp_groups),
                len(batch_groups),
                total_sub_batches,
                [sum(bg['hsdp_group'].records_numel) if hasattr(bg['hsdp_group'], 'records_numel') else sum(
                    r.param.numel() for r in bg['hsdp_group'].records) for bg in batch_groups],
                max_batch_numel
            )
            self._hsdp_batches[group_key] = batch_groups

    def _build_sub_batch_assignment(
            self,
            sub_batch_entries: List[Dict],
            hsdp_group: Any,
    ) -> Optional[HSDPGroupAssignment]:
        """Build an HSDPGroupAssignment for one sub-batch and update broadcast info."""
        records = [e["record"] for e in sub_batch_entries]
        if not records:
            return None

        device_mesh = records[0].param.device_mesh

        replicate_group_ranks, replicate_sizes = get_multi_dim_logical_info(
            device_mesh,
            hsdp_group.comm_key.replicate_mesh_dims,
        )

        # When hsdp_replica_count is set, remap coordinates into
        # sub-groups: each original coord maps to (coord % sub_size)
        # within its sub-group, and the effective group size shrinks.
        if self.hsdp_replica_count is not None:
            replicate_group_ranks = tuple(
                r % self.hsdp_replica_count for r in replicate_group_ranks
            )
            replicate_sizes = tuple(
                min(s, self.hsdp_replica_count) for s in replicate_sizes
            )

        # Greedy owner assignment on this sub-batch's records.
        owner_by_index = build_owner_by_size(
            records=records,
            replicate_sizes=replicate_sizes,
        )

        owned_records = select_owned_records(
            records=records,
            owner_by_index=owner_by_index,
            replicate_group_ranks=replicate_group_ranks,
        )

        is_shard_for_ns = (
                hsdp_group.comm_key.has_shard_group
                and hsdp_group.layout_spec.is_last2d_sharded
        )

        hsdp_assign = HSDPGroupAssignment(
            owned_records=owned_records,
            all_records=records,
            owner_by_index=owner_by_index,
            replicate_group_ranks=replicate_group_ranks,
            replicate_sizes=replicate_sizes,
            replicate_pgs=hsdp_group.replicate_pgs,
            shard_pgs=hsdp_group.shard_pgs,
            is_shard=is_shard_for_ns,
            layout_spec=hsdp_group.layout_spec,
        )
        logger.debug_rank0(f'[Hyper-optimizer] hsdp_assign: {hsdp_assign}')

        # Build broadcast reverse mapping for replicated groups
        if hsdp_assign.is_replicated and hsdp_assign.replicate_pgs:
            for record in hsdp_assign.all_records:
                src_coord = hsdp_assign.owner_rank_coord(record)
                if not src_coord or any(c < 0 for c in src_coord):
                    continue
                self._param_to_broadcast_info[record.param] = (
                    src_coord, hsdp_assign.replicate_pgs,
                )

        return hsdp_assign

    def _build_param_broadcast_info(self) -> None:
        """Build per-batch HSDP assignments and param broadcast reverse mapping.

        When ``hsdp_replica_count`` is set, replicate coordinates and sizes
        are remapped so that owner assignment and broadcast happen within
        sub-groups of size ``hsdp_replica_count`` instead of the full
        replicate group.

        Sub-group broadcast already covers all ranks (each rank belongs to
        exactly one sub-group), so param and state broadcast use the same
        sub-group mapping — no separate full-group path is needed.
        """
        self._hsdp_assignment_batches: Dict[int, Dict] = {}
        self._param_to_broadcast_info: Dict[
            torch.nn.Parameter, Tuple[Tuple[int, ...], Tuple[dist.ProcessGroup, ...]]
        ] = {}

        for group_key, batch_groups in self._hsdp_batches.items():
            no_comm_params = self._hsdp_grouping[group_key][0]
            assignment_batch_groups = []

            for bg in batch_groups:
                hsdp_group = bg["hsdp_group"]
                sub_batch_assigns: List[HSDPGroupAssignment] = []

                for sub_batch_entries in bg["sub_batches"]:
                    hsdp_assign = self._build_sub_batch_assignment(sub_batch_entries, hsdp_group)
                    if hsdp_assign is not None:
                        sub_batch_assigns.append(hsdp_assign)

                assignment_batch_groups.append({
                    "hsdp_group": hsdp_group,
                    "sub_batches": sub_batch_assigns,
                })

            self._hsdp_assignment_batches[group_key] = {
                "no_comm": no_comm_params,
                "batch_groups": assignment_batch_groups,
            }

    def _broadcast_replicate_params_after_step(self) -> None:
        """Broadcast updated params from assigned rank to replicate-group peers."""
        self._broadcast_op_fused(target="param")

    def _broadcast_state_fused_for_ckpt(self) -> None:
        """Broadcast optimizer state before checkpoint save."""
        state_keys = ["momentum_buffer"] if self.is_muon else ["exp_avg", "exp_avg_sq"]
        self._broadcast_op_fused(target="state", state_keys=state_keys)

    def _collect_broadcast_tensors(
            self,
            target: str,
            state_keys: Optional[List[str]] = None,
    ) -> Dict[Tuple, List[torch.Tensor]]:
        """Collect tensors to broadcast, grouped by (src_coord, dtype, replicate_pgs).

        Args:
            target: "param" or "state".
            state_keys: State dict keys to broadcast when target="state".

        Returns:
            Dict mapping (src_coord, dtype, replicate_pgs) to list of local tensors.
        """
        rank_dtype_tensors = defaultdict(list)

        for p, (src_coord, replicate_pgs) in self._param_to_broadcast_info.items():
            if target == "param":
                local_tensor = to_local_if_dtensor(p.data)
                rank_dtype_tensors[(src_coord, local_tensor.dtype, replicate_pgs)].append(local_tensor)

            elif target == "state" and state_keys:
                param_state = self.state.setdefault(p, {})
                for key in state_keys:
                    if key in param_state:
                        state_tensor = param_state[key]
                        local_tensor = to_local_if_dtensor(state_tensor)
                    else:
                        local_tensor = torch.empty_like(p, dtype=torch.float32)
                        param_state[key] = local_tensor
                        local_tensor = to_local_if_dtensor(local_tensor)

                    rank_dtype_tensors[(src_coord, local_tensor.dtype, replicate_pgs)].append(local_tensor)

        return rank_dtype_tensors

    @staticmethod
    def _compute_broadcast_batches(
            tensors: List[torch.Tensor],
            alignment_elements: int,
            max_broadcast_elements: int
    ) -> List[Tuple[List[Tuple[torch.Tensor, int, int, int]], int]]:
        """Split tensors into memory-capped batches with alignment padding.

        Args:
            tensors: List of tensors to batch.
            alignment_elements: Alignment granularity in elements.
            max_broadcast_elements: Maximum elements per batch.

        Returns:
            List of (batch_offsets, batch_total_size) tuples.
            Each batch_offsets entry is (tensor, offset, actual_numel, padded_numel).
        """
        batches = []
        current_batch = []
        current_total_size = 0

        for t in tensors:
            actual_numel = t.numel()
            padded_numel = ((actual_numel + alignment_elements - 1) // alignment_elements) * alignment_elements

            if current_batch and current_total_size + padded_numel > max_broadcast_elements:
                batches.append((current_batch, current_total_size))
                current_batch = []
                current_total_size = 0

            current_batch.append((t, current_total_size, actual_numel, padded_numel))
            current_total_size += padded_numel

        if current_batch:
            batches.append((current_batch, current_total_size))

        return batches

    @staticmethod
    def _hierarchical_broadcast_buffer(
            batch_buffer: torch.Tensor,
            src_coord: Tuple[int, ...],
            replicate_pgs: Tuple[dist.ProcessGroup, ...],
            local_coord: Tuple[int, ...]
    ) -> None:
        """Broadcast a buffer dimension-by-dimension across replicate groups.

        Args:
            batch_buffer: Contiguous buffer to broadcast.
            src_coord: Source coordinate tuple.
            replicate_pgs: Tuple of ProcessGroups (one per dimension).
            local_coord: Local rank coordinate tuple.
        """

        for dim_idx, pg in enumerate(replicate_pgs):
            if pg is None:
                continue

            participate = True
            for subsequent_dim in range(dim_idx + 1, len(replicate_pgs)):
                if local_coord[subsequent_dim] != src_coord[subsequent_dim]:
                    participate = False
                    break

            if not participate:
                continue

            src_rank_in_pg = src_coord[dim_idx]
            global_src_rank = dist.get_global_rank(pg, src_rank_in_pg)
            dist.broadcast(batch_buffer, src=global_src_rank, group=pg)

    @staticmethod
    def _hierarchical_broadcast_buffer_async(
            batch_buffer: torch.Tensor,
            src_coord: Tuple[int, ...],
            replicate_pgs: Tuple[dist.ProcessGroup, ...],
            local_coord: Tuple[int, ...],
    ) -> List[dist.Work]:
        """Async version of _hierarchical_broadcast_buffer.

        Same dimension-by-dimension relay logic, but each dist.broadcast uses
        async_op=True.  Returns a list of Work handles to wait on later.

        Note: dimensions within a single buffer are still sequential (dim N+1
        depends on dim N completing), but different buffers can overlap.
        """
        broadcast_ops: List[Tuple[dist.ProcessGroup, int]] = []

        for dim_idx, pg in enumerate(replicate_pgs):
            if pg is None:
                continue

            participate = True
            for subsequent_dim in range(dim_idx + 1, len(replicate_pgs)):
                if local_coord[subsequent_dim] != src_coord[subsequent_dim]:
                    participate = False
                    break

            if not participate:
                continue

            src_rank_in_pg = src_coord[dim_idx]
            global_src_rank = dist.get_global_rank(pg, src_rank_in_pg)
            broadcast_ops.append((pg, global_src_rank))

        handles: List[dist.Work] = []
        for op_idx, (pg, global_src_rank) in enumerate(broadcast_ops):
            handle = dist.broadcast(
                batch_buffer, src=global_src_rank, group=pg, async_op=True,
            )
            if op_idx + 1 < len(broadcast_ops):
                # The next mesh dimension relays the data received here.
                # It must not read the shared buffer until this hop finishes.
                handle.wait()
            else:
                # Only the final hop may overlap with subsequent computation.
                handles.append(handle)

        return handles

    def _broadcast_op_fused(
            self,
            target: str,
            state_keys: Optional[List[str]] = None,
    ) -> None:
        """Fused hierarchical broadcast for param or state across replicate groups.

        Groups tensors by (src_coord, dtype, replicate_pgs), packs into contiguous buffers
        with 512-byte alignment, and broadcasts dimension-by-dimension in memory-capped batches.

        Args:
            target: "param" or "state".
            state_keys: State dict keys to broadcast when target="state".
        """
        device = get_current_device()

        alignment = 512  # bytes
        rank_dtype_tensors = self._collect_broadcast_tensors(target, state_keys)

        for (src_coord, dtype, replicate_pgs), tensors in rank_dtype_tensors.items():
            if not tensors:
                continue

            local_coord = tuple(
                dist.get_rank(group=pg) if pg is not None else -1
                for pg in replicate_pgs
            )

            element_size = torch.empty(0, dtype=dtype, device=device).element_size()
            alignment_elements = alignment // element_size
            max_broadcast_bytes = getattr(self, "replicate_broadcast_max_bytes", 512 * 1024 * 1024)
            max_broadcast_elements = max_broadcast_bytes // element_size
            max_broadcast_elements = max(
                alignment_elements,
                (max_broadcast_elements // alignment_elements) * alignment_elements,
            )

            batches = self._compute_broadcast_batches(tensors, alignment_elements, max_broadcast_elements)
            if not batches:
                continue

            max_batch_size = max(batch_size for _, batch_size in batches)
            buffer = torch.empty(max_batch_size, dtype=dtype, device=device)

            for batch_tensor_offsets, batch_total_size in batches:
                batch_buffer = buffer[:batch_total_size]

                # Pack: owner rank
                if local_coord == src_coord:
                    for t, offset, actual_numel, padded_numel in batch_tensor_offsets:
                        batch_buffer[offset:offset + actual_numel].copy_(t.view(-1))
                        if padded_numel > actual_numel:
                            batch_buffer[offset + actual_numel:offset + padded_numel].zero_()

                # Hierarchical Broadcast
                self._hierarchical_broadcast_buffer(batch_buffer, src_coord, replicate_pgs, local_coord)

                # Unpack: copy buffer data back to individual tensors
                for t, offset, actual_numel, _ in batch_tensor_offsets:
                    t.view(-1).copy_(batch_buffer[offset:offset + actual_numel])

            buffer.untyped_storage().resize_(0)
            del buffer

    def cleanup_synced_state(self) -> None:
        """Release optimizer state for non-owned params after checkpoint saving.

        In HSDP mode, _broadcast_state_fused_for_ckpt broadcasts state to all
        ranks so every rank can save a complete checkpoint. This method removes
        the non-owned state to restore per-rank memory savings. Must be called
        after the checkpoint has been fully written to disk.
        """
        params_to_remove = []
        for assignment_info in self._hsdp_assignment_batches.values():
            for bg in assignment_info["batch_groups"]:
                for hsdp_info in bg["sub_batches"]:
                    if not hsdp_info.is_replicated:
                        continue
                    for record in hsdp_info.all_records:
                        if not hsdp_info.is_owned(record) and record.param in self.state:
                            params_to_remove.append(record.param)

        for p in params_to_remove:
            self.state[p].clear()
            del self.state[p]

        empty_accelerator_cache()


class AsyncReplicateBroadcaster:
    """Incremental async replicate broadcast with HSDP-group-level flushing.

    Designed to overlap HSDP replicate-group broadcasts with Muon NS
    computation.  After all sub_batches for an HSDP group are done, the
    caller invokes flush_group(hsdp_assign) to issue an async hierarchical
    broadcast — overlapping with the next HSDP group's NS computation.

    Flush is driven by the caller at HSDP-group boundaries (not by a
    threshold) so that all ranks issue the same collective operations in
    the same order, which is required by NCCL/HCCL.

    Usage::

        broadcaster = AsyncReplicateBroadcaster(optimizer)
        for hsdp_assign in hsdp_assignments:
            # ... NS compute + apply for this group ...
            broadcaster.flush_group(hsdp_assign)
        broadcaster.wait_all()
    """

    def __init__(
            self,
            optimizer: BaseDistributedOptimizer,
    ) -> None:
        self._optimizer = optimizer

        # Inflight async broadcasts: list of (buffer, batch_offsets, handles)
        # Buffer must stay alive until handles are waited on.
        self._inflight: List[
            Tuple[
                torch.Tensor,
                List[Tuple[torch.Tensor, int, int, int]],
                List[dist.Work],
            ]
        ] = []
        # Allow one batch of replicate communication to overlap with the
        # next batch's compute, but avoid unbounded inflight buffer growth.
        self._max_inflight_batches = 1

    def flush_group(
            self,
            hsdp_assign: Any,
            records: Optional[List] = None,
    ) -> None:
        """Flush params belonging to the given HSDP group.

        Can be called per sub_batch (passing only the records in that
        sub_batch) so that the async replicate broadcast overlaps with
        the next sub_batch's NS computation.  When *records* is None,
        all records in the group are flushed (backward compatible).

        All ranks must call this at the same point in the execution flow
        to ensure collective communication consistency.
        """
        if not hsdp_assign.is_replicated or not hsdp_assign.replicate_pgs:
            return

        flush_records = records if records is not None else hsdp_assign.all_records

        # Collect local tensors for the given records, grouped by
        # (src_coord, dtype, replicate_pgs) — same logic as
        # _collect_broadcast_tensors but scoped to the provided records.
        rank_dtype_tensors: Dict[
            Tuple[Tuple[int, ...], torch.dtype, Tuple[dist.ProcessGroup, ...]],
            List[torch.Tensor],
        ] = defaultdict(list)

        for record in flush_records:
            src_coord = hsdp_assign.owner_rank_coord(record)
            if not src_coord or any(c < 0 for c in src_coord):
                continue
            local_tensor = to_local_if_dtensor(record.param.data)
            key = (src_coord, local_tensor.dtype, hsdp_assign.replicate_pgs)
            rank_dtype_tensors[key].append(local_tensor)

        for key, tensors in rank_dtype_tensors.items():
            if tensors:
                self._flush_key(key, tensors, async_op=True)

        while len(self._inflight) > self._max_inflight_batches:
            self._wait_and_release_oldest()

    def _flush_key(
            self,
            key: Tuple[Tuple[int, ...], torch.dtype, Tuple[dist.ProcessGroup, ...]],
            tensors: List[torch.Tensor],
            async_op: bool = True,
    ) -> None:
        """Pack and broadcast tensors for one broadcast key."""
        device = get_current_device()
        src_coord, dtype, replicate_pgs = key
        alignment = 512  # bytes

        local_coord = tuple(
            dist.get_rank(group=pg) if pg is not None else -1
            for pg in replicate_pgs
        )

        element_size = torch.empty(0, dtype=dtype, device=device).element_size()
        alignment_elements = alignment // element_size
        max_broadcast_bytes = getattr(
            self._optimizer, "replicate_broadcast_max_bytes", 512 * 1024 * 1024,
        )
        max_broadcast_elements = max_broadcast_bytes // element_size
        max_broadcast_elements = max(
            alignment_elements,
            (max_broadcast_elements // alignment_elements) * alignment_elements,
        )

        # pylint: disable=protected-access
        batches = BaseDistributedOptimizer._compute_broadcast_batches(
            tensors, alignment_elements, max_broadcast_elements,
        )
        if not batches:
            return

        buffer: torch.Tensor
        if not async_op:
            max_batch_size = max(batch_size for _, batch_size in batches)
            buffer = torch.empty(max_batch_size, dtype=dtype, device=device)

        for batch_tensor_offsets, batch_total_size in batches:
            if async_op:
                batch_buffer = torch.empty(batch_total_size, dtype=dtype, device=device)
            else:
                batch_buffer = buffer[:batch_total_size]

            # Pack: owner rank
            if local_coord == src_coord:
                for t, offset, actual_numel, padded_numel in batch_tensor_offsets:
                    batch_buffer[offset:offset + actual_numel].copy_(t.view(-1))
                    if padded_numel > actual_numel:
                        batch_buffer[offset + actual_numel:offset + padded_numel].zero_()

            if async_op:
                handles = BaseDistributedOptimizer._hierarchical_broadcast_buffer_async(
                    batch_buffer, src_coord, replicate_pgs, local_coord,
                )
                if handles:
                    # Pin buffer + offsets until wait_all unpacks them.
                    self._inflight.append((batch_buffer, batch_tensor_offsets, handles))
                else:
                    # No async work was enqueued on this rank, so unpack and free now.
                    for t, offset, actual_numel, _ in batch_tensor_offsets:
                        t.view(-1).copy_(batch_buffer[offset:offset + actual_numel])
                    batch_buffer.untyped_storage().resize_(0)
                    del batch_buffer
            else:
                BaseDistributedOptimizer._hierarchical_broadcast_buffer(
                    batch_buffer, src_coord, replicate_pgs, local_coord,
                )
                # Unpack immediately for sync path
                for t, offset, actual_numel, _ in batch_tensor_offsets:
                    t.view(-1).copy_(batch_buffer[offset:offset + actual_numel])

        if not async_op:
            # Sync path: buffer can be freed immediately
            buffer.untyped_storage().resize_(0)
            del buffer

    def _wait_and_release_oldest(self) -> None:
        """Wait, unpack, and release the oldest inflight async batch."""
        batch_buffer, batch_tensor_offsets, handles = self._inflight.pop(0)
        for handle in handles:
            handle.wait()

        # Unpack after the async broadcast completes.
        for t, offset, actual_numel, _ in batch_tensor_offsets:
            t.view(-1).copy_(batch_buffer[offset:offset + actual_numel])

        batch_buffer.untyped_storage().resize_(0)
        del batch_buffer

    def wait_all(self) -> None:
        """Wait for all inflight async broadcasts and unpack results."""
        while self._inflight:
            self._wait_and_release_oldest()
