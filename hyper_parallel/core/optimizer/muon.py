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

"""Muon optimizer with HSDP shard-group-aware communication."""

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from hyper_parallel.core.optimizer.optimizer import AsyncReplicateBroadcaster, BaseDistributedOptimizer
from hyper_parallel.core.optimizer.dtensor_compat import to_local_if_dtensor
from hyper_parallel.core.optimizer.sharding_category import (
    HSDPGroupAssignment,
    fused_allgather_dtensor_params,
    build_owner_by_size,
    chunk_update_by_layout,
)
from hyper_parallel.platform import get_platform


def zeropower_via_newtonschulz5(ns_inputs: torch.Tensor, steps: int) -> torch.Tensor:
    """Matrix orthogonalization with preallocated matmul buffers."""
    mat_x = ns_inputs
    if ns_inputs.size(-2) > ns_inputs.size(-1):
        mat_x = mat_x.mT

    # Create a new tensor so later in-place updates are safe.
    mat_x = mat_x / (torch.norm(mat_x, dim=[-2, -1], keepdim=True) + 1e-7)

    coeffs = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]

    # Preallocate temporary buffers to avoid loop-time allocations.
    n_size = mat_x.size(-2)
    buf_a = torch.empty(mat_x.shape[:-2] + (n_size, n_size), dtype=mat_x.dtype, device=mat_x.device)
    buf_a2 = torch.empty_like(buf_a)
    buf_b = torch.empty_like(buf_a)
    buf_bx = torch.empty_like(mat_x)

    for coeff_a, coeff_b, coeff_c in coeffs[:steps]:
        # A = X @ X.T
        torch.matmul(mat_x, mat_x.mT, out=buf_a)

        # A2 = A @ A
        torch.matmul(buf_a, buf_a, out=buf_a2)

        # B = b * A + c * A2
        buf_b.copy_(buf_a).mul_(coeff_b)
        buf_a2.mul_(coeff_c)
        buf_b.add_(buf_a2)

        # BX = B @ X
        torch.matmul(buf_b, mat_x, out=buf_bx)

        # X = a * X + BX
        mat_x.mul_(coeff_a).add_(buf_bx)

    if ns_inputs.size(-2) > ns_inputs.size(-1):
        mat_x = mat_x.mT

    return mat_x


def adjust_lr_wd_for_muon(lr: float, matched_adamw_rms: float, param_shape: torch.Size) -> float:
    """Scale learning rate for 2D Muon parameters based on tensor dimensions."""
    dim_a, dim_b = param_shape[-2:]
    adjusted_ratio = math.sqrt(max(dim_a, dim_b)) * matched_adamw_rms
    return lr * adjusted_ratio


def adjust_lr_wd_for_muon_conv(lr: float, matched_adamw_rms: float, param_shape: torch.Size) -> float:
    """Scale learning rate for 3D convolutional Muon parameters."""
    dim_a, dim_b, dim_c = param_shape[:]
    adjusted_ratio = math.sqrt(max(dim_a, dim_b, dim_c)) * matched_adamw_rms
    return lr * adjusted_ratio


class Muon(BaseDistributedOptimizer):
    """Muon optimizer with HSDP shard-group-aware Newton-Schulz orthogonalization.

    Implements the Muon optimizer which uses Newton-Schulz iteration for matrix
    orthogonalization of gradient updates, with HSDP-aware communication for
    sharded parameters.
    """

    def __init__(
            self,
            params,
            lr: float = 2e-2,
            weight_decay: float = 0.1,
            matched_adamw_rms: float = 0.2,
            momentum: float = 0.95,
            nesterov: bool = True,
            ns_steps: int = 5,
            hsdp_replica_count: Optional[Union[int, Tuple[int, ...]]] = None,
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "matched_adamw_rms": matched_adamw_rms,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
        }
        super().__init__(params, defaults, is_muon=True, hsdp_replica_count=hsdp_replica_count)

        self._group_dtensor_by_mesh()
        deduced_count = self._auto_deduce_replica_count()
        if deduced_count is None:
            self.hsdp_replica_count = None
        elif self.hsdp_replica_count is None:
            self.hsdp_replica_count = deduced_count
        self._split_replicate_groups()
        self._build_hsdp_batch()
        self._build_param_broadcast_info()
        self._classify_parameters_for_step()

    @torch.no_grad()
    def step(self, closure=None) -> Optional[float]:
        """
        Perform a single optimization step.
        De-duplication is controlled by the caller: ``param_to_ns_input`` should already contain only the owned 
        params (via ``hsdp_assign.owned_params``). The caller is responsible for broadcasting the updated params to 
        replica peers via ``AsyncReplicateBroadcaster.flush_group``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        broadcaster = AsyncReplicateBroadcaster(self)

        num_groups = len(self.param_groups)

        for group in self.param_groups:
            group['step'] = (group.get('step') or 0) + 1

        # Compute momentum only for no-comm params upfront.
        no_comm_ns: Dict[int, Dict] = {}
        for group_idx in range(num_groups):
            info = self._hsdp_assignment_batches.get(group_idx)
            if not info:
                # No HSDP info — all params are no_comm.
                unshard_params = self.unshard_params_by_group.get(group_idx, [])
                no_comm_ns[group_idx] = self._update_muon_momentum(
                    self.param_groups[group_idx], unshard_params
                )
            else:
                no_comm_params = info.get("no_comm", [])
                if no_comm_params:
                    no_comm_ns[group_idx] = self._update_muon_momentum(
                        self.param_groups[group_idx], no_comm_params
                    )
                else:
                    no_comm_ns[group_idx] = {}

        # Process no-comm params.
        for group_idx in range(num_groups):
            if no_comm_ns[group_idx]:
                self._process_unshard_params(self.param_groups[group_idx], no_comm_ns[group_idx])

        # Flatten nested batch into a linear schedule.
        group_linear_batches: Dict[int, List[HSDPGroupAssignment]] = {}
        max_num_batches = 0
        for group_idx in range(num_groups):
            info = self._hsdp_assignment_batches.get(group_idx)
            linear_batches = []
            if info:
                for bg in info.get("batch_groups", []):
                    linear_batches.extend(bg.get("sub_batches", []))
            group_linear_batches[group_idx] = linear_batches
            max_num_batches = max(max_num_batches, len(linear_batches))

        # Process batches: compute momentum per-batch for owned params only (HSDP de-duplication).
        for batch_idx in range(max_num_batches):
            for group_idx in range(num_groups):
                linear_batches = group_linear_batches[group_idx]
                if batch_idx >= len(linear_batches):
                    continue

                hsdp_assign = linear_batches[batch_idx]
                group = self.param_groups[group_idx]

                # Compute momentum for this assignment's owned params only.
                ns_inputs = self._update_muon_momentum(group, hsdp_assign.owned_params)
                if not hsdp_assign.is_shard:
                    if ns_inputs:
                        self._process_unshard_params(group, ns_inputs)
                else:
                    self._process_shard_params(
                        group, ns_inputs, [hsdp_assign], group_idx,
                        buffer_cache={},
                    )

                # Flush broadcasts after each assignment.
                broadcaster.flush_group(hsdp_assign)

        broadcaster.wait_all()
        return loss

    def _classify_parameters_for_step(self) -> None:
        """Classify params by whether the last two dims are sharded.

        unshard: run Newton-Schulz locally.
        shard: all-gather before Newton-Schulz.

        Reads from self._hsdp_assignment_batches which is organized by batch.
        """
        self.unshard_params_by_group: Dict[int, List] = {}
        self.shard_params_by_group: Dict[int, List] = {}
        self.shard_assignments_by_group: Dict[int, List[HSDPGroupAssignment]] = {}
        # Per group: record.index -> shard coord that computes NS.
        self._shard_compute_coord: Dict[int, Dict[int, Tuple[int, ...]]] = {}

        for group_idx, group in enumerate(self.param_groups):
            assignment_info = self._hsdp_assignment_batches.get(group_idx)

            unshard_params = []
            shard_hsdp_assignments: List[HSDPGroupAssignment] = []

            if assignment_info:
                unshard_params.extend(assignment_info["no_comm"])
                for bg in assignment_info["batch_groups"]:
                    for hsdp_assign in bg["sub_batches"]:
                        if hsdp_assign.is_shard:
                            shard_hsdp_assignments.append(hsdp_assign)
                        else:
                            unshard_params.extend(hsdp_assign.owned_params)
            else:
                unshard_params.extend(group["params"])

            # Shard params only include locally owned params; replica ownership is enforced at apply time.
            shard_params = [p for a in shard_hsdp_assignments for p in a.owned_params]

            # Greedily assign NS compute across shard ranks.
            self._shard_compute_coord[group_idx] = {}
            for hsdp_assign in shard_hsdp_assignments:
                shard_sizes, _, _, _ = self._get_shard_info(hsdp_assign)
                compute_by_index = build_owner_by_size(
                    records=hsdp_assign.owned_records,
                    replicate_sizes=shard_sizes,
                )
                self._shard_compute_coord[group_idx].update(compute_by_index)

            self.unshard_params_by_group[group_idx] = unshard_params
            self.shard_assignments_by_group[group_idx] = shard_hsdp_assignments
            self.shard_params_by_group[group_idx] = shard_params

    def _update_muon_momentum(
            self,
            group: Dict[str, Any],
            params: List[torch.Tensor],
    ) -> Dict[torch.nn.Parameter, torch.Tensor]:
        """Compute first-order momentum and return bfloat16 NS inputs."""
        momentum = group['momentum']
        nesterov = group['nesterov']

        # Pre-filter params with valid grads and ensure momentum buffers exist
        valid_params = []
        grads = []
        bufs = []
        for p in params:
            g = p.grad
            if g is None:
                continue
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            valid_params.append(p)
            grads.append(g)
            bufs.append(state["momentum_buffer"])

        if not valid_params:
            return {}

        # Strip DTensor wrappers for elementwise foreach ops
        local_grads = [to_local_if_dtensor(g) for g in grads]
        local_bufs = [to_local_if_dtensor(b) for b in bufs]

        # Fused momentum update: buf = momentum * buf + grad
        # pylint: disable=protected-access
        torch._foreach_mul_(local_bufs, momentum)
        torch._foreach_add_(local_bufs, local_grads)

        # Nesterov: u = momentum * buf + grad  (out-of-place, keeps buf intact)
        if nesterov:
            local_us = torch._foreach_mul(local_bufs, momentum)
            torch._foreach_add_(local_us, local_grads)
        else:
            local_us = list(local_bufs)

        # Cast to bfloat16 for NS iteration
        if local_us[0].dtype == torch.bfloat16:
            local_us_bf = local_us
        else:
            local_us_bf = [u.to(torch.bfloat16) for u in local_us]

        return dict(zip(valid_params, local_us_bf))

    def _process_unshard_params(
            self,
            group: Dict[str, Any],
            param_to_ns_input: Dict[torch.nn.Parameter, torch.Tensor],
    ) -> None:
        """Process un-sharded params: shape-group -> memory-batch -> NS -> local update."""
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        rms = group["matched_adamw_rms"]
        ns_steps = group["ns_steps"]

        shape_groups = self._group_by_shape(list(param_to_ns_input.keys()))
        for _, p_list in shape_groups.items():
            safe_batches = self._split_into_memory_safe_batches(p_list, shard_size=1)

            for sub_batch in safe_batches:
                updates_dict, adjusted_lr = self._compute_batched_ns_updates(
                    sub_batch, param_to_ns_input, lr, rms, ns_steps, no_shard=True
                )

                # Fused batched apply — all params in the same sub_batch share
                # the same adjusted_lr, so we can use foreach ops.
                local_params = [to_local_if_dtensor(p.data) for p in sub_batch]
                local_updates = [updates_dict[p].view(lp.shape) for p, lp in zip(sub_batch, local_params)]

                if weight_decay != 0.0:
                    # pylint: disable=protected-access
                    torch._foreach_mul_(local_params, 1 - lr * weight_decay)
                # pylint: disable=protected-access
                torch._foreach_add_(local_params, local_updates, alpha=-adjusted_lr)

    def _gather_and_compute_shard_updates(
            self,
            valid_params: List[torch.nn.Parameter],
            param_to_ns_input: Dict[torch.nn.Parameter, torch.Tensor],
            hsdp_assign: HSDPGroupAssignment,
            shard_compute_coord: Dict[int, Tuple[int, ...]],
            shard_sizes: Tuple[int, ...],
            local_coords: Tuple[int, ...],
            shard_pgs: Tuple[dist.ProcessGroup, ...],
            total_shard_size: int,
            lr: float,
            rms: float,
            ns_steps: int,
            buffer_cache: Optional[Dict],
    ) -> Tuple[
        Dict[torch.nn.Parameter, torch.Tensor],
        Dict[torch.nn.Parameter, Tuple[int, ...]],
    ]:
        """Gather NS inputs and compute updates for locally-assigned shard params.

        Returns:
            (my_updates, param_compute_coord) for this HSDP assignment.
        """
        param_to_index = {record.param: record.index for record in hsdp_assign.owned_records}

        my_params: List[torch.nn.Parameter] = []
        my_param_ids: set = set()
        param_compute_coord: Dict[torch.nn.Parameter, Tuple[int, ...]] = {}
        my_indices: set = set()

        for idx, p in enumerate(valid_params):
            p_index = param_to_index[p]
            compute_coord = shard_compute_coord.get(p_index, (0,) * len(shard_sizes))
            param_compute_coord[p] = compute_coord
            if compute_coord == local_coords:
                my_params.append(p)
                my_param_ids.add(id(p))
                my_indices.add(idx)

        # Fused all-gather full NS inputs; keep only tensors computed locally.
        gathered_inputs: Dict[torch.nn.Parameter, torch.Tensor] = {}
        local_inputs = [param_to_ns_input[p] for p in valid_params]
        gathered_list = fused_allgather_dtensor_params(
            local_inputs, shard_pgs, hsdp_assign.layout_spec, buffer_cache=buffer_cache,
            keep_indices=my_indices,
        )
        for p, full_inp in zip(valid_params, gathered_list):
            if id(p) in my_param_ids:
                gathered_inputs[p] = full_inp
        local_inputs.clear()
        gathered_list.clear()

        # Compute NS updates in shape groups and memory-safe batches.
        my_updates: Dict[torch.nn.Parameter, torch.Tensor] = {}
        if my_params:
            shape_groups = self._group_by_shape(my_params)
            for _, p_list in shape_groups.items():
                safe_batches = self._split_into_memory_safe_batches(p_list, shard_size=total_shard_size)
                for sub_batch in safe_batches:
                    updates_dict, _ = self._compute_batched_ns_updates(
                        sub_batch, gathered_inputs, lr, rms, ns_steps
                    )
                    for p in sub_batch:
                        my_updates[p] = updates_dict[p].contiguous()
                    del updates_dict
            gathered_inputs.clear()

        return my_updates, param_compute_coord

    def _process_shard_params(
            self,
            group: Dict[str, Any],
            param_to_ns_input: Dict[torch.nn.Parameter, torch.Tensor],
            hsdp_assignments: List[HSDPGroupAssignment],
            group_idx: int,
            buffer_cache: Optional[Dict] = None,
    ) -> None:
        """Process sharded params with greedy shard-group compute assignment."""
        platform = get_platform()
        device = torch.npu.current_device() if torch.npu.is_available() else torch.cuda.current_device()

        lr = group["lr"]
        weight_decay = group["weight_decay"]
        rms = group["matched_adamw_rms"]
        ns_steps = group["ns_steps"]

        shard_compute_coord = self._shard_compute_coord.get(group_idx, {})

        for hsdp_assign in hsdp_assignments:
            owned_params = hsdp_assign.owned_params
            valid_params = [p for p in owned_params if p in param_to_ns_input]

            shard_sizes, local_coords, shard_pgs, total_shard_size = self._get_shard_info(hsdp_assign)

            # Gather NS inputs and compute updates assigned to this shard coordinate.
            if valid_params:
                my_updates, param_compute_coord = self._gather_and_compute_shard_updates(
                    valid_params, param_to_ns_input, hsdp_assign,
                    shard_compute_coord, shard_sizes, local_coords,
                    shard_pgs, total_shard_size,
                    lr, rms, ns_steps, buffer_cache,
                )

                # Fused broadcast full updates within the shard group, then batched apply.
                self._fused_broadcast_and_apply(
                    valid_params, my_updates, param_compute_coord,
                    lr, weight_decay, rms,
                    shard_pgs, shard_sizes, local_coords, total_shard_size,
                    hsdp_assign, platform, device,
                )

    def _group_by_shape(
            self,
            params: List[torch.nn.Parameter],
    ) -> Dict[tuple, List[torch.nn.Parameter]]:
        """Group parameters by their last-2-dim shape (A, B) for batched NS.

        [1024, 1024], [1024, 1, 1024], and [3, 1024, 1024] all map to
        key (1024, 1024) for maximum batch merging.
        """
        groups = defaultdict(list)

        for p in params:
            shape = tuple(p.shape)

            if len(shape) == 2:
                core_shape = (shape[0], shape[1])
            elif len(shape) == 3 and shape[1] == 1:
                core_shape = (shape[0], shape[2])
            elif len(shape) >= 3:
                core_shape = (shape[-2], shape[-1])
            else:
                raise ValueError('1D parameters are not supported in Muon')

            groups[core_shape].append(p)

        return groups

    def _split_into_memory_safe_batches(
            self,
            p_list: List[torch.nn.Parameter],
            shard_size: int = 1,
    ) -> List[List[torch.nn.Parameter]]:
        """Split parameters into memory-safe batches to prevent OOM during NS.

        The per-batch element limit is scaled down by shard_size to account
        for the memory amplification from allgather.
        """
        max_numel_per_batch = 512 * 1024 * 1024 // shard_size

        batches = []
        current_batch = []
        current_count = 0

        for p in p_list:
            p_count = p.numel()
            if current_batch and current_count + p_count > max_numel_per_batch:
                batches.append(current_batch)
                current_batch = [p]
                current_count = p_count
            else:
                current_batch.append(p)
                current_count += p_count

        if current_batch:
            batches.append(current_batch)

        return batches

    def _compute_batched_ns_updates(
            self,
            p_list: List[torch.nn.Parameter],
            ns_inputs: Dict[torch.nn.Parameter, torch.Tensor],
            lr: float,
            rms: float,
            ns_steps: int,
            no_shard: bool = False
    ) -> Tuple[Dict[torch.nn.Parameter, torch.Tensor], float]:
        """Batched Newton-Schulz update for mixed 2D / Conv3D / 3D parameters.

        Normalizes all inputs to 3D, concatenates along dim 0, runs a single
        NS iteration, then slices results back to original shapes.

        Returns:
            updates_dict: per-parameter NS-orthogonalized updates.
            adjusted_lr: a single scalar — all params in the same batch share
                the same shape group and optimizer hyper-params, so their
                adjusted_lr is identical.
        """
        updates_dict = {}

        if not p_list:
            return updates_dict, 0.0

        inputs_3d = []
        slice_sizes = []
        shapes_info = []

        for p in p_list:
            origin_shape = tuple(getattr(p, 'local_shape', None) or p.to_local().shape) if no_shard else tuple(p.shape)
            ns_input = ns_inputs[p].view(origin_shape)

            is_conv = False
            if len(origin_shape) == 2:
                inp_3d = ns_input.unsqueeze(0)
                n_dim = 1
            elif len(origin_shape) == 3 and origin_shape[1] == 1:
                inp_3d = ns_input.squeeze(1).unsqueeze(0)
                is_conv = True
                n_dim = 1
            else:
                inp_3d = ns_input
                n_dim = origin_shape[0]

            inputs_3d.append(inp_3d)
            slice_sizes.append(n_dim)
            shapes_info.append((origin_shape, is_conv))

        merged_input = torch.cat(inputs_3d, dim=0)
        merged_update = zeropower_via_newtonschulz5(merged_input, steps=ns_steps)
        del merged_input

        current_idx = 0
        for i, p in enumerate(p_list):
            n_dim = slice_sizes[i]
            origin_shape, is_conv = shapes_info[i]

            update = merged_update[current_idx: current_idx + n_dim]
            current_idx += n_dim

            if is_conv:
                update = update.squeeze(0).unsqueeze(1)
            elif len(origin_shape) == 2:
                update = update.squeeze(0)

            updates_dict[p] = update
        del merged_update

        # Compute adjusted_lr once — all params share the same shape group
        ref_shape, is_conv = shapes_info[0]
        if is_conv:
            adjusted_lr = adjust_lr_wd_for_muon_conv(lr, rms, ref_shape)
        else:
            adjusted_lr = adjust_lr_wd_for_muon(lr, rms, ref_shape)

        return updates_dict, adjusted_lr

    def _fused_broadcast_and_apply(
            self,
            valid_params: List[torch.nn.Parameter],
            my_updates: Dict[torch.nn.Parameter, torch.Tensor],
            param_compute_coord: Dict[torch.nn.Parameter, Tuple[int, ...]],
            lr: float,
            weight_decay: float,
            rms: float,
            shard_pgs: Tuple[dist.ProcessGroup, ...],
            shard_sizes: Tuple[int, ...],
            local_coords: Tuple[int, ...],
            total_shard_size: int,
            hsdp_assign: HSDPGroupAssignment,
            platform: Any,
            device: torch.device,
    ) -> None:
        """Fused broadcast + batched apply for shard-group updates (Optimized for low Free time)."""
        coord_groups: Dict[Tuple[int, ...], List[torch.nn.Parameter]] = defaultdict(list)
        for p in valid_params:
            coord_groups[param_compute_coord[p]].append(p)

        all_local_params: List[torch.Tensor] = []
        all_update_shards: List[torch.Tensor] = []
        all_adjusted_lrs: List[float] = []

        # Phase 1: Batched Pack
        alignment_bytes = 512
        element_size = torch.empty(0, dtype=torch.bfloat16, device=device).element_size()
        alignment_elements = max(1, alignment_bytes // element_size)

        pack_buffers: Dict[Tuple[int, ...], torch.Tensor] = {}
        coord_param_offsets: Dict[Tuple[int, ...], List[Tuple[int, int, int]]] = {}

        for coord, coord_params in coord_groups.items():
            is_compute_rank = coord == local_coords
            param_offsets: List[Tuple[int, int, int]] = []
            total_padded_numel = 0

            for p in coord_params:
                actual_numel = p.numel()
                padded_numel = ((actual_numel + alignment_elements - 1) // alignment_elements) * alignment_elements
                param_offsets.append((total_padded_numel, actual_numel, padded_numel))
                total_padded_numel += padded_numel

            coord_param_offsets[coord] = param_offsets
            pack_buffer = torch.empty(total_padded_numel, dtype=torch.bfloat16, device=device)

            if is_compute_rank:
                for p, (offset, actual_numel, padded_numel) in zip(coord_params, param_offsets):
                    update = my_updates[p]
                    pack_buffer[offset:offset + actual_numel].copy_(update.reshape(-1))
                    if padded_numel > actual_numel:
                        pack_buffer[offset + actual_numel:offset + padded_numel].zero_()

            pack_buffers[coord] = pack_buffer

        # Phase 2: Async Batched Relay Broadcast
        if total_shard_size > 1:
            self._batched_relay_broadcast(
                pack_buffers, shard_pgs, shard_sizes, local_coords, platform
            )

        # Phase 3: Batched Unpack & Apply
        layout_spec = hsdp_assign.layout_spec
        for coord, coord_params in coord_groups.items():
            pack_buffer = pack_buffers[coord]
            param_offsets = coord_param_offsets[coord]

            for p, (offset, actual_numel, _) in zip(coord_params, param_offsets):
                full_update = pack_buffer[offset:offset + actual_numel].view(tuple(p.shape))
                update_to_apply = chunk_update_by_layout(full_update, p, layout_spec)

                origin_shape = tuple(p.shape)
                if len(origin_shape) == 3 and origin_shape[1] == 1:
                    adjusted_lr = adjust_lr_wd_for_muon_conv(lr, rms, origin_shape)
                else:
                    adjusted_lr = adjust_lr_wd_for_muon(lr, rms, origin_shape)

                all_local_params.append(to_local_if_dtensor(p.data))
                all_update_shards.append(update_to_apply.view(to_local_if_dtensor(p.data).shape))
                all_adjusted_lrs.append(adjusted_lr)

        # Batched Apply
        if not all_local_params:
            return

        if weight_decay != 0.0:
            coeff = 1.0 - lr * weight_decay
            # pylint: disable=protected-access
            torch._foreach_mul_(all_local_params, coeff)

        lr_group_map: Dict[float, Tuple[List[torch.Tensor], List[torch.Tensor]]] = defaultdict(lambda: ([], []))
        for local_p, update_shard, adj_lr in zip(all_local_params, all_update_shards, all_adjusted_lrs):
            params_list, updates_list = lr_group_map[adj_lr]
            params_list.append(local_p)
            updates_list.append(update_shard)

        for adj_lr, (params_list, updates_list) in lr_group_map.items():
            if params_list:
                # pylint: disable=protected-access
                torch._foreach_add_(params_list, updates_list, alpha=-adj_lr)

    @staticmethod
    def _get_shard_info(
            hsdp_assign: HSDPGroupAssignment,
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[dist.ProcessGroup, ...], int]:
        """Extract shard topology from HSDPGroupAssignment.

        Returns:
            shard_sizes: Size of each shard mesh dimension.
            local_coords: Current rank's coordinate in each shard mesh dim.
            shard_pgs: ProcessGroup for each shard mesh dim.
            total_shard_size: Product of all shard_sizes.
        """
        shard_pgs = hsdp_assign.shard_pgs
        shard_sizes = tuple(
            dist.get_world_size(pg) if pg is not None else 1
            for pg in shard_pgs
        )
        local_coords = tuple(
            dist.get_rank(pg) if pg is not None else 0
            for pg in shard_pgs
        )
        total_shard_size = 1
        for s in shard_sizes:
            total_shard_size *= s

        return shard_sizes, local_coords, shard_pgs, total_shard_size

    @staticmethod
    def _batched_relay_broadcast(
            tensor_dict: Dict[Tuple[int, ...], torch.Tensor],
            shard_pgs: Tuple[dist.ProcessGroup, ...],
            shard_sizes: Tuple[int, ...],
            local_coords: Tuple[int, ...],
            platform: Any,
    ) -> None:
        """
        Batched asynchronous multi-dimensional relay broadcast.
        By operating asynchronously within each dimension, we eliminate CPU overhead bubbles
        while strictly preserving the multidimensional relay dependency.
        """
        for dim_idx, pg in enumerate(shard_pgs):
            if pg is None or shard_sizes[dim_idx] <= 1:
                continue

            work_handles = []

            for coord, tensor in tensor_dict.items():
                aligned = all(
                    local_coords[sub_dim] == coord[sub_dim]
                    for sub_dim in range(dim_idx + 1, len(shard_pgs))
                )
                if not aligned:
                    continue

                src_rank_in_pg = coord[dim_idx]
                global_src_rank = platform.get_global_rank(pg, src_rank_in_pg)

                work = dist.broadcast(tensor, src=global_src_rank, group=pg, async_op=True)
                if work is not None:
                    work_handles.append(work)

            for work in work_handles:
                work.wait()
