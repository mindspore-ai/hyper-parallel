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
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from hyper_parallel.core.optimizer.optimizer import BaseDistributedOptimizer, to_local_if_dtensor
from hyper_parallel.core.optimizer.sharding_category import (
    HSDPGroupAssignment,
    OptimizerHSDPAssignment,
    allgather_dtensor_param,
    chunk_update_by_layout,
)
from hyper_parallel.platform import get_platform


def zeropower_via_newtonschulz5(ns_inputs: torch.Tensor, steps: int) -> torch.Tensor:
    """Matrix orthogonalization using Newton-Schulz iteration.

    Args:
        ns_inputs: Input matrix/tensor to orthogonalize.
        steps: Number of Newton-Schulz iteration steps to run.

    Returns:
        Orthogonalized tensor with the same shape as input.
    """
    mat_x = ns_inputs
    if ns_inputs.size(-2) > ns_inputs.size(-1):
        mat_x = mat_x.mT

    mat_x = mat_x / (torch.norm(mat_x, dim=[-2, -1], keepdim=True) + 1e-7)

    coeffs = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]
    for coeff_a, coeff_b, coeff_c in coeffs[:steps]:
        mat_a = mat_x @ mat_x.mT
        mat_b = coeff_b * mat_a + coeff_c * mat_a @ mat_a
        mat_x = coeff_a * mat_x + mat_b @ mat_x

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


def _unflatten_coord(flat_idx: int, sizes: Tuple[int, ...]) -> Tuple[int, ...]:
    """Convert a flat index into a multi-dimensional coordinate (row-major).

    Args:
        flat_idx: Flattened 1D index.
        sizes: Size of each dimension.

    Returns:
        Multi-dimensional coordinate tuple.

    Example:
        _unflatten_coord(5, (2, 4)) -> (1, 1)
    """
    coord = []
    for size in reversed(sizes):
        coord.append(flat_idx % size)
        flat_idx //= size
    return tuple(reversed(coord))


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
            ns_steps: int = 5
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "matched_adamw_rms": matched_adamw_rms,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
        }
        super().__init__(params, defaults, is_muon=True)
        self._classify_parameters_for_step()

    def _classify_parameters_for_step(self) -> None:
        """Classify parameters into un_shard and shard groups based on HSDP topology.

        un_shard: last 2 tensor dims are not sharded -> local Newton-Schulz.
        shard: last 2 tensor dims are sharded -> allgather before Newton-Schulz.
        """
        self.unshard_params_by_group: Dict[int, List] = {}
        self.shard_params_by_group: Dict[int, List] = {}
        self.shard_assignments_by_group: Dict[int, List[HSDPGroupAssignment]] = {}

        for group_idx, group in enumerate(self.param_groups):
            assignment: Optional[OptimizerHSDPAssignment] = self.replicate_param_assignment.get(group_idx)

            unshard_params = []
            shard_hsdp_assignments: List[HSDPGroupAssignment] = []

            if assignment:
                unshard_params.extend(assignment.no_comm)
                for hsdp_assign in assignment.hsdp:
                    if hsdp_assign.is_shard:
                        shard_hsdp_assignments.append(hsdp_assign)
                    else:
                        unshard_params.extend(hsdp_assign.owned_params)
            else:
                unshard_params.extend(group["params"])

            shard_params = [p for a in shard_hsdp_assignments for p in a.owned_params]

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

        ns_inputs = {}
        for p in params:
            g = p.grad
            if g is None:
                continue

            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)

            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(g)
            g_updated = g.add(buf, alpha=momentum) if nesterov else buf

            ns_inputs[p] = to_local_if_dtensor(g_updated).bfloat16()

        return ns_inputs

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
        max_numel_per_batch = 400 * 1024 * 1024 // shard_size

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
    ) -> Tuple[Dict[torch.nn.Parameter, torch.Tensor], Dict[torch.nn.Parameter, float]]:
        """Batched Newton-Schulz update for mixed 2D / Conv3D / 3D parameters.

        Normalizes all inputs to 3D, concatenates along dim 0, runs a single
        NS iteration, then slices results back to original shapes.
        """
        updates_dict = {}
        lrs_dict = {}

        if not p_list:
            return updates_dict, lrs_dict

        inputs_3d = []
        slice_sizes = []
        shapes_info = []

        for p in p_list:
            origin_shape = tuple(p.local_shape) if no_shard else tuple(p.shape)
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

        current_idx = 0
        for i, p in enumerate(p_list):
            n_dim = slice_sizes[i]
            origin_shape, is_conv = shapes_info[i]

            update = merged_update[current_idx: current_idx + n_dim]
            current_idx += n_dim

            if is_conv:
                update = update.squeeze(0).unsqueeze(1)
                adj_lr = adjust_lr_wd_for_muon_conv(lr, rms, origin_shape)
            elif len(origin_shape) == 2:
                update = update.squeeze(0)
                adj_lr = adjust_lr_wd_for_muon(lr, rms, origin_shape)
            else:
                adj_lr = adjust_lr_wd_for_muon(lr, rms, origin_shape)

            updates_dict[p] = update
            lrs_dict[p] = adj_lr

        return updates_dict, lrs_dict

    def _apply_local_update(
            self,
            param: torch.nn.Parameter,
            update: torch.Tensor,
            lr: float,
            weight_decay: float,
            adjusted_lr: float,
    ) -> None:
        """Apply weight decay and parameter update in-place."""
        local_param = to_local_if_dtensor(param.data)
        update_to_apply = update.view(local_param.shape)
        local_param.mul_(1 - lr * weight_decay)
        local_param.add_(update_to_apply, alpha=-adjusted_lr)

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
                updates_dict, lrs_dict = self._compute_batched_ns_updates(
                    sub_batch, param_to_ns_input, lr, rms, ns_steps, no_shard=True
                )

                for p in sub_batch:
                    self._apply_local_update(p, updates_dict[p], lr, weight_decay, lrs_dict[p])

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

    def _process_shard_params(
            self,
            group: Dict[str, Any],
            param_to_ns_input: Dict[torch.nn.Parameter, torch.Tensor],
            hsdp_assignments: List[HSDPGroupAssignment],
    ) -> None:
        """Process sharded params with shard-group de-duplication and broadcast.

        Flow per HSDP group:
          1. Allgather along each shard axis to reconstruct full tensor.
          2. De-duplicate: assign each batch to one shard-rank for NS computation.
          3. Multi-dim relay broadcast to distribute results.
          4. Chunk update back to local shard and apply.
        """
        platform = get_platform()
        device = torch.npu.current_device()

        lr = group["lr"]
        weight_decay = group["weight_decay"]
        rms = group["matched_adamw_rms"]
        ns_steps = group["ns_steps"]

        for hsdp_assign in hsdp_assignments:
            valid_params = [p for p in hsdp_assign.owned_params if p in param_to_ns_input]
            if not valid_params:
                continue

            shard_sizes, local_coords, shard_pgs, total_shard_size = self._get_shard_info(hsdp_assign)
            shape_groups = self._group_by_shape(valid_params)

            for _, p_list in shape_groups.items():
                safe_batches = self._split_into_memory_safe_batches(p_list, shard_size=total_shard_size)

                for batch_idx, sub_batch in enumerate(safe_batches):
                    # Round-robin assign this batch to a shard-rank coordinate
                    compute_flat = batch_idx % total_shard_size
                    compute_coord = _unflatten_coord(compute_flat, shard_sizes)
                    is_compute_rank = local_coords == compute_coord

                    # 1. Allgather + de-duplication
                    gathered_inputs = {}
                    for p in sub_batch:
                        local_inp = param_to_ns_input[p]
                        full_inp = allgather_dtensor_param(local_inp, shard_pgs, hsdp_assign.layout_spec)

                        if is_compute_rank:
                            gathered_inputs[p] = full_inp
                        else:
                            del full_inp

                    # 2. Compute NS only on the assigned rank
                    updates_to_broadcast: Dict[torch.nn.Parameter, torch.Tensor] = {}
                    lrs_to_broadcast: Dict[torch.nn.Parameter, torch.Tensor] = {}

                    if is_compute_rank:
                        updates_dict, lrs_dict = self._compute_batched_ns_updates(
                            sub_batch, gathered_inputs, lr, rms, ns_steps
                        )
                        for p in sub_batch:
                            updates_to_broadcast[p] = updates_dict[p].contiguous()
                            lrs_to_broadcast[p] = torch.tensor(
                                lrs_dict[p], dtype=torch.float64, device=device
                            )

                    # 3. Multi-dim relay broadcast
                    for p in sub_batch:
                        if not is_compute_rank:
                            global_shape = tuple(p.shape)
                            updates_to_broadcast[p] = torch.empty(
                                global_shape, dtype=torch.bfloat16, device=device
                            )
                            lrs_to_broadcast[p] = torch.empty((), dtype=torch.float64, device=device)

                        if total_shard_size > 1:
                            self._relay_broadcast(
                                updates_to_broadcast[p],
                                lrs_to_broadcast[p],
                                shard_pgs,
                                shard_sizes,
                                local_coords,
                                compute_coord,
                                platform,
                            )

                        # 4. Chunk back to local shard and apply
                        final_update = updates_to_broadcast[p]
                        adjusted_lr = lrs_to_broadcast[p].item()

                        update_to_apply = chunk_update_by_layout(final_update, p, hsdp_assign.layout_spec)
                        self._apply_local_update(p, update_to_apply, lr, weight_decay, adjusted_lr)

    @staticmethod
    def _relay_broadcast(
            update_tensor: torch.Tensor,
            lr_tensor: torch.Tensor,
            shard_pgs: Tuple[dist.ProcessGroup, ...],
            shard_sizes: Tuple[int, ...],
            local_coords: Tuple[int, ...],
            compute_coord: Tuple[int, ...],
            platform: Any,
    ) -> None:
        """Multi-dimensional relay broadcast within shard group.

        Broadcasts from compute_coord to all ranks, dimension by dimension.
        A rank participates in dimension d only if its coordinates in all
        higher dimensions (d+1, d+2, ...) match compute_coord.
        """
        for dim_idx, pg in enumerate(shard_pgs):
            if pg is None or shard_sizes[dim_idx] <= 1:
                continue

            # Only ranks aligned with compute_coord in higher dims participate
            aligned = all(
                local_coords[sub_dim] == compute_coord[sub_dim]
                for sub_dim in range(dim_idx + 1, len(shard_pgs))
            )
            if not aligned:
                continue

            src_rank_in_pg = compute_coord[dim_idx]
            global_src_rank = platform.get_global_rank(pg, src_rank_in_pg)

            dist.broadcast(update_tensor, src=global_src_rank, group=pg)
            dist.broadcast(lr_tensor, src=global_src_rank, group=pg)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[float]:
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            The loss value if closure is provided, otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group_idx, group in enumerate(self.param_groups):
            group['step'] = (group.get('step') or 0) + 1

            unshard_params = self.unshard_params_by_group.get(group_idx, [])
            shard_params = self.shard_params_by_group.get(group_idx, [])
            shard_hsdp_assignments = self.shard_assignments_by_group.get(group_idx, [])

            unshard_ns_inputs = self._update_muon_momentum(group, unshard_params)
            shard_ns_inputs = self._update_muon_momentum(group, shard_params)

            if unshard_ns_inputs:
                self._process_unshard_params(group, unshard_ns_inputs)

            if shard_ns_inputs and shard_hsdp_assignments:
                self._process_shard_params(group, shard_ns_inputs, shard_hsdp_assignments)

        self._broadcast_replicate_params_after_step()

        return loss
