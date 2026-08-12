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
from collections.abc import Callable, Iterable
from collections import defaultdict
from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist

from hyper_parallel.core.optimizer.optimizer import AsyncReplicateBroadcaster, BaseDistributedOptimizer
from hyper_parallel.core.optimizer.dtensor_compat import to_local_if_dtensor
from hyper_parallel.core.optimizer.sharding_category import (
    HSDPGroupAssignment,
    ParamShardMeta,
    build_owner_by_size,
)
from hyper_parallel.core.optimizer.muon_shard import (
    _debug_param_shard_metadata,
    build_pad_ns_inputs,
    build_param_shard_metadata_for_group,
    chunk_update_by_layout,
    fused_allgather_dtensor_params,
)

logger = logging.getLogger(__name__)

# Legacy: single-coefficient quintic NS (Keller Jordan / Moonlight).
# Same (a, b, c) applied every step; typically 5 steps.
_NS_LEGACY_COEFFS: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315)

# Asym5: 5-step asymmetric NS (dion). Each step uses different coefficients.
_NS_ASYM5_COEFFS: Tuple[Tuple[float, float, float], ...] = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)


@dataclass
class NSInputTransform:
    """Prepared NS tensors and the callback that restores their updates."""

    tensors: List[torch.Tensor]
    restore: Callable[[List[torch.Tensor], torch.Tensor], None]


@dataclass(frozen=True)
class MuonPostUpdateContext:
    """Context supplied to an optional parameter post-update callback."""

    param_fqn: Optional[str]
    logical_shape: Tuple[int, ...]
    lr: float
    weight_decay: float
    step: int


def zeropower_via_newtonschulz5(
        ns_inputs: torch.Tensor,
        steps: int,
        ns_variant: str = "asym5",
        epsilon: float = 1e-10,
        ns_coefficients: Optional[Sequence[Tuple[float, float, float]]] = None,
) -> torch.Tensor:
    """Newton-Schulz orthogonalization with preallocated matmul buffers."""
    mat_x = ns_inputs
    transposed = ns_inputs.size(-2) > ns_inputs.size(-1)
    if transposed:
        mat_x = mat_x.mT

    # Normalize input before Newton-Schulz iteration.
    mat_x = mat_x / (mat_x.norm(dim=(-2, -1), keepdim=True) + epsilon)

    n_size = mat_x.size(-2)
    buf_a = torch.empty(mat_x.shape[:-2] + (n_size, n_size), dtype=mat_x.dtype, device=mat_x.device)
    buf_b = torch.empty_like(buf_a)
    buf_c = torch.empty(mat_x.shape, dtype=mat_x.dtype, device=mat_x.device)

    if ns_variant == "legacy":
        step_coeffs = [_NS_LEGACY_COEFFS] * steps
    elif ns_variant == "asym5":
        step_coeffs = _NS_ASYM5_COEFFS[:steps]
    elif ns_variant == "custom":
        if ns_coefficients is None:
            raise ValueError("ns_coefficients must be provided when ns_variant='custom'")
        if steps > len(ns_coefficients):
            raise ValueError(
                f"ns_steps={steps} exceeds the {len(ns_coefficients)} provided coefficient groups"
            )
        step_coeffs = tuple(ns_coefficients[:steps])
    else:
        raise ValueError(
            f"ns_variant must be 'legacy', 'asym5', or 'custom', got {ns_variant!r}"
        )

    for coeff_a, coeff_b, coeff_c in step_coeffs:
        torch.matmul(mat_x, mat_x.mT, out=buf_a)
        torch.matmul(buf_a, buf_a, out=buf_b)
        buf_a.mul_(coeff_b)
        buf_b.mul_(coeff_c)
        buf_a.add_(buf_b)
        torch.matmul(buf_a, mat_x, out=buf_c)
        mat_x.mul_(coeff_a).add_(buf_c)

    del buf_a, buf_b, buf_c

    if transposed:
        mat_x = mat_x.mT

    return mat_x


def compute_muon_slice_scale(
        slice_tensor: torch.Tensor,
        matched_adamw_rms: float,
        zero_rms_scale_mode: str = "zero",
) -> float:
    """Compute Muon scale from the logical matrix dims of a reshaped slice."""
    if not matched_adamw_rms:
        return 1.0 if zero_rms_scale_mode == "use_lr" else 0.0
    shape = tuple(slice_tensor.shape)
    if len(shape) == 3 and shape[1] == 1:
        logical_dims = (shape[0], shape[2])
    else:
        logical_dims = shape[-2:]
    return math.sqrt(max(logical_dims)) * matched_adamw_rms


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
            ns_variant: str = "asym5",
            ns_coefficients: Optional[Sequence[Tuple[float, float, float]]] = None,
            ns_epsilon: float = 1e-10,
            zeropower_fn: Optional[Callable[..., torch.Tensor]] = None,
            momentum_update_fn: Optional[Callable[..., torch.Tensor]] = None,
            reshape_fn: Optional[Callable[..., Any]] = None,
            ns_transform_fn: Optional[Callable[..., Optional[NSInputTransform]]] = None,
            post_update_fn: Optional[Callable[..., None]] = None,
            zero_rms_scale_mode: str = "zero",
            apply_lr_in_update: bool = False,
            hsdp_replica_count: Optional[Union[int, Tuple[int, ...]]] = None,
    ):
        if ns_variant not in ("legacy", "asym5", "custom"):
            raise ValueError(
                f"ns_variant must be 'legacy', 'asym5', or 'custom', got {ns_variant!r}"
            )
        if ns_epsilon < 0.0 or not math.isfinite(ns_epsilon):
            raise ValueError(f"ns_epsilon must be a finite non-negative value, got {ns_epsilon}")
        if zero_rms_scale_mode not in ("zero", "use_lr"):
            raise ValueError(
                "zero_rms_scale_mode must be 'zero' or 'use_lr', "
                f"got {zero_rms_scale_mode!r}"
            )
        normalized_coefficients = self._validate_ns_coefficients(ns_variant, ns_steps, ns_coefficients)
        if not isinstance(momentum, (list, tuple)):
            momentum = [momentum]
        if len(momentum) == 1:
            momentum = (momentum[0], momentum[0])
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "matched_adamw_rms": matched_adamw_rms,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "ns_variant": ns_variant,
            "ns_coefficients": normalized_coefficients,
            "ns_epsilon": ns_epsilon,
            "zero_rms_scale_mode": zero_rms_scale_mode,
            "apply_lr_in_update": apply_lr_in_update,
        }
        super().__init__(params, defaults, is_muon=True, hsdp_replica_count=hsdp_replica_count)
        self.reshape_fn = reshape_fn
        self.zeropower_fn = zeropower_fn
        self.momentum_update_fn = momentum_update_fn
        self.ns_transform_fn = ns_transform_fn
        self.post_update_fn = post_update_fn

        self._group_dtensor_by_mesh()
        self._build_param_shard_metadata()
        deduced_count = self._auto_deduce_replica_count()
        if deduced_count is None:
            self.hsdp_replica_count = None
        elif self.hsdp_replica_count is None:
            self.hsdp_replica_count = deduced_count
        self._split_replicate_groups()
        self._build_hsdp_batch()
        self._build_param_broadcast_info()
        self._classify_parameters_for_step()

    @staticmethod
    def _validate_ns_coefficients(
            ns_variant: str,
            ns_steps: int,
            ns_coefficients: Optional[Sequence[Tuple[float, float, float]]],
    ) -> Optional[Tuple[Tuple[float, float, float], ...]]:
        """Validate and normalize externally supplied Newton-Schulz coefficients."""
        if ns_variant != "custom":
            if ns_coefficients is not None:
                raise ValueError("ns_coefficients can only be provided when ns_variant='custom'")
            return None
        if not ns_coefficients:
            raise ValueError("ns_coefficients must be provided when ns_variant='custom'")
        if ns_steps > len(ns_coefficients):
            raise ValueError(
                f"ns_steps={ns_steps} exceeds the {len(ns_coefficients)} provided coefficient groups"
            )

        normalized = []
        for index, coefficients in enumerate(ns_coefficients):
            if len(coefficients) != 3:
                raise ValueError(
                    f"ns_coefficients[{index}] must contain exactly three values, got {coefficients!r}"
                )
            normalized_coefficients = tuple(float(value) for value in coefficients)
            if not all(math.isfinite(value) for value in normalized_coefficients):
                raise ValueError(f"ns_coefficients[{index}] must contain only finite values")
            normalized.append(normalized_coefficients)
        return tuple(normalized)

    def __str__(self):
        return super().__repr__()

    __repr__ = __str__

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
                group = self.param_groups[group_idx]
                self._process_unshard_params(group, no_comm_ns[group_idx])
                if self.post_update_fn is not None:
                    self._run_post_update_fn(group, no_comm_ns[group_idx].keys())

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
                if self.post_update_fn is not None:
                    self._run_post_update_fn(group, ns_inputs.keys())

                # Flush broadcasts after each assignment.
                broadcaster.flush_group(hsdp_assign)

        broadcaster.wait_all()
        return loss

    def _run_post_update_fn(
            self,
            group: Dict[str, Any],
            params: Iterable[torch.nn.Parameter],
    ) -> None:
        """Run the optional model callback on updated owner parameters before broadcast."""
        if self.post_update_fn is None:
            return
        context_values = {
            "lr": group["lr"],
            "weight_decay": group["weight_decay"],
            "step": group["step"],
        }
        for param in params:
            local_tensor = to_local_if_dtensor(param.data)
            context = MuonPostUpdateContext(
                param_fqn=getattr(param, "model_name", None),
                logical_shape=tuple(param.shape),
                **context_values,
            )
            self.post_update_fn(param, local_tensor, context)

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
        momentum1, momentum2 = group["momentum"]
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

        if self.momentum_update_fn is not None:
            local_grads = [to_local_if_dtensor(grad) for grad in grads]
            local_bufs = [to_local_if_dtensor(buffer) for buffer in bufs]
            custom_updates = list(self.momentum_update_fn(
                local_grads,
                local_bufs,
                momentum1,
                momentum2,
                nesterov,
            ))
            if len(custom_updates) != len(valid_params):
                raise ValueError(
                    "momentum_update_fn must return one update tensor per input gradient, "
                    f"got {len(custom_updates)} updates for {len(valid_params)} gradients"
                )
            custom_updates = [update.to(torch.bfloat16) for update in custom_updates]
            return dict(zip(valid_params, custom_updates))

        # Match muon_update_core():
        # m_for_update = grad + momentum1 * m_old
        # m_new = grad + momentum2 * m_old
        local_grads = [to_local_if_dtensor(g) for g in grads]
        local_bufs = [to_local_if_dtensor(b) for b in bufs]
        # pylint: disable=protected-access
        local_us = torch._foreach_add(local_grads, local_bufs, alpha=momentum1)

        torch._foreach_mul_(local_bufs, momentum2)
        torch._foreach_add_(local_bufs, local_grads)

        if nesterov:
            torch._foreach_mul_(local_us, momentum1)
            torch._foreach_add_(local_us, local_grads)

        if local_us[0].dtype == torch.bfloat16:
            local_us_bf = local_us
        else:
            local_us = list(local_us)
            for i, u in enumerate(local_us):
                local_us[i] = u.to(torch.bfloat16)
            local_us_bf = local_us

        return dict(zip(valid_params, local_us_bf))

    def _process_unshard_params(
            self,
            group: Dict[str, Any],
            param_to_ns_input: Dict[torch.nn.Parameter, torch.Tensor],
    ) -> None:
        """Process un-sharded params: shape-group -> memory-batch -> NS -> local update."""
        lr = group["lr"]
        weight_decay = group["weight_decay"]

        # empty params are not processed in muon, for uneven shard
        param_to_ns_input = {
            p: ns_input for p, ns_input in param_to_ns_input.items() if ns_input.numel() > 0
        }
        if not param_to_ns_input:
            return

        shape_groups = self._group_by_shape(list(param_to_ns_input.keys()))
        for _, p_list in shape_groups.items():
            safe_batches = self._split_into_memory_safe_batches(p_list, shard_size=1)

            for sub_batch in safe_batches:
                updates_dict = self._compute_batched_ns_updates(
                    sub_batch, param_to_ns_input, group, no_shard=True
                )

                local_params = [to_local_if_dtensor(p.data) for p in sub_batch]
                local_updates = [updates_dict[p].view(lp.shape) for p, lp in zip(sub_batch, local_params)]

                if weight_decay != 0.0:
                    # pylint: disable=protected-access
                    torch._foreach_mul_(local_params, 1 - lr * weight_decay)
                # pylint: disable=protected-access
                apply_alpha = 1.0 if group["apply_lr_in_update"] else -lr
                torch._foreach_add_(local_params, local_updates, alpha=apply_alpha)

    def _gather_and_compute_shard_updates(
            self,
            valid_params: List[torch.nn.Parameter],
            param_to_ns_input: Dict[torch.nn.Parameter, torch.Tensor],
            hsdp_assign: HSDPGroupAssignment,
            shard_compute_coord: Dict[int, Tuple[int, ...]],
            group: Dict[str, Any],
            buffer_cache: Optional[Dict],
    ) -> Tuple[
        Dict[torch.nn.Parameter, torch.Tensor],
        Dict[torch.nn.Parameter, Tuple[int, ...]],
    ]:
        """Gather NS inputs and compute updates for locally-assigned shard params.

        Returns:
            (my_updates, param_compute_coord) for this HSDP assignment.
        """
        shard_sizes, local_coords, shard_pgs, total_shard_size = self._get_shard_info(hsdp_assign)
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
        local_inputs = build_pad_ns_inputs(
            valid_params,
            param_to_ns_input,
            self._param_shard_metadata,
        )
        param_shard_metadata = [self._param_shard_metadata.get(p) for p in valid_params]
        gathered_list = fused_allgather_dtensor_params(
            local_inputs, shard_pgs, hsdp_assign.layout_spec,
            param_shard_metadata=param_shard_metadata,
            buffer_cache=buffer_cache,
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
                    updates_dict = self._compute_batched_ns_updates(
                        sub_batch, gathered_inputs, group
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
        shard_compute_coord = self._shard_compute_coord.get(group_idx, {})

        for hsdp_assign in hsdp_assignments:
            owned_params = hsdp_assign.owned_params
            # Communication membership is static per assignment. Ranks with empty
            # local shards still participate via build_pad_ns_inputs().
            valid_params = owned_params

            shard_sizes, _, shard_pgs, total_shard_size = self._get_shard_info(hsdp_assign)

            # uneven shard check
            shard_pgs_rank = []
            for pg in shard_pgs:
                pg_rank = torch.distributed.get_process_group_ranks(pg)
                shard_pgs_rank.append(pg_rank)

            for p in valid_params:
                if tuple(p.shape)[0] % shard_sizes[0] != 0:
                    logger.debug_rank0(
                        "[hyper-optimizer uneven-shard] p %s placement %s device_mesh %s "
                        "local_shape %s global_shape %s pg_rank %s shard_sizes %s",
                        p.model_name,
                        p.placements,
                        p.device_mesh,
                        p.to_local().shape,
                        p.shape,
                        shard_pgs_rank,
                        shard_sizes,
                    )

            if valid_params:
                safe_batches = self._split_into_memory_safe_batches(
                    valid_params, shard_size=total_shard_size,
                )
                for sub_batch in safe_batches:
                    my_updates, param_compute_coord = self._gather_and_compute_shard_updates(
                        sub_batch, param_to_ns_input, hsdp_assign,
                        shard_compute_coord, group, buffer_cache,
                    )

                    self._fused_broadcast_and_apply(
                        sub_batch, my_updates, param_compute_coord,
                        group, hsdp_assign,
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
            core_shape = self._shape_to_core_shape(tuple(p.shape))
            groups[core_shape].append(p)

        return groups

    @staticmethod
    def _shape_to_core_shape(shape: Tuple[int, ...]) -> Tuple[int, int]:
        if len(shape) == 2:
            return (shape[0], shape[1])
        if len(shape) == 3 and shape[1] == 1:
            return (shape[0], shape[2])
        if len(shape) >= 3:
            return (shape[-2], shape[-1])
        raise ValueError('1D parameters are not supported in Muon')

    def _reshape_ns_input(
            self,
            param: torch.nn.Parameter,
            ns_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Return the contiguous NS input and any reshape views used for NS."""
        working_input = ns_input if ns_input.is_contiguous() else ns_input.contiguous()
        if self.reshape_fn is None:
            return working_input, [working_input]

        param_fqn = param.model_name
        if param_fqn is None:
            return working_input, [working_input]

        reshaped_inputs = list(self.reshape_fn(param_fqn, working_input))
        if not reshaped_inputs:
            return working_input, [working_input]

        if reshaped_inputs[0].shape != working_input.shape:
            logger.info_rank0(
                "Reshape %s from %s to %s",
                param_fqn,
                working_input.shape,
                reshaped_inputs[0].shape,
            )

        for reshaped_input in reshaped_inputs:
            assert (
                    reshaped_input.untyped_storage().data_ptr() == working_input.untyped_storage().data_ptr()
            ), "reshape_fn must return views that share storage with the working NS input tensor."

        return working_input, reshaped_inputs

    def _prepare_ns_transform(
            self,
            param: torch.nn.Parameter,
            ns_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, NSInputTransform]:
        """Route one parameter through custom, zero-copy, or identity NS preparation."""
        working_input = ns_input if ns_input.is_contiguous() else ns_input.contiguous()
        param_fqn = getattr(param, "model_name", None)
        if self.ns_transform_fn is not None and param_fqn is not None:
            transform = self.ns_transform_fn(param_fqn, working_input)
            if transform is not None:
                self._validate_ns_transform(transform, working_input)
                return working_input, transform

        _, reshaped_inputs = self._reshape_ns_input(param, working_input)

        def restore_view_updates(updates: List[torch.Tensor], output: torch.Tensor) -> None:
            """Write reshaped NS updates back into the views and the working input."""
            for reshaped_input, update in zip(reshaped_inputs, updates):
                reshaped_input.copy_(update.contiguous().view_as(reshaped_input))
            if output.untyped_storage().data_ptr() != working_input.untyped_storage().data_ptr():
                output.copy_(working_input)

        return working_input, NSInputTransform(tensors=reshaped_inputs, restore=restore_view_updates)

    @staticmethod
    def _validate_ns_transform(transform: NSInputTransform, working_input: torch.Tensor) -> None:
        """Validate a model-provided reversible NS transform."""
        if not isinstance(transform, NSInputTransform):
            raise ValueError(
                "ns_transform_fn must return NSInputTransform or None, "
                f"got {type(transform).__name__}"
            )
        if not transform.tensors:
            raise ValueError("NSInputTransform.tensors must not be empty")
        for index, tensor in enumerate(transform.tensors):
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"NSInputTransform.tensors[{index}] must be a torch.Tensor")
            if tensor.device != working_input.device:
                raise ValueError("NSInputTransform tensors must be on the same device as the NS input")
            if tensor.dtype != working_input.dtype:
                raise ValueError("NSInputTransform tensors must have the same dtype as the NS input")
            if tensor.dim() < 2:
                raise ValueError("NSInputTransform tensors must have at least two dimensions")
        if not callable(transform.restore):
            raise ValueError("NSInputTransform.restore must be callable")

    def _compute_batched_ns_outputs_for_tensors(
            self,
            tensor_list: List[torch.Tensor],
            ns_steps: int,
            ns_variant: str = "asym5",
            ns_coefficients: Optional[Sequence[Tuple[float, float, float]]] = None,
            ns_epsilon: float = 1e-10,
    ) -> List[torch.Tensor]:
        """Run batched NS on mixed-shape tensors and restore their original shapes."""
        if not tensor_list:
            return []

        inputs_3d = []
        slice_sizes = []
        shapes_info = []

        for tensor in tensor_list:
            origin_shape = tuple(tensor.shape)
            is_conv = False
            if len(origin_shape) == 2:
                inp_3d = tensor.unsqueeze(0)
                n_dim = 1
            elif len(origin_shape) == 3 and origin_shape[1] == 1:
                inp_3d = tensor.squeeze(1).unsqueeze(0)
                is_conv = True
                n_dim = 1
            elif len(origin_shape) == 3:
                inp_3d = tensor
                n_dim = origin_shape[0]
            else:
                inp_3d = tensor.reshape(-1, origin_shape[-2], origin_shape[-1])
                n_dim = inp_3d.shape[0]

            inputs_3d.append(inp_3d)
            slice_sizes.append(n_dim)
            shapes_info.append((origin_shape, is_conv))

        merged_input = torch.cat(inputs_3d, dim=0)
        squeeze_batch = merged_input.shape[0] == 1
        if squeeze_batch:
            merged_input = merged_input.squeeze(0)

        if self.zeropower_fn is None:
            merged_update = zeropower_via_newtonschulz5(
                merged_input,
                steps=ns_steps,
                ns_variant=ns_variant,
                epsilon=ns_epsilon,
                ns_coefficients=ns_coefficients,
            )
        else:
            merged_update = self.zeropower_fn(merged_input, steps=ns_steps)
        del merged_input

        if squeeze_batch:
            merged_update = merged_update.unsqueeze(0)

        outputs = []
        current_idx = 0
        for n_dim, (origin_shape, is_conv) in zip(slice_sizes, shapes_info):
            update = merged_update[current_idx: current_idx + n_dim]
            current_idx += n_dim

            if is_conv:
                update = update.squeeze(0).unsqueeze(1)
            elif len(origin_shape) == 2:
                update = update.squeeze(0)
            elif len(origin_shape) >= 4:
                update = update.reshape(origin_shape)

            outputs.append(update)

        del merged_update
        return outputs

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
            group: Dict[str, Any],
            no_shard: bool = False
    ) -> Dict[torch.nn.Parameter, torch.Tensor]:
        """Route batched NS through the native or reversible-transform path."""
        if self.ns_transform_fn is None:
            return self._compute_batched_ns_updates_fast(p_list, ns_inputs, group, no_shard)
        return self._compute_batched_ns_updates_with_transform(p_list, ns_inputs, group, no_shard)

    def _compute_batched_ns_updates_fast(
            self,
            p_list: List[torch.nn.Parameter],
            ns_inputs: Dict[torch.nn.Parameter, torch.Tensor],
            group: Dict[str, Any],
            no_shard: bool = False,
    ) -> Dict[torch.nn.Parameter, torch.Tensor]:
        """Compute native reshape/view NS updates without transform bookkeeping."""
        updates_dict = {}
        if not p_list:
            return updates_dict

        reshape_groups: Dict[Tuple[int, int], List[torch.Tensor]] = defaultdict(list)
        origin_shapes: Dict[torch.nn.Parameter, Tuple[int, ...]] = {}
        working_inputs: Dict[torch.nn.Parameter, torch.Tensor] = {}

        for param in p_list:
            local_shape = getattr(param, "local_shape", None)
            if local_shape is None:
                local_shape = to_local_if_dtensor(param.data).shape
            origin_shape = tuple(local_shape) if no_shard else tuple(param.shape)
            ns_input = ns_inputs[param].view(origin_shape)
            origin_shapes[param] = origin_shape
            working_input, reshaped_inputs = self._reshape_ns_input(param, ns_input)
            working_inputs[param] = working_input
            for reshaped_input in reshaped_inputs:
                core_shape = self._shape_to_core_shape(tuple(reshaped_input.shape))
                reshape_groups[core_shape].append(reshaped_input)

        for tensor_list in reshape_groups.values():
            reshaped_updates = self._compute_batched_ns_outputs_for_tensors(
                tensor_list,
                group["ns_steps"],
                ns_variant=group["ns_variant"],
                ns_coefficients=group["ns_coefficients"],
                ns_epsilon=group["ns_epsilon"],
            )
            for reshaped_input, reshaped_update in zip(tensor_list, reshaped_updates):
                slice_scale = compute_muon_slice_scale(
                    reshaped_update,
                    group["matched_adamw_rms"],
                    zero_rms_scale_mode=group["zero_rms_scale_mode"],
                )
                if group["apply_lr_in_update"]:
                    slice_scale *= -group["lr"]
                reshaped_update.mul_(slice_scale)
                reshaped_input.copy_(reshaped_update.contiguous().view_as(reshaped_input))

        for param in p_list:
            ns_input = ns_inputs[param].view(origin_shapes[param])
            working_input = working_inputs[param]
            if working_input.untyped_storage().data_ptr() != ns_input.untyped_storage().data_ptr():
                ns_input.copy_(working_input)
            updates_dict[param] = ns_input
        return updates_dict

    def _compute_batched_ns_updates_with_transform(
            self,
            p_list: List[torch.nn.Parameter],
            ns_inputs: Dict[torch.nn.Parameter, torch.Tensor],
            group: Dict[str, Any],
            no_shard: bool = False,
    ) -> Dict[torch.nn.Parameter, torch.Tensor]:
        """Batched Newton-Schulz update for mixed 2D / Conv3D / 3D parameters.

        Normalizes all inputs to 3D, concatenates along dim 0, runs a single
        NS iteration, then slices results back to original shapes.

        """
        updates_dict = {}

        if not p_list:
            return updates_dict

        rms = group["matched_adamw_rms"]
        ns_steps = group["ns_steps"]
        ns_variant = group["ns_variant"]
        ns_coefficients = group["ns_coefficients"]
        ns_epsilon = group["ns_epsilon"]
        zero_rms_scale_mode = group["zero_rms_scale_mode"]
        apply_lr_in_update = group["apply_lr_in_update"]

        reshape_groups: Dict[Any, List[Tuple[torch.Tensor, int, int]]] = defaultdict(list)
        origin_shapes: Dict[torch.nn.Parameter, Tuple[int, ...]] = {}
        working_inputs: Dict[torch.nn.Parameter, torch.Tensor] = {}
        transforms: List[NSInputTransform] = []
        transform_updates: List[List[Optional[torch.Tensor]]] = []

        for p in p_list:
            origin_shape = tuple(to_local_if_dtensor(p.data).shape) if no_shard else tuple(p.shape)
            ns_input = ns_inputs[p].view(origin_shape)
            origin_shapes[p] = origin_shape

            working_input, transform = self._prepare_ns_transform(p, ns_input)
            working_inputs[p] = working_input
            transform_index = len(transforms)
            transforms.append(transform)
            transform_updates.append([None] * len(transform.tensors))
            for tensor_index, transformed_input in enumerate(transform.tensors):
                core_shape = self._shape_to_core_shape(tuple(transformed_input.shape))
                reshape_groups[core_shape].append((transformed_input, transform_index, tensor_index))

        for _, tensor_records in reshape_groups.items():
            tensor_list = [record[0] for record in tensor_records]
            reshaped_updates = self._compute_batched_ns_outputs_for_tensors(
                tensor_list,
                ns_steps,
                ns_variant=ns_variant,
                ns_coefficients=ns_coefficients,
                ns_epsilon=ns_epsilon,
            )

            # scale updates
            for (_, transform_index, tensor_index), reshaped_update in zip(tensor_records, reshaped_updates):
                slice_scale = compute_muon_slice_scale(
                    reshaped_update,
                    rms,
                    zero_rms_scale_mode=zero_rms_scale_mode,
                )
                if apply_lr_in_update:
                    slice_scale *= -group["lr"]
                reshaped_update.mul_(slice_scale)
                transform_updates[transform_index][tensor_index] = reshaped_update

        for transform, updates, working_input in zip(transforms, transform_updates, working_inputs.values()):
            if any(update is None for update in updates):
                raise RuntimeError("Missing Newton-Schulz update for a transformed input")
            transform.restore(updates, working_input)

        for p in p_list:
            ns_input = ns_inputs[p].view(origin_shapes[p])
            working_input = working_inputs[p]
            if working_input.untyped_storage().data_ptr() != ns_input.untyped_storage().data_ptr():
                ns_input.copy_(working_input)
            updates_dict[p] = ns_input

        return updates_dict

    def _fused_broadcast_and_apply(
            self,
            valid_params: List[torch.nn.Parameter],
            my_updates: Dict[torch.nn.Parameter, torch.Tensor],
            param_compute_coord: Dict[torch.nn.Parameter, Tuple[int, ...]],
            group: Dict[str, Any],
            hsdp_assign: HSDPGroupAssignment,
    ) -> None:
        """Fused broadcast and apply for shard-group updates."""
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        shard_sizes, local_coords, shard_pgs, total_shard_size = self._get_shard_info(hsdp_assign)
        device = to_local_if_dtensor(valid_params[0].data).device

        coord_groups: Dict[Tuple[int, ...], List[torch.nn.Parameter]] = defaultdict(list)
        for p in valid_params:
            coord_groups[param_compute_coord[p]].append(p)

        all_local_params: List[torch.Tensor] = []
        all_update_shards: List[torch.Tensor] = []

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

        if total_shard_size > 1:
            self._batched_relay_broadcast(
                pack_buffers, shard_pgs, shard_sizes, local_coords
            )

        layout_spec = hsdp_assign.layout_spec
        for coord, coord_params in coord_groups.items():
            pack_buffer = pack_buffers[coord]
            param_offsets = coord_param_offsets[coord]

            for p, (offset, actual_numel, _) in zip(coord_params, param_offsets):
                full_update = pack_buffer[offset:offset + actual_numel].view(tuple(p.shape))
                update_to_apply = chunk_update_by_layout(
                    full_update,
                    p,
                    layout_spec,
                    self._param_shard_metadata.get(p),
                )

                local_param = to_local_if_dtensor(p.data)
                all_local_params.append(local_param)
                all_update_shards.append(update_to_apply.view(local_param.shape))

        if not all_local_params:
            return

        if weight_decay != 0.0:
            coeff = 1.0 - lr * weight_decay
            # pylint: disable=protected-access
            torch._foreach_mul_(all_local_params, coeff)

        # Slice-wise Muon scaling has already been applied during NS postprocess.
        # pylint: disable=protected-access
        apply_alpha = 1.0 if group["apply_lr_in_update"] else -lr
        torch._foreach_add_(all_local_params, all_update_shards, alpha=apply_alpha)

    def _build_param_shard_metadata(self) -> None:
        """Build shard metadata once during optimizer init."""
        self._param_shard_metadata: Dict[torch.nn.Parameter, ParamShardMeta] = {}

        for _, hsdp_groups in self._hsdp_grouping.values():
            for hsdp_group in hsdp_groups:
                if hsdp_group.layout_spec is None or not hsdp_group.layout_spec.shard_axes:
                    continue
                group_param_to_meta = build_param_shard_metadata_for_group(hsdp_group)
                for param, shard_meta in group_param_to_meta.items():
                    self._param_shard_metadata[param] = shard_meta
                _debug_param_shard_metadata(hsdp_group, group_param_to_meta)

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
                global_src_rank = dist.get_global_rank(pg, src_rank_in_pg)

                work = dist.broadcast(tensor, src=global_src_rank, group=pg, async_op=True)
                if work is not None:
                    work_handles.append(work)

            for work in work_handles:
                work.wait()
