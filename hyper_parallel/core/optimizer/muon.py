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
from collections import defaultdict, deque
from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import

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
    prepare_muon_gather,
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


_MUON_ADVANCED_DEFAULTS: Dict[str, Any] = {
    "ns_coefficients": None,
    "ns_epsilon": 1e-10,
    "zeropower_fn": None,
    "momentum_update_fn": None,
    "reshape_fn": None,
    "ns_transform_fn": None,
    "post_update_fn": None,
    "zero_rms_scale_mode": "zero",
    "apply_lr_in_update": False,
    "fsdp_overlap": False,
    "fsdp_overlap_memory_budget_bytes": None,
}


@dataclass(frozen=True)
class _MuonAdvancedOptions:
    """Validated low-frequency Muon options."""

    ns_coefficients: Optional[Sequence[Tuple[float, float, float]]]
    ns_epsilon: float
    zeropower_fn: Optional[Callable[..., torch.Tensor]]
    momentum_update_fn: Optional[Callable[..., torch.Tensor]]
    reshape_fn: Optional[Callable[..., Any]]
    ns_transform_fn: Optional[Callable[..., Optional[NSInputTransform]]]
    post_update_fn: Optional[Callable[..., None]]
    zero_rms_scale_mode: str
    apply_lr_in_update: bool
    fsdp_overlap: bool
    fsdp_overlap_memory_budget_bytes: Optional[int]


def _resolve_muon_advanced_options(options: Dict[str, Any]) -> _MuonAdvancedOptions:
    """Validate and fill low-frequency Muon keyword options."""
    unknown = options.keys() - _MUON_ADVANCED_DEFAULTS.keys()
    if unknown:
        raise TypeError(f"Muon got unexpected keyword arguments: {', '.join(sorted(unknown))}")
    resolved = dict(_MUON_ADVANCED_DEFAULTS)
    resolved.update(options)
    return _MuonAdvancedOptions(**resolved)


_TransformNSState = Dict[str, Any]


def zeropower_via_newtonschulz5(
        ns_inputs: torch.Tensor,
        steps: int,
        ns_variant: str = "asym5",
        epsilon: float = 1e-10,
        ns_coefficients: Optional[Sequence[Tuple[float, float, float]]] = None,
) -> torch.Tensor:
    """Newton-Schulz orthogonalization with preallocated matmul buffers.

    Args:
        ns_inputs: Matrices to orthogonalize.
        steps: Number of Newton-Schulz iterations.
        ns_variant: Coefficient schedule name.
        epsilon: Normalization stability term.
        ns_coefficients: Optional custom coefficient schedule.
    """
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
    """Compute Muon scale from the logical matrix dims of a reshaped slice.

    Args:
        slice_tensor: Logical update slice.
        matched_adamw_rms: Target update RMS.
        zero_rms_scale_mode: Behavior when the target RMS is zero.
    """
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

    ADDITIONAL_CONFIG_KEYS = set(_MUON_ADVANCED_DEFAULTS)

    def __init__(
            self,
            params: Any,
            lr: float = 2e-2,
            weight_decay: float = 0.1,
            matched_adamw_rms: float = 0.2,
            momentum: Union[float, List[float], Tuple[float, ...]] = 0.95,
            nesterov: bool = True,
            ns_steps: int = 5,
            ns_variant: str = "asym5",
            hsdp_replica_count: Optional[Union[int, Tuple[int, ...]]] = None,
            **advanced_options: Any,
    ) -> None:
        """Initialize Muon and build parameter-identity runtime caches.

        Args:
            params: Parameters or parameter groups optimized by Muon.
            lr: Learning rate.
            weight_decay: Decoupled weight-decay coefficient.
            matched_adamw_rms: RMS scale matched to an AdamW update.
            momentum: Momentum coefficient or coefficient pair.
            nesterov: Whether to use Nesterov momentum.
            ns_steps: Newton-Schulz iteration count.
            ns_variant: Newton-Schulz coefficient variant.
            ns_coefficients: Optional custom Newton-Schulz coefficients.
            ns_epsilon: Numerical stability term used during normalization.
            zeropower_fn: Optional Newton-Schulz implementation override, also supported with
                fsdp_overlap. The callback must compute independently of other parameters and return
                a tensor with the input shape, dtype, and device. It runs on the caller's compute
                stream; work on internal streams must be joined to that stream before returning.
            momentum_update_fn: Optional momentum update implementation override.
            reshape_fn: Optional logical-matrix reshape function.
            ns_transform_fn: Optional Newton-Schulz input transformation.
            post_update_fn: Optional callback applied after a parameter update.
            zero_rms_scale_mode: Scaling behavior when matched AdamW RMS is zero.
            apply_lr_in_update: Whether the update callback applies the learning rate.
            hsdp_replica_count: Optional optimizer-state replica group size.
            fsdp_overlap: Pipeline single-axis FSDP communication with Newton-Schulz.
            fsdp_overlap_memory_budget_bytes: Estimated temporary storage budget for queued
                buckets and prepared inputs. An explicit budget also controls serial buckets for fair comparisons.
                The estimate uses native NS workspace; reserve extra memory for custom zeropower_fn workspace.
        """
        advanced = _resolve_muon_advanced_options(advanced_options)
        if not isinstance(advanced.fsdp_overlap, bool):
            raise TypeError("fsdp_overlap must be a bool")
        budget = advanced.fsdp_overlap_memory_budget_bytes
        if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0):
            raise ValueError("fsdp_overlap_memory_budget_bytes must be a positive integer")
        self.fsdp_overlap = advanced.fsdp_overlap
        self.fsdp_overlap_memory_budget_bytes = budget
        self._overlap_streams = {}
        self._pipeline_plans = {}
        if ns_variant not in ("legacy", "asym5", "custom"):
            raise ValueError(
                f"ns_variant must be 'legacy', 'asym5', or 'custom', got {ns_variant!r}"
            )
        if advanced.ns_epsilon < 0.0 or not math.isfinite(advanced.ns_epsilon):
            raise ValueError(f"ns_epsilon must be a finite non-negative value, got {advanced.ns_epsilon}")
        if advanced.zero_rms_scale_mode not in ("zero", "use_lr"):
            raise ValueError(
                "zero_rms_scale_mode must be 'zero' or 'use_lr', "
                f"got {advanced.zero_rms_scale_mode!r}"
            )
        normalized_coefficients = self._validate_ns_coefficients(
            ns_variant, ns_steps, advanced.ns_coefficients
        )
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
            "ns_epsilon": advanced.ns_epsilon,
            "zero_rms_scale_mode": advanced.zero_rms_scale_mode,
            "apply_lr_in_update": advanced.apply_lr_in_update,
        }
        super().__init__(params, defaults, is_muon=True, hsdp_replica_count=hsdp_replica_count)
        self.reshape_fn = advanced.reshape_fn
        self.zeropower_fn = advanced.zeropower_fn
        self.momentum_update_fn = advanced.momentum_update_fn
        self.ns_transform_fn = advanced.ns_transform_fn
        self.post_update_fn = advanced.post_update_fn
        self.reset_optimizer_parameters()

    def reset_optimizer_parameters(self) -> None:
        """Rebuild all parameter-identity caches from current param groups."""
        self._overlap_streams.clear()
        self._pipeline_plans = {}
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
        self._validate_overlap_configuration()

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

    def __str__(self) -> str:
        """Return the optimizer representation."""
        return super().__repr__()

    __repr__ = __str__

    @torch.no_grad()
    def step(
            self,
            closure: Optional[Callable[[], Any]] = None,
    ) -> Optional[float]:
        """Perform a single optimization step.

        De-duplication is controlled by the caller: ``param_to_ns_input`` should already contain only the owned
        params (via ``hsdp_assign.owned_params``). The caller is responsible for broadcasting the updated params to
        replica peers via ``AsyncReplicateBroadcaster.flush_group``.

        Args:
            closure: Optional callable that reevaluates the model and returns loss.

        Returns:
            The closure loss when provided, otherwise ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        broadcaster = AsyncReplicateBroadcaster(self)

        num_groups = len(self.param_groups)

        for group in self.param_groups:
            group['step'] = (group.get('step') or 0) + 1

        no_comm_ns = self._prepare_no_comm_momentum()

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

        local_work = self._prepare_local_work(group_linear_batches, no_comm_ns)

        # Process batches: compute momentum per-batch for owned params only (HSDP de-duplication).
        for batch_idx in range(max_num_batches):
            for group_idx in range(num_groups):
                linear_batches = group_linear_batches[group_idx]
                if batch_idx >= len(linear_batches):
                    continue

                hsdp_assign = linear_batches[batch_idx]
                group = self.param_groups[group_idx]

                # Compute momentum for this assignment's owned params only.
                if self._can_overlap_shard(hsdp_assign):
                    self._process_shard_params_pipelined(group, hsdp_assign, group_idx, local_work.get(group_idx))
                    continue
                self._process_serial_assignment(group, hsdp_assign, group_idx, broadcaster)

        for group_idx, work in local_work.items():
            while work["batches"]:
                self._run_local_ns_batch(self.param_groups[group_idx], work)
        broadcaster.wait_all()
        return loss

    def _process_serial_assignment(self, group, hsdp_assign, group_idx, broadcaster):
        """Compute and broadcast one assignment on the caller stream."""
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

    def _prepare_no_comm_momentum(self):
        """Prepare only the local inputs that are independent of shard communication."""
        no_comm_ns: Dict[int, Dict] = {}
        for group_idx, group in enumerate(self.param_groups):
            info = self._hsdp_assignment_batches.get(group_idx)
            if not info:
                # No HSDP info — all params are no_comm.
                unshard_params = self.unshard_params_by_group.get(group_idx, [])
                no_comm_ns[group_idx] = self._update_muon_momentum(
                    group, unshard_params
                )
            else:
                no_comm_params = info.get("no_comm", [])
                if no_comm_params:
                    no_comm_ns[group_idx] = self._update_muon_momentum(
                        group, no_comm_params
                    )
                else:
                    no_comm_ns[group_idx] = {}

        return no_comm_ns

    @staticmethod
    def _local_workspace_bytes(batches):
        """Reserve storage for the largest local NS batch."""
        return max((32 * sum(to_local_if_dtensor(param.data).numel() for param in params)
                    for params, _ in batches), default=0)

    def _prepare_local_work(self, group_linear_batches, no_comm_ns):
        """Move local NS batches into communication windows when overlap is eligible."""
        local_work = {}
        # Expert-axis shards live in local assignments, not necessarily in no_comm.
        for group_idx, assignments in group_linear_batches.items():
            group = self.param_groups[group_idx]
            if any(self._can_overlap_shard(assignment) for assignment in assignments):
                batches = [(list(batch), batch) for batch in self._unshard_ns_batches(no_comm_ns[group_idx])]
                remaining = []
                for assignment in assignments:
                    if not assignment.is_shard and not assignment.is_replicated:
                        params = [param for param in assignment.owned_params if param.grad is not None]
                        for shape_group in self._group_by_shape(params).values():
                            batches.extend((batch, None) for batch in
                                           self._split_into_memory_safe_batches(shape_group, shard_size=1))
                    else:
                        remaining.append(assignment)
                group_linear_batches[group_idx] = remaining
                local_work[group_idx] = {
                    "batches": deque(batches),
                    "workspace_bytes": self._local_workspace_bytes(batches),
                }
            elif no_comm_ns[group_idx]:
                self._process_unshard_params(group, no_comm_ns[group_idx])
                if self.post_update_fn is not None:
                    self._run_post_update_fn(group, no_comm_ns[group_idx].keys())

        return local_work

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
        self._shard_param_compute_coord = {}

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
                self._shard_param_compute_coord.update(
                    (record.param, compute_by_index[record.index]) for record in hsdp_assign.owned_records
                )

            self.unshard_params_by_group[group_idx] = unshard_params
            self.shard_assignments_by_group[group_idx] = shard_hsdp_assignments
            self.shard_params_by_group[group_idx] = shard_params

    def _update_muon_momentum(
            self,
            group: Dict[str, Any],
            params: List[torch.Tensor],
    ) -> Dict[torch.nn.Parameter, torch.Tensor]:
        """Compute first-order momentum and return bfloat16 NS inputs."""
        momentum = group["momentum"]
        if not isinstance(momentum, (list, tuple)):
            momentum = [momentum]
        if len(momentum) == 1:
            momentum = (momentum[0], momentum[0])
        momentum1, momentum2 = momentum
        # Pre-filter params with valid grads and ensure momentum buffers exist
        valid_params = []
        grads = []
        bufs = []
        for p in params:
            g = p.grad
            if g is None:
                continue
            if "momentum_buffer" not in self.state[p]:
                self.state[p]["momentum_buffer"] = torch.zeros_like(g)
            valid_params.append(p)
            grads.append(g)
            bufs.append(self.state[p]["momentum_buffer"])

        if not valid_params:
            return {}

        grads = [to_local_if_dtensor(grad) for grad in grads]
        bufs = [to_local_if_dtensor(buffer) for buffer in bufs]
        if self.momentum_update_fn is not None:
            custom_updates = list(self.momentum_update_fn(grads, bufs, momentum1, momentum2, group["nesterov"]))
            if len(custom_updates) != len(valid_params):
                raise ValueError(
                    "momentum_update_fn must return one update tensor per input gradient, "
                    f"got {len(custom_updates)} updates for {len(valid_params)} gradients"
                )
            custom_updates = [update.to(torch.bfloat16) for update in custom_updates]
            param_updates = dict(zip(valid_params, custom_updates))
            return param_updates

        # Match muon_update_core():
        # m_for_update = grad + momentum1 * m_old
        # m_new = grad + momentum2 * m_old
        # pylint: disable=protected-access
        local_us = torch._foreach_add(grads, bufs, alpha=momentum1)

        torch._foreach_mul_(bufs, momentum2)
        torch._foreach_add_(bufs, grads)

        if group["nesterov"]:
            torch._foreach_mul_(local_us, momentum1)
            torch._foreach_add_(local_us, grads)

        if local_us[0].dtype != torch.bfloat16:
            local_us = [update.to(torch.bfloat16) for update in local_us]

        param_updates = dict(zip(valid_params, local_us))
        return param_updates

    def _unshard_ns_batches(self, param_to_ns_input):
        """Yield original local NS batches, excluding empty local shards."""
        inputs = {param: value for param, value in param_to_ns_input.items() if value.numel() > 0}
        for params in self._group_by_shape(list(inputs)).values():
            for batch in self._split_into_memory_safe_batches(params, shard_size=1):
                yield {param: inputs[param] for param in batch}

    def _run_local_ns_batch(self, group, work):
        """Fill one communication window on the caller's compute stream."""
        if work is not None and work["batches"]:
            with torch.profiler.record_function("muon.pipeline.local_ns"):
                params, inputs = work["batches"].popleft()
                if inputs is None:
                    inputs = self._update_muon_momentum(group, params)
                self._process_unshard_params(group, inputs)

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

        my_updates = self._compute_shard_ns_updates(my_params, gathered_inputs, group, total_shard_size)
        return my_updates, param_compute_coord

    def _compute_shard_ns_updates(self, my_params, gathered_inputs, group, total_shard_size):
        """Keep identical shape grouping and NS numerics in serial and pipelined execution."""
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

        return my_updates

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
                if len(shard_pgs) == 1 and not hsdp_assign.is_replicated:
                    safe_batches = self._shard_pipeline_batches(valid_params, total_shard_size)
                else:
                    safe_batches = self._split_into_memory_safe_batches(
                        valid_params, shard_size=total_shard_size, min_batch_size=total_shard_size,
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

    def _validate_overlap_configuration(self):
        """Reject rank-divergent schedules once, before any optimizer collective is submitted."""
        checked = set()
        for assignments in self.shard_assignments_by_group.values():
            for assignment in assignments:
                if assignment.is_replicated or len(assignment.shard_pgs) != 1:
                    continue
                pg = assignment.shard_pgs[0]
                if pg is None or pg in checked:
                    continue
                checked.add(pg)
                device = to_local_if_dtensor(assignment.owned_params[0].data).device
                config = torch.tensor(
                    [int(self.fsdp_overlap), self.fsdp_overlap_memory_budget_bytes or -1,
                     int(self._can_overlap_shard(assignment))], dtype=torch.int64, device=device,
                )
                gathered = torch.empty(dist.get_world_size(pg) * 3, dtype=torch.int64, device=device)
                dist.all_gather_into_tensor(gathered, config, group=pg)
                values = gathered.view(-1, 3).cpu()
                if not torch.equal(values, values[0].expand_as(values)):
                    raise ValueError("Muon FSDP overlap settings and callback eligibility must agree across ranks")

    def _pipeline_parameter_bytes(self, param, shard_size):
        """Estimate native NS and padded communication storage for one parameter."""
        meta = self._param_shard_metadata[param]
        padded_numel = math.prod(meta.stage_metas[0].pad_shape)
        padded_numel = ((padded_numel + 255) // 256) * 256
        return 32 * param.numel() + 2 * padded_numel * (shard_size + 1) + 1024

    def _can_overlap_shard(self, assignment):
        """Allow local NS overrides while retaining single-axis FSDP scheduling."""
        if not self.fsdp_overlap or not assignment.is_shard or assignment.is_replicated:
            return False
        if len(assignment.shard_pgs) != 1 or assignment.shard_pgs[0] is None:
            return False
        if any(callback is not None for callback in (
                self.reshape_fn, self.momentum_update_fn,
                self.ns_transform_fn, self.post_update_fn)):
            return False
        params = assignment.owned_params
        return bool(params) and to_local_if_dtensor(params[0].data).device.type in ("cuda", "npu")

    def _pipeline_prepared_bytes(self, params):
        """Reserve BF16 inputs, padding, and prepacked sends for the assignment."""
        return sum(
            6 * (((math.prod(self._param_shard_metadata[param].stage_metas[0].pad_shape) + 255) // 256) * 256)
            for param in params
        )

    def _pipeline_bucket_bytes(self, params, shard_size):
        """Bound shared communication storage plus the largest compute-owner workspace."""
        owners = getattr(self, "_shard_param_compute_coord", {})
        if any(param not in owners for param in params):
            return sum(self._pipeline_parameter_bytes(param, shard_size) for param in params)
        workspace = defaultdict(int)
        communication = 0
        for param in params:
            workspace[owners[param]] += 32 * param.numel()
            padded = math.prod(self._param_shard_metadata[param].stage_metas[0].pad_shape)
            padded = ((padded + 255) // 256) * 256
            # Every rank retains a full gather and full owner-broadcast buffers.
            communication += 4 * padded * shard_size + 2 * padded + 1024
        return communication + max(workspace.values(), default=0)

    def _shard_pipeline_batches(self, params, shard_size, *, use_overlap_budget=False):
        """Keep owner waves intact when two buckets and prepared inputs fit the budget."""
        batches = self._split_into_memory_safe_batches(
            params, shard_size=shard_size, min_batch_size=shard_size,
        )
        budget = self.fsdp_overlap_memory_budget_bytes
        if budget is None:
            if not use_overlap_budget:
                return batches
            budget = 512 * 1024 * 1024
        capacity = max(0, (budget - self._pipeline_prepared_bytes(params)) // 2)
        owners = getattr(self, "_shard_param_compute_coord", {})
        storage = {}
        for param in params:
            if param in owners:
                workspace = 32 * param.numel()
                storage[param] = (owners[param], workspace,
                                  self._pipeline_bucket_bytes([param], shard_size) - workspace)
            else:
                storage[param] = (None, 0, self._pipeline_parameter_bytes(param, shard_size))
        result = []
        for batch in batches:
            current, workspace, communication = [], defaultdict(int), 0
            for param in batch:
                owner, ns_bytes, comm_bytes = storage[param]
                workspace[owner] += ns_bytes
                communication += comm_bytes
                if current and communication + max(workspace.values()) > capacity:
                    result.append(current)
                    current, workspace, communication = [], defaultdict(int), comm_bytes
                    workspace[owner] = ns_bytes
                current.append(param)
            if current:
                result.append(current)
        return result

    def _prepare_pipeline_buckets(self, batches, assignment, compute_coord, ns_inputs):
        """Cache static plans without enqueueing device work for later buckets."""
        key = (id(assignment), tuple(tuple(params) for params in batches))
        plans = self._pipeline_plans.get(key)
        if plans is None:
            _, local_coords, _, _ = self._get_shard_info(assignment)
            indices = {record.param: record.index for record in assignment.owned_records}
            plans = []
            for params in batches:
                coords = {param: compute_coord[indices[param]] for param in params}
                plans.append({
                    "params": params, "coords": coords,
                    "keep": {index for index, param in enumerate(params) if coords[param] == local_coords},
                    "stage_metas": [self._param_shard_metadata[param].stage_metas[0] for param in params],
                    "gather_plan": {},
                })
            self._pipeline_plans[key] = plans
        return [dict(plan, ns_inputs=ns_inputs, shard_metadata=self._param_shard_metadata) for plan in plans]

    @staticmethod
    def _launch_pipeline_gather(bucket, assignment, cache, previous, comm_stream):
        """Pack just this bucket on the communication stream, then launch its gather."""
        if previous is not None:
            comm_stream.wait_event(previous["ns_done"])
        if "prepared" not in bucket:
            local_inputs = build_pad_ns_inputs(bucket["params"], bucket["ns_inputs"], bucket["shard_metadata"])
            # Buckets retain every input until the caller stream has joined the last apply.
            # This keeps compute-stream allocations alive across communication reads.
            bucket["prepared"] = prepare_muon_gather(
                local_inputs, assignment.shard_pgs[0], assignment.layout_spec.shard_axes[0][1],
                bucket["stage_metas"], bucket["keep"], cache, bucket["gather_plan"],
            )
        bucket["pending"] = bucket["prepared"].launch(assignment.shard_pgs[0], cache)

    @staticmethod
    def _retire_pipeline_bucket(bucket):
        """Bound queued NS workspace after communication has been submitted."""
        if bucket is not None:
            bucket["ns_done"].synchronize()

    def _enqueue_pipeline_ns(self, bucket, group, shard_size, backend, compute_stream):
        """Wait for this gather only, preserving overlap with earlier broadcasts."""
        with torch.profiler.record_function("muon.pipeline.ns"):
            gathered = bucket["pending"].finish()
            inputs = {param: tensor for param, tensor in zip(bucket["params"], gathered) if tensor is not None}
            bucket["pending"].gathered_view.record_stream(compute_stream)
            bucket["updates"] = self._compute_shard_ns_updates(list(inputs), inputs, group, shard_size)
            bucket["ns_done"] = backend.Event()
            bucket["ns_done"].record(compute_stream)

    def _enqueue_pipeline_broadcast(self, bucket, assignment, backend, comm_stream):
        """Submit broadcasts after output production without waiting on the compute stream."""
        with backend.stream(comm_stream), torch.profiler.record_function("muon.pipeline.broadcast"):
            comm_stream.wait_event(bucket["ns_done"])
            if bucket["reuse_after"] is not None:
                comm_stream.wait_event(bucket["reuse_after"])
            for update in bucket["updates"].values():
                update.record_stream(comm_stream)
            active_coords = set(bucket["coords"].values())
            for key in tuple(bucket["cache"]):
                if key[1] not in active_coords:
                    del bucket["cache"][key]
            packed = self._pack_shard_updates(
                bucket["params"], bucket["updates"], bucket["coords"], assignment, bucket["cache"],
            )
            bucket["packed"] = packed
            pg = assignment.shard_pgs[0]
            bucket["works"] = [
                dist.broadcast(tensor, src=dist.get_global_rank(pg, coord[0]), group=pg, async_op=True)
                for coord, tensor in packed[0].items()
            ]

    def _enqueue_pipeline_apply(self, bucket, assignment, group, backend, update_stream):
        """Consume broadcast outputs on the update stream, leaving NS independent."""
        with backend.stream(update_stream), torch.profiler.record_function("muon.pipeline.apply"):
            for work in bucket["works"]:
                work.wait()
            for tensor in bucket["packed"][0].values():
                tensor.record_stream(update_stream)
            self._apply_shard_updates(bucket["packed"], group, assignment)
            bucket["apply_done"] = backend.Event()
            bucket["apply_done"].record(update_stream)

    def _process_shard_params_pipelined(self, group, assignment, group_idx, local_work=None):
        """Run G0, G1, B0, G2, B1 with two communication slots and budgeted host lookahead."""
        shard_size = dist.get_world_size(assignment.shard_pgs[0])
        batches = self._shard_pipeline_batches(assignment.owned_params, shard_size, use_overlap_budget=True)
        budget = self.fsdp_overlap_memory_budget_bytes or 512 * 1024 * 1024
        prepared_bytes = self._pipeline_prepared_bytes(assignment.owned_params)
        peak_bucket_bytes = max(self._pipeline_bucket_bytes(batch, shard_size) for batch in batches)
        if len(batches) < 2 or prepared_bytes + 2 * peak_bucket_bytes > budget:
            if local_work is not None:
                while local_work["batches"]:
                    self._run_local_ns_batch(group, local_work)
            inputs = self._update_muon_momentum(group, assignment.owned_params)
            self._process_shard_params(group, inputs, [assignment], group_idx, buffer_cache={})
            return
        if local_work is not None:
            local_reserve = local_work["workspace_bytes"]
            if prepared_bytes + 2 * peak_bucket_bytes + local_reserve <= budget:
                prepared_bytes += local_reserve
            else:
                while local_work["batches"]:
                    self._run_local_ns_batch(group, local_work)
                local_work = None
        queue_depth = min(len(batches), (budget - prepared_bytes) // peak_bucket_bytes)
        device = to_local_if_dtensor(batches[0][0].data).device
        backend = getattr(torch, device.type)
        compute_stream = backend.current_stream(device)
        if device not in self._overlap_streams:
            self._overlap_streams[device] = (backend.Stream(device=device), backend.Stream(device=device))
        comm_stream, update_stream = self._overlap_streams[device]
        inputs = self._update_muon_momentum(group, assignment.owned_params)
        buckets = self._prepare_pipeline_buckets(batches, assignment, self._shard_compute_coord[group_idx], inputs)
        ready = backend.Event()
        ready.record(compute_stream)
        comm_stream.wait_event(ready)
        update_stream.wait_event(ready)
        gather_caches = [{}, {}]
        broadcast_caches = [{}, {}]
        try:
            self._run_pipeline_buckets(
                buckets, assignment, group, shard_size, backend, compute_stream,
                comm_stream, update_stream, gather_caches, broadcast_caches, queue_depth, local_work,
            )
        except Exception:
            # Drain while cross-stream storage remains alive if a step fails midway.
            backend.synchronize(device)
            raise

    def _run_pipeline_buckets(
            self, buckets, assignment, group, shard_size, backend, compute_stream,
            comm_stream, update_stream, gather_caches, broadcast_caches, queue_depth, local_work,
    ):
        """Submit the gather, compute, and apply stages within the workspace budget."""
        with backend.stream(comm_stream), torch.profiler.record_function("muon.pipeline.gather"):
            self._launch_pipeline_gather(buckets[0], assignment, gather_caches[0], None, comm_stream)
        self._run_local_ns_batch(group, local_work)
        for index, current in enumerate(buckets):
            self._enqueue_pipeline_ns(current, group, shard_size, backend, compute_stream)
            if index:
                self._enqueue_pipeline_apply(buckets[index - 1], assignment, group, backend, update_stream)
            if index + 1 < len(buckets):
                next_slot = (index + 1) % 2
                previous = buckets[index - 1] if index else None
                with backend.stream(comm_stream), torch.profiler.record_function("muon.pipeline.gather"):
                    self._launch_pipeline_gather(
                        buckets[index + 1], assignment, gather_caches[next_slot], previous, comm_stream,
                    )
            current["cache"] = broadcast_caches[index % 2]
            current["reuse_after"] = buckets[index - 2]["apply_done"] if index >= 2 else None
            self._enqueue_pipeline_broadcast(current, assignment, backend, comm_stream)
            self._run_local_ns_batch(group, local_work)
            # Keep host submission ahead within the workspace budget; the two
            # communication slots retain their independent device-side reuse events.
            if queue_depth - 1 <= index < len(buckets) - 1:
                retired = buckets[index - queue_depth + 1]
                self._retire_pipeline_bucket(retired)
                for key in ("pending", "updates", "packed", "works"):
                    retired.pop(key, None)
        self._enqueue_pipeline_apply(buckets[-1], assignment, group, backend, update_stream)
        compute_stream.wait_event(buckets[-1]["apply_done"])

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
            if reshaped_input.untyped_storage().data_ptr() != working_input.untyped_storage().data_ptr():
                raise ValueError("reshape_fn must return views that share storage with the working NS input tensor")

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
            """Write reshaped NS updates back into the views and the working input.

            Args:
                updates: Orthogonalized updates for the logical matrix views.
                output: Destination for the reconstructed update.
            """
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

        state = {"inputs_3d": [], "slice_sizes": [], "shapes_info": [], "restored_updates": [], "current_idx": 0}

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

            state["inputs_3d"].append(inp_3d)
            state["slice_sizes"].append(n_dim)
            state["shapes_info"].append((origin_shape, is_conv))

        merged_input = torch.cat(state["inputs_3d"], dim=0)

        if merged_input.shape[0] == 1:
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

        if merged_input.dim() == 2:
            merged_update = merged_update.unsqueeze(0)
        del merged_input

        for n_dim, (origin_shape, is_conv) in zip(state["slice_sizes"], state["shapes_info"]):
            update = merged_update[state["current_idx"]:state["current_idx"] + n_dim]
            state["current_idx"] += n_dim

            if is_conv:
                update = update.squeeze(0).unsqueeze(1)
            elif len(origin_shape) == 2:
                update = update.squeeze(0)
            elif len(origin_shape) >= 4:
                update = update.reshape(origin_shape)

            state["restored_updates"].append(update)

        del merged_update
        return state["restored_updates"]

    def _split_into_memory_safe_batches(
            self,
            p_list: List[torch.nn.Parameter],
            shard_size: int = 1,
            min_batch_size: int = 1,
    ) -> List[List[torch.nn.Parameter]]:
        """Split parameters into memory-safe batches to prevent OOM during NS.

        The per-batch element limit is scaled down by shard_size to account
        for the memory amplification from allgather. Sharded paths prioritize
        filling the compute-owner ranks before applying this soft limit.
        """
        max_numel_per_batch = 512 * 1024 * 1024 // shard_size
        min_batch_size = max(1, min_batch_size)

        batches = []
        current_batch = []
        current_count = 0

        for p in p_list:
            p_count = p.numel()
            if (
                    len(current_batch) >= min_batch_size
                    and current_count + p_count > max_numel_per_batch
            ):
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
        if not p_list:
            return {}
        state = {"reshape_groups": defaultdict(list), "origin_shapes": {}, "working_inputs": {}, "restored": {}}
        for param in p_list:
            local_shape = getattr(param, "local_shape", None)
            if local_shape is None:
                local_shape = to_local_if_dtensor(param.data).shape
            state["origin_shapes"][param] = tuple(local_shape) if no_shard else tuple(param.shape)
            working_input, reshaped_inputs = self._reshape_ns_input(
                param, ns_inputs[param].view(state["origin_shapes"][param])
            )
            state["working_inputs"][param] = working_input
            for reshaped_input in reshaped_inputs:
                state["reshape_groups"][self._shape_to_core_shape(tuple(reshaped_input.shape))].append(
                    reshaped_input
                )
        for tensor_list in state["reshape_groups"].values():
            updates = self._compute_batched_ns_outputs_for_tensors(
                tensor_list, group["ns_steps"], group["ns_variant"], group["ns_coefficients"], group["ns_epsilon"]
            )
            for reshaped_input, update in zip(tensor_list, updates):
                scale = compute_muon_slice_scale(
                    update, group["matched_adamw_rms"], zero_rms_scale_mode=group["zero_rms_scale_mode"]
                )
                if group["apply_lr_in_update"]:
                    scale *= -group["lr"]
                update.mul_(scale)
                reshaped_input.copy_(update.contiguous().view_as(reshaped_input))

        for param in p_list:
            update = ns_inputs[param].view(state["origin_shapes"][param])
            working_input = state["working_inputs"][param]
            if working_input.untyped_storage().data_ptr() != update.untyped_storage().data_ptr():
                update.copy_(working_input)
            state["restored"][param] = update
        return state["restored"]

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

        if not p_list:
            return {}

        state: _TransformNSState = {
            "reshape_groups": defaultdict(list),
            "origin_shapes": {},
            "working_inputs": {},
            "transforms": {},
            "transform_updates": [],
            "restored": {},
        }

        for p in p_list:
            state["origin_shapes"][p] = (
                tuple(to_local_if_dtensor(p.data).shape) if no_shard else tuple(p.shape)
            )
            prepared = self._prepare_ns_transform(p, ns_inputs[p].view(state["origin_shapes"][p]))
            state["working_inputs"][p] = prepared[0]
            state["transforms"][p] = prepared[1]
            state["transform_updates"].append([None] * len(prepared[1].tensors))

            for tensor_index, transformed_input in enumerate(prepared[1].tensors):
                state["reshape_groups"][self._shape_to_core_shape(tuple(transformed_input.shape))].append(
                    (transformed_input, len(state["transform_updates"]) - 1, tensor_index)
                )

        for tensor_records in state["reshape_groups"].values():
            reshaped_updates = self._compute_batched_ns_outputs_for_tensors(
                [record[0] for record in tensor_records],
                group["ns_steps"],
                ns_variant=group["ns_variant"],
                ns_coefficients=group["ns_coefficients"],
                ns_epsilon=group["ns_epsilon"],
            )

            # Scale updates.
            for record, reshaped_update in zip(tensor_records, reshaped_updates):
                slice_scale = compute_muon_slice_scale(
                    reshaped_update,
                    group["matched_adamw_rms"],
                    zero_rms_scale_mode=group["zero_rms_scale_mode"],
                )

                if group["apply_lr_in_update"]:
                    slice_scale *= -group["lr"]
                reshaped_update.mul_(slice_scale)
                state["transform_updates"][record[1]][record[2]] = reshaped_update

        for p, reshaped_updates in zip(p_list, state["transform_updates"]):

            if any(update is None for update in reshaped_updates):
                raise RuntimeError("Missing Newton-Schulz update for a transformed input")
            state["transforms"][p].restore(reshaped_updates, state["working_inputs"][p])

        for p in p_list:
            reshaped_update = ns_inputs[p].view(state["origin_shapes"][p])

            if (state["working_inputs"][p].untyped_storage().data_ptr()
                    != reshaped_update.untyped_storage().data_ptr()):
                reshaped_update.copy_(state["working_inputs"][p])
            state["restored"][p] = reshaped_update

        return state["restored"]

    def _fused_broadcast_and_apply(
            self,
            valid_params: List[torch.nn.Parameter],
            my_updates: Dict[torch.nn.Parameter, torch.Tensor],
            param_compute_coord: Dict[torch.nn.Parameter, Tuple[int, ...]],
            group: Dict[str, Any],
            hsdp_assign: HSDPGroupAssignment,
    ) -> None:
        """Fused broadcast and apply for shard-group updates."""
        packed = self._pack_shard_updates(valid_params, my_updates, param_compute_coord, hsdp_assign, {})
        shard_sizes, local_coords, shard_pgs, total_shard_size = self._get_shard_info(hsdp_assign)
        if total_shard_size > 1:
            self._batched_relay_broadcast(packed[0], shard_pgs, shard_sizes, local_coords)
        self._apply_shard_updates(packed, group, hsdp_assign)

    def _pack_shard_updates(self, valid_params, my_updates, param_compute_coord, hsdp_assign, buffer_cache):
        """Pack complete owner updates using slot-local communication storage."""
        _, local_coords, _, _ = self._get_shard_info(hsdp_assign)
        device = to_local_if_dtensor(valid_params[0].data).device

        coord_groups: Dict[Tuple[int, ...], List[torch.nn.Parameter]] = defaultdict(list)
        for p in valid_params:
            coord_groups[param_compute_coord[p]].append(p)

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
            cache_key = ("broadcast", coord)
            pack_buffer = buffer_cache.get(cache_key)
            if pack_buffer is None or pack_buffer.numel() < total_padded_numel:
                pack_buffer = torch.empty(total_padded_numel, dtype=torch.bfloat16, device=device)
                buffer_cache[cache_key] = pack_buffer
            pack_buffer = pack_buffer[:total_padded_numel]

            if is_compute_rank:
                for p, (offset, actual_numel, padded_numel) in zip(coord_params, param_offsets):
                    update = my_updates[p]
                    pack_buffer[offset:offset + actual_numel].copy_(update.reshape(-1))
                    if padded_numel > actual_numel:
                        pack_buffer[offset + actual_numel:offset + padded_numel].zero_()

            pack_buffers[coord] = pack_buffer

        return pack_buffers, coord_groups, coord_param_offsets

    def _apply_shard_updates(self, packed, group, hsdp_assign):
        """Apply received updates on the caller's stream after communication has completed."""
        pack_buffers, coord_groups, coord_param_offsets = packed
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        all_local_params = []
        all_update_shards = []
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
