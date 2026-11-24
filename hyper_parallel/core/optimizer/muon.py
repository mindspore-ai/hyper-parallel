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
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from functools import partial
from itertools import chain, groupby
import logging
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union

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
    get_or_alloc_buffer,
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


@dataclass(frozen=True)
class _ShardBatch:
    """One batch and its original assignment boundary."""

    group_idx: int
    assignment: HSDPGroupAssignment
    params: List[torch.nn.Parameter]
    last_in_assignment: bool


@dataclass
class _GatheredShardBatch:
    """Gather results owned by the communication stream."""

    batch: _ShardBatch
    inputs: Dict[torch.nn.Parameter, torch.Tensor]
    compute_coords: Dict[torch.nn.Parameter, Tuple[int, ...]]
    ready: Any


_PackedShardUpdates = Dict[
    Tuple[int, ...], Tuple[torch.Tensor, List[Tuple[torch.nn.Parameter, int, int]]]
]
_AssignmentEntry = Tuple[int, HSDPGroupAssignment]
_LocalQueueKey = Tuple[int, torch.device]
_LocalBatches = Deque[List[torch.nn.Parameter]]


@dataclass
class _ShardPipelineBuffers:
    """Scratch storage owned by the communication stream for one pipeline run."""

    gather: Dict = field(default_factory=dict)
    owner_slots: Tuple[Dict, Dict] = field(default_factory=lambda: ({}, {}))
    broadcast: Dict = field(default_factory=dict)


@dataclass
class _BroadcastShardBatch:
    """Broadcast outputs waiting to be consumed by the apply stream."""

    batch: _ShardBatch
    packed: _PackedShardUpdates
    works: List[dist.Work]


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


def _resolve_muon_advanced_options(options: Dict[str, Any]) -> _MuonAdvancedOptions:
    """Validate and fill low-frequency Muon keyword options."""
    unknown = options.keys() - _MUON_ADVANCED_DEFAULTS.keys()
    if unknown:
        raise TypeError(f"Muon got unexpected keyword arguments: {', '.join(sorted(unknown))}")
    resolved = dict(_MUON_ADVANCED_DEFAULTS)
    resolved.update(options)
    return _MuonAdvancedOptions(**resolved)


_TransformNSState = Dict[str, Any]


def _resolve_ns_coefficients(
        steps: int,
        ns_variant: str = "asym5",
        ns_coefficients: Optional[Sequence[Tuple[float, float, float]]] = None,
) -> Tuple[Tuple[float, float, float], ...]:
    """Resolve the coefficient prefix shared by native NS and backend construction."""
    if ns_variant == "legacy":
        return (_NS_LEGACY_COEFFS,) * steps
    if ns_variant == "asym5":
        return _NS_ASYM5_COEFFS[:steps]
    if ns_variant == "custom":
        if ns_coefficients is None:
            raise ValueError("ns_coefficients must be provided when ns_variant='custom'")
        if steps > len(ns_coefficients):
            raise ValueError(
                f"ns_steps={steps} exceeds the {len(ns_coefficients)} provided coefficient groups"
            )
        return tuple(tuple(row) for row in ns_coefficients[:steps])
    raise ValueError(f"ns_variant must be 'legacy', 'asym5', or 'custom', got {ns_variant!r}")


def zeropower_via_newtonschulz5(
        ns_inputs: torch.Tensor,
        steps: int,
        ns_variant: str = "asym5",
        epsilon: float = 1e-10,
        ns_coefficients: Optional[Sequence[Tuple[float, float, float]]] = None,
) -> torch.Tensor:
    """Newton-Schulz orthogonalization with preallocated matmul buffers.

    Args:
        ns_inputs: Matrix or matrix batch to orthogonalize.
        steps: Requested iteration count; asym5 uses at most five iterations.
        ns_variant: Built-in legacy/asym5 schedule or an explicit custom schedule.
        epsilon: Normalization epsilon added to the input norm.
        ns_coefficients: Coefficient triples required by the custom variant.

    Returns:
        Orthogonalized updates with the input shape, dtype and device.
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

    step_coeffs = _resolve_ns_coefficients(steps, ns_variant, ns_coefficients)

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
            momentum_mode: str = "legacy",
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
            zeropower_fn: Optional ``fn(local_tensor, steps=ns_steps)`` override.
                Return a writable tensor with the same shape, dtype and device.
                The callback owns its coefficient schedule and epsilon; these
                group options are not forwarded. Recreate matching callback
                configuration when restoring an optimizer checkpoint.
            momentum_update_fn: Optional momentum update implementation override.
            reshape_fn: Optional logical-matrix reshape function.
            ns_transform_fn: Optional Newton-Schulz input transformation.
            post_update_fn: Optional callback applied after a parameter update.
            fsdp_overlap: Overlap computation and communication on a single
                CUDA or NPU shard axis. Independent local batches can fill
                additional communication windows. Set consistently on all ranks.
                Reshape and NS transform callbacks may run in a different order
                across parameters. Custom momentum and post-update callbacks
                use the serial schedule.
            zero_rms_scale_mode: Scaling behavior when matched AdamW RMS is zero.
            apply_lr_in_update: Whether the update callback applies the learning rate.
            hsdp_replica_count: Optional optimizer-state replica group size.
            momentum_mode: Momentum formula, either "legacy" or "vanilla".
                Both modes support distinct update and buffer coefficients.
        """
        advanced = _resolve_muon_advanced_options(advanced_options)
        if not isinstance(advanced.fsdp_overlap, bool):
            raise TypeError("fsdp_overlap must be a bool")
        if momentum_mode not in ("legacy", "vanilla"):
            raise ValueError(
                f"momentum_mode must be 'legacy' or 'vanilla', got {momentum_mode!r}"
            )
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
        self._update_muon_momentum_mode = momentum_mode
        self.ns_transform_fn = advanced.ns_transform_fn
        self.post_update_fn = advanced.post_update_fn
        self.fsdp_overlap = advanced.fsdp_overlap
        self._shard_pipeline_streams: Dict[torch.device, Tuple[Any, Any]] = {}
        self.reset_optimizer_parameters()

    def reset_optimizer_parameters(self) -> None:
        """Rebuild all parameter-identity caches from current param groups."""
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

        for group in self.param_groups:
            group['step'] = (group.get('step') or 0) + 1

        self._run_step_assignments(broadcaster)
        broadcaster.wait_all()
        return loss

    def _run_step_assignments(self, broadcaster: AsyncReplicateBroadcaster) -> None:
        """Submit shard and independent local batches in one schedule."""
        assignments: Iterable[_AssignmentEntry] = self._step_assignments()
        local_batches: Dict[_LocalQueueKey, _LocalBatches] = {}
        if self.fsdp_overlap and self.momentum_update_fn is None and self.post_update_fn is None:
            assignments, local_batches = self._plan_overlap_batches(assignments)
        self._process_no_comm_params(local_batches)

        for key, run in groupby(assignments, self._shard_pipeline_key):
            if key is None:
                for group_idx, assignment in run:
                    self._process_serial_assignment(group_idx, assignment, broadcaster)
            else:
                device, _, _ = key
                self._process_shard_pipeline(self._iter_shard_batches(run), device, broadcaster, local_batches)
        for (group_idx, _), queue in local_batches.items():
            while queue:
                self._run_local_batch(group_idx, queue)

    def _plan_overlap_batches(
            self, assignments: Iterable[_AssignmentEntry],
    ) -> Tuple[List[_AssignmentEntry], Dict[_LocalQueueKey, _LocalBatches]]:
        """Extract updates that need neither shard collectives nor replica synchronization."""
        entries = list(assignments)
        queues: Dict[_LocalQueueKey, _LocalBatches] = {}
        for entry in entries:
            pipeline_key = self._shard_pipeline_key(entry)
            if pipeline_key is not None:
                queues.setdefault((entry[0], pipeline_key[0]), deque())

        remaining = []
        for group_idx, assignment in entries:
            if not assignment.is_shard and not assignment.is_replicated and assignment.owned_records:
                params = assignment.owned_params
                device = to_local_if_dtensor(params[0]).device
                queue = queues.get((group_idx, device))
                if queue is not None:
                    self._queue_local_batches(params, queue)
                    continue
            remaining.append((group_idx, assignment))
        return remaining, queues

    def _queue_local_batches(self, params: List[torch.nn.Parameter], queue: _LocalBatches) -> None:
        """Plan local work while keeping pending entries free of temporary tensors."""
        for shape_group in self._group_by_shape([param for param in params if param.grad is not None]).values():
            queue.extend(self._split_into_memory_safe_batches(shape_group, shard_size=1))

    def _process_no_comm_params(self, local_batches: Dict[_LocalQueueKey, _LocalBatches]) -> None:
        """Use the same local queue for parameters outside HSDP assignments."""
        no_comm_inputs = []
        for group_idx, group in enumerate(self.param_groups):
            info = self._hsdp_assignment_batches.get(group_idx)
            params = info.get("no_comm", []) if info else self.unshard_params_by_group.get(group_idx, [])
            if local_batches and params:
                eligible = defaultdict(list)
                eager = []
                for param in params:
                    device = to_local_if_dtensor(param).device
                    if (group_idx, device) in local_batches:
                        eligible[device].append(param)
                    else:
                        eager.append(param)
                for device, device_params in eligible.items():
                    self._queue_local_batches(device_params, local_batches[(group_idx, device)])
                params = eager
            no_comm_inputs.append(self._update_muon_momentum(group, params) if params else {})

        for group, ns_inputs in zip(self.param_groups, no_comm_inputs):
            if ns_inputs:
                self._process_unshard_params(group, ns_inputs)
                if self.post_update_fn is not None:
                    self._run_post_update_fn(group, ns_inputs.keys())

    def _run_local_batch(self, group_idx: int, queue: Optional[_LocalBatches]) -> None:
        """Submit one local batch on the caller's compute stream."""
        if queue:
            with torch.profiler.record_function("muon.pipeline.local_ns"):
                group = self.param_groups[group_idx]
                self._process_unshard_params(group, self._update_muon_momentum(group, queue.popleft()))

    def _step_assignments(self) -> Iterator[_AssignmentEntry]:
        """Yield assignments in the original round-robin optimizer group order."""
        by_group = []
        for group_idx in range(len(self.param_groups)):
            batch_groups = self._hsdp_assignment_batches.get(group_idx, {}).get("batch_groups", [])
            by_group.append(list(chain.from_iterable(batch_group["sub_batches"] for batch_group in batch_groups)))
        for batch_idx in range(max(map(len, by_group), default=0)):
            for group_idx, assignments in enumerate(by_group):
                if batch_idx < len(assignments):
                    yield group_idx, assignments[batch_idx]

    def _process_serial_assignment(
            self, group_idx: int, assignment: HSDPGroupAssignment, broadcaster: AsyncReplicateBroadcaster,
            overlap_work: Optional[Callable[[], None]] = None,
    ) -> None:
        """Keep whole-assignment callback inputs and the serial shard path."""
        group = self.param_groups[group_idx]
        ns_inputs = self._update_muon_momentum(group, assignment.owned_params)
        if assignment.is_shard:
            self._process_shard_params(
                group, ns_inputs, [assignment], group_idx, buffer_cache={}, overlap_work=overlap_work,
            )
        elif ns_inputs:
            self._process_unshard_params(group, ns_inputs)
        if self.post_update_fn is not None:
            self._run_post_update_fn(group, ns_inputs.keys())
        broadcaster.flush_group(assignment)

    def _shard_pipeline_key(
            self, entry: Tuple[int, HSDPGroupAssignment],
    ) -> Optional[Tuple[torch.device, Tuple[dist.ProcessGroup, ...], Tuple[dist.ProcessGroup, ...]]]:
        """Only look ahead across consecutive assignments with the same communication groups."""
        _, assignment = entry
        if not self.fsdp_overlap or self.momentum_update_fn is not None or self.post_update_fn is not None:
            return None
        if not assignment.is_shard or len(assignment.shard_pgs) != 1 or not assignment.owned_records:
            return None
        pg = assignment.shard_pgs[0]
        if pg is None or dist.get_world_size(pg) <= 1:
            return None
        device = to_local_if_dtensor(assignment.owned_records[0].param).device
        if device.type not in ("cuda", "npu"):
            return None
        return device, assignment.shard_pgs, assignment.replicate_pgs

    def _iter_shard_batches(self, assignments: Iterable[Tuple[int, HSDPGroupAssignment]]) -> Iterator[_ShardBatch]:
        """Keep the serial shard batch membership and replica flush boundaries."""
        for group_idx, assignment in assignments:
            shard_size = dist.get_world_size(assignment.shard_pgs[0])
            batches = self._split_into_memory_safe_batches(assignment.owned_params, shard_size, shard_size)
            for batch_idx, params in enumerate(batches):
                yield _ShardBatch(group_idx, assignment, params, batch_idx == len(batches) - 1)

    def _prepare_shard_batch(
            self, batch: _ShardBatch, backend: Any, stream: Any, buffers: _ShardPipelineBuffers, slot: int,
    ) -> _GatheredShardBatch:
        """Produce this batch's momentum and gathered inputs on the communication stream."""
        with backend.stream(stream), torch.profiler.record_function("muon.pipeline.gather"):
            ns_inputs = self._update_muon_momentum(self.param_groups[batch.group_idx], batch.params)
            gathered, coords = self._gather_shard_inputs(
                batch.params, ns_inputs, batch.assignment, self._shard_compute_coord[batch.group_idx],
                buffer_cache=buffers.gather, fallback_dtype=torch.bfloat16,
                owner_buffer_cache=buffers.owner_slots[slot],
            )
            ready = backend.Event()
            ready.record(stream)
        return _GatheredShardBatch(batch, gathered, coords, ready)

    def _process_shard_pipeline(
            self, batches: Iterator[_ShardBatch], device: torch.device, broadcaster: AsyncReplicateBroadcaster,
            local_batches: Dict[_LocalQueueKey, _LocalBatches],
    ) -> None:
        """Overlap each batch's NS with its neighbors' collectives using two in-flight batches."""
        first = next(batches)
        second = next(batches, None)
        if second is None:
            queue = local_batches.get((first.group_idx, device))
            overlap_work = partial(self._run_local_batch, first.group_idx, queue) if queue else None
            self._process_serial_assignment(first.group_idx, first.assignment, broadcaster, overlap_work)
            return

        backend = getattr(torch, device.type)
        compute_stream = backend.current_stream(device)
        if device not in self._shard_pipeline_streams:
            self._shard_pipeline_streams[device] = backend.Stream(device=device), backend.Stream(device=device)
        comm_stream, apply_stream = self._shard_pipeline_streams[device]
        comm_stream.wait_stream(compute_stream)
        apply_stream.wait_stream(compute_stream)
        buffers = _ShardPipelineBuffers()
        try:
            apply_done = self._run_shard_pipeline(
                chain((first, second), batches), backend, compute_stream, comm_stream, apply_stream,
                broadcaster, local_batches, device, buffers,
            )
        except Exception:
            # The caller still owns the scratch storage while queued work drains.
            backend.synchronize(device)
            raise
        compute_stream.wait_event(apply_done)
        # The allocator can reuse these communication-stream buffers as soon as this run returns.
        comm_stream.wait_event(apply_done)

    def _run_shard_pipeline(
            self, batches: Iterator[_ShardBatch], backend: Any, compute_stream: Any, comm_stream: Any,
            apply_stream: Any, broadcaster: AsyncReplicateBroadcaster,
            local_batches: Dict[_LocalQueueKey, _LocalBatches],
            device: torch.device, buffers: _ShardPipelineBuffers,
    ) -> Any:
        """Enqueue the stages while keeping the collective order identical across ranks."""
        first = next(batches)
        current = self._prepare_shard_batch(first, backend, comm_stream, buffers, 0)
        self._run_local_batch(first.group_idx, local_batches.get((first.group_idx, device)))
        previous = None
        apply_done = None
        for index, following in enumerate(chain(batches, (None,))):
            with torch.profiler.record_function("muon.pipeline.ns"):
                compute_stream.wait_event(current.ready)
                batch = current.batch
                shard_size = dist.get_world_size(batch.assignment.shard_pgs[0])
                group = self.param_groups[batch.group_idx]
                updates = self._compute_shard_updates(current.inputs, group, shard_size)
                ns_done = backend.Event()
                ns_done.record(compute_stream)

            if previous is not None:
                local_group = previous.batch.group_idx
                # The next shard NS is already queued on the compute stream.
                # Independent local work can cover the rest of the older broadcast.
                self._run_local_batch(local_group, local_batches.get((local_group, device)))
                apply_done = self._apply_pipeline_batch(previous, backend, apply_stream, compute_stream, broadcaster)

            # Put the next gather ahead of this broadcast on the communication
            # stream so the next NS can run while this broadcast is in flight.
            # At this point the target owner buffer's previous broadcast pack is already queued.
            prefetched = self._prepare_shard_batch(
                following, backend, comm_stream, buffers, (index + 1) % 2,
            ) if following else None
            with backend.stream(comm_stream), torch.profiler.record_function("muon.pipeline.broadcast"):
                comm_stream.wait_event(ns_done)
                if apply_done is not None:
                    comm_stream.wait_event(apply_done)
                pg = batch.assignment.shard_pgs[0]
                packed = self._pack_shard_updates(
                    batch.params, updates, current.compute_coords, (dist.get_rank(pg),), buffers.broadcast,
                )
                works = self._launch_shard_axis_broadcasts(packed, pg, 0, ())
            previous = _BroadcastShardBatch(batch, packed, works)
            current = prefetched

        self._run_local_batch(batch.group_idx, local_batches.get((batch.group_idx, device)))
        return self._apply_pipeline_batch(previous, backend, apply_stream, compute_stream, broadcaster)

    def _apply_pipeline_batch(
            self,
            pending: _BroadcastShardBatch,
            backend: Any,
            stream: Any,
            compute_stream: Any,
            broadcaster: AsyncReplicateBroadcaster,
    ) -> Any:
        """Make the apply event the release point for the broadcast slot."""
        batch = pending.batch
        with backend.stream(stream), torch.profiler.record_function("muon.pipeline.apply"):
            for work in pending.works:
                work.wait()
            self._apply_shard_updates(pending.packed, self.param_groups[batch.group_idx], batch.assignment)
            done = backend.Event()
            done.record(stream)
        if batch.last_in_assignment and batch.assignment.is_replicated:
            compute_stream.wait_event(done)
            broadcaster.flush_group(batch.assignment)
        return done

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

        if self._update_muon_momentum_mode == "vanilla":
            # Match vanilla implementation in Muon paper & PyTorch implementation:
            # m_for_update = (1 - momentum1) * grad + momentum1 * m_old
            # m_new = (1 - momentum2) * grad + momentum2 * m_old
            # Fold the Nesterov interpolation into one lerp with momentum1 squared.
            # pylint: disable=protected-access
            update_momentum = momentum1 ** 2 if group["nesterov"] else momentum1
            local_us = torch._foreach_lerp(grads, bufs, weight=update_momentum)
            torch._foreach_lerp_(bufs, grads, weight=1 - momentum2)
        elif self._update_muon_momentum_mode == "legacy":
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
        else:
            raise ValueError(
                "Muon._update_muon_momentum_mode must be 'vanilla' or 'legacy', "
                f"got {self._update_muon_momentum_mode!r}"
            )

        if local_us[0].dtype != torch.bfloat16:
            local_us = [update.to(torch.bfloat16) for update in local_us]

        param_updates = dict(zip(valid_params, local_us))
        return param_updates

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
            overlap_work: Optional[Callable[[], None]] = None,
    ) -> Tuple[
        Dict[torch.nn.Parameter, torch.Tensor],
        Dict[torch.nn.Parameter, Tuple[int, ...]],
    ]:
        """Gather NS inputs and compute updates for locally-assigned shard params.

        Returns:
            (my_updates, param_compute_coord) for this HSDP assignment.
        """
        gathered_inputs, compute_coords = self._gather_shard_inputs(
            valid_params, param_to_ns_input, hsdp_assign, shard_compute_coord, buffer_cache,
            overlap_work=overlap_work,
        )
        shard_size = self._get_shard_info(hsdp_assign)[-1]
        return self._compute_shard_updates(gathered_inputs, group, shard_size), compute_coords

    def _gather_shard_inputs(
            self,
            params: List[torch.nn.Parameter],
            param_to_ns_input: Dict[torch.nn.Parameter, torch.Tensor],
            hsdp_assign: HSDPGroupAssignment,
            shard_compute_coord: Dict[int, Tuple[int, ...]],
            buffer_cache: Optional[Dict],
            fallback_dtype: Optional[torch.dtype] = None,
            overlap_work: Optional[Callable[[], None]] = None,
            owner_buffer_cache: Optional[Dict] = None,
    ) -> Tuple[Dict[torch.nn.Parameter, torch.Tensor], Dict[torch.nn.Parameter, Tuple[int, ...]]]:
        """Gather a fused buffer and retain only locally assigned NS inputs."""
        shard_sizes, local_coords, shard_pgs, _ = self._get_shard_info(hsdp_assign)
        param_to_index = {record.param: record.index for record in hsdp_assign.owned_records}
        param_compute_coord: Dict[torch.nn.Parameter, Tuple[int, ...]] = {}
        my_indices: set = set()

        for idx, p in enumerate(params):
            p_index = param_to_index[p]
            compute_coord = shard_compute_coord.get(p_index, (0,) * len(shard_sizes))
            param_compute_coord[p] = compute_coord
            if compute_coord == local_coords:
                my_indices.add(idx)

        local_inputs = build_pad_ns_inputs(
            params,
            param_to_ns_input,
            self._param_shard_metadata,
            fallback_dtype=fallback_dtype,
        )
        param_shard_metadata = [self._param_shard_metadata.get(p) for p in params]
        gathered_list = fused_allgather_dtensor_params(
            local_inputs, shard_pgs, hsdp_assign.layout_spec,
            param_shard_metadata=param_shard_metadata,
            buffer_cache=buffer_cache,
            keep_indices=my_indices,
            while_gathering=overlap_work,
            owner_buffer_cache=owner_buffer_cache,
        )
        return {p: full_input for p, full_input in zip(params, gathered_list)
                if full_input is not None}, param_compute_coord

    def _compute_shard_updates(
            self,
            gathered_inputs: Dict[torch.nn.Parameter, torch.Tensor],
            group: Dict[str, Any],
            shard_size: int,
    ) -> Dict[torch.nn.Parameter, torch.Tensor]:
        """Use the same NS shape groups for serial and overlapping shard batches."""
        my_updates: Dict[torch.nn.Parameter, torch.Tensor] = {}
        for params in self._group_by_shape(list(gathered_inputs)).values():
            for batch in self._split_into_memory_safe_batches(params, shard_size=shard_size):
                updates = self._compute_batched_ns_updates(batch, gathered_inputs, group)
                my_updates.update(updates)
        return my_updates

    def _process_shard_params(
            self,
            group: Dict[str, Any],
            param_to_ns_input: Dict[torch.nn.Parameter, torch.Tensor],
            hsdp_assignments: List[HSDPGroupAssignment],
            group_idx: int,
            buffer_cache: Optional[Dict] = None,
            overlap_work: Optional[Callable[[], None]] = None,
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
                        getattr(p, "model_name", None),
                        p.placements,
                        p.device_mesh,
                        p.to_local().shape,
                        p.shape,
                        shard_pgs_rank,
                        shard_sizes,
                    )

            if valid_params:
                safe_batches = self._split_into_memory_safe_batches(
                    valid_params, shard_size=total_shard_size, min_batch_size=total_shard_size
                )
                for sub_batch in safe_batches:
                    my_updates, param_compute_coord = self._gather_and_compute_shard_updates(
                        sub_batch, param_to_ns_input, hsdp_assign,
                        shard_compute_coord, group, buffer_cache, overlap_work,
                    )

                    self._fused_broadcast_and_apply(
                        sub_batch, my_updates, param_compute_coord,
                        group, hsdp_assign, overlap_work,
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
            overlap_work: Optional[Callable[[], None]] = None,
    ) -> None:
        """Fused broadcast and apply for shard-group updates."""
        shard_sizes, local_coords, shard_pgs, total_shard_size = self._get_shard_info(hsdp_assign)
        packed = self._pack_shard_updates(valid_params, my_updates, param_compute_coord, local_coords)
        if total_shard_size > 1:
            self._batched_relay_broadcast(packed, shard_pgs, shard_sizes, local_coords, overlap_work)
        self._apply_shard_updates(packed, group, hsdp_assign)

    @staticmethod
    def _pack_shard_updates(
            params: List[torch.nn.Parameter],
            my_updates: Dict[torch.nn.Parameter, torch.Tensor],
            param_compute_coord: Dict[torch.nn.Parameter, Tuple[int, ...]],
            local_coords: Tuple[int, ...],
            buffer_cache: Optional[Dict] = None,
    ) -> _PackedShardUpdates:
        """Pack BF16 updates into one aligned buffer per compute owner."""
        device = to_local_if_dtensor(params[0].data).device
        coord_groups: Dict[Tuple[int, ...], List[torch.nn.Parameter]] = defaultdict(list)
        for p in params:
            coord_groups[param_compute_coord[p]].append(p)

        alignment_elements = 256  # 512 bytes in BF16
        offsets_by_coord = {}
        sizes = []
        for coord, coord_params in coord_groups.items():
            param_offsets = []
            total_padded_numel = 0
            for p in coord_params:
                actual_numel = p.numel()
                padded_numel = ((actual_numel + alignment_elements - 1) // alignment_elements) * alignment_elements
                param_offsets.append((p, total_padded_numel, actual_numel, padded_numel))
                total_padded_numel += padded_numel

            offsets_by_coord[coord] = param_offsets
            sizes.append(total_padded_numel)

        if buffer_cache is None:
            buffers = (torch.empty(size, dtype=torch.bfloat16, device=device) for size in sizes)
        else:
            total_size = sum(sizes)
            buffer = get_or_alloc_buffer(buffer_cache, device, total_size, torch.bfloat16, device)
            buffers = buffer[:total_size].split(sizes)

        packed: _PackedShardUpdates = {}
        for (coord, param_offsets), pack_buffer in zip(offsets_by_coord.items(), buffers):
            if coord == local_coords:
                for p, offset, actual_numel, padded_numel in param_offsets:
                    update = my_updates[p]
                    pack_buffer[offset:offset + actual_numel].view(update.shape).copy_(update)
                    if padded_numel > actual_numel:
                        pack_buffer[offset + actual_numel:offset + padded_numel].zero_()
            packed[coord] = pack_buffer, [(p, offset, count) for p, offset, count, _ in param_offsets]
        return packed

    def _apply_shard_updates(
            self, packed: _PackedShardUpdates, group: Dict[str, Any], hsdp_assign: HSDPGroupAssignment,
    ) -> None:
        """Apply the local shard of each update after its broadcast completes."""
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        all_local_params: List[torch.Tensor] = []
        all_update_shards: List[torch.Tensor] = []
        for pack_buffer, param_offsets in packed.values():
            for p, offset, actual_numel in param_offsets:
                full_update = pack_buffer[offset:offset + actual_numel].view(tuple(p.shape))
                update_to_apply = chunk_update_by_layout(
                    full_update,
                    p,
                    hsdp_assign.layout_spec,
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

    @classmethod
    def _batched_relay_broadcast(
            cls,
            packed: _PackedShardUpdates,
            shard_pgs: Tuple[dist.ProcessGroup, ...],
            shard_sizes: Tuple[int, ...],
            local_coords: Tuple[int, ...],
            overlap_work: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        Batched asynchronous multi-dimensional relay broadcast.
        By operating asynchronously within each dimension, we eliminate CPU overhead bubbles
        while strictly preserving the multidimensional relay dependency.
        """
        for dim_idx, pg in enumerate(shard_pgs):
            if pg is None or shard_sizes[dim_idx] <= 1:
                continue

            works = cls._launch_shard_axis_broadcasts(packed, pg, dim_idx, local_coords[dim_idx + 1:])
            try:
                if works and overlap_work is not None:
                    overlap_work()
            finally:
                for work in works:
                    work.wait()

    @staticmethod
    def _launch_shard_axis_broadcasts(
            packed: _PackedShardUpdates,
            pg: dist.ProcessGroup,
            dim_idx: int,
            downstream_coords: Tuple[int, ...],
    ) -> List[dist.Work]:
        """Submit the buffers whose source is present on this relay axis."""
        works = []
        for coord, (tensor, _) in packed.items():
            if coord[dim_idx + 1:] != downstream_coords:
                continue
            src = dist.get_global_rank(pg, coord[dim_idx])
            work = dist.broadcast(tensor, src=src, group=pg, async_op=True)
            if work is not None:
                works.append(work)
        return works
