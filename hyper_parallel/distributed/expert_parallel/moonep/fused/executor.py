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
"""MoonEP placement and FSDP lifecycle for the independent sparse executor."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any, Callable

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.multicore.modules.routed_experts.arena import PhaseArena
from hyper_parallel.core.multicore.modules.routed_experts.descriptor import RouteDescriptor
from hyper_parallel.core.multicore.modules.routed_experts.compute import NativeExpertCompute
from hyper_parallel.core.multicore.modules.routed_experts.forward import forward_vector_budget
from hyper_parallel.core.multicore.modules.routed_experts.readiness import upload_metadata
from hyper_parallel.core.multicore.modules.routed_experts.transport import transport_worker_count, vector_core_count
from hyper_parallel.core.multicore.modules.routed_experts.workspace import RingSpec, RingWorkspace
from hyper_parallel.distributed.expert_parallel.moonep.checkpoint import bind_execution, is_replaying
from hyper_parallel.distributed.expert_parallel.moonep.fused.autograd import SparseExpertFunction
from hyper_parallel.distributed.expert_parallel.moonep.fused.communicators import reserve_communicators
from hyper_parallel.distributed.expert_parallel.moonep.fused.ownership import validate_parameter_ownership
from hyper_parallel.distributed.expert_parallel.moonep.fused.plan import compile_local_route
from hyper_parallel.distributed.expert_parallel.moonep.fused.pool import check_dense
from hyper_parallel.distributed.expert_parallel.moonep.fused.transfers import TransferRound, compile_transfer_rounds
from hyper_parallel.distributed.expert_parallel.moonep.fused.weights import TransferDomain, WeightLease
from hyper_parallel.distributed.expert_parallel.moonep.planning import select_allocation


def _transfer_quota_bytes(gate: torch.Tensor, down: torch.Tensor, replica_capacity: int) -> int:
    """Avoid a 256 MiB staging floor when every small replica fits in 2 MiB."""
    role_elements = gate.shape[1] * gate.shape[2] + down.shape[1] * down.shape[2]
    small_quota = 2 * 1024 * 1024
    if replica_capacity == 0 or role_elements * replica_capacity * 4 <= small_quota:
        return small_quota
    return 64 * 1024 * 1024


@dataclass(frozen=True)
class ExecutionKey:
    """One original forward and its replay/backward at a fixed parameter version."""

    layer: int
    invocation: int
    version: int


@dataclass(frozen=True)
class SparseExecution:
    """Immutable placement and small metadata reused by replay and backward."""

    route: RouteDescriptor
    order: torch.Tensor
    inverse_order: torch.Tensor
    compute: NativeExpertCompute
    weights: tuple[TransferRound, ...]
    gate_gradients: tuple[TransferRound, ...]
    down_gradients: tuple[TransferRound, ...]


class SparseAllGatherExtension:
    """Restore FSDP-owned parameters, then notify the independent replica broker."""

    def __init__(self, executor: Any, runtime: Any) -> None:
        """Bind a registered expert executor to its root runtime."""
        self.executor, self.runtime = executor, runtime

    def fsdp_pre_all_gather(self, _state: Any) -> dict:
        """Keep FSDP's own mixed-precision and all-gather input storage."""
        return {}

    def fsdp_post_all_gather(self, _state: Any) -> None:
        """Continue replica prefetch only after FSDP has restored home weights."""
        self.runtime.prefetch_fsdp_expert(self.executor)

    def fsdp_post_prefetch(self, _state: Any) -> None:
        """Continue bounded replica prefetch when complete home weights are reused."""
        self.runtime.prefetch_fsdp_expert(self.executor)


class SparseGradientExtension:
    """Validate that autograd returned complete owner gradients before FSDP RS."""

    def __init__(self, executor: Any, runtime: Any) -> None:
        """Bind a registered expert executor to its root runtime."""
        self.executor, self.runtime = executor, runtime

    def fsdp_pre_gradient_reduce(self, _scheduler: Any, context: Any) -> Any:
        """Publish one dW role to the dedicated gradient communication domain."""
        key, role, gradient = context
        return self.runtime.start_gradient(key, role, gradient)

    def fsdp_post_gradient_reduce(self, _scheduler: Any) -> None:
        """Never add replica dW twice; only reject an unfinished custom backward."""
        if self.executor.layer in self.runtime.backward_layers:
            raise RuntimeError("FSDP gradient reduction overtook sparse expert gradient completion")


class SparseMoonEPRuntime:
    """Fixed registered forward order and controlled reverse-order backward.

    Every forward executes the complete registered executor sequence once. All
    ranks enter the controlled backward together. Admission checks this supported
    schedule; arbitrary rank divergence during autograd still requires job abort.
    """

    controlled_backward = True

    def __init__(self, executors: list) -> None:
        """Register only generic FSDP extensions; native state is initialized lazily."""
        if not executors:
            raise ValueError("sparse MoonEP requires expert modules")
        self.executors = executors
        self.version = self.sequence = 0
        self.executions: dict[ExecutionKey, tuple[Any, SparseExecution]] = {}
        self.trace: list[ExecutionKey] = []
        self.tickets: dict[ExecutionKey, WeightLease] = {}
        self.statistics = Counter()
        self.backward_layers = set()
        self.gradient_roles = {}
        self.gradient_outputs = {}
        self.gradient_accumulators = {}
        self.gradient_staging = {}
        self.backward_active = False
        self.backward_token: tuple[ExecutionKey, ...] | None = None
        self.backward_position = 0
        self.poisoned = False
        self.closed = False
        self._initialization_failed = False
        self.workspace = self.weight_domain = self.gradient_domain = None
        self.arena = None
        self.activation_stream = None
        self.forward_vector_workers = 0
        self.groups = ()
        self._group_lease = None
        self.home_geometry = None
        self.home_device = None
        for layer, executor in enumerate(executors):
            executor.runtime, executor.layer = self, layer
            scheduler = getattr(executor.owner, "hsdp_scheduler", None)
            if scheduler is not None:
                if len(scheduler.modules) != 1 or scheduler.cell is not executor.owner:
                    raise ValueError("sparse MoonEP experts must form an independent FSDP unit")
                scheduler.enable_param_grad_gate()
                scheduler.register_all_gather_extension(SparseAllGatherExtension(executor, self))
                scheduler.register_gradient_extension(SparseGradientExtension(executor, self))

    @staticmethod
    def _validate_root_topology(root: Any, group_size: int, configuration: tuple) -> None:
        """Reject invalid root extents before inspecting individual expert groups."""
        if (any(isinstance(value, bool) or not isinstance(value, int) for value in configuration)
                or root.ranks != group_size or not 1 <= root.ranks < 2**30
                or root.experts < 1 or root.experts != root.ranks * root.home_count
                or not 0 < root.top_k <= root.experts or root.replica_capacity < 0):
            raise ValueError("sparse experts require a valid EP root and expert partition")

    @staticmethod
    def _validate_peer_topology(peer: Any, configuration: tuple, expected_members: tuple[int, ...],
                                group_size: int, local_rank: int) -> None:
        """Check group metadata without changing collective membership or lookup order."""
        local = peer.ranks, peer.experts, peer.home_count, peer.top_k, peer.replica_capacity
        group = dist.group.WORLD if peer.group is None else peer.group
        members = tuple(dist.get_process_group_ranks(group))
        if (any(isinstance(value, bool) or not isinstance(value, int) for value in local)
                or local != configuration or members != expected_members
                or dist.get_world_size(peer.group) != group_size
                or isinstance(peer.rank, bool) or not isinstance(peer.rank, int)
                or peer.rank != local_rank or dist.get_rank(peer.group) != peer.rank):
            raise ValueError("all sparse experts must share EP membership, local rank, and expert partition")

    def _validate_topology(self, executor: Any) -> tuple[int, ...]:
        """Validate stable host metadata for every owner without restoring any weights."""
        root = self.executors[0]
        group_size, local_rank = dist.get_world_size(root.group), dist.get_rank(root.group)
        group = dist.group.WORLD if root.group is None else root.group
        expected_members = tuple(dist.get_process_group_ranks(group))
        configuration = root.ranks, root.experts, root.home_count, root.top_k, root.replica_capacity
        if executor is not root:
            raise ValueError("sparse initialization requires the first registered expert")
        self._validate_root_topology(root, group_size, configuration)
        for peer in self.executors:
            self._validate_peer_topology(peer, configuration, expected_members, group_size, local_rank)
            if peer.token_dedup != root.token_dedup:
                raise ValueError("all sparse experts must use the same device-counted transport mode")
            if (peer.forward_ratio, peer.backward_ratio, peer.swiglu_limit) != (
                    root.forward_ratio, root.backward_ratio, root.swiglu_limit):
                raise ValueError("all sparse experts must share compact ratios and activation semantics")
        return expected_members

    def validate_home(self, home: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Check restored full views against startup geometry before count admission."""
        gate, down = home
        geometry = tuple(gate.shape), tuple(down.shape), gate.stride(), down.stride()
        if (self.home_geometry is None or geometry != self.home_geometry
                or gate.dtype != torch.bfloat16 or down.dtype != torch.bfloat16
                or gate.device != self.home_device or down.device != self.home_device):
            raise ValueError("sparse home shape, stride, dtype, or device differs from initial admission")
        for value in home:
            check_dense(value, tuple(value.shape), torch.bfloat16, self.home_device,
                        transposed=value.stride(-2) == 1)

    def _initial_home_manifest(self, executor: Any, home: tuple[torch.Tensor, torch.Tensor]) -> tuple:
        """Validate local home geometry and ownership within initialization admission."""
        gate, down = home
        if (gate.dtype != torch.bfloat16 or down.dtype != gate.dtype or gate.device.type != "npu"
                or down.device != gate.device):
            raise ValueError("sparse executor requires same-device NPU BF16 home weights after FSDP unshard")
        gate_shape, down_shape = tuple(gate.shape), tuple(down.shape)
        if (len(gate_shape) != 3 or len(down_shape) != 3
                or any(size < 1 for size in (*gate_shape, *down_shape))
                or gate_shape[0] != executor.home_count or down_shape[0] != executor.home_count
                or gate_shape[1] != down_shape[2] or gate_shape[2] != 2 * down_shape[1]):
            raise ValueError("sparse home weights must have positive [home,H,2I] and [home,I,H] shapes")
        manifest = (executor.ranks, executor.experts, executor.top_k, executor.replica_capacity,
                    gate_shape, down_shape, gate.stride(), down.stride(), len(self.executors),
                    executor.token_dedup, executor.forward_ratio, executor.backward_ratio,
                    executor.swiglu_limit)
        validate_parameter_ownership(tuple(peer.owner for peer in self.executors))
        return manifest

    def initialize(self, executor: Any, home: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Create disjoint token, weight, and gradient communication resources once."""
        if self.closed:
            raise RuntimeError("sparse runtime has been closed")
        if self.poisoned:
            raise RuntimeError("sparse runtime is poisoned; whole-job termination is required")
        if self.workspace is not None:
            return
        if not dist.is_initialized():
            raise ValueError("sparse executor requires initialized distributed communication")
        root = self.executors[0]
        manifest, admission_error = None, None
        try:
            members = self._validate_topology(executor)
            manifest = self._initial_home_manifest(executor, home)
            self._group_lease = reserve_communicators(members)
            manifest = (*manifest, self._group_lease.slot)
        except Exception as exc:
            admission_error = f"{type(exc).__name__}: {exc}"
        gathered = [None] * dist.get_world_size(root.group)
        try:
            dist.all_gather_object(gathered, (manifest, admission_error), group=root.group)
        except BaseException:
            self.poisoned = True
            self._release_group_lease()
            raise
        failures = tuple((rank, item[1]) for rank, item in enumerate(gathered) if item[1] is not None)
        if failures:
            self.poisoned = True
            self._release_group_lease()
            raise ValueError(f"sparse expert home/ownership admission failed: {failures}")
        if any(item[0] != manifest for item in gathered):
            self.poisoned = True
            self._release_group_lease()
            raise ValueError("sparse executor model manifests differ across ranks")
        try:
            self._initialize_resources(executor, home, manifest)
        except BaseException:
            self.poisoned = True
            self._initialization_failed = True
            raise

    def _initialize_resources(self, executor: Any, home: tuple[torch.Tensor, torch.Tensor],
                              manifest: tuple) -> None:
        """Build resources once; local construction errors require whole-job termination."""
        gate, down = home
        self.groups = self._group_lease.initialize()
        payload_groups, control_groups = self.groups[:2], self.groups[2:]
        self.workspace = RingWorkspace(RingSpec(executor.ranks, gate.shape[1]), executor.group,
                                       control_group=control_groups[0])
        self.arena = PhaseArena(gate.device)
        quota_bytes = _transfer_quota_bytes(gate, down, executor.replica_capacity)
        self.weight_domain = TransferDomain(payload_groups[0], executor.rank, gate.device,
                                            quota_bytes=quota_bytes, control_group=control_groups[1])
        self.gradient_domain = TransferDomain(payload_groups[1], executor.rank, gate.device, dtype=torch.float32,
                                              quota_bytes=quota_bytes, control_group=control_groups[2])
        shapes = tuple(gate.shape[1:]), tuple(down.shape[1:])
        transposed = gate.stride(-2) == 1, down.stride(-2) == 1
        error = None
        try:
            vector_cores = vector_core_count()
            transport_workers = transport_worker_count(executor.ranks, vector_cores)
            self.forward_vector_workers = forward_vector_budget(transport_workers, vector_cores)
            self.statistics.update(logical_peers=executor.ranks, transport_workers=transport_workers,
                                   vector_cores=vector_cores)
            if (shapes != ((2048, 1536), (768, 2048))
                    or transposed[0] != transposed[1]):
                self.forward_vector_workers = 0
            if self.forward_vector_workers:
                self.activation_stream = torch.npu.Stream(device=gate.device)
            self.weight_domain.admit(executor.home_count, executor.replica_capacity, shapes, transposed)
            self.gradient_domain.admit(executor.home_count, executor.replica_capacity, shapes, transposed,
                                       caller_accumulator=True, caller_staging=True)
        except Exception as exc:
            error = exc
        self.workspace.check_preflight(error)
        self.home_geometry = manifest[4:8]
        self.home_device = gate.device
        for peer in self.executors:
            peer.shapes = shapes
            peer.transposed = gate.stride(-2) == 1, down.stride(-2) == 1

    def new_key(self, executor: Any) -> ExecutionKey:
        """Assign a fresh invocation while preserving original forward order."""
        if self.poisoned:
            raise RuntimeError("sparse runtime is poisoned")
        if (executor.layer == 0 and self.arena is not None
                and not (self.trace or self.executions or self.tickets or self.backward_active)):
            self._retire_backward_scratch()
        self.sequence += 1
        key = ExecutionKey(executor.layer, self.sequence, self.version)
        self.trace.append(key)
        return key

    def _retire_backward_scratch(self) -> None:
        """Drop completed phase storage before dense backward or the next step."""
        if self.arena is None:
            return
        regions = ("selections", "incoming", "down_gradient", "gate_gradient",
                   "activation", "project", "forward_activation")
        self.arena.release_regions(regions)
        self.statistics["scratch_reserved_bytes"] = self.arena.reserved_bytes

    def forward_header(self, executor: Any, key: ExecutionKey, failed: bool) -> tuple[int, ...]:
        """Describe one original call for the existing counts all-gather.

        The final field carries local host validation failures. It is checked
        together with all ranks' call identities before placement or W/T work.
        """
        expected = (key.invocation - 1) % len(self.executors)
        invalid = (self.backward_active or self.poisoned or self.closed or key.version != self.version
                   or key.invocation < 1
                   or key.invocation != self.sequence or key.layer != executor.layer
                   or len(self.trace) > len(self.executors) or not self.trace or self.trace[-1] != key
                   or executor.layer != expected or self.executors[expected] is not executor)
        return key.layer, key.invocation, key.version, len(self.executors), expected, int(failed or invalid)

    def admit_forward_headers(self, headers: torch.Tensor) -> None:
        """Validate host headers from the same completed counts all-gather."""
        if headers.device.type != "cpu" or headers.dtype != torch.int64 or headers.ndim != 2:
            raise ValueError("forward admission headers must be a CPU int64 matrix")
        rows = headers.tolist()
        if (not rows or any(len(row) != 6 or row[-1] != 0 for row in rows)
                or any(row[:-1] != rows[0][:-1] for row in rows[1:])):
            self.poisoned = True
            raise RuntimeError(f"sparse forward calls violate the registered rank-consistent order: {rows}")

    def prepare_backward_resources(self, key: ExecutionKey) -> None:
        """Restore retired replica buffers before replay without reading parameters.

        Args:
            key: First admitted sparse backward execution, also reused by replay.

        The same admission is reused by demand preparation without checkpointing.
        Repeated calls after restoration neither allocate nor communicate.
        """
        if self.weight_domain.parked:
            error = None
            try:
                self.validate_backward(key)
                if self.backward_position != 0:
                    raise RuntimeError("retired weights may resume only at the first sparse backward demand")
            except Exception as exc:
                error = exc
            try:
                self.weight_domain.restore_storage(error)
            except BaseException:
                self.poisoned = True
                raise
            self.statistics["weight_storage_restores"] += 1

    @torch.no_grad()
    def prepare(self, key: ExecutionKey) -> WeightLease:
        """Copy remote experts from home parameter views already restored by FSDP."""
        self.prepare_backward_resources(key)
        if key in self.tickets:
            return self.tickets[key]
        if len(self.tickets) >= 2:
            raise RuntimeError("demanded weight lease exceeds current plus one speculative lease")
        executor, plan = self.executions[key]
        ticket = self.weight_domain.materialize(plan.route.active_experts, executor.home_count,
                                               executor.read_weights, plan.weights,
                                               executor.shapes, executor.transposed)
        self.tickets[key] = ticket
        self.statistics["materializations"] += 1
        self.statistics["max_live_tickets"] = max(self.statistics["max_live_tickets"], len(self.tickets))
        return ticket

    def consume(self, key: ExecutionKey) -> tuple[list[torch.Tensor], list[torch.Tensor], Any]:
        """Bind fresh weight views and return the fence guarding their contents.

        Restoration has installed valid storage before this call returns. View
        construction and pointer-table preparation may precede its device writes;
        the caller must wait for the returned event before the first weight read.
        Keeping that wait after token submission lets W and T overlap.
        """
        ticket = self.prepare(key)
        executor, plan = self.executions[key]
        current = torch.npu.current_stream(self.workspace.device)
        home = executor.read_weights()
        result = []
        for role, tensor in zip(("gate", "down"), home):
            if not tensor.untyped_storage().nbytes():
                raise RuntimeError("FSDP home storage was released before sparse expert consumption")
            result.append([tensor[expert % executor.home_count] if expert // executor.home_count == executor.rank
                           else ticket.remote[expert, role] for expert in plan.route.active_experts])
            for value in result[-1]:
                value.record_stream(current)
        return result[0], result[1], ticket.ready

    def release(self, key: ExecutionKey) -> None:
        """Join the last W reader onto the stream that FSDP uses for reshard."""
        ticket = self.tickets.pop(key, None)
        if ticket is not None:
            self.weight_domain.release_lease(ticket)

    def retire_forward_scratch(self, key: ExecutionKey, *, requires_backward: bool = False) -> None:
        """Release temporary projection storage after the last original forward.

        Saved X/Z/returned values have independent storage. C has joined V and
        T before arena.end; allocator stream records protect queued consumers.
        Dropping only arena references keeps these regions out of the loss peak.
        Replay and backward regrow them during their normal phase admission.
        W retires only when the caller confirms the returned result needs backward;
        grad mode alone does not cover frozen inputs and parameters.
        """
        if is_replaying() or key.layer != len(self.executors) - 1:
            return
        if (key.version != self.version or key.invocation != self.sequence
                or not self.trace or self.trace[-1] != key):
            self.poisoned = True
            raise RuntimeError("forward scratch retirement requires the current final execution")
        self.arena.release_regions(("project", "forward_activation"))
        self.statistics["scratch_reserved_bytes"] = self.arena.reserved_bytes
        if requires_backward:
            try:
                if self.tickets:
                    raise RuntimeError("weight retirement requires every forward lease to be released")
                self.statistics["weight_retired_bytes"] = self.weight_domain.retire_storage()
            except BaseException:
                self.poisoned = True
                raise

    def prefetch_key(self, executor: Any) -> ExecutionKey | None:
        """Select one admitted invocation without waiting for a future lease release."""
        if (not self.backward_active or self.weight_domain is None
                or self.weight_domain.parked or len(self.tickets) >= 2):
            return None
        matches = [key for key in self.trace if key.layer == executor.layer]
        if len(matches) == 1 and matches[0] not in self.tickets:
            return matches[0]
        return None

    def prefetch_fsdp_expert(self, executor: Any) -> None:
        """Prepare at most the current and one speculative replica lease."""
        key = self.prefetch_key(executor)
        if key is not None:
            self.prepare(key)
            self.statistics["fsdp_ep_prefetches"] += 1

    def start_gradient(self, key: ExecutionKey, role: str, gradient: torch.Tensor) -> Any:
        """Issue each role in fixed down/gate order on the independent G domain."""
        expected = "gate" if self.gradient_roles.get(key) == "down" else "down"
        if role != expected or self.gradient_roles.get(key) == "gate":
            raise RuntimeError("gradient roles must be published exactly once in down/gate order")
        self.gradient_roles[key] = role
        executor, plan = self.executions[key]
        rounds = plan.down_gradients if role == "down" else plan.gate_gradients
        return self.gradient_domain.return_gradient(role, plan.route.active_experts,
                                                    executor.home_count, gradient, rounds,
                                                    output=self.gradient_outputs[key][role],
                                                    accumulator=self.gradient_accumulators[key],
                                                    staging=self.gradient_staging[key])

    def complete(self, key: ExecutionKey) -> None:
        """Retire one completed execution after all complete gradients were joined."""
        if not self.trace or self.trace[-1] != key:
            raise RuntimeError("sparse executor requires reverse original-forward backward order")
        if self.backward_active:
            self.validate_backward(key)
            self.backward_position += 1
        self.trace.pop()
        route = self.executions[key][1].route
        for phase in (False, True, "readiness"):
            self.workspace.metadata.pop((route, phase), None)
        self.workspace.metadata.pop((route, "host_transport", self.workspace.spec.block_rows), None)
        del self.executions[key]
        self.backward_layers.discard(key.layer)
        self.gradient_roles.pop(key, None)
        self.gradient_outputs.pop(key, None)
        self.gradient_accumulators.pop(key, None)
        self.gradient_staging.pop(key, None)
        if (self.backward_active
                and not (self.trace or self.executions or self.tickets or self.backward_layers
                         or self.gradient_roles or self.gradient_outputs
                         or self.gradient_accumulators or self.gradient_staging)):
            # Dense/root backward continues after the last sparse node. Its
            # allocations no longer need these completed temporary regions.
            self._retire_backward_scratch()

    def _append_backward_trace_errors(self, errors: list[str]) -> None:
        """Report retained execution identity errors in their original diagnostic order."""
        if len(self.trace) != len(self.executors):
            errors.append("backward requires one complete registered forward sequence")
        for previous, key in zip((None, *self.trace[:-1]), self.trace):
            if (key.version != self.version or key.layer != (key.invocation - 1) % len(self.executors)
                    or previous is not None and key.invocation != previous.invocation + 1):
                errors.append("trace contains an invalid version, layer, or invocation")
                break
        if self.trace and (self.trace[0].layer != 0 or self.trace[-1].invocation != self.sequence):
            errors.append("trace does not cover the complete most recent forward")
        if (len(set(self.trace)) != len(self.trace) or set(self.executions) != set(self.trace)
                or not set(self.tickets).issubset(self.trace)):
            errors.append("trace and live executions or leases disagree")

    def _append_backward_gradient_errors(self, errors: list[str]) -> None:
        """Report unfinished gradient roles after execution-trace validation."""
        if (self.backward_layers or self.gradient_roles or self.gradient_outputs
                or self.gradient_accumulators or self.gradient_staging):
            errors.append("a previous gradient phase is unfinished")

    def _backward_errors(self, loss: torch.Tensor, retain_graph: bool, create_graph: bool) -> tuple[str, ...]:
        """Collect local host errors without leaving peers before entry admission."""
        errors = []
        if retain_graph is not False or create_graph is not False:
            errors.append("retain_graph and create_graph must both be False")
        if self.backward_active or self.poisoned or self.closed:
            errors.append("runtime is nested, poisoned, or closed")
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1 or not loss.requires_grad:
            errors.append("loss must be a scalar tensor requiring gradients")
        self._append_backward_trace_errors(errors)
        self._append_backward_gradient_errors(errors)
        return tuple(errors)

    def validate_backward(self, key: ExecutionKey) -> None:
        """Validate one node against the immutable token admitted at backward entry.

        This local guard poisons unexpected execution. It is not a new collective
        and cannot make arbitrary rank divergence inside autograd symmetric.
        """
        token = self.backward_token
        if (not self.backward_active or self.poisoned or token is None
                or self.backward_position >= len(token) or token[self.backward_position] != key
                or not self.trace or self.trace[-1] != key or key.version != self.version):
            self.poisoned = True
            raise RuntimeError("sparse backward node does not match the controlled execution token")

    def backward(self, loss: torch.Tensor, *, retain_graph: bool = False, create_graph: bool = False) -> None:
        """Admit one complete trace on CPU, then run its non-retained backward.

        All ranks must enter this method together after the same complete fixed
        forward schedule. Only this entry adds a control collective; individual
        backward layers reuse their existing phase admission.
        """
        if self.workspace is None:
            raise RuntimeError("sparse backward requires an initialized original forward")
        errors = self._backward_errors(loss, retain_graph, create_graph)
        trace = tuple((key.layer, key.invocation, key.version) for key in self.trace)
        parked = self.weight_domain is not None and self.weight_domain.parked
        state = self.version, self.sequence, len(self.executors), trace, parked
        gathered = [None] * self.executors[0].ranks
        try:
            dist.all_gather_object(gathered, (state, errors), group=self.workspace.control_group)
        except BaseException:
            self.poisoned = True
            raise
        failures = tuple((rank, item[1]) for rank, item in enumerate(gathered) if item[1])
        if failures or any(item[0] != state for item in gathered):
            self.poisoned = True
            raise RuntimeError(f"sparse backward admission failed: rank errors={failures}, traces disagree="
                               f"{any(item[0] != state for item in gathered)}")
        self.backward_token = tuple(reversed(self.trace))
        self.backward_position = 0
        self.backward_active = True
        try:
            loss.backward(retain_graph=False, create_graph=False)
            if self.backward_position != len(self.backward_token) or self.trace:
                raise RuntimeError("controlled backward did not complete every admitted sparse execution")
        except BaseException:
            self.poisoned = True
            raise
        finally:
            self.backward_active = False
            self.backward_token = None
            self.backward_position = 0

    def finish_step(self) -> None:
        """Validate completion and retire B-only scratch before the optimizer update."""
        if (self.poisoned or self.backward_active or self.backward_token is not None
                or self.trace or self.executions or self.tickets
                or self.backward_layers or self.gradient_roles or self.gradient_outputs
                or self.gradient_accumulators or self.gradient_staging):
            raise RuntimeError("sparse MoonEP step still owns live or failed execution state")
        self._retire_backward_scratch()
        self.version += 1

    def _release_group_lease(self) -> None:
        """Release admission-only or fully drained communicator ownership."""
        if self._group_lease is not None:
            self._group_lease.release()
            self._group_lease = None

    def close(self) -> None:
        """Drain execution storage and return session-owned communicators for reuse.

        Torch WORLD teardown destroys the cached process groups. Per-runtime
        weight, gradient, token-window and activation storage is released here.
        """
        if self.closed:
            return
        if self._initialization_failed:
            # Initialization has submitted no domain payloads or arena phases.
            # Collective resource ownership may differ across ranks after failure.
            self.weight_domain = self.gradient_domain = None
            self.arena = self.activation_stream = None
            raise RuntimeError("sparse resource initialization failed; local unsubmitted references were dropped, "
                               "but collective resources require whole-job termination")
        self.finish_step()
        if self.workspace is not None:
            self.weight_domain.drain()
            self.gradient_domain.drain()
            self.workspace.close()
            self.arena.close()
            self.arena = None
            self.activation_stream = None
            self._release_group_lease()
            self.workspace = None
            self.weight_domain = self.gradient_domain = None
            self.groups = ()
        self.closed = True


class SparseMoonEPExecutor:
    """Provide real token ranges and replica weights to independent native kernels."""

    def __init__(self, owner: torch.nn.Module, read_weights: Callable, *, ep_group: Any, ep_size: int,
                 num_experts: int, top_k: int, replica_capacity: int | None = None,
                 token_dedup: bool = False,
                 forward_ratio: float | None = None,
                 backward_ratio: float | None = None,
                 swiglu_limit: float | None = None) -> None:
        """Bind the existing expert owner without registering any replacement parameters."""
        if ep_size < 1 or num_experts % ep_size or not 0 < top_k <= num_experts:
            raise ValueError("invalid expert partition or top-k")
        self.owner, self.read_weights, self.group = owner, read_weights, ep_group
        self.ranks, self.experts, self.top_k = ep_size, num_experts, top_k
        if not isinstance(token_dedup, bool):
            raise ValueError("token_dedup must be a boolean")
        self.token_dedup = token_dedup
        self.forward_ratio = float(0.75 if forward_ratio is None else forward_ratio)
        self.backward_ratio = float(1.0 if backward_ratio is None else backward_ratio)
        if not (0.0 <= self.forward_ratio <= 1.0 and 0.0 <= self.backward_ratio <= 1.0):
            raise ValueError("MoonEP compact unique ratios must be between zero and one")
        if swiglu_limit is not None and (isinstance(swiglu_limit, bool)
                                         or not isinstance(swiglu_limit, (int, float))
                                         or not math.isfinite(swiglu_limit) or swiglu_limit <= 0):
            raise ValueError("swiglu_limit must be a finite positive number or None")
        self.swiglu_limit = None if swiglu_limit is None else float(swiglu_limit)
        self.rank, self.home_count = dist.get_rank(ep_group), num_experts // ep_size
        self.replica_capacity = self.home_count if replica_capacity is None else replica_capacity
        if (isinstance(self.replica_capacity, bool) or not isinstance(self.replica_capacity, int)
                or self.replica_capacity < 0):
            raise ValueError("replica capacity must be a nonnegative integer")
        self.runtime = None
        self.layer = -1
        self.shapes = None
        self.transposed = None

    def plan(self, ids: torch.Tensor, *, key: ExecutionKey | None = None,
             error: Exception | None = None) -> SparseExecution:
        """Admit original call identity with counts, then compile only actual ranges."""
        key = self.runtime.trace[-1] if key is None else key
        selections, flat = ids.numel(), None
        local = None
        if error is None:
            try:
                if self.shapes is None:
                    raise ValueError("sparse expert weight shapes have not been admitted")
                if ids.dtype != torch.int64:
                    raise ValueError("sparse expert IDs must be int64 before counting")
                if selections > 2**20 or not 1 <= self.experts <= 2**20:
                    raise ValueError("sparse route counts exceed the exact admitted histogram extent")
                flat = ids.reshape(-1)
                # Fixed bins avoid bincount's device-to-host maximum/shape query.
                # Both IDs and integer bin counts are exact in FP32 at this bound.
                local = torch.histc(flat.float(), bins=self.experts, min=-0.5, max=self.experts - 0.5).long()
            except Exception as exc:
                error = exc
        if error is not None:
            local = torch.zeros(self.experts, dtype=torch.int64, device=ids.device)
        header = self.runtime.forward_header(self, key, error is not None)
        payload = torch.cat((local, upload_metadata(torch.tensor(header, dtype=torch.int64), ids.device)))
        # histc omits invalid IDs. Carry that device-side failure in the same
        # admission header so all peers reject before any token payload starts.
        payload[-1].bitwise_or_((local.sum() != selections).long())
        gathered = payload.new_empty(self.ranks * payload.numel())
        dist.all_gather_into_tensor(gathered, payload, group=self.group)
        host = gathered.reshape(self.ranks, payload.numel()).cpu()
        self.runtime.admit_forward_headers(host[:, self.experts:])
        try:
            source, allocation, fallback = select_allocation(host[:, :self.experts], self.replica_capacity)
            route = compile_local_route(source, allocation, self.rank)
            # Expert IDs fit exactly in FP32; INT64 argsort falls back to AiCPU.
            order = torch.argsort(flat.float(), stable=True)
            inverse_order = torch.empty_like(order)
            inverse_order.scatter_(0, order, torch.arange(order.numel(), dtype=order.dtype, device=order.device))
            compute = NativeExpertCompute(route.boundaries, ids.device, vector_stream=self.runtime.activation_stream,
                                          vector_workers=self.runtime.forward_vector_workers,
                                          swiglu_limit=self.swiglu_limit)
            # The existing count-header admission already rejects missing geometry on every rank.
            gate, down = tuple(self.shapes)
            role_sizes = (("gate", gate[0] * gate[1]), ("down", down[0] * down[1]))
            self.runtime.statistics["plans"] += 1
            self.runtime.statistics["planner_fallbacks"] += int(fallback)
            source_rows = int(source.sum(axis=1).max())
            if source_rows % self.top_k:
                raise ValueError("source selection rows must be a multiple of top-k")
            weight_quota = self.runtime.weight_domain.quota_elements
            gradient_quota = self.runtime.gradient_domain.quota_elements
            return SparseExecution(route, order, inverse_order, compute,
                                   compile_transfer_rounds(allocation, role_sizes, quota_elements=weight_quota),
                                   compile_transfer_rounds(allocation, role_sizes[:1], reverse=True,
                                                           quota_elements=gradient_quota),
                                   compile_transfer_rounds(allocation, role_sizes[1:], reverse=True,
                                                           quota_elements=gradient_quota))
        except BaseException as exc:
            self.runtime.poisoned = True
            raise RuntimeError(
                "sparse plan preparation failed after count admission; whole-job termination is required "
                "because peer preparation failures are not collectively converged"
            ) from exc


    def _validate_inputs(self, hidden: torch.Tensor, ids: torch.Tensor, scores: torch.Tensor,
                         home: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Check current home views and source tensors inside count admission."""
        self.runtime.validate_home(home)
        if (hidden.ndim < 2 or hidden.shape[-1] != home[0].shape[1] or hidden.dtype != torch.bfloat16
                or hidden.device != home[0].device or ids.ndim != 2 or ids.shape != scores.shape
                or ids.shape[1] != self.top_k or ids.shape[0] != hidden.numel() // hidden.shape[-1]
                or ids.device != hidden.device or scores.device != hidden.device
                or ids.dtype != torch.int64 or scores.dtype not in (torch.bfloat16, torch.float32)
                or not scores.is_contiguous()):
            raise ValueError("sparse token, expert ID and BF16/FP32 score arguments violate the admitted contract")

    def __call__(self, hidden: torch.Tensor, ids: torch.Tensor, scores: torch.Tensor, *,
                 expert_weights: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        """Execute one original or replayed invocation under its retained placement."""
        if self.runtime is None:
            raise RuntimeError("configure_moonep must run after wrapping the model with FSDP")
        home = self.read_weights() if expert_weights is None else expert_weights
        self.runtime.initialize(self, home)
        key = bind_execution(self, lambda: self.runtime.new_key(self))
        error = None
        try:
            self._validate_inputs(hidden, ids, scores, home)
        except Exception as exc:
            error = exc
        if key not in self.runtime.executions:
            self.runtime.executions[key] = self, self.plan(ids, key=key, error=error)
        elif error is not None:
            self.runtime.poisoned = True
            raise RuntimeError(
                "sparse replay argument validation failed; whole-job termination is required "
                "because replay errors are not collectively converged"
            ) from error
        result = SparseExpertFunction.apply(hidden.reshape(-1, hidden.shape[-1]), scores, *home, key, self.runtime)
        requires_backward = torch.is_grad_enabled() and result.requires_grad
        self.runtime.retire_forward_scratch(key, requires_backward=requires_backward)
        if not requires_backward:
            self.runtime.complete(key)
        return result.reshape_as(hidden)
