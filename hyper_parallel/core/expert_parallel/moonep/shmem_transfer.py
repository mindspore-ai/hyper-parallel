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
"""Managed MoonEP expert banks over HyperParallel's private SHMEM heap.

The bank completion event includes native compute and gradient return. It is
independent of the native token workspace event, whose lease can end earlier.
"""
from __future__ import annotations

import ctypes
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from types import ModuleType
import threading
import weakref
from typing import TYPE_CHECKING, Any

import torch

from hyper_parallel.core.multicore import _collective, shmem
from hyper_parallel.core.multicore.shmem import _lifecycle
from hyper_parallel.core.multicore.modules import module as module_resources
from hyper_parallel.core.multicore.modules.mega_moe.heap_manager import _WorkspaceLayout
from hyper_parallel.core.multicore.modules.mega_moe import function as native
from hyper_parallel.core.multicore.modules.mega_moe.route import restore_topk_output

from .allocator import load_native_extension, native_library_path

if TYPE_CHECKING:
    from hyper_parallel.core.multicore.modules.mega_moe.workspace import MegaMoeWorkspace

_POOLS: dict[tuple[Any, ...], Any] = {}

_ROOT_TRANSFER_EVENTS = weakref.WeakKeyDictionary()


@lru_cache(maxsize=1)
def load_bridge() -> ModuleType:
    """Load transfers only for the MegaMoE transport with its active SHMEM DSO."""
    library = native_library_path("libhp_moonep_shared_kernels.so")
    ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    return load_native_extension("_moonep_shared_bridge")


@contextmanager
def _ordered_transfers(pool: Any):
    """Order same-root collective transfers without holding native compute."""
    with pool.manager.access():
        previous, event = _ROOT_TRANSFER_EVENTS.get(pool.manager, (None, None))
        # The bank lease already waits for all transfers from the same pool.
        if previous is not None and previous() is not pool:
            torch.npu.current_stream(pool.device).wait_event(event)
        try:
            yield
        finally:
            if event is None:
                event = torch.npu.Event()
            event.record(torch.npu.current_stream(pool.device))
            _ROOT_TRANSFER_EVENTS[pool.manager] = (weakref.ref(pool), event)


class SharedExpertPool:
    """Register a single reusable expert bank with the existing HP heap manager."""

    def __init__(self, hidden: torch.Tensor, engine: Any, local_experts: int,
                 reservations: tuple = ()) -> None:
        """Reserve the bank together with optional lazy token workspace owners."""
        spec = engine.spec
        self.group, self.rank, self.ranks = spec.ep_group, spec.rank_id, spec.ep_size
        self.local_experts = local_experts
        self.hidden, self.intermediate = spec.hidden_size, spec.intermediate_size
        self.blocks = spec.num_cube_cores
        self.device, self.dtype = hidden.device, hidden.dtype
        self.manager = engine.heap_manager
        self.weights: tuple[torch.Tensor, ...] = ()
        self.completion_event = None
        self.used = False
        self.in_use = False
        self.closed = False
        self.lock = threading.RLock()
        self.cache_token = None
        self.cache_generation = None
        self.generation = 0
        self.graphs = weakref.WeakSet()
        self.owners = weakref.WeakSet()
        self.statistics = {"prefetches": 0, "gradient_returns": 0,
                           "backward_prefetches": 0, "backward_cache_hits": 0,
                           "remote_prefetches": 0, "remote_gradient_returns": 0}
        self.closing = False
        self.capacity_floor = local_experts
        self.expert_weight_layout = spec.expert_weight_layout
        self.key = {"kind": "moonep_expert_bank", "layout": self.expert_weight_layout}
        signature = ("moonep_expert_bank", local_experts, self.hidden, self.intermediate,
                     str(self.dtype), self.expert_weight_layout)
        elements = 6 * local_experts * self.hidden * self.intermediate
        layout = _WorkspaceLayout(elements * hidden.element_size() + 1024, signature, local_experts)
        self.manager.reserve((*reservations, (self.key, layout)), allow_resize=True)
        self.manager.bind(self, engine.spec, self.key)
        try:
            _collective.collective_call(self._allocate, stage="MoonEP bank construction", group=self.group)
        except Exception:
            self.manager.close_workspace(self)
            raise
        _install_runtime_cleanup()

    def _allocate(self) -> None:
        allocated = []
        # Persistent buffers must remain writable after an inference-only first call.
        with torch.inference_mode(False), torch.no_grad():
            try:
                for shape in self._bank_shapes():
                    allocated.append(shmem.empty(shape, dtype=self.dtype, alignment=512))
            finally:
                self.weights = tuple(allocated)
            if self.completion_event is None:
                self.completion_event = torch.npu.Event()
        self.cache_token = None
        self.cache_generation = None

    def _free(self) -> None:
        for tensor in self.weights:
            shmem.free(tensor)
        self.weights = ()
        self.cache_token = None
        self.cache_generation = None

    def _bank_shapes(self):
        if self.expert_weight_layout == "EOI":
            return ((2 * self.local_experts, 2 * self.intermediate, self.hidden),
                    (2 * self.local_experts, self.hidden, self.intermediate))
        return ((2 * self.local_experts, self.hidden, 2 * self.intermediate),
                (2 * self.local_experts, self.intermediate, self.hidden))

    def describe_restart_allocations(
        self, _spec: Any, dtype: torch.dtype, _receive_capacity: int,
    ) -> tuple[Any, ...]:
        """Participate in HP's atomic allocation replacement, without rebuilding runtime."""
        return tuple(_lifecycle._AllocationReplacement(
            tensor, _lifecycle._AllocationSpec(shape, dtype, 512))
                     for tensor, shape in zip(self.weights, self._bank_shapes()))

    def validate_restart_allocations(
        self, _spec: Any, dtype: torch.dtype, _current_capacity: int, replacements: tuple[Any, ...],
    ) -> bool:
        """Keep bank leases independent of the token workspace's native completion."""
        if self.in_use or self.closing:
            raise RuntimeError("Active or closing MoonEP bank prevents heap growth")
        if len(replacements) != len(self.weights):
            raise RuntimeError("MoonEP bank restart allocation mapping changed")
        for tensor, replacement, shape in zip(self.weights, replacements, self._bank_shapes()):
            if replacement.current is not tensor or tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise RuntimeError("MoonEP bank restart mapping does not match committed storage")
        return bool(self.weights)

    def validate_restarted_allocations(
        self, _spec: Any, dtype: torch.dtype, _receive_capacity: int, tensors: tuple[torch.Tensor, ...],
    ) -> None:
        """Validate every replacement before the HeapManager commits any owner."""
        if len(tensors) != len(self._bank_shapes()):
            raise RuntimeError("MoonEP bank restart returned incomplete storage")
        if any(tuple(tensor.shape) != shape or tensor.dtype != dtype
               for tensor, shape in zip(tensors, self._bank_shapes())):
            raise RuntimeError("MoonEP bank restart returned an incompatible allocation")

    def bind_restarted_allocations(
        self, _spec: Any, _dtype: torch.dtype, _device: Any, _receive_capacity: int,
        tensors: tuple[torch.Tensor, ...],
    ) -> None:
        """Invalidate staging generations when HP replaces the symmetric heap."""
        self.weights = tuple(tensors)
        self.cache_token = None
        self.cache_generation = None

    def claim(self) -> None:
        """Order bank reuse after compute and all peer gradient reads."""
        with self.manager.access(), self.lock:
            if self.closed or self.in_use:
                raise RuntimeError("MoonEP expert banks require serial, open execution")
            if self.used:
                torch.npu.current_stream(self.device).wait_event(self.completion_event)
            self.in_use = True

    def release(self) -> None:
        """Record the complete weight/gradient lease on the current stream."""
        with self.manager.access(), self.lock:
            if not self.in_use:
                raise RuntimeError("MoonEP expert banks released without a lease")
            self.completion_event.record(torch.npu.current_stream(self.device))
            self.used = True
            self.in_use = False

    def prefetch(self, weights: tuple[torch.Tensor, ...], copies: torch.Tensor,
                 extra: int, *, token: Any = None) -> tuple[torch.Tensor, ...]:
        """Publish current home weights and fetch this invocation's borrowed experts."""
        if not self.in_use:
            raise RuntimeError("MoonEP prefetch requires an active bank lease")
        self.cache_token = None
        self.cache_generation = None
        bridge = load_bridge()
        for bank, weight in zip(self.weights, weights):
            bank[:self.local_experts].copy_(weight)
        if extra:
            with _ordered_transfers(self):
                for bank in self.weights:
                    bridge.prefetch(bank, bank[self.local_experts:], copies[self.rank],
                                    self.rank, self.ranks, self.blocks)
        self.generation += 1
        self.cache_generation = self.generation
        self.cache_token = token
        self.statistics["prefetches"] += 1
        self.statistics["remote_prefetches"] += int(bool(extra))
        return tuple(bank.detach() for bank in self.weights)

    def return_generated_gradients(self, gradients: tuple[torch.Tensor, ...],
                                   copies: torch.Tensor, active: bool = True) -> tuple[torch.Tensor, ...]:
        """Sum borrowed gradients into the HP native producer's home gradients."""
        bridge = load_bridge()
        home = []
        with _ordered_transfers(self) if active else nullcontext():
            for gradient, bank in zip(gradients, self.weights):
                owner = gradient[:self.local_experts]
                slots = bank[self.local_experts:]
                if active:
                    slots.copy_(gradient[self.local_experts:])
                    bridge.reduce(owner, owner, slots, copies, self.rank, self.blocks)
                home.append(owner)
        self.statistics["gradient_returns"] += 1
        self.statistics["remote_gradient_returns"] += int(active)
        self.cache_token = None
        self.cache_generation = None
        return tuple(home)

    def track_graph(self, ctx: Any) -> None:
        """Keep the bank's owner alive while its autograd graph can replay."""
        self.graphs.add(ctx)

    def release_graph(self, ctx: Any) -> None:
        """Retain explicit retained graphs and release completed graph ownership."""
        keep = getattr(torch._C._autograd, "_get_current_graph_task_keep_graph", None)
        if keep is not None and not keep():
            self.graphs.discard(ctx)

    def close(self) -> None:
        """Collectively release the bank after its last native/autograd consumer."""
        if self.closed:
            return
        with self.manager.access():
            if self.in_use or self.graphs:
                raise RuntimeError("Cannot close MoonEP banks while a call or graph owns them")
            if self.used:
                self.completion_event.synchronize()
            shmem.host_barrier()
            self._free()
            self.closed = True


def _install_runtime_cleanup() -> None:
    """Release banks before HP's sole Runtime reference, preserving its cleanup callback.

    HP exposes one automatic callback, already owned by its resource manager.
    Chaining the last runtime release keeps native group teardown and all of its
    rank-convergence checks in place; registering a second callback would replace it.
    """
    manager = module_resources._RESOURCE_MANAGER
    release = manager.runtime_release
    if getattr(release, "_moonep_cleanup", False):
        return

    def release_after_banks() -> None:
        """Release bank owners before the original HP runtime release."""
        close_expert_pools()
        release()

    release_after_banks._moonep_cleanup = True
    manager.runtime_release = release_after_banks


def close_expert_pools() -> None:
    """Join the existing HP automatic teardown without creating another runtime."""
    for pool in tuple(_POOLS.values()):
        pool.manager.close_workspace(pool)
    _POOLS.clear()
    _ROOT_TRANSFER_EVENTS.clear()


def get_expert_pool(hidden: torch.Tensor, engine: Any, local_experts: int, owner: Any,
                    reservations: tuple = ()) -> SharedExpertPool:
    """Select one bank for a compatible ordered EP root and expert geometry."""
    spec = engine.spec
    key = (tuple(shmem.root_members(spec.ep_group)), local_experts, spec.hidden_size,
           spec.intermediate_size, spec.expert_weight_layout, hidden.dtype, hidden.device)
    pool = _POOLS.get(key)
    if pool is None or pool.closed:
        pool = SharedExpertPool(hidden, engine, local_experts, reservations)
        _POOLS[key] = pool
    elif reservations:
        pool.manager.reserve(reservations, allow_resize=True)
    pool.owners.add(owner)
    return pool


class _TransferContext:
    """Save home weights and acquire borrowed slots only during execution."""

    def __init__(
        self, ctx: Any, weights: tuple[torch.Tensor, ...], copies: torch.Tensor,
        resources: Any, extra: int, workspace: MegaMoeWorkspace,
    ) -> None:
        """Retain the current home parameters and the native workspace lifetime."""
        self.workspace = workspace
        self.context = weakref.ref(ctx)
        self.needs_input_grad = ctx.needs_input_grad
        self.home_weights = weights
        self.copies = copies
        self.resources = resources
        self.extra = extra
        self.token = object()
        self.leased = False

    def save_for_backward(self, *tensors: torch.Tensor) -> None:
        """Save home weights while borrowed views remain owned by the shared bank."""
        # Borrowed views share mutable staging storage across all MoE layers.
        # Saving home parameters lets checkpoint and FSDP restore their lifetime.
        self.weights = tensors[3:5]
        # Checkpoint restores the recomputed generation even if hooks clone tensors.
        # This CPU tag avoids a device synchronization or rank-local identity test.
        generation = torch.tensor(
            self.resources.cache_generation, dtype=torch.int64, device="cpu"
        )
        self.context().save_for_backward(
            *tensors[:3], *self.home_weights, *tensors[5:], self.copies, generation
        )
        self.home_weights = ()

    @property
    def saved_tensors(self) -> tuple:
        """Unpack before acquiring slots so checkpoint replay can use the same bank."""
        # Unpacking may run non-reentrant recomputation. Do not hold slots here.
        tensors = self.context().saved_tensors
        self.copies = tensors[-2]
        self.resources.claim()
        self.leased = True
        if (
            self.resources.cache_token is self.token
            or tensors[-1].item() == self.resources.cache_generation
        ):
            # HP may replace the heap during another layer or checkpoint replay.
            # The generation identifies the data; acquire views of the current bank.
            self.weights = tuple(bank.detach() for bank in self.resources.weights)
            self.resources.statistics["backward_cache_hits"] += 1
        else:
            self.workspace.wait_for_reuse()
            self.weights = self.resources.prefetch(
                tensors[3:5], self.copies, self.extra, token=self.token
            )
            self.resources.statistics["backward_prefetches"] += 1
        return (*tensors[:3], *self.weights, *tensors[5:-2])

    def maybe_clear_saved_tensors(self) -> None:
        """Follow the wrapped autograd context's saved-state lifetime."""
        self.context().maybe_clear_saved_tensors()


class _FusedTransferFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden: torch.Tensor,
        weight1: torch.Tensor,
        weight2: torch.Tensor,
        route: Any,
        plan: Any,
        workspace: MegaMoeWorkspace,
        permutation: tuple,
        topk_weights: torch.Tensor | None,
        workspace_claimed: bool,
        copies: torch.Tensor,
        resources: Any,
        extra: int,
    ) -> torch.Tensor:
        """Run the original HP forward while leased borrowed weights are valid."""
        ctx.transfer_resources = resources
        ctx.original_shapes = weight1.shape, weight2.shape
        ctx.original_dtypes = weight1.dtype, weight2.dtype
        transfer = _TransferContext(ctx, (weight1, weight2), copies, resources, extra, workspace)
        ctx.transfer = transfer
        resources.track_graph(ctx)
        resources.claim()
        try:
            weights = resources.prefetch((weight1, weight2), copies, extra, token=transfer.token)
            return native._MegaMoeFunction.forward(
                transfer,
                hidden,
                *weights,
                route,
                plan,
                workspace,
                permutation,
                topk_weights,
                workspace_claimed,
            )
        finally:
            resources.release()

    @staticmethod
    def backward(ctx: Any, output_gradient: torch.Tensor) -> tuple:
        """Sum generated expert gradients back to the original parameter owners."""
        transfer = ctx.transfer
        completed = False
        try:
            gradients = native._MegaMoeFunction.backward(transfer, output_gradient)
            # Native backward owns these gradients; accumulate without cloning.
            owners = ctx.transfer_resources.return_generated_gradients(
                gradients[1:3], transfer.copies, transfer.extra > 0
            )
            owners = tuple(
                gradient.to(dtype).reshape(shape)
                for gradient, dtype, shape in zip(
                    owners, ctx.original_dtypes, ctx.original_shapes
                )
            )
            completed = True
            return (gradients[0], *owners, *gradients[3:], None, None, None)
        finally:
            if transfer.leased:
                transfer.resources.cache_token = None
                transfer.resources.cache_generation = None
                transfer.resources.release()
                if completed:
                    transfer.resources.release_graph(ctx)
                transfer.leased = False


def execute_shared(
    owner: Any, hidden: torch.Tensor, ids: torch.Tensor, probs: torch.Tensor,
    weights: tuple[torch.Tensor, torch.Tensor], counts: torch.Tensor, copies: torch.Tensor,
    plan_tokens: int, maximum_received: int,
) -> torch.Tensor:
    """Use the original HP route preparation, plan, kernels and pull lease order."""
    executor = owner._get_virtual_executor(hidden, plan_tokens, maximum_received)
    flat = hidden.reshape(-1, owner.hidden_size)
    engine = owner._persistent_engine(flat, executor)
    # Register the bank before HP prepares/publishes the current dynamic route.
    # Its restart descriptors preserve it if token-capacity agreement grows the heap.
    with torch.inference_mode(False), torch.no_grad():
        pool = get_expert_pool(flat, engine, owner.local_experts, owner)
    owner._transfer_pool = pool
    engine, route, claimed = owner._prepare_native_route(
        flat, ids, probs, counts, weights, executor=executor, resources=engine)
    pull = owner.dispatch_mode == "pull"
    try:
        plan = engine.plan_for_tokens(route.metadata.plan_tokens, flat.device)
        output = _FusedTransferFunction.apply(
            flat, weights[0].contiguous(), weights[1].contiguous(), route.metadata,
            plan, engine.workspace, (route.routed_tokens, ids, route.unpermute_mapping),
            probs if pull else None, pull, copies, pool, owner.local_experts,
        )
    finally:
        if claimed:
            engine.workspace.release()
    if not pull:
        output = restore_topk_output(output, route.unpermute_mapping, probs)
    return output.reshape_as(hidden)
