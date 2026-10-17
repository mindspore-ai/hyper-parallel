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
"""Muon momentum-only optimizer-state swap adapter."""

# pylint: disable=protected-access
# pylint: disable=forbidden-backend-import

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch

from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.core.optimizer.optimizer import AsyncReplicateBroadcaster
from hyper_parallel.core.optimizer.sharding_category import HSDPGroupAssignment
from hyper_parallel.core.optimizer.swap_optimizer_base import (
    StateSwapAdapter,
    SwapOptimizer as CoreSwapOptimizer,
    SwapSlot,
)

#: Muon's only persistent optimizer state that participates in swap.
MUON_STATE_KEYS = ("momentum_buffer",)

_NO_COMM = "no_comm"
_HSDP = "hsdp"


@dataclass
class MuonSwapUnit:
    """One atomic Muon update unit and the momentum slots it needs.

    A unit is never split by the pipeline: its parameters run through momentum,
    Newton-Schulz and the parameter update as one indivisible piece of work, so
    a single matrix is never partially swapped.  ``kind`` only selects which
    Muon execution path consumes the unit, and ``hsdp_assign`` keeps the
    original HSDP communication boundary intact.

    Attributes:
        group_index: Index of the owning parameter group, derived from
            ``schedule_key``.
        kind: ``"no_comm"`` for locally orthonormalized params, ``"hsdp"`` for
            an HSDP shard/replicate assignment. Derived from ``schedule_key``.
        params: Parameters this unit updates, in schedule order.
        slots: Swappable momentum-buffer slots for ``params``.
        schedule_key: Rank-independent group, kind, assignment and sub-batch indices.
        logical_nbytes: Full logical momentum bytes, independent of local ownership.
        hsdp_assign: The original assignment for ``"hsdp"`` units, else ``None``.
        post_update_params: Parameters whose post-update callbacks run after this
            unit. Only the final unit of the original Muon update boundary owns
            this tuple, in the original parameter order.
        flushes_broadcast: Whether this unit commits its assignment's parameter
            broadcast.  Only the last unit of an assignment sets it, so an
            assignment split into several NS batches still issues the single
            flush the bare schedule issues.
    """

    params: List[torch.nn.Parameter]
    slots: List[SwapSlot]
    schedule_key: Tuple[int, str, int, int]
    logical_nbytes: int
    hsdp_assign: Optional[HSDPGroupAssignment] = None
    post_update_params: Tuple[torch.nn.Parameter, ...] = ()
    flushes_broadcast: bool = False

    @property
    def group_index(self) -> int:
        """Return the owning parameter-group index from the schedule identity."""
        return self.schedule_key[0]

    @property
    def kind(self) -> str:
        """Return the execution-path kind from the schedule identity."""
        return self.schedule_key[1]


class MuonSwapAdapter(StateSwapAdapter):
    """Swap adapter that offloads and prefetches Muon's momentum buffers.

    The adapter is the swap *coordinator* for Muon, never a second Muon
    implementation.  It replays the schedule of the unmodified ``Muon`` and
    calls that optimizer's existing execution methods -- momentum,
    Newton-Schulz, all-gather, broadcast and the parameter write-back all stay
    in ``muon.py`` as the single source of truth.  What the adapter owns is
    (a) building the atomic update-unit schedule in the original order, and
    (b) making sure the momentum buffer each unit reads is materialized on the
    device only for the duration of that unit.
    """

    @staticmethod
    def default_state_keys() -> Tuple[str, ...]:
        """Return the swap state keys a Muon optimizer owns."""
        return MUON_STATE_KEYS

    def validate(self) -> None:
        """Reject configurations the momentum-only swap path cannot preserve.

        Raises:
            ValueError: If a configured state key is not one Muon owns, or a
                parameter holds momentum state the runtime cannot rebind, or a
                custom momentum update would lose its original call boundary.
        """
        self._validate_configured_state_keys()
        if self.optimizer.momentum_update_fn is not None:
            raise ValueError(
                "Muon swap optimizer does not support momentum_update_fn because "
                "state partitioning changes its call boundary."
            )
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                self._validate_param(param)

    def _validate_configured_state_keys(self) -> None:
        """Reject a configured swap key that is not Muon state.

        ``_configured_state_keys`` shares a vocabulary with the other families
        and silently drops keys none of them implement, so Muon has to check its
        own selection explicitly: asking to swap ``exp_avg`` on a Muon optimizer
        must fail rather than quietly swap nothing.

        Raises:
            ValueError: If the configured selection names a foreign state key.
        """
        if self.config.state_keys is None:
            return
        unavailable = sorted(set(self.config.state_keys) - set(self.default_state_keys()))
        if unavailable:
            raise ValueError(
                f"Requested state keys {unavailable} are not available for {type(self.optimizer)!r}; "
                f"Muon only swaps {self.default_state_keys()}."
            )

    def _validate_param(self, param: torch.nn.Parameter) -> None:
        """Validate one parameter's momentum state is swap-compatible."""
        state = self.optimizer.state.get(param)
        if not state:
            return
        buffer = state.get("momentum_buffer")
        if buffer is None:
            return
        if getattr(param, "grad", None) is not None and getattr(param.grad, "is_sparse", False):
            raise ValueError("Muon swap optimizer only supports dense gradients.")
        local = self._runtime_local_tensor(buffer)
        if not isinstance(local, torch.Tensor):
            raise ValueError(
                f"Muon swap optimizer requires tensor momentum state, got {type(buffer)!r}."
            )
        if local.layout != torch.strided:
            raise ValueError("Muon swap optimizer requires dense momentum state.")
        if not local.is_contiguous():
            raise ValueError("Muon swap optimizer requires contiguous momentum state.")

    @staticmethod
    def _runtime_local_tensor(tensor: Any) -> Any:
        """Return the local shard of a possibly distributed tensor."""
        to_local = getattr(tensor, "to_local", None)
        return to_local() if callable(to_local) else tensor

    def _scheduled_param_ids(self) -> Optional[set[int]]:
        """Return parameters this rank owns in the current Muon schedule.

        ``None`` means the optimizer has no HSDP schedule and therefore all
        parameters are local.  With HSDP metadata present, only no-comm
        parameters and assignment owners may carry Muon state on this rank.
        """
        schedule = getattr(self.optimizer, "_hsdp_assignment_batches", None)
        # ``Muon`` initialises this as an empty dict in non-HSDP runs, so an
        # empty schedule carries no ownership metadata at all: every parameter
        # is local and pre-existing momentum state must be kept.
        if not schedule:
            return None
        scheduled: set[int] = set()
        for group_index, group in enumerate(self.optimizer.param_groups):
            info = schedule.get(group_index)
            if info is None:
                scheduled.update(id(param) for param in group["params"])
                continue
            scheduled.update(id(param) for param in info.get("no_comm", []))
            for batch_group in info.get("batch_groups", []):
                for assignment in batch_group.get("sub_batches", []):
                    scheduled.update(id(param) for param in assignment.owned_params)
        return scheduled

    def _params_with_local_state(self, params: Iterable[Any]) -> Iterable[Any]:
        """Keep checkpoint-restored momentum only for local schedule owners."""
        scheduled_ids = self._scheduled_param_ids()
        if scheduled_ids is None:
            return params
        return tuple(param for param in params if id(param) in scheduled_ids)

    def initial_slots(self) -> Iterable[SwapSlot]:
        """Discover only local-owner momentum state before the first step."""
        scheduled_ids = self._scheduled_param_ids()
        slots = []
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if scheduled_ids is not None and id(param) not in scheduled_ids:
                    # State may have been primed on every rank before this
                    # wrapper was created.  Drop non-owner entries before the
                    # swap runtime publishes any slots.
                    self.optimizer.state.pop(param, None)
                    continue
                state = self.optimizer.state.get(param)
                if not state:
                    continue
                for key in self._configured_state_keys():
                    if key in state and (id(param), key) not in self._slots:
                        self._slots[(id(param), key)] = self._make_slot(key, state[key])
                slots.extend(self._slots_for_params([param]))
        return tuple(slots)

    # ------------------------------------------------------------------ schedule
    def prepare_step(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Advance steps, register momentum slots and build the unit schedule.

        The unit schedule mirrors ``Muon.step()`` exactly: the no-comm params of
        every group first, then the flattened HSDP assignment schedule, keeping
        the per-group-then-per-batch nesting so collective order across ranks is
        unchanged.

        Returns:
            Step context holding the ordered units and the step's broadcaster.

        Raises:
            ValueError: If extra step arguments are passed.
        """
        if args or kwargs:
            raise ValueError("Muon swap optimizer step does not support closure or extra arguments.")
        for group in self.optimizer.param_groups:
            group['step'] = (group.get('step') or 0) + 1

        units: List[MuonSwapUnit] = []
        num_groups = len(self.optimizer.param_groups)
        for group_index in range(num_groups):
            no_comm_params = self._no_comm_params(group_index)
            no_comm_batches = list(self._iter_unshard_batches(no_comm_params))
            for batch_index, params in enumerate(no_comm_batches):
                post_update_params = no_comm_params if batch_index == len(no_comm_batches) - 1 else ()
                units.append(self._make_unit(
                    (group_index, _NO_COMM, -1, batch_index), params, params,
                    post_update_params=post_update_params,
                ))

        linear_batches = [
            self._hsdp_linear_batches(group_index) for group_index in range(num_groups)
        ]
        max_num_batches = max((len(batches) for batches in linear_batches), default=0)
        for batch_index in range(max_num_batches):
            for group_index in range(num_groups):
                group_batches = linear_batches[group_index]
                if batch_index >= len(group_batches):
                    continue
                units.extend(self._make_hsdp_units(group_index, group_batches[batch_index], batch_index))

        return {
            "units": units,
            "broadcaster": AsyncReplicateBroadcaster(self.optimizer),
        }

    def _no_comm_params(self, group_index: int) -> List[torch.nn.Parameter]:
        """Return the params of ``group_index`` that need no HSDP collective.

        This reads the same private schedule metadata ``Muon.step()`` reads, and
        falls back to every parameter in the group when no HSDP assignment
        metadata was built for it -- the same "no HSDP info means all params are
        no_comm" rule the bare step applies.
        """
        info = getattr(self.optimizer, "_hsdp_assignment_batches", {}).get(group_index)
        if not info:
            return list(self.optimizer.unshard_params_by_group.get(group_index, []))
        return list(info.get("no_comm", []))

    def _hsdp_linear_batches(self, group_index: int) -> List[HSDPGroupAssignment]:
        """Flatten one group's nested HSDP schedule into a linear batch list."""
        info = getattr(self.optimizer, "_hsdp_assignment_batches", {}).get(group_index)
        linear_batches: List[HSDPGroupAssignment] = []
        if info:
            for batch_group in info.get("batch_groups", []):
                linear_batches.extend(batch_group.get("sub_batches", []))
        return linear_batches

    def _iter_unshard_batches(
            self,
            params: List[torch.nn.Parameter],
    ) -> Iterator[List[torch.nn.Parameter]]:
        """Plan logical batches using Muon's existing shape and memory rules."""
        for shape_group in self.optimizer._group_by_shape(params).values():
            yield from self.optimizer._split_into_memory_safe_batches(shape_group, shard_size=1)

    def _make_unit(
            self,
            schedule_key: Tuple[int, str, int, int],
            logical_params: Iterable[torch.nn.Parameter],
            params: Iterable[torch.nn.Parameter],
            hsdp_assign: Optional[HSDPGroupAssignment] = None,
            post_update_params: Iterable[torch.nn.Parameter] = (),
            flushes_broadcast: bool = False,
    ) -> MuonSwapUnit:
        """Project a logical unit onto local gradient-bearing owners and slots."""
        logical_nbytes = sum(
            math.prod(param.shape) * self._runtime_local_tensor(param).element_size()
            for param in logical_params
        )
        params = [param for param in params if param.grad is not None]
        post_update_params = tuple(
            param for param in post_update_params if param.grad is not None
        )
        # HSDP assignments contain every logical parameter, but only the local
        # owner participates in this rank's momentum update.  Register state
        # only for parameters that are actually present in this atomic unit;
        # otherwise a non-owner would retain a device/host slot indefinitely.
        for param in params:
            self._ensure_momentum_state(param)
        return MuonSwapUnit(
            params=params,
            slots=self._slots_for_params(params),
            schedule_key=schedule_key,
            logical_nbytes=logical_nbytes,
            hsdp_assign=hsdp_assign,
            post_update_params=post_update_params,
            flushes_broadcast=flushes_broadcast,
        )

    def _make_hsdp_units(
            self,
            group_index: int,
            hsdp_assign: HSDPGroupAssignment,
            assignment_index: int,
    ) -> List[MuonSwapUnit]:
        """Build the units for one HSDP assignment, in assignment order.

        Sharded assignments stay atomic because the shard executor consumes the
        entire assignment. For unsharded assignments, every rank derives all
        owners' NS batches before selecting its own local batch.
        """
        if hsdp_assign.is_shard:
            return [
                self._make_unit(
                    (group_index, _HSDP, assignment_index, 0),
                    hsdp_assign.all_params, hsdp_assign.owned_params,
                    hsdp_assign, post_update_params=hsdp_assign.owned_params,
                    flushes_broadcast=True,
                )
            ]
        params_by_owner: Dict[Tuple[int, ...], List[torch.nn.Parameter]] = {}
        for record in hsdp_assign.all_records:
            owner = hsdp_assign.owner_by_index.get(record.index)
            # Frozen records remain in all_records but have no assigned owner.
            # () is a valid owner when there are no replicate dimensions.
            if owner is None:
                continue
            params_by_owner.setdefault(owner, []).append(record.param)
        batches_by_owner = {
            owner: list(self._iter_unshard_batches(params_by_owner[owner]))
            for owner in sorted(params_by_owner)
        }
        batch_count = max((len(batches) for batches in batches_by_owner.values()), default=1)
        local_batches = batches_by_owner.get(hsdp_assign.replicate_group_ranks, [])
        units = []
        for index in range(batch_count):
            logical_params = [
                param for batches in batches_by_owner.values() if index < len(batches)
                for param in batches[index]
            ]
            local_params = local_batches[index] if index < len(local_batches) else []
            post_update_params = hsdp_assign.owned_params if index == batch_count - 1 else ()
            units.append(self._make_unit(
                (group_index, _HSDP, assignment_index, index),
                logical_params, local_params, hsdp_assign,
                post_update_params=post_update_params,
                flushes_broadcast=index == batch_count - 1,
            ))
        return units

    def _ensure_momentum_state(self, param: torch.nn.Parameter) -> None:
        """Create the host-resident momentum slot for one parameter.

        First-step state is created host-side on purpose: the design requires
        that a freshly created momentum buffer never becomes device-resident
        outside its own update unit.  Parameters without a gradient keep Muon's
        original skip semantics and get no slot at all.

        The device shell published into ``state[param]`` is what
        ``Muon._update_muon_momentum`` reads during this step.  That method only
        allocates momentum when the key is missing, so a registered slot is
        written in place and never replaced.
        """
        if self._runtime_local_tensor(param.grad).layout != torch.strided:
            raise ValueError("Muon swap optimizer only supports dense gradients.")
        state = self.optimizer.state[param]
        for key in self._configured_state_keys():
            if (id(param), key) in self._slots:
                continue
            device_tensor = self.runtime.make_empty_state_like(param)
            if not self._runtime_local_tensor(device_tensor).is_contiguous():
                raise ValueError("Muon swap optimizer requires contiguous momentum state.")
            state[key] = device_tensor
            slot = self._make_slot(key, device_tensor)
            if not slot.swappable:
                # Too small or otherwise ineligible: keep ordinary device state.
                device_tensor.zero_()
                self._slots[(id(param), key)] = slot
                continue
            slot.cpu_tensor = self.runtime.make_zero_cpu_tensor_like(param)
            slot.state = "host"
            self._slots[(id(param), key)] = slot
            self.runtime.release_device_storage(slot)

    def _slots_for_params(self, params: Iterable[torch.nn.Parameter]) -> List[SwapSlot]:
        """Return the registered momentum slots for ``params``, in order.

        Only slots the adapter already owns are returned: they carry the swap
        state (host mirror, released device storage, pending event) that the
        runtime manipulates, so rebuilding a slot here would drop that state.
        Momentum state materialized before the wrapper was created is registered
        separately by :meth:`initial_slots`.
        """
        keys = self._configured_state_keys()
        slots = []
        seen = set()
        for param in params:
            for key in keys:
                slot = self._slots.get((id(param), key))
                if slot is None or id(slot) in seen:
                    continue
                slots.append(slot)
                seen.add(id(slot))
        return slots

    # ------------------------------------------------------------------ execution
    def iter_update_units(self, step_context: Dict[str, Any]) -> List[MuonSwapUnit]:
        """Return the units built by :meth:`prepare_step`."""
        return step_context["units"]

    def partition_costs(self, units: Sequence[MuonSwapUnit], step_context: Any) -> Sequence[int]:
        """Return logical momentum bytes without consulting local slots or gradients."""
        del step_context
        return [unit.logical_nbytes for unit in units]

    def step_batch(self, batch: List[MuonSwapUnit], step_context: Dict[str, Any]) -> None:
        """Run one prefetched batch of Muon update units.

        The runtime has already waited for this batch's H2D before calling in,
        so every slot the batch reads is device-resident here.
        """
        broadcaster = step_context["broadcaster"]
        for unit in batch:
            self._run_unit(unit, broadcaster)

    def _run_unit(self, unit: MuonSwapUnit, broadcaster: AsyncReplicateBroadcaster) -> None:
        """Run one atomic update unit with the matching Muon method."""
        kind = unit.kind
        if kind not in (_NO_COMM, _HSDP):
            raise ValueError(f"Unknown Muon swap unit kind {kind!r}.")
        group = self.optimizer.param_groups[unit.group_index]
        ns_inputs = self.optimizer._update_muon_momentum(group, unit.params)
        if kind == _NO_COMM:
            self._apply_unshard_update(group, ns_inputs)
        else:
            hsdp_assign = unit.hsdp_assign
            if hsdp_assign is None:
                raise RuntimeError("Muon swap HSDP unit is missing its group assignment.")
            if hsdp_assign.is_shard:
                self.optimizer._process_shard_params(
                    group, ns_inputs, [hsdp_assign], unit.group_index,
                    buffer_cache={},
                )
            else:
                self._apply_unshard_update(group, ns_inputs)
        if self.optimizer.post_update_fn is not None and unit.post_update_params:
            self.optimizer._run_post_update_fn(group, unit.post_update_params)
        # Flush at the global assignment boundary, after owner callbacks, even
        # when this rank has no local updates in the last logical unit.
        if unit.flushes_broadcast:
            broadcaster.flush_group(unit.hsdp_assign)

    def _apply_unshard_update(
            self,
            group: Dict[str, Any],
            ns_inputs: Dict[torch.nn.Parameter, torch.Tensor],
    ) -> None:
        """Run the unsharded NS and parameter update for one unit."""
        if not ns_inputs:
            return
        # Muon's own helper re-groups by shape and re-splits into memory-safe
        # batches; a unit's params are already one such batch, so the split is a
        # single group and the boundary cannot drift.
        self.optimizer._process_unshard_params(group, ns_inputs)

    def finish_step(self, step_context: Any) -> None:
        """Wait for every parameter broadcast issued during this step."""
        step_context["broadcaster"].wait_all()


def build_muon_swap_adapter(optimizer: Any, config: Any, runtime: Any) -> MuonSwapAdapter:
    """Build the Muon swap adapter for ``optimizer``.

    Args:
        optimizer: Muon optimizer instance.
        config: Resolved ``SwapOptimizerConfig``.
        runtime: Shared ``PipelineSwapRuntime``.

    Returns:
        The Muon swap adapter.

    Raises:
        ValueError: If ``optimizer`` is not a Muon optimizer.
    """
    if not isinstance(optimizer, Muon):
        raise ValueError(
            "Swap muon only supports hyper_parallel.core.optimizer.muon.Muon. "
            f"Got {type(optimizer)!r}."
        )
    return MuonSwapAdapter(optimizer, config, runtime)


def swap_muon(optimizer: Any, config: Any) -> Any:
    """Wrap a Muon optimizer with momentum-only state swap.

    Args:
        optimizer: Muon optimizer instance.
        config: Resolved ``SwapOptimizerConfig``. Packed swap preserves Muon's
            logical units and only coalesces their momentum transfers.

    Returns:
        The shared core swap-optimizer wrapper.
    """
    return CoreSwapOptimizer(
        optimizer,
        config,
        adapter_factory=build_muon_swap_adapter,
    )
