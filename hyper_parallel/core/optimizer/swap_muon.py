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
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

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
        group_index: Index of the owning parameter group.
        kind: ``"no_comm"`` for locally orthonormalized params, ``"hsdp"`` for
            an HSDP shard/replicate assignment.
        params: Parameters this unit updates, in schedule order.
        slots: Swappable momentum-buffer slots for ``params``.
        hsdp_assign: The original assignment for ``"hsdp"`` units, else ``None``.
        flushes_broadcast: Whether this unit commits its assignment's parameter
            broadcast.  Only the last unit of an assignment sets it, so an
            assignment split into several NS batches still issues the single
            flush the bare schedule issues.
    """

    group_index: int
    kind: str
    params: List[torch.nn.Parameter]
    slots: List[SwapSlot]
    hsdp_assign: Optional[HSDPGroupAssignment] = None
    flushes_broadcast: bool = False


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

    supported_cls = (Muon,)

    @staticmethod
    def default_state_keys() -> Tuple[str, ...]:
        """Return the swap state keys a Muon optimizer owns."""
        return MUON_STATE_KEYS

    def validate(self) -> None:
        """Reject configurations the momentum-only swap path cannot preserve.

        Raises:
            ValueError: If a configured state key is not one Muon owns, or a
                parameter holds momentum state the runtime cannot rebind.
        """
        self._validate_configured_state_keys()
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
        if not local.is_contiguous():
            raise ValueError("Muon swap optimizer requires contiguous momentum state.")

    @staticmethod
    def _runtime_local_tensor(tensor: Any) -> Any:
        """Return the local shard of a possibly distributed tensor."""
        to_local = getattr(tensor, "to_local", None)
        return to_local() if callable(to_local) else tensor

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
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                self._ensure_momentum_state(param)

        units: List[MuonSwapUnit] = []
        num_groups = len(self.optimizer.param_groups)
        for group_index in range(num_groups):
            for params in self._iter_no_comm_batches(group_index):
                units.append(self._make_unit(group_index, _NO_COMM, params))

        linear_batches = [
            self._hsdp_linear_batches(group_index) for group_index in range(num_groups)
        ]
        max_num_batches = max((len(batches) for batches in linear_batches), default=0)
        for batch_index in range(max_num_batches):
            for group_index in range(num_groups):
                group_batches = linear_batches[group_index]
                if batch_index >= len(group_batches):
                    continue
                units.extend(self._make_hsdp_units(group_index, group_batches[batch_index]))

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

    def _iter_no_comm_batches(
            self,
            group_index: int,
    ) -> Iterator[List[torch.nn.Parameter]]:
        """Yield the atomic no-comm NS batches of one group.

        The yielded lists are exactly the shape-group / memory-safe batches
        ``Muon._process_unshard_params`` iterates internally, so the outer swap
        batch boundary never disagrees with the inner NS batch boundary.  A rank
        whose local params produce no batch simply contributes no no-comm unit.
        """
        no_comm_params = [
            param for param in self._no_comm_params(group_index)
            if getattr(param, "grad", None) is not None
        ]
        if not no_comm_params:
            return
        for _, shape_group in self.optimizer._group_by_shape(no_comm_params).items():
            yield from self.optimizer._split_into_memory_safe_batches(shape_group, shard_size=1)

    def _make_unit(
            self,
            group_index: int,
            kind: str,
            params: Iterable[torch.nn.Parameter],
            hsdp_assign: Optional[HSDPGroupAssignment] = None,
            flushes_broadcast: bool = False,
    ) -> MuonSwapUnit:
        """Build one update unit plus its momentum slots."""
        params = list(params)
        return MuonSwapUnit(
            group_index=group_index,
            kind=kind,
            params=params,
            slots=self._slots_for_params(params),
            hsdp_assign=hsdp_assign,
            flushes_broadcast=flushes_broadcast,
        )

    def _make_hsdp_units(
            self,
            group_index: int,
            hsdp_assign: HSDPGroupAssignment,
    ) -> List[MuonSwapUnit]:
        """Build the units for one HSDP assignment, in assignment order.

        An assignment that has no locally owned parameter still produces one
        (empty) unit, so every rank walks the same schedule length and reaches
        the same collective positions.  Communication membership is static per
        assignment, so a sharded assignment keeps the batches
        ``Muon._process_shard_params`` would all-gather; an unsharded assignment
        keeps the shape/memory batches ``Muon._process_unshard_params`` would
        iterate internally.
        """
        owned_params = [
            param for param in hsdp_assign.owned_params
            if getattr(param, "grad", None) is not None
        ]
        if not owned_params:
            return [self._make_unit(group_index, _HSDP, [], hsdp_assign, flushes_broadcast=True)]
        if hsdp_assign.is_shard:
            return [
                self._make_unit(group_index, _HSDP, owned_params, hsdp_assign, flushes_broadcast=True)
            ]
        _, _, _, total_shard_size = self.optimizer._get_shard_info(hsdp_assign)
        ns_batches = self.optimizer._split_into_memory_safe_batches(
            owned_params, shard_size=total_shard_size, min_batch_size=total_shard_size
        )
        return [
            self._make_unit(
                group_index, _HSDP, ns_batch, hsdp_assign,
                flushes_broadcast=index == len(ns_batches) - 1,
            )
            for index, ns_batch in enumerate(ns_batches)
        ]

    def initial_slots(self) -> Iterable[SwapSlot]:
        """Discover momentum state materialized before the swap wrapper existed.

        Muon creates momentum lazily, but a caller may run steps before wrapping
        (or load a checkpoint) and leave device-resident buffers behind.  Those
        are registered here so the wrapper can offload them before the first
        swapped step.

        Returns:
            The slots registered for the pre-existing momentum state.
        """
        slots = []
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                state = self.optimizer.state.get(param)
                if not state:
                    continue
                for key in self._configured_state_keys():
                    if key in state and (id(param), key) not in self._slots:
                        self._slots[(id(param), key)] = self._make_slot(key, state[key])
                slots.extend(self._slots_for_params([param]))
        return tuple(slots)

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
        if getattr(param, "grad", None) is None:
            return
        state = self.optimizer.state[param]
        for key in self._configured_state_keys():
            if (id(param), key) in self._slots:
                continue
            device_tensor = torch.empty_like(param, memory_format=torch.preserve_format)
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
        group = self.optimizer.param_groups[unit.group_index]
        ns_inputs = self._momentum_and_ns_inputs(group, unit)
        if unit.kind == _NO_COMM:
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
            # The assignment's parameters are broadcast once, after the last unit
            # that updates them -- exactly where the bare ``step()`` flushes too.
            # Every rank derives the flag from the same assignment metadata, so
            # the collective sequence stays identical across ranks.
            if unit.flushes_broadcast:
                broadcaster.flush_group(hsdp_assign)
        if self.optimizer.post_update_fn is not None:
            self.optimizer._run_post_update_fn(group, unit.params)

    def _momentum_and_ns_inputs(
            self,
            group: Dict[str, Any],
            unit: MuonSwapUnit,
    ) -> Dict[torch.nn.Parameter, torch.Tensor]:
        """Compute one unit's momentum and return its bf16 NS inputs.

        Units without a gradient are skipped, matching the bare step's
        behaviour for a parameter that never receives one.
        """
        if not any(getattr(param, "grad", None) is not None for param in unit.params):
            return {}
        return self.optimizer._update_muon_momentum(group, unit.params)

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

    Muon swaps states one tensor at a time: a packed A/B staging arena is
    rebound as a whole batch, which cannot be reconciled with per-unit
    Newton-Schulz buffers.  Requesting packed swap is therefore rejected rather
    than silently downgraded.

    Args:
        optimizer: Muon optimizer instance.
        config: Resolved ``SwapOptimizerConfig``; ``packed_swap`` must be False.

    Returns:
        The shared core swap-optimizer wrapper.

    Raises:
        ValueError: If ``packed_swap`` is enabled for a Muon optimizer.
    """
    if getattr(config, "packed_swap", False):
        raise ValueError(
            "Muon swap optimizer requires packed_swap=False: Muon momentum is "
            "swapped tensor by tensor."
        )
    return CoreSwapOptimizer(
        optimizer,
        config,
        adapter_factory=build_muon_swap_adapter,
    )
