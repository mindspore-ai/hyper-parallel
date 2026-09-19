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
"""Torch-only Muon optimizer state swap.

Unlike the Adam/AdamW swap optimizer this backend does not partition the
optimizer state into a prefetch pipeline.  ``Muon.step()`` computes its
Newton-Schulz updates and drives HSDP communication as one indivisible unit, so
the whole state is handed to the device once, the wrapped optimizer runs
untouched, and the state is handed back to the host once.

The offload half of that round trip is completed before ``step()`` returns.  The
D2H copies therefore do not overlap the next step's forward/backward pass, but
their device storage is released before that pass starts.  This keeps Muon's
momentum out of the forward/backward memory budget.
"""

import contextlib
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.optimizer.swap_optimizer_base import (
    MUON_STATE_KEYS,
    SUPPORTED_STATE_KEYS,
    OptimizerSwapAdapter,
    PipelineSwapRuntime,
    SwapSlot,
    UpdateUnit,
    validate_state_keys,
)


def _muon_cls() -> type:
    """Return hyper-parallel's ``Muon`` class without importing it at module load."""
    # pylint: disable=import-outside-toplevel,cyclic-import
    from hyper_parallel.core.optimizer.muon import Muon as muon_cls

    return muon_cls


@dataclass(frozen=True)
class SwapMuonConfig:
    """Configuration for Muon optimizer state swap.

    Muon keeps exactly one optimizer state tensor per parameter
    (``momentum_buffer``), so ``state_keys`` selects from that single slot and
    the whole state travels tensor by tensor.  There is no packing option:
    staging a batch's state in one shared device buffer only pays off when an
    optimizer owns several tensors per parameter, since the arena costs as much
    as the state it replaces.

    ``pipelined`` and ``packed_swap`` are accepted only so a configuration
    written for the Adam/AdamW wrapper can be reused verbatim; both are ignored.
    """

    swap_times: int = 16
    state_keys: Optional[Sequence[str]] = None
    min_numel: int = 1024
    pipelined: bool = True
    # Accepted and ignored, so an Adam-shaped config can be passed through.  The
    # runtime never sees this value: staging is disabled for Muon outright rather
    # than left off by default, because an arena would be allocated at wrap time
    # from this field alone.
    packed_swap: bool = False
    # Muon hands its whole state over at once, so there is nothing to partition.
    # Declared rather than patched so the frozen config keeps a stable field set.
    supports_pipelined: bool = False

    def __post_init__(self) -> None:
        if self.swap_times <= 0:
            raise ValueError("SwapMuonConfig.swap_times must be positive.")
        if self.min_numel < 0:
            raise ValueError("SwapMuonConfig.min_numel must be non-negative.")
        invalid = sorted(set(self.state_keys or ()) - set(SUPPORTED_STATE_KEYS))
        if invalid:
            raise ValueError(
                "SwapMuonConfig.state_keys only supports optimizer state slots "
                f"{SUPPORTED_STATE_KEYS}, but got {invalid}."
            )
        object.__setattr__(self, "state_keys", validate_state_keys(self.state_keys))


class MuonSwapAdapter(OptimizerSwapAdapter):
    """Adapter that swaps all of Muon's optimizer state in a single hand-over."""

    # Muon's step is already an indivisible batch: splitting the state across
    # pipeline batches would run Newton-Schulz on partial state.
    supports_pipelined = False

    @classmethod
    def matches(cls, optimizer: Any) -> bool:
        """Return whether ``optimizer`` is hyper-parallel's Muon."""
        return isinstance(optimizer, _muon_cls())

    def validate(self) -> None:
        """Reject options the swap path cannot preserve.

        ``ns_transform_fn`` hands Newton-Schulz a zero-copy view that must share
        storage with the working input.  Swapped state lives in host mirrors and
        staging buffers, so that sharing cannot be guaranteed; failing here keeps
        the mismatch from surfacing as a silent numerical difference.
        """
        if self.optimizer.ns_transform_fn is not None:
            raise ValueError("Swap Muon does not support ns_transform_fn.")

    @staticmethod
    def _default_state_keys() -> Tuple[str, ...]:
        return MUON_STATE_KEYS

    def prepare_step(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Collect the optimizer state this step needs before the update runs.

        Nothing is transferred here.  The whole-batch runtime binds the slots to
        device tensors and drives the H2D itself.  It also defensively settles a
        pending offload, although the Muon finish hook normally drains every D2H
        before returning from the preceding step.
        """
        if args or kwargs:
            raise ValueError("Swap Muon step does not support closure or extra arguments.")
        units = self._collect_units()
        self.runtime.set_whole_batch_units(units)
        return {"units": units}

    def _collect_units(self) -> List[UpdateUnit]:
        """Build one update unit per group from the state known before the update."""
        units: List[UpdateUnit] = []
        for group_index, group in enumerate(self.optimizer.param_groups):
            owned = []
            for param in group["params"]:
                state = self.optimizer.state.get(param)
                if not state or "momentum_buffer" not in state:
                    continue
                slot = self._prepare_slot(param, state)
                if slot is not None:
                    owned.append(slot)
            if owned:
                units.append(UpdateUnit(
                    adapter_index=group_index,
                    param=group["params"][0],
                    grad=None,
                    slots=owned,
                ))
        return units

    def bind_slots_for_device(self, units: Sequence[UpdateUnit]) -> None:
        """Point the optimizer state at the tensor each slot will be updated in.

        The runtime has already moved values into each slot's device tensor, so
        this rebuilds the storage if a previous offload released it and then
        repoints the state mapping at it.  Muon updates that tensor in place,
        which is what the D2H copy afterwards carries back to the mirror.
        """
        for unit in units:
            for slot in unit.slots:
                param = self._slot_owner(slot)
                if param is None:
                    continue
                device_tensor = self.runtime.ensure_device_storage(slot)
                slot.bind_tensor(device_tensor)
                slot.state = "device"
                slot.event = None
                # The mapping is written unconditionally: after a checkpoint load
                # the state may hold a host view or be missing the key entirely.
                self.optimizer.state[param][slot.name] = self.runtime.storage_tensor(slot.tensor)

    def iter_update_units(self, step_context: Dict[str, Any]) -> List[UpdateUnit]:
        """Return this step's update unit."""
        return step_context["units"]

    def refresh_swappable_slots(self, batch: Sequence[UpdateUnit]) -> None:
        """Adopt momentum buffers the wrapped step allocated on the device.

        Muon allocates ``momentum_buffer`` with ``zeros_like`` inside its own
        ``step``, so the first update writes into a tensor the pre-step batch
        never saw.  Any such buffer becomes a slot of its own, with a host mirror
        for the offload copy to fill.
        """
        del batch
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                state = self.optimizer.state.get(param)
                if not state or "momentum_buffer" not in state:
                    continue
                self._adopt_step_slot(param, state)

    def _slot_owner(self, slot: SwapSlot) -> Optional[Any]:
        """Return the live parameter a slot belongs to.

        The ownership recorded on the slot can go stale: loading a checkpoint
        rebuilds the slot registry from the saved state, and those slots are not
        created through this adapter's discovery path, so they never recorded an
        owner.  The registry key is the source of truth instead.
        """
        for (param_id, key), registered in self._slots.items():
            if registered is slot and key == slot.name:
                for group in self.optimizer.param_groups:
                    for param in group["params"]:
                        if id(param) == param_id:
                            slot.owner_param = param
                            return param
        return slot.owner_param

    def step_batch(self, batch: List[UpdateUnit], step_context: Dict[str, Any]) -> Any:
        """Run the wrapped Muon step once for the whole optimizer state."""
        del batch, step_context
        return self.optimizer.step()

    def finish_step(self, step_context: Any) -> Any:
        """Finish D2H and release momentum storage before the next forward.

        The first Muon step creates momentum lazily, so the whole-batch runtime
        cannot enqueue those newly discovered slots from its pre-step unit list.
        ``offload_initial_slots`` closes that first-step gap; on later steps it
        ignores slots whose D2H is already pending.  ``synchronize`` then drains
        the regular whole-batch offload and releases its device storage.
        """
        del step_context
        self.runtime.offload_initial_slots(self.all_slots())
        self.runtime.synchronize()

    def _prepare_slot(self, param: Any, state: Dict[str, Any]) -> Optional[SwapSlot]:
        """Return ``param``'s swap slot, creating it on first discovery.

        A known slot is returned untouched: whether its values currently sit on
        the device or in its host mirror is the runtime's business, and the slot's
        own state records which.
        """
        key = "momentum_buffer"
        slot = self._slots.get((id(param), key))
        if slot is not None and slot.swappable:
            if self.runtime.storage_tensor(slot.tensor).device.type == "cpu":
                # Parked on the host mirror: after a checkpoint load the state
                # mapping points here, so point it back at the slot as well.
                self.optimizer.state[param][key] = slot.tensor
            return slot
        self._slots.pop((id(param), key), None)
        slot = self._make_slot(key, state[key])
        slot.owner_param = param
        self._slots[(id(param), key)] = slot
        if not slot.swappable:
            return None
        if slot.cpu_tensor is None:
            slot.cpu_tensor = self.runtime.make_cpu_tensor(slot.tensor)
        # Park on the mirror until ``bind_slots_for_device`` rebuilds the device
        # side, so a step that runs without a transfer still reads defined values.
        slot.bind_tensor(slot.cpu_tensor)
        slot.state = "host"
        slot.event = None
        # A checkpoint load can have rebuilt the state mapping to hold a host
        # view, so it is pointed at the slot's tensor here as well.
        self.optimizer.state[param][key] = slot.tensor
        return slot

    def _adopt_step_slot(self, param: Any, state: Dict[str, Any]) -> Optional[SwapSlot]:
        """Register a momentum buffer the wrapped step just allocated on device."""
        key = "momentum_buffer"
        known = self._slots.get((id(param), key))
        if known is not None:
            return known
        slot = self._make_slot(key, state[key])
        slot.owner_param = param
        self._slots[(id(param), key)] = slot
        if not slot.swappable:
            return None
        if slot.cpu_tensor is None:
            slot.cpu_tensor = self.runtime.make_cpu_tensor(slot.tensor)
        # The slot keeps pointing at the live device tensor: the offload that
        # follows has to copy out of it, and ``bind_slots_for_device`` repoints
        # the state mapping at whatever tensor the slot holds next step.
        return slot


class SwapMuonOptimizer(torch.optim.Optimizer):
    """Torch optimizer wrapper for Muon state swap."""

    _is_swap_optimizer = True
    _is_swap_muon_optimizer = True
    _adapters = (MuonSwapAdapter,)

    def __init__(self, optimizer: Any, config: Optional[Any] = None) -> None:
        """Wrap ``optimizer`` with whole-state Muon swap."""
        # Mirrors the Adam/AdamW wrapper: the wrapped optimizer already owns
        # param_groups/state/defaults, so ``torch.optim.Optimizer.__init__`` is
        # deliberately skipped while inheriting for scheduler and isinstance
        # compatibility.
        if not MuonSwapAdapter.matches(optimizer):
            raise ValueError(
                "Swap Muon only supports hyper_parallel.core.optimizer.muon.Muon, "
                f"got {type(optimizer)!r}."
            )
        self.optimizer = optimizer
        resolved = config or SwapMuonConfig()
        if not isinstance(resolved, SwapMuonConfig):
            raise ValueError(
                f"Swap Muon expects a SwapMuonConfig, got {type(resolved)!r}."
            )
        # Staging is disabled outright rather than merely defaulted off: the
        # runtime reads ``packed_swap`` when it decides whether to build host
        # packed buffers and a staging arena, so an inherited Adam config that
        # enables packing would otherwise allocate both at wrap time.  A new
        # config is built rather than mutating the caller's, which is frozen.
        if resolved.packed_swap:
            resolved = SwapMuonConfig(
                swap_times=resolved.swap_times,
                state_keys=resolved.state_keys,
                min_numel=resolved.min_numel,
                pipelined=resolved.pipelined,
            )
        self.config = resolved
        self.runtime = self._build_runtime(self.config)
        self.adapter = self._build_adapter()
        self.runtime.adapter = self.adapter
        self.adapter.validate()
        # Muon allocates its momentum buffer lazily on the first step, but a
        # caller may have already run one step before wrapping.
        initial_slots = tuple(self.adapter.initial_slots())
        self.runtime.offload_initial_slots(initial_slots)

    @staticmethod
    def _build_runtime(config: Any) -> Any:
        """Return the Torch-only core swap runtime."""
        return PipelineSwapRuntime(config)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the base optimizer."""
        return getattr(self.optimizer, name)

    @property
    def param_groups(self) -> Any:
        """Proxy parameter groups."""
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value: Any) -> None:
        """Replace the wrapped optimizer's parameter groups."""
        self.optimizer.param_groups = value

    @property
    def state(self) -> Any:
        """Proxy optimizer state."""
        return self.optimizer.state

    @property
    def defaults(self) -> Any:
        """Proxy optimizer defaults."""
        return self.optimizer.defaults

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """Proxy param group addition."""
        self.optimizer.add_param_group(param_group)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Proxy gradient clearing."""
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Optional[Any] = None) -> Any:
        """Run one Muon step, handing all optimizer state device-side for its duration."""
        if closure is not None:
            raise ValueError("Swap Muon does not support closure.")
        with self._no_grad_context():
            step_context = self.adapter.prepare_step()
            units = self.adapter.iter_update_units(step_context)
            batches = self.runtime.partition(units)
            self.runtime.run_pipeline(batches, step_context, self.adapter.step_batch)
            return self.adapter.finish_step(step_context)

    def synchronize(self) -> None:
        """Settle transfers explicitly; normally a no-op after ``step`` returns."""
        self.runtime.synchronize()

    def state_dict(self) -> Dict[str, Any]:
        """Return optimizer state dict using host mirrors for swapped tensors."""
        self.runtime.synchronize()
        return self.adapter.checkpoint_state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state dict while keeping swapped tensors on host mirrors.

        Loading only rewrites host mirrors and slot metadata.  A normal Muon step
        has no transfer in flight when it returns.
        """
        self.adapter.load_checkpoint_state_dict(state_dict)

    def _build_adapter(self) -> MuonSwapAdapter:
        return MuonSwapAdapter(self.optimizer, self.config, self.runtime)

    @contextlib.contextmanager
    def _no_grad_context(self) -> Iterator[None]:
        with torch.no_grad():
            yield


def swap_muon(optimizer: Any, config: Optional[Any] = None) -> Any:
    """Wrap a Muon optimizer with whole-state swap.

    Args:
        optimizer: A ``hyper_parallel.core.optimizer.muon.Muon`` instance.
        config: A :class:`SwapMuonConfig`, or an ``SwapOptimizerConfig`` whose
            transport options are copied over.

    Returns:
        A Torch-only :class:`SwapMuonOptimizer` wrapper.
    """
    resolved = _resolve_config(config)
    return SwapMuonOptimizer(optimizer, resolved)


def _resolve_config(config: Optional[Any]) -> SwapMuonConfig:
    """Normalize a caller-supplied config into a staging-free ``SwapMuonConfig``.

    Staging is forced off for every entry point.  The runtime reads
    ``packed_swap`` when it decides whether to build host packed buffers and a
    staging arena, so a config that enables packing -- notably an Adam-shaped one
    handed over for convenience -- would otherwise make the Muon path allocate
    both at wrap time, even though Muon has no packed transport.  A new config is
    built instead of mutating the caller's, which is frozen.
    """
    if config is None:
        return SwapMuonConfig()

    if isinstance(config, SwapMuonConfig):
        if not config.packed_swap:
            return config
        return SwapMuonConfig(
            swap_times=config.swap_times,
            state_keys=config.state_keys,
            min_numel=config.min_numel,
            pipelined=config.pipelined,
        )

    # Keep the public factory's Adam-shaped config reusable without importing
    # its defining module back into this one.  That module lazily imports
    # ``swap_muon`` for dispatch, so a concrete type check would create a cycle.
    required_fields = ("swap_times", "state_keys", "min_numel", "packed_swap")
    if all(hasattr(config, field) for field in required_fields):
        return SwapMuonConfig(
            swap_times=config.swap_times,
            state_keys=config.state_keys,
            min_numel=config.min_numel,
            pipelined=getattr(config, "pipelined", True),
        )

    raise ValueError(
        "Swap Muon expects a SwapMuonConfig or SwapOptimizerConfig, "
        f"got {type(config)!r}."
    )


def is_swap_muon_optimizer(optimizer: Any) -> bool:
    """Return whether ``optimizer`` is a Muon swap wrapper."""
    if bool(getattr(optimizer, "_is_swap_muon_optimizer", False)):
        return True
    return isinstance(getattr(optimizer, "adapter", None), MuonSwapAdapter)


def muon_state_keys() -> Tuple[str, ...]:
    """Return the optimizer state keys the Muon swap path can offload."""
    return MUON_STATE_KEYS


__all__ = [
    "MUON_STATE_KEYS",
    "MuonSwapAdapter",
    "SwapMuonConfig",
    "SwapMuonOptimizer",
    "is_swap_muon_optimizer",
    "muon_state_keys",
    "swap_muon",
]
