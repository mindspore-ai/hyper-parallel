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
"""Model-runtime cleanup adapters separate from recovery scheduling and transport."""

from typing import Any, Callable

import torch

from hyper_parallel.core.activation_memory.swap import SwapManager
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.random import OffsetBasedRNGTracker
from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy
from hyper_parallel.core.shard._op_dispatch import _OP_DISPATCHER
from hyper_parallel.high_availability.validation import validate_recovery_model


class ModelRecoveryAdapter:
    """Bind persistent local state for TP/EP, CP and optional HSDP modules.

    TP/EP collective work is owned by the interrupted autograd execution and
    released when the SDK unwinds that body. HSDP additionally retains work,
    views and bucket references in schedulers, handled by its optional adapter.
    """

    def __init__(self, model: torch.nn.Module, reboot: bool = False) -> None:
        """Check materialized storage and collect optional scheduler cleanup hooks."""
        validate_recovery_model(model)
        if reboot and any(isinstance(parameter, DTensor) for parameter in model.parameters()):
            if _OP_DISPATCHER._rng_tracker is None:  # pylint: disable=protected-access
                # Survivors already have a tracker. A replacement must not
                # trigger the lazy tracker's world broadcast on first use.
                tracker = OffsetBasedRNGTracker(run_state_sync=False)
                _OP_DISPATCHER._rng_tracker = tracker  # pylint: disable=protected-access
        self._schedulers: list[Any] = []
        self._swap_modules = []
        self._has_cpu_offload = False
        self._cp_async_states = {}
        self._model_modules = tuple(model.modules())
        seen = set()
        for module in self._model_modules:
            cp_state = vars(module).get("_context_parallel_async_state")
            if cp_state is not None:
                self._cp_async_states[id(cp_state)] = cp_state
            if "_swap_group_name" in vars(module):
                self._swap_modules.append(module)
            scheduler = getattr(module, "hsdp_scheduler", None)
            if scheduler is None or id(scheduler) in seen:
                continue
            state = scheduler.hsdp_state
            self._has_cpu_offload |= isinstance(state.offload_policy, CPUOffloadPolicy)
            self._schedulers.append(scheduler)
            seen.add(id(scheduler))

    def abort_iteration(self) -> None:
        """Clear retained strategy state after device work abort and main-thread ACK."""
        for scheduler in self._schedulers:
            scheduler.abort_iter_state()

    def reset_after_device_restart(self, synchronize: Callable[[], None]) -> None:
        """Drop swap and CP state after the device becomes usable.

        Args:
            synchronize: Device fence for CPU-offload transfers and asynchronous
                CP buffers when activation swap has not already fenced the device.
        """
        # Swap reset already synchronizes the device, covering concurrent CP
        # and CPU-offload transfers too. The elif avoids a duplicate fence
        # when these features are enabled together.
        if self._swap_modules:
            SwapManager().reset_after_device_restart(self._swap_modules)
        elif self._cp_async_states or self._has_cpu_offload:
            synchronize()
        for state in self._cp_async_states.values():
            state.reset_after_device_restart()
        for module in self._model_modules:
            vars(module).pop("_context_parallel_output_layout_stack", None)
