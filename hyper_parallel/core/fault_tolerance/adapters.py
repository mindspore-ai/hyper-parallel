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

from typing import Any

import torch

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.random import OffsetBasedRNGTracker
from hyper_parallel.core.shard._op_dispatch import _OP_DISPATCHER


class ModelRecoveryAdapter:
    """Bind persistent local state for unsharded, TP, EP and optional HSDP modules.

    TP/EP collective work is owned by the interrupted autograd execution and
    released when the SDK unwinds that body. HSDP additionally retains work,
    views and bucket references in schedulers, handled by its optional adapter.
    """

    def __init__(self, model: torch.nn.Module, reboot: bool = False) -> None:
        """Check materialized storage and collect optional scheduler cleanup hooks."""
        parameters = list(model.parameters())
        if getattr(model, "_hp_external_state", None) is not None:
            raise ValueError("model-owned external state requires its own recovery provider")
        if not parameters or any(parameter.device.type == "meta" for parameter in parameters):
            raise ValueError("recovery requires materialized persistent local parameters")
        if any(parameter.dtype == torch.float16 for parameter in parameters):
            raise ValueError("FP16 loss scaling is not supported by process recovery")
        if reboot and any(isinstance(parameter, DTensor) for parameter in parameters):
            if _OP_DISPATCHER._rng_tracker is None:  # pylint: disable=protected-access
                # Survivors already have a tracker. A replacement must not
                # trigger the lazy tracker's world broadcast on first use.
                tracker = OffsetBasedRNGTracker(run_state_sync=False)
                _OP_DISPATCHER._rng_tracker = tracker  # pylint: disable=protected-access
        self._schedulers: list[Any] = []
        seen = set()
        for module in model.modules():
            scheduler = getattr(module, "hsdp_scheduler", None)
            if scheduler is None or id(scheduler) in seen:
                continue
            state = scheduler.hsdp_state
            if state.offload_policy is not None and type(state.offload_policy).__name__ != "OffloadPolicy":
                raise ValueError("HSDP CPU offload is not supported by process recovery")
            self._schedulers.append(scheduler)
            seen.add(id(scheduler))

    def abort_iteration(self) -> None:
        """Clear retained strategy state after device work abort and main-thread ACK."""
        for scheduler in self._schedulers:
            scheduler.abort_iter_state()
