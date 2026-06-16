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
"""Warmup-phase executor.

During warmup, the executor records the execution trace while applying
online greedy eviction to keep device memory within budget.  Every
input tensor that was evicted in an earlier op is faulted back
synchronously (demand-paging) before dispatch.

Physical state transitions are delegated to
:class:`~offload.runtime.residency.ResidencyManager`.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.utils._pytree import tree_flatten

from hyper_parallel.auto_parallel.hyper_offload.execution.base import BaseExecutor
from hyper_parallel.auto_parallel.hyper_offload.runtime.timer import DeviceTimer
from hyper_parallel.auto_parallel.hyper_offload.execution.warmup.tracker import ActivationTracker
from hyper_parallel.auto_parallel.hyper_offload.ir.replay import OpGuide
from hyper_parallel.auto_parallel.hyper_offload.ir.trace import AccessKind, ActivationTrace, StorageAccess, TraceOp
from hyper_parallel.auto_parallel.hyper_offload.runtime.bandwidth import profile_transfer_bandwidth
from hyper_parallel.auto_parallel.hyper_offload.runtime.residency import ResidencyManager

logger = logging.getLogger(__name__)


def iter_tensors(value: Any) -> list[torch.Tensor]:
    """Return tensor leaves from an arbitrary pytree."""
    leaves, _ = tree_flatten(value)
    return [leaf for leaf in leaves if isinstance(leaf, torch.Tensor)]


class WarmupExecutor(BaseExecutor):
    """Executor for the warmup phase.

    Records the execution trace while applying online greedy eviction
    to keep device memory within the configured budget.  Evicts the
    **oldest** activations first when the memory budget is exceeded
    (oldest-first within the same op, largest-sized entries are
    preferred as tie-breaker to minimise eviction count).
    """

    def __init__(
        self,
        residency_manager: ResidencyManager,
        memory_limit_bytes: int,
    ) -> None:
        super().__init__(residency_manager)
        self._memory_limit_bytes = memory_limit_bytes
        self._tracker = ActivationTracker()
        self._timer = DeviceTimer()
        self._guide: list[OpGuide] = []
        self._ops: list[TraceOp] = []
        #: sid -> the op index that first produced this activation.
        self._sid_produced_at_op: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Eviction policy
    # ------------------------------------------------------------------

    def _enforce_budget(self, protected_sids: set[int]) -> None:
        """Evict warmup activations until resident bytes fit the configured budget."""
        while self.residency_manager.resident_bytes > self._memory_limit_bytes:
            # Greedy: evict oldest first; within the same op, evict largest first.
            victim_sid: int | None = None
            victim_key: tuple[int, int] | None = None

            for sid, produced_at_op in self._sid_produced_at_op.items():
                if sid in protected_sids:
                    continue
                size = self.residency_manager.device_resident_size(sid)
                if size is None:
                    continue
                key = (produced_at_op, -size)
                if victim_key is None or key < victim_key:
                    victim_key = key
                    victim_sid = sid

            if victim_sid is None:
                raise RuntimeError(
                    "Warmup memory budget exceeded but no evictable activation found. "
                    f"resident_bytes={self.residency_manager.resident_bytes}, "
                    f"limit={self._memory_limit_bytes}, "
                    f"protected_sids={protected_sids}"
                )

            self.residency_manager.copy_d2h(victim_sid)
            self.residency_manager.release_device(victim_sid)

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_op_begin(self, func, args, kwargs) -> None:
        """Before op: enforce memory budget and fault inputs back to device."""
        super().on_op_begin(func, args, kwargs)

        protected_sids: set[int] = set()
        for t in iter_tensors((args, kwargs)):
            if (sid := self._tracker.get_activation_sid(t)) is not None:
                protected_sids.add(sid)
        self._enforce_budget(protected_sids)

        self._timer.start()

    def on_op_end(self, result) -> Any:
        """After op: record trace, residency metadata, and return shadowed result."""
        op_duration_ms = self._timer.stop()

        func, args, kwargs = self._last_func, self._last_args, self._last_kwargs

        self._tracker.register_op_activations(iter_tensors((args, kwargs)), iter_tensors(result))

        op = TraceOp(name=func.__name__, duration_ms=op_duration_ms)

        # --- Detect mutated (write) input tensors via func._schema ---
        mutated_tensor_ids: set[int] = set()
        if hasattr(func, "_schema") and func._schema.is_mutable:  # pylint: disable=protected-access
            flat_args = iter_tensors((args, {}))
            for idx, arg_info in enumerate(func._schema.arguments):  # pylint: disable=protected-access
                if arg_info.alias_info is not None and arg_info.alias_info.is_write and idx < len(flat_args):  # pylint: disable=protected-access
                    mutated_tensor_ids.add(id(flat_args[idx]))

        seen: set[tuple[int, AccessKind]] = set()

        # --- Input accesses ---
        for t in iter_tensors((args, kwargs)):
            sid = self._tracker.get_activation_sid(t)
            if sid is None:
                continue
            is_mutated = id(t) in mutated_tensor_ids
            kind = AccessKind.WRITE if is_mutated else AccessKind.READ
            if (sid, kind) not in seen:
                seen.add((sid, kind))
                op.accesses.append(StorageAccess(self.op_idx, sid, kind))

        # --- Output accesses + bindings ---
        leaves, _ = tree_flatten(result)
        output_bindings: dict[int, int] = {}

        for leaf_index, t in enumerate(leaves):
            if not isinstance(t, torch.Tensor):
                continue
            sid = self._tracker.get_activation_sid(t)
            if sid is None:
                continue

            if (sid, AccessKind.WRITE) not in seen:
                seen.add((sid, AccessKind.WRITE))
                op.accesses.append(StorageAccess(self.op_idx, sid, AccessKind.WRITE))

            # Track the op that first produced this activation (used by
            # eviction policy: oldest-first).
            if sid not in self._sid_produced_at_op:
                self._sid_produced_at_op[sid] = self.op_idx

            output_bindings[leaf_index] = sid

        guide = OpGuide(
            name=func.__name__,
            output_leaf_count=len(leaves),
            output_bindings=output_bindings,
        )

        self._ops.append(op)
        self._guide.append(guide)

        return self.apply_shadows(result, output_bindings)

    def finish(self) -> tuple[ActivationTrace, list[OpGuide]]:
        """Finish warmup and return recorded trace and replay guide."""
        d2h, h2d = profile_transfer_bandwidth()

        trace = ActivationTrace(
            ops=list(self._ops),
            storage_sizes=self._tracker.storage_sizes,
            retained_sids=set(self.retained_sids),
            memory_limit_bytes=self._memory_limit_bytes,
            d2h_bandwidth_gbps=d2h,
            h2d_bandwidth_gbps=h2d,
        )
        guide = self._guide

        self.reset()
        return trace, guide

    def reset(self) -> None:
        """Reset."""
        self._sid_produced_at_op.clear()
        self._ops = []
        self._guide = []
        self._tracker.clear_activations()
        super().reset()
