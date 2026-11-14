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
"""Replay-phase executor.

The executor executes residency schedule actions (D2H/H2D, device/host
release) and validates replayed outputs against the warmup trace.
Physical state transitions are delegated to
:class:`~offload.runtime.residency.ResidencyManager`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from torch.utils._pytree import tree_flatten

from hyper_parallel.auto_parallel.hyper_offload.execution.base import BaseExecutor
from hyper_parallel.auto_parallel.hyper_offload.execution.framework import dispatch_framework_read
from hyper_parallel.auto_parallel.hyper_offload.execution.tensor import ShadowTensor
from hyper_parallel.auto_parallel.hyper_offload.ir.replay import OpGuide
from hyper_parallel.auto_parallel.hyper_offload.ir.schedule import (
    ResidencyActionType,
    ResidencySchedule,
)
from hyper_parallel.auto_parallel.hyper_offload.runtime.residency import (
    ResidencyManager,
)

logger = logging.getLogger(__name__)


class ReplayExecutor(BaseExecutor):
    """Executor for the replay phase.

    Executes residency schedule actions (D2H/H2D, device/host release)
    and validates replayed outputs against the warmup trace.
    """

    def __init__(
        self,
        residency_manager: ResidencyManager,
        schedule: ResidencySchedule,
        guide: list[OpGuide],
    ) -> None:
        """Bind a recorded trace and select the storages requiring replay management."""
        super().__init__(residency_manager)
        self.native_saved_replay = False
        self._schedule = schedule
        self._guide = guide
        # Transfer storage keeps wrapper identity; unrelated activations remain
        # native. Wrapper replay can add detach and virtual FSDP-read records
        # that native replay omits, so on_op_begin advances over those entries.
        # Keep legacy hand-built guides on their original contract.
        transfer_sids = {
            action.storage_id
            for phase in (schedule.pre, schedule.post)
            for actions in phase.values()
            for action in actions
            if action.kind in (ResidencyActionType.COPY_D2H, ResidencyActionType.COPY_H2D)
        }
        self._managed_sids = self._select_managed_sids(guide, transfer_sids)
        self._requires_dispatch = self._managed_sids is None or bool(transfer_sids)
        self._replay_output_bindings = [self._managed_bindings(op.output_bindings) for op in guide]
        self._replay_input_bindings = [
            self._managed_bindings(op.input_bindings) if op.input_bindings is not None else None
            for op in guide
        ]
        self._active_ops = [self._op_requires_management(index) for index in range(len(guide))]
        self._framework_ops = [op.name.startswith("framework_read:") for op in guide]
        self._dispatch_names = [op.name.removeprefix("framework_read:") for op in guide]
        logger.info(
            "Replay management: transfer_storages=%d, managed_storages=%s, wrapped_outputs=%d/%d",
            len(transfer_sids),
            len(self._managed_sids) if self._managed_sids is not None else "all",
            sum(len(bindings) for bindings in self._replay_output_bindings),
            sum(len(op.output_bindings) for op in guide),
        )

    @property
    def requires_dispatch(self) -> bool:
        """Whether replay needs tensor management for any planned transfer."""
        return self._requires_dispatch

    def _select_managed_sids(self, guide: list[OpGuide], transfer_sids: set[int]) -> set[int] | None:
        """Retain legacy ownership and expand transfer-boundary dependencies."""
        if any(op.input_bindings is None for op in guide):
            return None
        managed_sids = set(transfer_sids)
        for op_idx, op in enumerate(guide):
            if op.name == "detach.default" or op.name.startswith("framework_read:"):
                actions = self._schedule.pre_actions(op_idx) + self._schedule.post_actions(op_idx)
                if any(action.storage_id in transfer_sids for action in actions):
                    managed_sids.update(op.input_bindings.values())
        return managed_sids

    def _managed_bindings(self, bindings: dict[int, int]) -> dict[int, int]:
        if self._managed_sids is None:
            return bindings
        return {index: sid for index, sid in bindings.items() if sid in self._managed_sids}

    def _input_bindings(self, leaves: list[Any]) -> dict[int, int]:
        return {
            index: value.storage_id
            for index, value in enumerate(leaves)
            if isinstance(value, ShadowTensor)
            and (self._managed_sids is None or value.storage_id in self._managed_sids)
        }

    def _op_requires_management(self, index: int) -> bool:
        return (
            self._managed_sids is None
            or bool(self._replay_input_bindings[index])
            or bool(self._replay_output_bindings[index])
            or self._has_actions(index)
        )

    def _has_actions(self, op_idx: int) -> bool:
        return any(
            self._managed_sids is None or action.storage_id in self._managed_sids
            for action in self._schedule.pre_actions(op_idx) + self._schedule.post_actions(op_idx)
        )

    def _is_elidable(self, op_idx: int) -> bool:
        name = self._guide[op_idx].name
        return not self._active_ops[op_idx] and (
            name == "detach.default" or name.startswith("framework_read:")
        )

    def _next_op(self, name: str) -> int:
        index = self.op_idx + 1
        while (
            index < len(self._guide)
            and name != self._guide[index].name
            and name != self._dispatch_names[index]
            and self._is_elidable(index)
        ):
            index += 1
        return index

    def dispatch(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Run unrelated native operations without building ownership pytrees.

        Args:
            func: Replayed tensor operation.
            args: Positional operation inputs.
            kwargs: Keyword operation inputs.
        """
        if self.in_opaque_region:
            return func(*args, **kwargs)
        index = self.op_idx + 1
        if index < len(self._guide) and func.__name__ == self._dispatch_names[index]:
            if not self._active_ops[index]:
                self.op_idx = index
                return func(*args, **(kwargs or {}))
        else:
            index = self._next_op(func.__name__)
        return self._dispatch_slow(func, args, kwargs or {}, index)

    def _dispatch_slow(self, func: Callable, args: tuple, kwargs: dict, index: int) -> Any:
        """Handle elided records and managed ownership boundaries."""
        if self._managed_sids is not None and func.__name__ == "detach.default":
            leaves, _ = tree_flatten((args, kwargs))
            if not self._input_bindings(leaves) and (
                index >= len(self._guide)
                or self._dispatch_names[index] != "detach.default"
            ):
                # Checkpoint recomputation can add native detaches when fewer
                # outputs are subclasses. They own no planned transfer state.
                return func(*args, **kwargs)
        if index < len(self._guide) and self._framework_ops[index] and func.__name__ == self._dispatch_names[index]:
            if not self._active_ops[index]:
                self.op_idx = index
                return func(*args, **(kwargs or {}))
            self.op_idx = index - 1
            return dispatch_framework_read(
                self, func, args, kwargs or {}, native_inputs=self._replay_input_bindings[index] == {},
            )
        if (
            index < len(self._guide)
            and not self._active_ops[index]
            and not func.__name__.startswith("framework_read:")
        ):
            if func.__name__ != self._guide[index].name:
                raise RuntimeError(
                    f"replay operator differs at op={index}: expected={self._guide[index].name}, actual={func.__name__}"
                )
            self.op_idx = index
            return func(*args, **kwargs)
        self.op_idx = index - 1
        return super().dispatch(func, args, kwargs)

    def execute_opaque_op(
        self,
        func_name: str,
        fn: Callable,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """Run an opaque boundary and bind its native replay outputs.

        Args:
            func_name: Name recorded for the opaque boundary.
            fn: Native callable inside the boundary.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        if self.native_saved_replay or not self.requires_dispatch:
            return fn(*args, **kwargs)
        return super().execute_opaque_op(func_name, fn, args, kwargs)

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_op_begin(self, func: Callable, args: tuple, kwargs: dict) -> None:
        """Before op: execute pre-actions."""
        super().on_op_begin(func, args, kwargs)
        if self.op_idx >= len(self._guide):
            raise RuntimeError(
                "replay op count exceeds warmup trace: "
                f"op_idx={self.op_idx}, "
                f"trace_ops={len(self._guide)}, "
                f"op={func.__name__}"
            )

        guide = self._guide[self.op_idx]
        actual_bindings = self._input_bindings(self._last_flat_inputs[0])
        while (
            (
                func.__name__ != guide.name
                or actual_bindings != (self._replay_input_bindings[self.op_idx] or {})
            )
            and self._is_elidable(self.op_idx)
        ):
            self.op_idx += 1
            if self.op_idx >= len(self._guide):
                raise RuntimeError(
                    f"replay operator differs at op={self.op_idx}: actual={func.__name__}"
                )
            guide = self._guide[self.op_idx]

        if guide.input_bindings is not None and func.__name__ != guide.name:
            raise RuntimeError(
                f"replay operator differs at op={self.op_idx}: expected={guide.name}, actual={func.__name__}"
            )
        if guide.input_bindings is not None:
            expected_bindings = self._replay_input_bindings[self.op_idx]
            if actual_bindings != expected_bindings:
                raise RuntimeError(
                    f"replay input bindings differ at op={self.op_idx}: "
                    f"name={guide.name}, "
                    f"expected_bindings={expected_bindings}, "
                    f"actual_bindings={actual_bindings}"
                )

        for action in self._schedule.pre_actions(self.op_idx):
            if action.kind == ResidencyActionType.COPY_H2D:
                self.residency_manager.copy_h2d(action.storage_id)
            elif action.kind == ResidencyActionType.RELEASE_HOST:
                self.residency_manager.release_host(action.storage_id)
            else:
                raise RuntimeError(
                    f"unsupported pre action {action} at op={self.op_idx}"
                )

    def on_op_end(self, result: Any) -> Any:
        """After op: validate output structure, shadow outputs, then execute post-actions.

        The order is critical: :meth:`apply_shadows` must run **before**
        post-actions so that output tensors are registered in the residency
        table (via :meth:`bind`) before the scheduler tries to copy or
        release them.  The previous order (post-actions → apply_shadows)
        caused ``COPY_D2H`` / ``RELEASE_DEVICE`` to silently skip output
        sids that had not yet been bound, leaving stale device buffers
        and missing host copies.
        """
        func = self._last_func
        op_guide = self._guide[self.op_idx]
        leaves, output_spec = tree_flatten(result)
        if len(leaves) != op_guide.output_leaf_count:
            raise RuntimeError(
                "replay output structure differs from warmup trace: "
                f"op_idx={self.op_idx}, "
                f"name={op_guide.name}, "
                f"expected_leaves={op_guide.output_leaf_count}, "
                f"actual_leaves={len(leaves)}, "
                f"op={func.__name__}"
            )

        # 1. Shadow outputs first — binds PhysicalBuffers so that
        #    post-actions can find and operate on them.
        result = self.apply_shadows(
            result,
            self._replay_output_bindings[self.op_idx],
            flattened=(leaves, output_spec),
        )

        # 2. Execute post-actions (COPY_D2H, RELEASE_DEVICE, RELEASE_HOST).
        for action in self._schedule.post_actions(self.op_idx):
            sid = action.storage_id
            if action.kind == ResidencyActionType.COPY_D2H:
                self.residency_manager.copy_d2h(sid)
            elif action.kind == ResidencyActionType.RELEASE_DEVICE:
                self.residency_manager.release_device(sid)
            elif action.kind == ResidencyActionType.TRY_RELEASE_DEVICE:
                self.residency_manager.try_release_device(sid)
            elif action.kind == ResidencyActionType.RELEASE_HOST:
                self.residency_manager.release_host(sid)
            else:
                raise RuntimeError(
                    f"unsupported post action {action} at op={self.op_idx}"
                )

        # Sparse replay may execute many native ops before the next managed
        # boundary. Do not retain its input tensors throughout that interval.
        self._last_func = None
        self._last_args = None
        self._last_kwargs = None
        self._last_flat_inputs = None
        return result

    def finish(self) -> None:
        """Reject truncated steps as well as steps with additional operations."""
        while self.op_idx + 1 < len(self._guide) and self._is_elidable(self.op_idx + 1):
            self.op_idx += 1
        if self.op_idx + 1 != len(self._guide):
            raise RuntimeError(
                f"replay op count differs from warmup trace: expected={len(self._guide)}, actual={self.op_idx + 1}"
            )
