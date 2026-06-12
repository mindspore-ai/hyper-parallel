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
from typing import Any

from torch.utils._pytree import tree_flatten

from hyper_parallel.auto_parallel.hyper_offload.execution.base import BaseExecutor
from hyper_parallel.auto_parallel.hyper_offload.ir.replay import OpGuide
from hyper_parallel.auto_parallel.hyper_offload.ir.schedule import ResidencyActionType, ResidencySchedule
from hyper_parallel.auto_parallel.hyper_offload.runtime.residency import ResidencyManager

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
        super().__init__(residency_manager)
        self._schedule = schedule
        self._guide = guide

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_op_begin(self, func, args, kwargs) -> None:
        """Before op: execute pre-actions."""
        super().on_op_begin(func, args, kwargs)
        if self.op_idx >= len(self._guide):
            raise RuntimeError(
                "replay op count exceeds warmup trace: "
                f"op_idx={self.op_idx}, "
                f"trace_ops={len(self._guide)}, "
                f"op={func.__name__}"
            )

        for action in self._schedule.pre_actions(self.op_idx):
            if action.kind == ResidencyActionType.COPY_H2D:
                self.residency_manager.copy_h2d(action.storage_id)
            else:
                raise RuntimeError(f"unsupported pre action {action} at op={self.op_idx}")

    def on_op_end(self, result) -> Any:
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
        leaves, _ = tree_flatten(result)
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
        result = self.apply_shadows(result, op_guide.output_bindings)

        # 2. Execute post-actions (COPY_D2H, RELEASE_DEVICE, RELEASE_HOST).
        for action in self._schedule.post_actions(self.op_idx):
            sid = action.storage_id
            if action.kind == ResidencyActionType.COPY_D2H:
                self.residency_manager.copy_d2h(sid)
            elif action.kind == ResidencyActionType.RELEASE_DEVICE:
                self.residency_manager.release_device(sid)
            elif action.kind == ResidencyActionType.RELEASE_HOST:
                self.residency_manager.release_host(sid)
            else:
                raise RuntimeError(f"unsupported post action {action} at op={self.op_idx}")

        return result
