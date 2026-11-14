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
"""Activation offload session built on TorchDispatchMode."""

from __future__ import annotations

import contextvars
import logging
from types import TracebackType

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode_stack

from hyper_parallel.auto_parallel.hyper_offload.api.config import OffloadConfig
from hyper_parallel.auto_parallel.hyper_offload.api.scope import (
    dispatch_framework_read,
    has_managed_inputs,
    inside_fsdp_runtime,
)
from hyper_parallel.auto_parallel.hyper_offload.execution.saved_tensors import (
    SavedNative, SavedTensorRecorder, SavedTensorReplay,
)
from hyper_parallel.auto_parallel.hyper_offload.execution.replay import ReplayExecutor
from hyper_parallel.auto_parallel.hyper_offload.execution.warmup import WarmupExecutor
from hyper_parallel.auto_parallel.hyper_offload.planning.peak import (
    PeakResidencyPlanner,
)
from hyper_parallel.auto_parallel.hyper_offload.runtime.residency import (
    ResidencyManager,
)

logger = logging.getLogger(__name__)

# Context variable tracking the active OffloadSession (set on __enter__).
_active_session: contextvars.ContextVar[OffloadSession | None] = contextvars.ContextVar("_active_session", default=None)


def _pack_native(tensor: torch.Tensor) -> SavedNative:
    return SavedNative(tensor.detach(), tensor._version)


def _unpack_native(packed: SavedNative) -> torch.Tensor:
    if packed.tensor._version != packed.version:
        raise RuntimeError("a tensor saved for backward was modified in place")
    return packed.tensor


class ActivationDispatchMode(TorchDispatchMode):
    """Dispatch mode that delegates to a :class:`BaseExecutor`.

    The executor (warmup or replay) is selected by
    :class:`OffloadSession` and never swapped while this mode
    is active.
    """

    def __init__(self, session: OffloadSession, *, record_framework_ops: bool = False) -> None:
        """Bind this dispatch mode to the active offload session."""
        super().__init__()
        self.session = session
        self._executor = session.executor
        self._record_framework_ops = record_framework_ops
        self._replay_framework_ops = record_framework_ops and isinstance(session.executor, ReplayExecutor)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # pylint: disable=unused-argument
        """Dispatch a torch operation."""
        # Opaque boundaries own their inner lifetimes. Native tensors need no
        # framework ownership check; subclasses still take the guarded path.
        opaque = self._executor.in_opaque_region
        if opaque and all(tensor_type is torch.Tensor for tensor_type in types):
            return func(*args, **(kwargs or {}))
        if self._replay_framework_ops and not opaque:
            return self._executor.dispatch(func, args, kwargs)
        if inside_fsdp_runtime():
            kwargs = kwargs or {}
            # Native redispatch may report Tensor itself (notably for detach).
            # Neither that case nor an empty set can contain a ShadowTensor.
            native_inputs = (
                all(tensor_type is torch.Tensor for tensor_type in types)
                or not has_managed_inputs(args, kwargs)
            )
            if native_inputs and not self._record_framework_ops:
                return func(*args, **kwargs)
            return dispatch_framework_read(self._executor, func, args, kwargs, native_inputs=native_inputs)
        return self._executor.dispatch(func, args, kwargs)


class OffloadSession:
    """Manage warmup and replay for activation residency control.

    Holds lifecycle state and a single active executor (:class:`WarmupExecutor`
    or :class:`ReplayExecutor`). The executor is switched during the
    warmup → replay transition in :meth:`__exit__`.
    """

    def __init__(self, config: OffloadConfig) -> None:
        """Initialize warmup execution and the configured residency planner."""
        self.mode: str = "warmup"

        self.planner = config.planner if config.planner is not None else PeakResidencyPlanner()
        self.residency_manager = ResidencyManager(
            max_host_bytes=config.max_offload_activation_mb * 1024**2,
        )
        self.executor: WarmupExecutor | ReplayExecutor = WarmupExecutor(
            residency_manager=self.residency_manager,
            memory_limit_bytes=config.max_resident_activation_mb * 1024**2,
        )

        self._dispatch_context = None
        self._session_token = None
        self._native_replay = False
        self._saved_recorder = None
        self._saved_replay = None
        self._saved_context = None

    # ------------------------------------------------------------------
    # Active session lookup (used by skip_offload decorator)
    # ------------------------------------------------------------------

    @staticmethod
    def get_active() -> OffloadSession | None:
        """Return the currently active OffloadSession, or ``None``."""
        session = _active_session.get()
        if session is not None:
            return session
        # Autograd carries dispatch modes into worker threads, but Python
        # context variables are not inherited by checkpoint recomputation.
        for mode in reversed(_get_current_dispatch_mode_stack()):
            if isinstance(mode, ActivationDispatchMode):
                return mode.session
        return None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> OffloadSession:
        """Enter the runtime context."""
        self.executor.reset()

        self._session_token = _active_session.set(self)
        if self._saved_replay is not None:
            self._saved_replay.reset()
            self.executor.native_saved_replay = True
            self._saved_context = torch.autograd.graph.saved_tensors_hooks(
                self._saved_replay.pack, self._saved_replay.unpack,
            )
            self._saved_context.__enter__()
            return self
        self._native_replay = self.mode == "replay" and not self.executor.requires_dispatch
        if self._native_replay:
            return self
        self._dispatch_context = ActivationDispatchMode(self, record_framework_ops=True)
        self._dispatch_context.__enter__()
        if self.mode == "warmup":
            self._saved_recorder = SavedTensorRecorder(self.executor)
            pack, unpack = self._saved_recorder.pack, self._saved_recorder.unpack
        else:
            pack, unpack = _pack_native, _unpack_native
        self._saved_context = torch.autograd.graph.saved_tensors_hooks(pack, unpack)
        self._saved_context.__enter__()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None,
        exc_val: BaseException | None, exc_tb: TracebackType | None,
    ) -> bool:
        """Exit the runtime context."""
        if self._saved_context is not None:
            self._saved_context.__exit__(exc_type, exc_val, exc_tb)
            self._saved_context = None
        if self._dispatch_context is not None:
            self._dispatch_context.__exit__(exc_type, exc_val, exc_tb)
            self._dispatch_context = None
        if self._session_token is not None:
            _active_session.reset(self._session_token)
            self._session_token = None

        if self._saved_replay is not None:
            self.executor.native_saved_replay = False
            if exc_type is not None:
                self.residency_manager.sync_all_transfers()
                self.executor.reset()
                self._saved_replay.reset()
            else:
                try:
                    self._saved_replay.finish()
                    self.residency_manager.wait_for_transfers()
                except BaseException:
                    self.residency_manager.sync_all_transfers()
                    self.executor.reset()
                    self._saved_replay.reset()
                    raise
            return False

        if self._native_replay:
            return False

        if exc_type is not None:
            self.residency_manager.sync_all_transfers()
            self.executor.reset()
            return False

        self.residency_manager.wait_for_transfers()

        if self.mode == "warmup":
            trace, guide = self.executor.finish()
            if self._saved_recorder is not None:
                SavedTensorReplay.add_planning_hints(self._saved_recorder, trace)
            schedule = self.planner.build(trace)
            if self._saved_recorder is not None:
                self._saved_replay = SavedTensorReplay.from_trace(
                    self.residency_manager, self._saved_recorder, trace, schedule,
                )
            # Native hooks own the complete replay contract. Retaining the
            # operator guide and its binding dictionaries only adds host memory
            # and garbage-collector traversal to every subsequent iteration.
            replay_guide = guide if self._saved_replay is None else []
            self.executor = ReplayExecutor(self.residency_manager, schedule, replay_guide)
            self._saved_recorder = None
            self.mode = "replay"
            if not self.executor.requires_dispatch:
                logger.info("HyperOffload plan has no transfers; subsequent steps use native execution")
        else:
            self.executor.finish()

        return False
