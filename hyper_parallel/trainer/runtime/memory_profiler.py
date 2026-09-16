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
"""Process-local allocator memory profiler used by Trainer runtimes."""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Optional

from hyper_parallel.trainer.config.training import MemoryConfig
from hyper_parallel.trainer.runtime.device import get_device_type, get_torch_device


logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Manage one allocator-history profiling session in the current process.

    Note:
        The module owns one singleton, so only one Trainer profiling session
        may be active in a process at a time.
    """

    def __init__(self) -> None:
        """Initialize an inactive profiler."""
        self.enable = False
        self.mem_info = False
        self.current_step = 0
        self.start_step = 0
        self.end_step = 0
        self.save_path = ""
        self.dump_ranks: list[int] = []
        self.stacks = "all"
        self.max_entries = sys.maxsize
        self.global_rank = 0
        self.tp_rank = 0
        self.dp_rank = 0
        self._session_active = False
        self._history_started = False
        self._device_api: Any = None
        self._memory_api: Any = None

    def reset(
            self,
            config: Optional[MemoryConfig],
            *,
            global_rank: int,
            tp_rank: int,
            dp_rank: int,
    ) -> None:
        """Replace the process-local session with a Trainer configuration.

        Args:
            config: Memory profiler configuration, or ``None`` to disable it.
            global_rank: Rank in the global process group.
            tp_rank: Rank in the tensor-parallel group.
            dp_rank: Rank in the data-parallel group.
        """
        self._discard_previous_session()
        if config is None:
            return

        self._validate(config)
        self.enable = config.enable
        self.mem_info = config.mem_info
        self.current_step = 0
        self.start_step = config.start_step
        self.end_step = config.end_step
        self.save_path = config.save_path
        self.dump_ranks = list(config.dump_ranks)
        self.stacks = config.stacks
        self.max_entries = sys.maxsize if config.max_entries is None else config.max_entries
        self.global_rank = global_rank
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self._session_active = self.enable or self.mem_info

        if self.enable:
            self._resolve_device_apis(require_history=True)
            # Match MindSpeed's session-relative control point: step zero runs
            # after distributed setup and before model construction.
            self.step()

    def step(self) -> None:
        """Advance the state machine before a Trainer training step starts."""
        if not self._session_active:
            return

        try:
            if self.enable:
                if self.current_step == self.start_step:
                    self._record()
                if self.current_step == self.end_step:
                    try:
                        self._dump_and_stop_history()
                    finally:
                        self.enable = False
                        if not self.mem_info and not self._history_started:
                            self._deactivate_session()

            if self.mem_info:
                self._log_memory_info()
        except BaseException:
            self.abort()
            raise
        finally:
            self.current_step += 1

        if not self.enable and not self.mem_info:
            self._deactivate_session()

    def stop(self) -> None:
        """Dump an active partial window and end a normal training session."""
        if not self._session_active and not self._history_started:
            return

        try:
            try:
                if self._history_started:
                    self._dump_and_stop_history()
            finally:
                self.enable = False
                self.mem_info = False
                if not self._history_started:
                    self._deactivate_session()
        except BaseException:
            self.abort()
            raise

    def abort(self) -> None:
        """Stop allocator history without dumping or masking a Trainer error."""
        self.enable = False
        self.mem_info = False
        try:
            self._stop_history()
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Failed to stop allocator memory history while aborting training.")
            return

        self._deactivate_session()

    def _discard_previous_session(self) -> None:
        if self._history_started:
            self._stop_history()
        self._deactivate_session()

    def _deactivate_session(self) -> None:
        self.enable = False
        self.mem_info = False
        self._session_active = False
        self._device_api = None
        self._memory_api = None

    @staticmethod
    def _validate(config: MemoryConfig) -> None:
        MemoryProfiler._validate_bool(config.enable, "enable")
        MemoryProfiler._validate_bool(config.mem_info, "mem_info")
        MemoryProfiler._validate_steps(config.start_step, config.end_step)
        MemoryProfiler._validate_save_path(config.save_path)
        MemoryProfiler._validate_dump_ranks(config.dump_ranks)
        MemoryProfiler._validate_stacks(config.stacks)
        MemoryProfiler._validate_max_entries(config.max_entries)

    @staticmethod
    def _validate_bool(value: bool, name: str) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"memory.{name} must be a bool")

    @staticmethod
    def _validate_steps(start_step: int, end_step: int) -> None:
        for name, value in (("start_step", start_step), ("end_step", end_step)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"memory.{name} must be an int")
        if start_step < 0:
            raise ValueError("memory.start_step must be non-negative")
        if end_step < start_step:
            raise ValueError("memory.end_step must be greater than or equal to memory.start_step")

    @staticmethod
    def _validate_save_path(save_path: str) -> None:
        if not isinstance(save_path, str) or not save_path:
            raise ValueError("memory.save_path must be a non-empty string")

    @staticmethod
    def _validate_dump_ranks(dump_ranks: list[int]) -> None:
        if not isinstance(dump_ranks, list):
            raise ValueError("memory.dump_ranks must be a list of non-negative integers")
        if any(
                isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
                for rank in dump_ranks
        ):
            raise ValueError("memory.dump_ranks must be a list of non-negative integers")

    @staticmethod
    def _validate_stacks(stacks: str) -> None:
        if stacks not in ("python", "all"):
            raise ValueError("memory.stacks must be 'python' or 'all'")

    @staticmethod
    def _validate_max_entries(max_entries: Optional[int]) -> None:
        if max_entries is not None and (
                isinstance(max_entries, bool)
                or not isinstance(max_entries, int)
                or max_entries <= 0
        ):
            raise ValueError("memory.max_entries must be a positive integer or None")

    def _record(self) -> None:
        self._history_started = True
        try:
            self._memory_api._record_memory_history(
                stacks=self.stacks,
                max_entries=self.max_entries,
            )
        except Exception:
            # The private allocator API does not expose whether a failed call
            # enabled history, so make a best-effort stop before propagating.
            self.abort()
            raise

    def _dump_and_stop_history(self) -> None:
        try:
            self._dump()
        finally:
            self._stop_history()

    def _dump(self) -> None:
        if self.global_rank not in self.dump_ranks:
            return
        os.makedirs(self.save_path, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d-%H-%M")
        file_path = os.path.join(
            self.save_path,
            f"snapshot_{timestamp}_{self.global_rank}.pickle",
        )
        self._memory_api._dump_snapshot(file_path)
        logger.info("Memory snapshot dumped to %s", file_path)

    def _stop_history(self) -> None:
        if not self._history_started:
            return
        self._memory_api._record_memory_history(enabled=None)
        self._history_started = False

    def _log_memory_info(self) -> None:
        self._resolve_device_apis(require_history=False)
        logger.info(
            "Memory usage step=%s global_rank=%s tp_rank=%s dp_rank=%s "
            "max_memory_reserved=%s max_memory_allocated=%s",
            self.current_step,
            self.global_rank,
            self.tp_rank,
            self.dp_rank,
            self._device_api.max_memory_reserved(),
            self._device_api.max_memory_allocated(),
        )
        self._device_api.reset_peak_memory_stats()

    def _resolve_device_apis(self, *, require_history: bool) -> None:
        if self._device_api is None:
            device_type = get_device_type()
            if device_type not in ("cuda", "npu"):
                raise RuntimeError(
                    "memory profiling requires a CUDA or NPU device, "
                    f"but the active device type is {device_type!r}"
                )
            self._device_api = get_torch_device()

        if not require_history or self._memory_api is not None:
            return
        self._memory_api = getattr(self._device_api, "memory", None)
        required_methods = ("_record_memory_history", "_dump_snapshot")
        missing = [
            method
            for method in required_methods
            if self._memory_api is None
            or not callable(getattr(self._memory_api, method, None))
        ]
        if missing:
            raise RuntimeError(
                "memory profiling backend is missing allocator API(s): "
                + ", ".join(missing)
            )


memory_profiler = MemoryProfiler()


__all__ = ["memory_profiler"]
