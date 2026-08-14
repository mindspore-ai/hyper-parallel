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
"""Optional PyTorch profiler callback for short training investigations."""

from typing import Any

from hyper_models.components.utils import helper

from .base import Callback, TrainerState


class ProfilingCallback(Callback):
    """Record a bounded CPU and accelerator trace on one distributed rank."""

    def __init__(self, trainer: Any) -> None:
        """Initialize the callback from ``TrainerConfig.profiling``.

        Args:
            trainer: Trainer that owns the callback lifecycle.

        Raises:
            ValueError: If an enabled profiling window or rank is invalid.
        """
        super().__init__(trainer)
        config = trainer.config.profiling
        self.enabled = config.enabled and trainer.global_rank == config.rank
        self.config = config
        self.profiler = None
        if not config.enabled:
            return
        if config.start_step < 1:
            raise ValueError("profiling.start_step must be at least 1")
        if config.end_step <= config.start_step:
            raise ValueError("profiling.end_step must be greater than profiling.start_step")
        if config.rank < 0 or config.rank >= trainer.world_size:
            raise ValueError(
                f"profiling.rank must be in [0, {trainer.world_size}), but got {config.rank}"
            )

    def on_train_begin(self, state: TrainerState, **kwargs: Any) -> None:
        """Create and start the profiler on the configured rank."""
        del state, kwargs
        if not self.enabled:
            return
        self.profiler = helper.create_profiler(
            start_step=self.config.start_step,
            end_step=self.config.end_step,
            trace_dir=self.config.trace_dir,
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
            with_modules=self.config.with_modules,
            global_rank=self.trainer.global_rank,
        )
        self.profiler.start()

    def on_step_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Advance the profiler schedule after one complete optimizer step."""
        del state, kwargs
        if self.profiler is not None:
            self.profiler.step()

    def on_train_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Stop the profiler and flush any pending trace output."""
        del state, kwargs
        if self.profiler is not None:
            self.profiler.stop()
            self.profiler = None
