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
"""Periodic Python and accelerator garbage collection callback."""

import gc
from multiprocessing.util import register_after_fork
from typing import Any, Optional

from hyper_parallel.trainer.runtime.device import empty_cache, get_device_type

from .base import Callback, TrainerState


class GarbageCollectionCallback(Callback):
    """Run Python GC and device cache cleanup on independent step cadences."""

    def __init__(self, trainer: Any) -> None:
        """Read cleanup intervals from the Trainer configuration.

        Args:
            trainer: Trainer that owns the callback lifecycle.
        """
        super().__init__(trainer)
        training_config = trainer.config.training
        self.gc_steps = training_config.gc_steps
        self.manual_gc = getattr(training_config, "manual_gc", False)
        self._gc_was_enabled: Optional[bool] = None
        if self.manual_gc and self.gc_steps <= 0:
            raise ValueError("manual_gc requires a positive gc_steps cleanup interval")
        if self.manual_gc:
            # Forked DataLoader workers do not run the parent's step callbacks.
            register_after_fork(self, type(self).close)
        self.empty_cache_steps = training_config.empty_cache_steps

    def on_train_begin(self, state: TrainerState, **kwargs: Any) -> None:
        """Save the automatic GC policy and switch to explicit collections."""
        del state, kwargs
        if self.manual_gc and self._gc_was_enabled is None:
            self._gc_was_enabled = gc.isenabled()
            gc.disable()
            gc.collect()

    def on_step_begin(self, state: TrainerState, **kwargs: Any) -> None:
        """Collect at step boundaries inside the step timing interval."""
        del kwargs
        if self.manual_gc and state.global_step > 0 and state.global_step % self.gc_steps == 0:
            gc.collect()

    def on_step_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Run configured cleanup operations after an optimizer step."""
        del kwargs
        if not self.manual_gc and self.gc_steps > 0 and state.global_step % self.gc_steps == 0:
            gc.collect()
        if (
            self.empty_cache_steps > 0
            and state.global_step % self.empty_cache_steps == 0
            and get_device_type() != "cpu"
        ):
            empty_cache()

    def close(self) -> None:
        """Restore the original process-local automatic GC policy once."""
        if self._gc_was_enabled is not None:
            if self._gc_was_enabled:
                gc.enable()
            else:
                gc.disable()
            self._gc_was_enabled = None

    def on_train_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Restore automatic GC when training finishes normally."""
        del state, kwargs
        self.close()
