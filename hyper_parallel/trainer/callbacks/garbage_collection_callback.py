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
from typing import Any

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
        self.empty_cache_steps = training_config.empty_cache_steps

    def on_step_end(self, state: TrainerState, **kwargs: Any) -> None:
        """Run configured cleanup operations after an optimizer step."""
        del kwargs
        if self.gc_steps > 0 and state.global_step % self.gc_steps == 0:
            gc.collect()
        if (
            self.empty_cache_steps > 0
            and state.global_step % self.empty_cache_steps == 0
            and get_device_type() != "cpu"
        ):
            empty_cache()
